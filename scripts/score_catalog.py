"""Run ABSA over the catalogue's reviews and record per-review aspect results.

This is the expensive pass -- around an hour of CPU inference for 37,000
reviews. It writes **per-review** rows rather than per-phone averages, on
purpose: aggregation is cheap and there are several things to try (shrinkage,
weighting, split-half reliability), and none of them should require re-running
inference.

Resumable. Results are appended per phone and a restart skips phones already
present in the output, so an interrupted run costs minutes rather than the
whole job.

    python scripts/build_catalog.py
    python scripts/score_catalog.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.inference.predictor import EmptyReviewError, ReviewTooLongError, load_predictor
from ml.preprocessing.transform import load_taxonomy

PROCESSED = REPO_ROOT / "data" / "processed"
CATALOG = REPO_ROOT / "data" / "catalog"

FIELDS = [
    "model_key",
    "review_id",
    "aspect",
    "polarity",
    "score",
    "confidence",
    "detection_confidence",
    "mentions",
    "p_negative",
    "p_neutral",
    "p_positive",
    "rating",
    # The sentences that produced this aspect's score. Captured here because
    # inference is the expensive step: recovering them later would mean a
    # second pass over the corpus, and a sampled second pass would only ever
    # explain some phones. A score a reader cannot trace is a score they have
    # to take on trust.
    "evidence",
]

# Evidence is joined with this and truncated, so one rambling review cannot
# blow up the CSV. Most aspects are carried by one or two sentences.
EVIDENCE_SEPARATOR = " || "
MAX_EVIDENCE_CHARS = 400


def completed_phones(path: Path) -> set[str]:
    """Phones already scored, so a restart can skip them."""
    if not path.exists():
        return set()
    try:
        done = pd.read_csv(path, usecols=["model_key"])
    except (pd.errors.EmptyDataError, ValueError):
        return set()
    return set(done["model_key"].unique())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROCESSED / "review_aspects.csv")
    parser.add_argument("--phones", type=Path, default=CATALOG / "phones.csv")
    parser.add_argument("--reviews", type=Path, default=PROCESSED / "phone_reviews.csv")
    parser.add_argument("--limit-phones", type=int, default=None, help="Smoke test.")
    parser.add_argument("--restart", action="store_true", help="Ignore existing output.")
    parser.add_argument(
        "--whole-review",
        action="store_true",
        help="Score each review as one unit instead of per sentence. Kept for "
        "comparison; sentence-level is 20.6 points better on mixed reviews.",
    )
    args = parser.parse_args()

    phones_path = args.phones
    reviews_path = args.reviews
    for path in (phones_path, reviews_path):
        if not path.exists():
            print(f"MISSING {path} -- run scripts/build_catalog.py first")
            return 1

    phones = pd.read_csv(phones_path)
    reviews = pd.read_csv(reviews_path)
    reviews["body"] = reviews["body"].fillna("").astype(str)

    if args.restart and args.out.exists():
        args.out.unlink()
    done = completed_phones(args.out)
    if done:
        print(f"Resuming: {len(done)} phones already scored.")

    order = list(phones["model_key"])
    if args.limit_phones:
        order = order[: args.limit_phones]
    todo = [key for key in order if key not in done]

    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    predictor = load_predictor(REPO_ROOT / "models", taxonomy)
    print(f"model: {predictor.model_name}")
    print(f"mode : {'whole-review' if args.whole_review else 'sentence-level'}")
    print(f"todo : {len(todo)} phones, {int(reviews.model_key.isin(todo).sum()):,} reviews\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    by_phone = dict(list(reviews.groupby("model_key")))

    started = time.perf_counter()
    scored = failed = 0

    with args.out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for index, key in enumerate(todo, start=1):
            group = by_phone.get(key)
            if group is None:
                continue

            rows = []
            for review in group.itertuples(index=False):
                try:
                    result = predictor.analyze(
                        review.body, by_sentence=not args.whole_review
                    )
                except (EmptyReviewError, ReviewTooLongError):
                    # Expected for stub text and for the rare 5,000+ character
                    # review; counted rather than silently skipped.
                    failed += 1
                    continue

                for prediction in result.aspects:
                    sentiment = prediction.sentiment
                    rows.append(
                        {
                            "model_key": key,
                            "review_id": review.review_id,
                            "aspect": prediction.aspect,
                            "polarity": sentiment.polarity,
                            "score": sentiment.score,
                            "confidence": sentiment.confidence,
                            "detection_confidence": prediction.detection_confidence,
                            "mentions": prediction.mentions,
                            "p_negative": sentiment.probabilities.get("negative"),
                            "p_neutral": sentiment.probabilities.get("neutral"),
                            "p_positive": sentiment.probabilities.get("positive"),
                            "rating": review.rating,
                            "evidence": EVIDENCE_SEPARATOR.join(prediction.evidence)[
                                :MAX_EVIDENCE_CHARS
                            ],
                        }
                    )
                scored += 1

            writer.writerows(rows)
            handle.flush()

            elapsed = time.perf_counter() - started
            rate = scored / elapsed if elapsed else 0
            remaining = int(reviews.model_key.isin(todo[index:]).sum())
            eta = remaining / rate / 60 if rate else 0
            print(
                f"  [{index:>3}/{len(todo)}] {key[:38]:<40}"
                f"{len(group):>4} reviews  {rate:5.1f}/s  ETA {eta:5.1f} min",
                flush=True,
            )

    minutes = (time.perf_counter() - started) / 60
    print(f"\nScored {scored:,} reviews, {failed} unusable, in {minutes:.1f} min")
    print(f"Wrote {args.out}")
    print("Next: python scripts/build_profiles.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
