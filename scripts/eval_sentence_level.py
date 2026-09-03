"""Compare whole-review and sentence-level inference.

Answers one question: does splitting a review into sentences before running the
model recover the per-aspect opinions that whole-review inference collapses?

The benchmark is composed from held-out M-ABSA test rows -- see
``ml/evaluation/multi_sentence.py`` for how, and for what it can and cannot
show.

    .venv/Scripts/python.exe scripts/eval_sentence_level.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.multi_sentence import build_benchmark, evaluate_mode
from ml.inference.predictor import load_predictor
from ml.preprocessing.transform import load_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reviews", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "models" / "metadata" / "sentence_level.json",
    )
    args = parser.parse_args()

    pairs_path = REPO_ROOT / "data" / "processed" / "asc_test.csv"
    if not pairs_path.exists():
        print(f"MISSING {pairs_path} -- run scripts/build_dataset.py first")
        return 1

    pairs = pd.read_csv(pairs_path)
    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    predictor = load_predictor(REPO_ROOT / "models", taxonomy)
    print(f"model: {predictor.model_name}")

    report: dict[str, dict] = {"model": predictor.model_name, "variants": {}}

    for variant, strip in (("punctuated", False), ("no-terminators", True)):
        reviews = build_benchmark(
            pairs,
            n_reviews=args.n_reviews,
            seed=args.seed,
            strip_terminators=strip,
        )
        mixed = sum(review.is_mixed for review in reviews)
        print(
            f"\n[{variant}] {len(reviews)} pseudo-reviews, "
            f"{mixed} mixed ({mixed / max(len(reviews), 1):.1%}), "
            f"{sum(len(r.gold) for r in reviews)} gold aspects"
        )

        results = {}
        for by_sentence in (False, True):
            started = time.perf_counter()
            result = evaluate_mode(predictor, reviews, by_sentence=by_sentence)
            elapsed = time.perf_counter() - started
            print(f"{result.summary()}  [{elapsed:.1f}s]")
            results[result.mode] = {
                "detection_recall": round(result.detection_recall, 4),
                "sentiment_accuracy": round(result.sentiment_accuracy, 4),
                "mixed_sentiment_accuracy": round(result.mixed_sentiment_accuracy, 4),
                "uniform_sentiment_accuracy": round(result.uniform_sentiment_accuracy, 4),
                "collapsed_rate": round(result.collapsed_rate, 4),
                "n_reviews": result.n_reviews,
                "n_gold_aspects": result.n_gold_aspects,
                "n_scored": result.n_scored,
                "seconds": round(elapsed, 1),
            }

        whole = results["whole-review"]["mixed_sentiment_accuracy"]
        sentence = results["sentence"]["mixed_sentiment_accuracy"]
        delta = sentence - whole
        print(f"  mixed-review accuracy: {whole:.4f} -> {sentence:.4f}  ({delta:+.4f})")

        report["variants"][variant] = {
            "n_reviews": len(reviews),
            "n_mixed": mixed,
            "results": results,
            "mixed_delta": round(delta, 4),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
