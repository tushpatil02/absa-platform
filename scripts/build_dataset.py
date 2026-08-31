"""Build the ABSA training datasets from raw M-ABSA files.

Pipeline::

    data/raw/mabsa/{phone,laptop}/{train,dev,test}.txt
        -> parse      (ml.preprocessing.parse)
        -> clean      (ml.preprocessing.clean)
        -> transform  (ml.preprocessing.transform)   86+108 categories -> 12 aspects
        -> dedup/split(ml.preprocessing.split)       grouped, leakage-asserted
        -> data/processed/

Outputs
-------
``asc_{split}.csv``
    One row per ``(review, aspect)``: the sentiment task.
``acd_{split}.csv``
    One row per review with 12 binary aspect columns: the detection task.
``build_report.json``
    Every count produced along the way, for docs and for the tests.

Usage::

    python scripts/build_dataset.py
    python scripts/build_dataset.py --domains phone
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from ml.preprocessing.clean import clean_text, is_usable  # noqa: E402
from ml.preprocessing.parse import ParseReport, parse_file  # noqa: E402
from ml.preprocessing.split import (  # noqa: E402
    assert_no_leakage,
    deduplicate_across_splits,
)
from ml.preprocessing.transform import (  # noqa: E402
    TransformReport,
    load_taxonomy,
    transform,
)

RAW_DIR = REPO_ROOT / "data" / "raw" / "mabsa"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TAXONOMY_PATH = REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml"

DEFAULT_DOMAINS = ("phone", "laptop")
SPLITS = ("train", "dev", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"Taxonomy: {len(taxonomy.aspect_ids)} aspects -> {', '.join(taxonomy.aspect_ids)}")

    # ---------------------------------------------------------------- parse --
    parse_report = ParseReport()
    raw_reviews = []
    for domain in args.domains:
        for split in SPLITS:
            path = RAW_DIR / domain / f"{split}.txt"
            if not path.exists():
                print(f"  MISSING {path} -- run scripts/download_data.py first", file=sys.stderr)
                return 1
            raw_reviews.extend(parse_file(path, domain, split, parse_report))
    print(f"\n[parse]     {parse_report.summary()}")

    # ---------------------------------------------------------------- clean --
    cleaned: dict[int, str] = {}
    dropped_unusable = 0
    kept_reviews = []
    for review in raw_reviews:
        text = clean_text(review.text)
        if not is_usable(text):
            dropped_unusable += 1
            continue
        cleaned[id(review)] = text
        kept_reviews.append(review)
    print(
        f"[clean]     kept={len(kept_reviews)}  dropped_unusable={dropped_unusable}  "
        f"(punctuation/negation/casing preserved)"
    )

    # ------------------------------------------------------------ transform --
    transform_report = TransformReport()
    try:
        pairs = transform(kept_reviews, taxonomy, transform_report, cleaned_text=cleaned)
    except KeyError as exc:
        print(f"\nTaxonomy error: {exc}", file=sys.stderr)
        return 1
    print(f"[transform] {transform_report.summary()}")

    # ---------------------------------------------------------- dedup/split --
    pairs, split_report = deduplicate_across_splits(pairs)
    print(f"[split]     {split_report.summary()}")

    assert_no_leakage(pairs)
    print("[split]     leakage assertion PASSED (no review id or text spans splits)")

    # --------------------------------------------------------------- write ---
    args.out.mkdir(parents=True, exist_ok=True)

    asc = pd.DataFrame(
        {
            "review_id": [p.review_id for p in pairs],
            "text": [p.text for p in pairs],
            "aspect": [p.aspect for p in pairs],
            "polarity": [p.polarity for p in pairs],
            "label": [taxonomy.polarity_to_id[p.polarity] for p in pairs],
            "domain": [p.domain for p in pairs],
            "split": [p.split for p in pairs],
            "terms": ["|".join(p.terms) for p in pairs],
        }
    )

    # ACD: pivot to one row per review with a binary column per aspect.
    aspect_columns = list(taxonomy.aspect_ids)
    by_review: dict[str, dict[str, object]] = {}
    for pair in pairs:
        row = by_review.setdefault(
            pair.review_id,
            {
                "review_id": pair.review_id,
                "text": pair.text,
                "domain": pair.domain,
                "split": pair.split,
                **{name: 0 for name in aspect_columns},
            },
        )
        row[pair.aspect] = 1
    acd = pd.DataFrame(list(by_review.values()), columns=["review_id", "text", "domain", "split", *aspect_columns])

    for split in SPLITS:
        asc_split = asc[asc["split"] == split]
        acd_split = acd[acd["split"] == split]
        asc_split.to_csv(args.out / f"asc_{split}.csv", index=False, encoding="utf-8")
        acd_split.to_csv(args.out / f"acd_{split}.csv", index=False, encoding="utf-8")
        print(f"[write]     {split:<5} asc={len(asc_split):>5} rows   acd={len(acd_split):>5} rows")

    # -------------------------------------------------------------- report ---
    aspect_stats = {}
    for aspect in aspect_columns:
        subset = asc[asc["aspect"] == aspect]
        counts = subset["polarity"].value_counts().to_dict()
        aspect_stats[aspect] = {
            "total": int(len(subset)),
            "positive": int(counts.get("positive", 0)),
            "negative": int(counts.get("negative", 0)),
            "neutral": int(counts.get("neutral", 0)),
        }

    report = {
        "domains": list(args.domains),
        "taxonomy": {
            "n_aspects": len(aspect_columns),
            "aspects": aspect_columns,
            "polarities": list(taxonomy.polarities),
        },
        "parse": {
            "total_lines": parse_report.total_lines,
            "parsed": parse_report.parsed,
            "blank_lines": parse_report.blank_lines,
            "skipped": parse_report.skipped,
            "invalid_polarity": len(parse_report.invalid_polarity),
        },
        "clean": {"kept": len(kept_reviews), "dropped_unusable": dropped_unusable},
        "transform": {
            "reviews_in": transform_report.reviews_in,
            "triplets_in": transform_report.triplets_in,
            "duplicate_triplets_removed": transform_report.duplicate_triplets_removed,
            "dropped_by_taxonomy": transform_report.triplets_dropped_by_taxonomy,
            "pairs_out": transform_report.pairs_out,
            "reviews_without_aspects": transform_report.reviews_without_aspects,
            "ties_resolved": transform_report.ties_resolved,
        },
        "split": {
            "pairs": split_report.counts,
            "reviews": split_report.review_counts,
            "duplicate_text_groups": split_report.duplicate_groups,
            "cross_split_duplicates_resolved": split_report.cross_split_duplicates_resolved,
            "rows_dropped_as_duplicates": split_report.rows_dropped_as_duplicates,
        },
        "polarity_distribution": {
            key: int(value) for key, value in asc["polarity"].value_counts().to_dict().items()
        },
        "aspect_distribution": aspect_stats,
    }
    (args.out / "build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------- summary ---
    print(f"\n{'ASPECT':<18}{'pairs':>7}{'pos':>7}{'neg':>7}{'neu':>6}{'%neg':>7}")
    print("-" * 52)
    for aspect, stats in sorted(aspect_stats.items(), key=lambda kv: -kv[1]["total"]):
        total = stats["total"]
        share = 100 * stats["negative"] / total if total else 0.0
        print(
            f"{aspect:<18}{total:>7}{stats['positive']:>7}"
            f"{stats['negative']:>7}{stats['neutral']:>6}{share:>6.1f}%"
        )
    print("-" * 52)
    print(f"{'TOTAL':<18}{len(asc):>7}")

    distribution = collections.Counter(asc["polarity"])
    total = len(asc)
    shares = "  ".join(
        f"{name} {100 * distribution[name] / total:.1f}%" for name in taxonomy.polarities
    )
    print(f"polarity: {shares}   (neutral n={distribution['neutral']})")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
