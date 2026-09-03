"""Build the candidate phone catalogue from the Amazon corpus.

Fast and model-free: it decides which phones exist and which reviews belong to
them. Sentiment scoring is a separate pass (``scripts/score_catalog.py``), so
this can be rerun and inspected in seconds.

    python scripts/download_phones.py
    python scripts/build_catalog.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.catalog.build import CatalogConfig, CatalogReport, build_candidates

RAW_DIR = REPO_ROOT / "data" / "raw" / "amazon"
OUT_DIR = REPO_ROOT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-reviews", type=int, default=CatalogConfig.min_reviews)
    parser.add_argument("--review-cap", type=int, default=CatalogConfig.review_cap)
    parser.add_argument("--seed", type=int, default=CatalogConfig.seed)
    parser.add_argument(
        "--allow-unpriced",
        action="store_true",
        help="Keep phones with no listed price. They cannot be ranked on the Price slider.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    items_path = args.raw_dir / "20191226-items.csv"
    reviews_path = args.raw_dir / "20191226-reviews.csv"
    for path in (items_path, reviews_path):
        if not path.exists():
            print(f"MISSING {path} -- run scripts/download_phones.py first")
            return 1

    items = pd.read_csv(items_path)
    reviews = pd.read_csv(reviews_path)

    config = CatalogConfig(
        min_reviews=args.min_reviews,
        require_price=not args.allow_unpriced,
        review_cap=args.review_cap,
        seed=args.seed,
    )
    report = CatalogReport()
    phones, sampled = build_candidates(items, reviews, config, report)

    print(f"[catalog]   {report.summary()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    phones_path = args.out_dir / "phones.csv"
    reviews_out = args.out_dir / "phone_reviews.csv"
    phones.to_csv(phones_path, index=False)
    sampled[
        ["review_id", "model_key", "asin", "name", "rating", "date", "verified", "title", "body"]
    ].to_csv(reviews_out, index=False)

    (args.out_dir / "catalog_report.json").write_text(
        json.dumps(report.__dict__ | {"config": config.__dict__}, indent=2), encoding="utf-8"
    )

    print()
    print(f"{'':2}{'phone':<34}{'price':>9}{'reviews':>9}{'scored':>8}{'stars':>7}")
    print("-" * 70)
    for row in phones.head(15).itertuples(index=False):
        print(
            f"  {row.name[:33]:<34}${row.price:>8,.0f}{row.reviews_total:>9,}"
            f"{row.reviews_sampled:>8,}{row.avg_rating:>7.2f}"
        )
    print("-" * 70)
    print(f"  {len(phones)} phones, {report.reviews_sampled:,} reviews to score")
    print(
        f"  price   min ${phones.price.min():,.0f}  "
        f"median ${phones.price.median():,.0f}  max ${phones.price.max():,.0f}"
    )
    print(
        f"  reviews median {phones.reviews_total.median():.0f} per phone  "
        f"(capped at {config.review_cap} for scoring)"
    )
    print(f"\nWrote {phones_path}\n      {reviews_out}")
    print("Next: python scripts/score_catalog.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
