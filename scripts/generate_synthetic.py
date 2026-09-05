"""Generate labelled synthetic reviews for 2025-2026 phones.

The Amazon corpus was scraped 2019-12-26 and no permissively-licensed
replacement covering 2025+ exists -- see ``ml/synthetic/reviews.py`` for what
the search turned up and why the Kaggle sets advertising "2025" and "2026" are
unusable.

Writes two things, kept strictly apart:

  data/synthetic/asc_synthetic.csv   training rows, same schema as asc_train.csv
  data/synthetic/acd_synthetic.csv   training rows, same schema as acd_train.csv
  data/synthetic/phones_synthetic.csv   catalogue entries, flagged simulated=1

Nothing here is ever written into `data/processed/` or `data/catalog/` by this
script. Mixing synthetic rows into the real splits would make every number in
docs/model.md unverifiable, so the merge happens explicitly and visibly at
train time instead -- see scripts/compare_synthetic.py.

    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n-reviews 8000 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.preprocessing.transform import load_taxonomy
from ml.synthetic.reviews import ASPECTS, generate, phone_quality, uniqueness

OUT_DIR = REPO_ROOT / "data" / "synthetic"

# Phones released 2024-2026, with an indicative price band.
#
# The NAMES are real; the PRICES are nominal and the REVIEWS are generated.
# That combination is only defensible because every downstream surface flags
# these rows as simulated -- phones_synthetic.csv carries simulated=1, the API
# passes it through, and the UI badges it. If that flagging is ever removed,
# this file becomes a set of fabricated claims about real purchasable products.
PHONES_2025: list[tuple[str, str, float]] = [
    ("Samsung Galaxy S25", "Samsung", 799.0),
    ("Samsung Galaxy S25 Plus", "Samsung", 999.0),
    ("Samsung Galaxy S25 Ultra", "Samsung", 1299.0),
    ("Samsung Galaxy S25 FE", "Samsung", 649.0),
    ("Samsung Galaxy A56", "Samsung", 499.0),
    ("Apple iPhone 16", "Apple", 799.0),
    ("Apple iPhone 16 Pro", "Apple", 999.0),
    ("Apple iPhone 16 Pro Max", "Apple", 1199.0),
    ("Apple iPhone 16e", "Apple", 599.0),
    ("Apple iPhone 17", "Apple", 849.0),
    ("Apple iPhone 17 Pro", "Apple", 1049.0),
    ("Google Pixel 9", "Google", 799.0),
    ("Google Pixel 9 Pro", "Google", 999.0),
    ("Google Pixel 9a", "Google", 499.0),
    ("Google Pixel 10", "Google", 799.0),
    ("Google Pixel 10 Pro", "Google", 999.0),
    ("OnePlus 13", "OnePlus", 899.0),
    ("OnePlus 13R", "OnePlus", 599.0),
    ("Xiaomi 15", "Xiaomi", 899.0),
    ("Xiaomi 15 Ultra", "Xiaomi", 1199.0),
    ("Xiaomi Redmi Note 14 Pro", "Xiaomi", 349.0),
    ("Nothing Phone 3", "Nothing", 799.0),
    ("Nothing Phone 3a", "Nothing", 379.0),
    ("Motorola Edge 60 Pro", "Motorola", 599.0),
    ("Motorola Razr 60 Ultra", "Motorola", 1299.0),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reviews", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--min-uniqueness",
        type=float,
        default=0.90,
        help="Fail rather than write if distinct-text share falls below this. "
        "The datasets this module replaces score 0.0022.",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    names = [name for name, _, _ in PHONES_2025]
    reviews = generate(names, args.n_reviews, seed=args.seed)

    unique = uniqueness(reviews)
    mixed = sum(review.is_mixed for review in reviews) / len(reviews)
    print(f"Generated {len(reviews):,} reviews across {len(names)} phones")
    print(f"  distinct text : {unique:.4f}   (floor {args.min_uniqueness})")
    print(f"  mixed polarity: {mixed:.1%}")

    if unique < args.min_uniqueness:
        print(
            f"\nFAILED: {unique:.4f} distinct is below the floor. A generator that "
            "repeats itself is worse than none -- it looks like data. Widen the "
            "slot vocabularies in ml/synthetic/reviews.py."
        )
        return 1

    # --- ASC rows: one per (review, aspect) ------------------------------
    asc = pd.DataFrame(
        [
            {
                "review_id": review.review_id,
                "text": review.text,
                "aspect": aspect,
                "polarity": polarity,
                "domain": "synthetic",
                "split": "synthetic",
                "terms": "",
            }
            for review in reviews
            for aspect, polarity in review.labels.items()
        ]
    )

    # --- ACD rows: one per review, multi-label ----------------------------
    acd_rows = []
    for review in reviews:
        row = {
            "review_id": review.review_id,
            "text": review.text,
            "domain": "synthetic",
            "split": "synthetic",
        }
        for aspect in taxonomy.aspect_ids:
            row[aspect] = int(aspect in review.labels)
        acd_rows.append(row)
    acd = pd.DataFrame(acd_rows)

    # --- per-review rows, the shape score_catalog.py consumes -------------
    review_rows = pd.DataFrame(
        [
            {
                "review_id": review.review_id,
                "model_key": f"sim:{review.phone.lower()}",
                "asin": review.review_id,
                "name": "simulated",
                "rating": None,
                "date": None,
                "verified": False,
                "title": "",
                "body": review.text,
            }
            for review in reviews
        ]
    )

    # --- catalogue entries, flagged --------------------------------------
    phones = pd.DataFrame(
        [
            {
                "model_key": f"sim:{name.lower()}",
                "name": name,
                "brand": brand,
                "price": price,
                "image": None,
                "url": None,
                "listings": 1,
                "reviews_total": sum(1 for r in reviews if r.phone == name),
                "avg_rating": None,
                "reviews_sampled": sum(1 for r in reviews if r.phone == name),
                # The flag every downstream surface keys on.
                "simulated": 1,
            }
            for name, brand, price in PHONES_2025
        ]
    )

    # The ground truth these reviews were generated from. Real data has no
    # equivalent, and it is what makes scripts/verify_recovery.py possible:
    # the pipeline's recovered scores can be checked against a known answer.
    truth = pd.DataFrame(
        [
            {"model_key": f"sim:{name.lower()}", "name": name, **phone_quality(name, args.seed)}
            for name, _, _ in PHONES_2025
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.out_dir / "ground_truth.csv", index=False)
    asc.to_csv(args.out_dir / "asc_synthetic.csv", index=False)
    review_rows.to_csv(args.out_dir / "phone_reviews_synthetic.csv", index=False)
    acd.to_csv(args.out_dir / "acd_synthetic.csv", index=False)
    phones.to_csv(args.out_dir / "phones_synthetic.csv", index=False)
    (args.out_dir / "generation_report.json").write_text(
        json.dumps(
            {
                "n_reviews": len(reviews),
                "n_asc_pairs": len(asc),
                "n_phones": len(PHONES_2025),
                "seed": args.seed,
                "distinct_text_share": round(unique, 4),
                "mixed_polarity_share": round(mixed, 4),
                "polarity_counts": asc.polarity.value_counts().to_dict(),
                "aspect_counts": asc.aspect.value_counts().to_dict(),
                "provenance": (
                    "Generated by ml/synthetic/reviews.py. Labels are exact by "
                    "construction. NEVER report accuracy measured on these rows: "
                    "that measures the generator. Training use only."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n  ASC pairs: {len(asc):,}")
    print(f"  {'aspect':<14}{'pairs':>8}   polarity mix")
    for aspect in ASPECTS:
        sub = asc[asc.aspect == aspect]
        share = sub.polarity.value_counts(normalize=True)
        mix = "  ".join(f"{p[:3]} {share.get(p, 0):.0%}" for p in ("negative", "neutral", "positive"))
        print(f"  {aspect:<14}{len(sub):>8}   {mix}")

    print(f"\nWrote {args.out_dir}")
    print("Next: python scripts/compare_synthetic.py   (does it actually help?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
