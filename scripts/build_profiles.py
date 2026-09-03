"""Aggregate per-review aspect scores into per-phone profiles.

Cheap: it reads the output of ``score_catalog.py`` and does arithmetic. Rerun
it freely while tuning aggregation; only ``score_catalog.py`` is expensive.

    python scripts/score_catalog.py
    python scripts/build_profiles.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.catalog.profiles import MIN_MENTIONS, build_profiles
from ml.recommender.price import price_to_score
from ml.recommender.similarity import AXES

PROCESSED = REPO_ROOT / "data" / "processed"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS)
    parser.add_argument("--scores", type=Path, default=PROCESSED / "review_aspects.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED / "phone_profiles.csv")
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"MISSING {args.scores} -- run scripts/score_catalog.py first")
        return 1

    scored = pd.read_csv(args.scores)
    phones = pd.read_csv(PROCESSED / "phones.csv")
    print(f"Loaded {len(scored):,} aspect rows for {scored.model_key.nunique()} phones")

    profiles, params = build_profiles(scored, min_mentions=args.min_mentions)

    print("\nShrinkage (empirical Bayes; large k = phones barely differ)")
    for entry in params:
        if entry.aspect in AXES or entry.aspect == "overall":
            print(entry.summary())

    # --- the Price axis, from listed price -------------------------------
    priced = phones[phones["price"] > 0]
    cheapest, dearest = float(priced["price"].min()), float(priced["price"].max())
    price_rows = pd.DataFrame(
        {
            "model_key": priced["model_key"],
            "aspect": "price",
            "score": np.round(price_to_score(priced["price"].to_numpy(), cheapest, dearest), 3),
            "raw_score": np.round(
                price_to_score(priced["price"].to_numpy(), cheapest, dearest), 3
            ),
            "mentions": priced["reviews_total"].values,
            "mean_confidence": None,
        }
    )
    # Replace any sentiment-derived price rows with the listed-price ones.
    profiles = pd.concat(
        [profiles[profiles["aspect"] != "price"], price_rows], ignore_index=True
    )
    print(f"\nPrice axis from listed price: ${cheapest:,.0f} -> 10.0, ${dearest:,.0f} -> 1.0")

    # --- which phones can be ranked on all five axes ----------------------
    coverage = profiles[profiles["aspect"].isin(AXES)].pivot(
        index="model_key", columns="aspect", values="score"
    )
    complete = coverage.dropna()
    print(f"\nPhones with all five axes: {len(complete)} of {len(phones)}")
    for axis in AXES:
        have = int(coverage[axis].notna().sum()) if axis in coverage else 0
        print(f"  {axis:<14}{have:>4} phones")

    print("\nScore spread across phones (shrunk)")
    print(f"  {'axis':<14}{'min':>7}{'median':>8}{'max':>7}{'sd':>7}{'range':>7}")
    for axis in AXES:
        if axis not in complete:
            continue
        column = complete[axis]
        print(
            f"  {axis:<14}{column.min():>7.2f}{column.median():>8.2f}"
            f"{column.max():>7.2f}{column.std():>7.2f}{column.max() - column.min():>7.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.out, index=False)
    (PROCESSED / "shrinkage.json").write_text(
        json.dumps(
            {
                "min_mentions": args.min_mentions,
                "price_bounds": {"cheapest": cheapest, "dearest": dearest},
                "aspects": [entry.__dict__ for entry in params],
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    print("Next: python scripts/evaluate_recommender.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
