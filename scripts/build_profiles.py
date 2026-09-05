"""Aggregate per-review aspect scores into the deployable catalogue.

Cheap: it reads the output of ``score_catalog.py`` and does arithmetic. Rerun it
freely while tuning aggregation; only ``score_catalog.py`` is expensive.

Writes three small files to ``data/catalog/``, which together are everything the
API needs:

    phones.csv           (from build_catalog.py)
    phone_profiles.csv   one row per (phone, aspect)
    phone_evidence.json  example sentences per phone

The ~45 MB ``review_aspects.csv`` stays in ``data/processed/`` as an
intermediate. Distilling the evidence here rather than at API startup keeps the
served artefacts small enough to ship and to read.

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

from ml.catalog.evidence import DEFAULT_PER_PHONE, select_evidence
from ml.catalog.profiles import MIN_MENTIONS, build_profiles
from ml.recommender.price import bounds_from, price_to_score
from ml.recommender.similarity import AXES

PROCESSED = REPO_ROOT / "data" / "processed"
CATALOG = REPO_ROOT / "data" / "catalog"
SYNTHETIC = REPO_ROOT / "data" / "synthetic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS)
    parser.add_argument("--evidence-per-phone", type=int, default=DEFAULT_PER_PHONE)
    parser.add_argument("--scores", type=Path, default=PROCESSED / "review_aspects.csv")
    parser.add_argument("--catalog-dir", type=Path, default=CATALOG)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Merge the simulated 2025-2026 phones from data/synthetic/ into the "
        "catalogue, flagged simulated=1. Their scores come from generated text, "
        "so they are excluded from the reliability gate and badged in the UI.",
    )
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"MISSING {args.scores} -- run scripts/score_catalog.py first")
        return 1

    phones_path = args.catalog_dir / "phones.csv"
    if not phones_path.exists():
        print(f"MISSING {phones_path} -- run scripts/build_catalog.py first")
        return 1

    scored = pd.read_csv(args.scores)
    phones = pd.read_csv(phones_path)

    # This script WRITES phones.csv, so a second --include-synthetic run would
    # otherwise read back its own output and append the same simulated phones
    # again: 236 rows became 261 with 25 duplicate keys, and every downstream
    # pivot then failed on "Index contains duplicate entries". Strip them here,
    # before anything -- including the price rows below -- consumes `phones`.
    if "simulated" in phones.columns:
        phones = phones[phones["simulated"].fillna(0) == 0].copy()
    phones = phones[~phones["model_key"].astype(str).str.startswith("sim:")].copy()
    phones["simulated"] = 0
    print(f"Loaded {len(scored):,} aspect rows for {scored.model_key.nunique()} phones")

    profiles, params = build_profiles(scored, min_mentions=args.min_mentions)

    print("\nShrinkage (empirical Bayes; large k = phones barely differ)")
    for entry in params:
        if entry.aspect in AXES or entry.aspect == "overall":
            print(entry.summary())

    # --- the Price axis, from listed price -------------------------------
    priced = phones[phones["price"] > 0]
    # Percentile anchors, not the extremes: one $14.99 feature phone otherwise
    # stretches the log scale until the middle half of the catalogue occupies
    # under 2 points of the 9 available.
    cheapest, dearest = bounds_from(priced["price"])
    price_scores = np.round(price_to_score(priced["price"].to_numpy(), cheapest, dearest), 3)
    price_rows = pd.DataFrame(
        {
            "model_key": priced["model_key"],
            "aspect": "price",
            "score": price_scores,
            # Identical by construction: there is nothing to shrink, because the
            # listed price is a recorded fact rather than an estimate.
            "raw_score": price_scores,
            "mentions": priced["reviews_total"].values,
            "mean_confidence": None,
        }
    )
    profiles = pd.concat(
        [profiles[profiles["aspect"] != "price"], price_rows], ignore_index=True
    )
    print(
        f"\nPrice axis from listed price: ${cheapest:,.0f} -> 10.0, "
        f"${dearest:,.0f} -> 1.0  (5th-95th percentile; outside clips)"
    )

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

    # --- simulated phones, kept separable at every step -------------------
    #
    # Merged here rather than in build_catalog.py so that `data/processed/
    # review_aspects.csv` -- the file the reliability gate reads -- never
    # contains a generated row. The gate measures whether real per-phone scores
    # are stable; including simulated ones would be measuring the generator.
    if args.include_synthetic:
        syn_phones_path = SYNTHETIC / "phones_synthetic.csv"
        syn_scores_path = SYNTHETIC / "review_aspects_synthetic.csv"
        if not syn_phones_path.exists() or not syn_scores_path.exists():
            print(
                f"\nMISSING synthetic inputs. Run:\n"
                f"  python scripts/generate_synthetic.py\n"
                f"  python scripts/score_catalog.py --phones {syn_phones_path} "
                f"--reviews {SYNTHETIC / 'phone_reviews_synthetic.csv'} "
                f"--out {syn_scores_path} --restart"
            )
            return 1

        syn_phones = pd.read_csv(syn_phones_path)
        syn_scored = pd.read_csv(syn_scores_path)
        syn_profiles, _ = build_profiles(syn_scored, min_mentions=args.min_mentions)

        # Price bounds stay derived from the REAL catalogue. Recomputing them
        # over the union would silently move every real phone's price score
        # because 2025 flagships cost more than 2019 stock -- the simulated
        # rows would be editing the real ones. Anything outside the real range
        # clips, which is what price_to_score already does.
        syn_price = pd.DataFrame(
            {
                "model_key": syn_phones["model_key"],
                "aspect": "price",
                "score": np.round(
                    price_to_score(syn_phones["price"].to_numpy(), cheapest, dearest), 3
                ),
                "raw_score": np.round(
                    price_to_score(syn_phones["price"].to_numpy(), cheapest, dearest), 3
                ),
                "mentions": syn_phones["reviews_total"].values,
                "mean_confidence": None,
            }
        )
        syn_profiles = pd.concat(
            [syn_profiles[syn_profiles["aspect"] != "price"], syn_price], ignore_index=True
        )

        profiles = pd.concat([profiles, syn_profiles], ignore_index=True)
        phones = pd.concat([phones, syn_phones], ignore_index=True)
        phones["simulated"] = phones["simulated"].fillna(0).astype(int)

        syn_complete = (
            syn_profiles[syn_profiles["aspect"].isin(AXES)]
            .pivot(index="model_key", columns="aspect", values="score")
            .dropna()
        )
        print(
            f"\n+ {len(syn_phones)} SIMULATED phones merged "
            f"({len(syn_complete)} rankable on all five axes)"
        )
        print("  Scores derived from generated text. Excluded from the reliability")
        print("  gate, flagged simulated=1 in phones.csv, badged in the UI.")
        scored = pd.concat([scored, syn_scored], ignore_index=True)

    # --- evidence ---------------------------------------------------------
    evidence = select_evidence(scored, per_phone=args.evidence_per_phone)
    sentences = sum(len(items) for items in evidence.values())
    print(f"\nEvidence: {sentences:,} sentences for {len(evidence)} phones")

    args.catalog_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = args.catalog_dir / "phone_profiles.csv"
    evidence_path = args.catalog_dir / "phone_evidence.json"
    profiles.to_csv(profiles_path, index=False)
    phones.to_csv(args.catalog_dir / "phones.csv", index=False)
    evidence_path.write_text(json.dumps(evidence, indent=1), encoding="utf-8")

    (args.catalog_dir / "shrinkage.json").write_text(
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

    size = evidence_path.stat().st_size / 1024
    print(f"\nWrote {profiles_path}")
    print(f"      {evidence_path}  ({size:,.0f} KB)")
    print("Next: python scripts/evaluate_recommender.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
