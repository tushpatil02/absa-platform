"""The gate: is the per-phone signal real?

Run this before trusting anything the recommender says. It answers two
questions that can both fail, and prints a verdict rather than a dashboard.

1. **Split-half reliability.** Split each phone's reviews at random, build the
   profile twice, correlate across phones. Low agreement means the score
   measures which reviews landed in which half, not the phone.

2. **Does it beat the stars?** Amazon already ships an average rating. If the
   aspect scores track it almost perfectly, the pipeline is an expensive
   re-derivation of a column that was already in the CSV.

A null simulation is printed alongside, showing how much apparent spread
identical phones produce for free. A real result has to clear it.

    python scripts/evaluate_recommender.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.reliability import null_spread, split_half_reliability, star_baseline
from ml.recommender.similarity import AXES

PROCESSED = REPO_ROOT / "data" / "processed"

# Price is excluded: it comes from the listed price, which is a recorded fact
# rather than an estimate, so split-half reliability is not meaningful for it.
SENTIMENT_AXES = tuple(axis for axis in AXES if axis != "price")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mentions", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores", type=Path, default=PROCESSED / "review_aspects.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED / "reliability.json")
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"MISSING {args.scores} -- run scripts/score_catalog.py first")
        return 1

    scored = pd.read_csv(args.scores)
    phones = pd.read_csv(PROCESSED / "phones.csv")

    print("=" * 74)
    print("1. SPLIT-HALF RELIABILITY  (Spearman-Brown corrected)")
    print("=" * 74)
    print("   Two independent halves of each phone's reviews must agree.")
    print("   >=0.80 strong   >=0.60 usable   >=0.40 weak   below that, noise\n")

    reliabilities = []
    for axis in SENTIMENT_AXES:
        result = split_half_reliability(
            scored, axis, min_mentions=args.min_mentions, repeats=args.repeats, seed=args.seed
        )
        reliabilities.append(result)
        print(result.summary())

    print()
    print("=" * 74)
    print("2. BEATING THE STAR RATING")
    print("=" * 74)
    print("   R2 near 1 means the aspect adds nothing over `avg_rating`.\n")

    baselines = []
    for axis in SENTIMENT_AXES:
        result = star_baseline(scored, phones, axis, min_mentions=args.min_mentions)
        baselines.append(result)
        print(result.summary())

    print()
    print("=" * 74)
    print("3. NULL BASELINE  (what identical phones produce for free)")
    print("=" * 74)
    typical = scored.groupby(["model_key", "aspect"]).size().median()
    null = null_spread(
        n_phones=int(scored.model_key.nunique()),
        reviews_per_phone=int(typical),
        seed=args.seed,
    )
    print(
        f"   {scored.model_key.nunique()} identical phones, {int(typical)} reviews each:\n"
        f"     range {null['range']:.2f}   sd {null['std']:.2f}   "
        f"p90-p10 {null['p90_minus_p10']:.2f}"
    )
    print("   Any observed spread smaller than this is not evidence of anything.")

    # --- verdict ---------------------------------------------------------
    print()
    print("=" * 74)
    usable = [r for r in reliabilities if r.verdict in ("strong", "usable")]
    redundant = [b for b in baselines if b.r_squared >= 0.90]

    if not usable:
        print("VERDICT: FAIL -- no aspect is reliable enough to rank on.")
        print("  The sliders would be reordering noise. Do not ship the recommender")
        print("  on these profiles; increase reviews per phone or improve the model.")
    elif redundant:
        names = ", ".join(b.aspect for b in redundant)
        print(f"VERDICT: PARTIAL -- {len(usable)}/{len(SENTIMENT_AXES)} axes reliable,")
        print(f"  but these merely restate the star rating: {names}")
    else:
        names = ", ".join(r.aspect for r in usable)
        print(f"VERDICT: PASS -- {len(usable)}/{len(SENTIMENT_AXES)} axes reliable ({names}),")
        print("  and none is redundant with the star rating.")
    print("=" * 74)

    args.out.write_text(
        json.dumps(
            {
                "min_mentions": args.min_mentions,
                "repeats": args.repeats,
                "seed": args.seed,
                "reliability": [r.__dict__ | {"verdict": r.verdict} for r in reliabilities],
                "star_baseline": [b.__dict__ | {"verdict": b.verdict} for b in baselines],
                "null_spread": null,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
