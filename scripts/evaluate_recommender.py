"""The gate: is the per-phone signal real?

Run this before trusting anything the recommender says. It answers two
questions that can both fail, and prints a verdict rather than a dashboard.

1. **Split-half reliability.** Split each phone's reviews at random, build the
   profile twice, correlate across phones. Low agreement means the score
   measures which reviews landed in which half, not the phone.

2. **Does it beat the stars?** Amazon already ships an average rating. If the
   aspect scores track it almost perfectly, the pipeline is an expensive
   re-derivation of a column that was already in the CSV.

3. **Is the spread bigger than noise?** For each aspect, a null is simulated
   using *that aspect's* within-phone variance and review counts, showing how
   much apparent spread identical phones produce for free. A null drawn with
   some fixed variance would answer a question about a different corpus while
   looking like a check.

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

from ml.evaluation.reliability import compare_to_null, split_half_reliability, star_baseline
from ml.recommender.similarity import AXES

PROCESSED = REPO_ROOT / "data" / "processed"
CATALOG = REPO_ROOT / "data" / "catalog"

# Price is excluded: it comes from the listed price, which is a recorded fact
# rather than an estimate, so split-half reliability is not meaningful for it.
SENTIMENT_AXES = tuple(axis for axis in AXES if axis != "price")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mentions", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores", type=Path, default=PROCESSED / "review_aspects.csv")
    parser.add_argument("--phones", type=Path, default=CATALOG / "phones.csv")
    parser.add_argument("--out", type=Path, default=CATALOG / "reliability.json")
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"MISSING {args.scores} -- run scripts/score_catalog.py first")
        return 1

    scored = pd.read_csv(args.scores)
    if not args.phones.exists():
        print(f"MISSING {args.phones} -- run scripts/build_catalog.py first")
        return 1
    phones = pd.read_csv(args.phones)

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
    print("3. OBSERVED SPREAD vs A MATCHED NULL")
    print("=" * 74)
    print("   Each null uses THIS aspect's within-phone noise and review counts.")
    print("   A null drawn with some fixed variance would answer a question about")
    print("   a different corpus while looking like a check.\n")

    nulls = []
    for axis in SENTIMENT_AXES:
        result = compare_to_null(scored, axis, min_mentions=args.min_mentions, seed=args.seed)
        nulls.append(result)
        print(result.summary())

    # --- verdict ---------------------------------------------------------
    print()
    print("=" * 74)
    usable = [r for r in reliabilities if r.verdict in ("strong", "usable")]
    redundant = [b for b in baselines if b.r_squared >= 0.90]
    indistinct = [n for n in nulls if n.verdict == "NOT ABOVE NULL"]

    if indistinct:
        names = ", ".join(n.aspect for n in indistinct)
        print(f"NOTE: spread indistinguishable from noise on: {names}")

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
                "null_comparison": [
                    n.__dict__ | {"ratio": n.ratio, "verdict": n.verdict} for n in nulls
                ],
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
