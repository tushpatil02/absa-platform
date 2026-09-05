"""Does the pipeline recover a quality it was never told?

Every other evaluation in this project is indirect. On real data there is no
ground truth for "how good is this phone's camera" -- only reviews, from which
a score is estimated. Split-half reliability asks whether that estimate is
*stable*; it cannot ask whether it is *right*, because nothing says what right
would be.

The simulated catalogue can. Each phone was assigned a latent 1-10 quality per
aspect (``ml/synthetic/reviews.py::phone_quality``), reviews were generated from
it, and the whole pipeline then ran over the text with no access to the latent
values:

    latent quality -> generated reviews -> detector -> sentiment -> shrinkage -> score

If the recovered scores correlate with the latent quality, the chain end to end
does what it claims. If they do not, something in it is broken in a way no
real-data metric would surface.

What this can and cannot show
-----------------------------
It validates the **pipeline**, not the models' fitness for real prose. The text
is compositional and its polarity is unambiguous by construction, so a high
correlation here is a floor -- a pipeline that fails this is definitely broken,
while one that passes may still do poorly on human writing. Treat it as a
wiring check with teeth, not as an accuracy claim.

    python scripts/verify_recovery.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.catalog.profiles import MIN_MENTIONS, build_profiles
from ml.recommender.similarity import AXES

SYNTHETIC = REPO_ROOT / "data" / "synthetic"

# Aspects with a latent value. `price` is excluded: its score comes from the
# listed price, not from review sentiment, so there is nothing to recover.
RECOVERABLE = tuple(axis for axis in AXES if axis != "price")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS)
    parser.add_argument("--truth", type=Path, default=SYNTHETIC / "ground_truth.csv")
    parser.add_argument("--scores", type=Path, default=SYNTHETIC / "review_aspects_synthetic.csv")
    parser.add_argument("--out", type=Path, default=SYNTHETIC / "recovery.json")
    args = parser.parse_args()

    for path in (args.truth, args.scores):
        if not path.exists():
            print(f"MISSING {path} -- run scripts/generate_synthetic.py, then")
            print("  scripts/score_catalog.py over the synthetic reviews.")
            return 1

    truth = pd.read_csv(args.truth).set_index("model_key")
    scored = pd.read_csv(args.scores)
    profiles, _ = build_profiles(scored, min_mentions=args.min_mentions)
    recovered = profiles.pivot(index="model_key", columns="aspect", values="score")

    print("=" * 72)
    print("RECOVERY OF LATENT QUALITY")
    print("=" * 72)
    print("  The pipeline never saw these values. It saw only generated text.\n")
    print(f"  {'aspect':<14}{'phones':>8}{'Spearman':>11}{'Pearson':>10}{'mean err':>11}   verdict")
    print("-" * 72)

    results = []
    for aspect in RECOVERABLE:
        if aspect not in recovered.columns:
            print(f"  {aspect:<14}{'--':>8}   no scores published at this floor")
            continue
        joined = pd.concat(
            [recovered[aspect].rename("recovered"), truth[aspect].rename("latent")], axis=1
        ).dropna()
        if len(joined) < 5:
            print(f"  {aspect:<14}{len(joined):>8}   too few phones to correlate")
            continue

        rho = stats.spearmanr(joined["latent"], joined["recovered"]).statistic
        r = stats.pearsonr(joined["latent"], joined["recovered"]).statistic
        error = (joined["recovered"] - joined["latent"]).abs().mean()
        verdict = "recovered" if rho >= 0.7 else ("weak" if rho >= 0.4 else "NOT RECOVERED")

        print(f"  {aspect:<14}{len(joined):>8}{rho:>11.3f}{r:>10.3f}{error:>11.2f}   {verdict}")
        results.append(
            {
                "aspect": aspect,
                "n_phones": len(joined),
                "spearman": round(float(rho), 4),
                "pearson": round(float(r), 4),
                "mean_abs_error": round(float(error), 3),
                "verdict": verdict,
            }
        )

    print()
    print("=" * 72)
    if not results:
        print("VERDICT: nothing to check -- no aspect published scores.")
    elif all(entry["spearman"] >= 0.7 for entry in results):
        print("VERDICT: PASS. Ordering is recovered on every aspect.")
        print("  Detection, sentiment, aggregation and shrinkage are wired correctly")
        print("  end to end. This is a floor, not an accuracy claim: the text is")
        print("  compositional and its polarity unambiguous by construction.")
    else:
        weak = ", ".join(e["aspect"] for e in results if e["spearman"] < 0.7)
        print(f"VERDICT: INCOMPLETE. Ordering is not recovered on: {weak}")
        print("  Something in the chain is losing the signal on those aspects.")
    print("=" * 72)

    print("\n  Mean absolute error is expected to be non-zero and is not a defect:")
    print("  shrinkage deliberately pulls scores toward the catalogue mean, so a")
    print("  phone at latent 9.4 should land below 9.4. Ordering is the claim.")

    args.out.write_text(
        json.dumps(
            {
                "protocol": (
                    "Latent per-aspect quality -> generated reviews -> full scoring "
                    "pipeline. The pipeline never sees the latent values."
                ),
                "min_mentions": args.min_mentions,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
