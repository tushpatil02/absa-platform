"""Does synthetic training data actually help? Measured on real held-out data.

The question this answers
-------------------------
`data/synthetic/` contains generated reviews with exact labels. Whether they
are *useful* is an empirical question, and it has exactly one honest form:

    train on real / train on real + synthetic
    evaluate BOTH on the same real held-out M-ABSA test split

Scoring on synthetic test data would measure the generator, not the model. The
test set here is never touched by synthetic rows, so the comparison is real
whichever way it comes out -- including if augmentation makes things worse,
which is a publishable result and not a failed experiment.

Sweeps the amount of synthetic data, because "does it help" and "how much helps"
are different questions and the second one usually has a peak.

    python scripts/compare_synthetic.py
    python scripts/compare_synthetic.py --seeds 0 1 2 --sizes 0 2500 5000 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.metrics import paired_bootstrap
from ml.preprocessing.transform import load_taxonomy
from ml.training.baseline import AspectPrefixEncoder, build_asc_baseline

PROCESSED = REPO_ROOT / "data" / "processed"
SYNTHETIC = REPO_ROOT / "data" / "synthetic"


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, encoder, seed: int):
    """Fit on `train`, return predictions for `test`. `test` is always real."""
    model = build_asc_baseline("logreg")
    # Reseed the classifier so seed sweeps measure training variance rather
    # than replaying one fixed fit.
    model.named_steps["clf"].set_params(random_state=seed)
    model.fit(encoder(train), train["polarity"])
    return model.predict(encoder(test))


def evaluate(train: pd.DataFrame, test: pd.DataFrame, encoder, seed: int) -> dict:
    """Fit on `train`, score on `test`. `test` is always real."""
    predicted = fit_predict(train, test, encoder, seed)

    labels = ["negative", "neutral", "positive"]
    per_class = f1_score(test["polarity"], predicted, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(test["polarity"], predicted, average="macro", zero_division=0)),
        "accuracy": float((predicted == test["polarity"]).mean()),
        **{f"f1_{name}": float(value) for name, value in zip(labels, per_class, strict=True)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[0, 1000, 2500, 5000, 9977])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "models" / "metadata" / "synthetic_augmentation.json"
    )
    args = parser.parse_args()

    synthetic_path = SYNTHETIC / "asc_synthetic.csv"
    if not synthetic_path.exists():
        print(f"MISSING {synthetic_path} -- run scripts/generate_synthetic.py first")
        return 1

    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    encoder = AspectPrefixEncoder(taxonomy.descriptions)

    real_train = pd.read_csv(PROCESSED / "asc_train.csv")
    real_test = pd.read_csv(PROCESSED / "asc_test.csv")
    synthetic = pd.read_csv(synthetic_path)

    print(f"real train {len(real_train):,}   real TEST {len(real_test):,} (never synthetic)")
    print(f"synthetic pool {len(synthetic):,}\n")

    header = (
        f"{'synthetic':>10}{'total train':>13}{'macro F1':>12}"
        f"{'sd':>8}{'accuracy':>10}{'neu F1':>9}   vs baseline"
    )
    print(header)
    print("-" * 78)

    results = []
    baseline_mean = None

    for size in args.sizes:
        runs = []
        for seed in args.seeds:
            if size:
                extra = synthetic.sample(n=min(size, len(synthetic)), random_state=seed)
                train = pd.concat([real_train, extra], ignore_index=True)
            else:
                train = real_train
            runs.append(evaluate(train, real_test, encoder, seed))

        macro = np.array([r["macro_f1"] for r in runs])
        mean, sd = float(macro.mean()), float(macro.std(ddof=1)) if len(macro) > 1 else 0.0
        if baseline_mean is None:
            baseline_mean = mean
            delta = "   (baseline)"
        else:
            diff = mean - baseline_mean
            # Two standard deviations of the baseline run-to-run spread is the
            # bar a difference has to clear before it means anything.
            marker = "" if abs(diff) > 2 * max(sd, 1e-9) else "  (within noise)"
            delta = f"  {diff:+.4f}{marker}"

        total = len(real_train) + min(size, len(synthetic))
        print(
            f"{size:>10,}{total:>13,}{mean:>12.4f}{sd:>8.4f}"
            f"{np.mean([r['accuracy'] for r in runs]):>10.4f}"
            f"{np.mean([r['f1_neutral'] for r in runs]):>9.4f}{delta}"
        )
        results.append(
            {
                "n_synthetic": size,
                "n_train_total": total,
                "seeds": args.seeds,
                "macro_f1_mean": round(mean, 4),
                "macro_f1_sd": round(sd, 4),
                "macro_f1_runs": [round(v, 4) for v in macro],
                "accuracy_mean": round(float(np.mean([r["accuracy"] for r in runs])), 4),
                "f1_neutral_mean": round(float(np.mean([r["f1_neutral"] for r in runs])), 4),
            }
        )

    best = max(results, key=lambda r: r["macro_f1_mean"])

    # The run-to-run standard deviation above is NOT the uncertainty that
    # matters. With a deterministic solver the real-only row has sd = 0.0000,
    # which would let any positive difference look significant. The dominant
    # uncertainty is test-set sampling on 1,493 examples, so the verdict comes
    # from a paired bootstrap over the test items.
    bootstrap = None
    if best["n_synthetic"]:
        extra = synthetic.sample(n=min(best["n_synthetic"], len(synthetic)), random_state=0)
        augmented = pd.concat([real_train, extra], ignore_index=True)
        pred_a = fit_predict(real_train, real_test, encoder, 0)
        pred_b = fit_predict(augmented, real_test, encoder, 0)
        bootstrap = paired_bootstrap(
            real_test["polarity"].to_numpy(),
            pred_a,
            pred_b,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
            n_resamples=5000,
            seed=0,
        )

    print()
    print("=" * 78)
    if best["n_synthetic"] == 0:
        print("VERDICT: synthetic data does NOT help. The best result uses none of it.")
        print("  Keep it out of training. Report this as a measured negative.")
    else:
        print(f"Best sweep point: n={best['n_synthetic']:,}")
        print("Paired bootstrap vs real-only (5,000 resamples of the test set):")
        print(f"  {bootstrap.summary()}")
        print()
        if bootstrap.includes_zero:
            print("VERDICT: NO MEASURABLE EFFECT. The interval includes zero, so the")
            print("  apparent gain is within test-set sampling noise. Do not ship this")
            print("  as an improvement -- the seed spread above would have said otherwise,")
            print("  and it is the wrong uncertainty to quote.")
        else:
            print("VERDICT: a real improvement. The interval excludes zero.")
    print("=" * 78)
    print("\nEvery figure above is on the REAL held-out test split. Synthetic rows")
    print("appear only in training, so this comparison is valid in both directions.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "protocol": (
                    "Train on real M-ABSA train, optionally augmented with N synthetic "
                    "pairs. Evaluate on the real held-out test split, which never "
                    "contains synthetic rows."
                ),
                "n_real_train": len(real_train),
                "n_real_test": len(real_test),
                "n_synthetic_pool": len(synthetic),
                "results": results,
                "paired_bootstrap_vs_real_only": (
                    {
                        "n_synthetic": best["n_synthetic"],
                        "point": round(bootstrap.point, 4),
                        "ci_lower": round(bootstrap.lower, 4),
                        "ci_upper": round(bootstrap.upper, 4),
                        "p_positive": round(bootstrap.p_positive, 3),
                        "includes_zero": bootstrap.includes_zero,
                        "verdict": bootstrap.verdict,
                    }
                    if bootstrap
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
