"""Build the model comparison table and select the final model.

Reads every ``models/metadata/*_results.json`` written by the training scripts,
prints one table per stage, and writes ``models/metadata/comparison.json`` for
the README.

Selection is objective and stated up front:

* **Stage A** (aspect detection) -- highest **micro F1** on test.
* **Stage B** (sentiment) -- highest **macro F1** on test.

Accuracy is deliberately not a criterion. It is still printed, because the gap
between it and macro F1 is exactly what this project is trying to show.

Also runs the aspect-conditioning diagnostic (see
``ml/evaluation/mixed_reviews.py``): overall metrics hide whether a model
conditions on the aspect at all, and that is the capability the product sells.

Usage::

    python scripts/compare_models.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

MODELS_DIR = REPO_ROOT / "models"
METADATA_DIR = MODELS_DIR / "metadata"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_results() -> list[dict]:
    """Load every result record written by the training scripts."""
    records = []
    for path in sorted(METADATA_DIR.glob("*results*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  WARNING: {path.name} is not valid JSON; skipping", file=sys.stderr)
            continue
        for entry in payload if isinstance(payload, list) else [payload]:
            entry["_source"] = path.name
            records.append(entry)
    return records


def is_multilabel(record: dict) -> bool:
    return "micro_f1" in record


def run_mixed_diagnostic() -> list[dict]:
    """Aspect-conditioning diagnostic for every sentiment model on disk."""
    from ml.evaluation.mixed_reviews import evaluate_mixed
    from ml.preprocessing.transform import load_taxonomy

    test_path = PROCESSED_DIR / "asc_test.csv"
    if not test_path.exists():
        return []

    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    test = pd.read_csv(test_path)
    rows = []

    # Baseline
    baseline_dir = MODELS_DIR / "baseline_sentiment_classifier"
    if (baseline_dir / "model.joblib").exists():
        import joblib

        from ml.training.baseline import AspectPrefixEncoder

        model = joblib.load(baseline_dir / "model.joblib")
        encoder = AspectPrefixEncoder(taxonomy.descriptions)
        result = evaluate_mixed(
            test, model.predict(encoder(test)), model="baseline (tfidf)", split="test"
        )
        rows.append(result)

    # Transformer
    transformer_dir = MODELS_DIR / "sentiment_classifier"
    if (transformer_dir / "metadata.json").exists():
        try:
            import numpy as np
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            metadata = json.loads((transformer_dir / "metadata.json").read_text(encoding="utf-8"))
            tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
            model = AutoModelForSequenceClassification.from_pretrained(transformer_dir).eval()

            predictions = []
            batch_size = 64
            aspect_texts = [taxonomy.descriptions[a] for a in test["aspect"]]
            with torch.no_grad():
                for start in range(0, len(test), batch_size):
                    encoded = tokenizer(
                        list(test["text"][start : start + batch_size]),
                        aspect_texts[start : start + batch_size],
                        truncation="only_first",
                        max_length=metadata.get("max_length", 128),
                        padding=True,
                        return_tensors="pt",
                    )
                    predictions.append(model(**encoded).logits.argmax(-1).numpy())
            result = evaluate_mixed(
                test,
                np.concatenate(predictions),
                model=f"transformer ({metadata.get('base_model', '?')})",
                split="test",
            )
            rows.append(result)
        except Exception as exc:
            print(f"  (skipped transformer diagnostic: {exc})", file=sys.stderr)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-diagnostic", action="store_true")
    args = parser.parse_args()

    records = load_results()
    if not records:
        print(
            "No results found in models/metadata/.\n"
            "Run scripts/train_baseline.py and/or scripts/train_transformer.py first.",
            file=sys.stderr,
        )
        return 1

    test_records = [r for r in records if r.get("split") == "test"]
    acd = [r for r in test_records if is_multilabel(r)]
    asc = [r for r in test_records if not is_multilabel(r)]

    comparison: dict = {}

    # ---------------------------------------------------------------- ACD ---
    if acd:
        print("=" * 76)
        print("STAGE A  Aspect Category Detection  (test split)")
        print("=" * 76)
        print(f"{'model':<30}{'MICRO F1':>10}{'macro F1':>10}{'subset':>9}{'P':>7}{'R':>7}")
        print("-" * 76)
        for record in sorted(acd, key=lambda r: -r["micro_f1"]):
            print(
                f"{record['model']:<30}{record['micro_f1']:>10.4f}{record['macro_f1']:>10.4f}"
                f"{record['subset_accuracy']:>9.4f}"
                f"{record['micro_precision']:>7.3f}{record['micro_recall']:>7.3f}"
            )
        best = max(acd, key=lambda r: r["micro_f1"])
        print(f"\n  SELECTED: {best['model']}  (micro F1 {best['micro_f1']:.4f})")
        comparison["aspect_detection"] = {
            "selected": best["model"],
            "metric": "micro_f1",
            "value": best["micro_f1"],
            "candidates": [
                {k: r[k] for k in ("model", "micro_f1", "macro_f1", "subset_accuracy")}
                for r in acd
            ],
        }

    # ---------------------------------------------------------------- ASC ---
    if asc:
        print("\n" + "=" * 76)
        print("STAGE B  Aspect Sentiment Classification  (test split)")
        print("=" * 76)
        print(f"{'model':<30}{'MACRO F1':>10}{'accuracy':>10}{'neg F1':>9}{'neu F1':>9}{'pos F1':>9}")
        print("-" * 76)
        for record in sorted(asc, key=lambda r: -r["macro_f1"]):
            per_class = {c["label"]: c["f1"] for c in record["per_class"]}
            print(
                f"{record['model']:<30}{record['macro_f1']:>10.4f}{record['accuracy']:>10.4f}"
                f"{per_class.get('negative', 0):>9.3f}{per_class.get('neutral', 0):>9.3f}"
                f"{per_class.get('positive', 0):>9.3f}"
            )
        best = max(asc, key=lambda r: r["macro_f1"])
        print(f"\n  SELECTED: {best['model']}  (macro F1 {best['macro_f1']:.4f})")

        # Make the accuracy trap explicit if it is present in the results.
        by_accuracy = max(asc, key=lambda r: r["accuracy"])
        if by_accuracy["model"] != best["model"]:
            print(
                f"\n  NOTE: '{by_accuracy['model']}' has higher accuracy "
                f"({by_accuracy['accuracy']:.4f} vs {best['accuracy']:.4f}) but lower macro F1. "
                f"Selecting on accuracy would pick the model that handles the minority class worse."
            )

        comparison["sentiment"] = {
            "selected": best["model"],
            "metric": "macro_f1",
            "value": best["macro_f1"],
            "candidates": [
                {
                    "model": r["model"],
                    "macro_f1": r["macro_f1"],
                    "accuracy": r["accuracy"],
                    "per_class_f1": {c["label"]: c["f1"] for c in r["per_class"]},
                }
                for r in asc
            ],
        }

    # --------------------------------------------------------- diagnostic ---
    if not args.skip_diagnostic:
        print("\n" + "=" * 76)
        print("DIAGNOSTIC  Does the model condition on the aspect?")
        print("=" * 76)
        print(
            "Overall metrics hide this. Mixed reviews carry different polarities for\n"
            "different aspects, so a model reading only overall tone is wrong by\n"
            "construction on half of them.\n"
        )
        diagnostics = run_mixed_diagnostic()
        if diagnostics:
            print(f"{'model':<34}{'mixed':>9}{'uniform':>10}{'gap':>9}{'collapsed':>11}")
            print("-" * 76)
            for result in diagnostics:
                print(
                    f"{result.model:<34}{result.mixed_accuracy:>9.4f}"
                    f"{result.uniform_accuracy:>10.4f}{result.gap:>+9.4f}"
                    f"{result.collapsed_rate:>11.4f}"
                )
            print(
                "\n  'collapsed' = share of mixed reviews given ONE polarity for every\n"
                "  aspect, i.e. the aspect was ignored entirely."
            )
            comparison["aspect_conditioning"] = [
                {
                    "model": r.model,
                    "mixed_accuracy": r.mixed_accuracy,
                    "uniform_accuracy": r.uniform_accuracy,
                    "gap": r.gap,
                    "collapsed_rate": r.collapsed_rate,
                    "n_mixed_reviews": r.n_mixed_reviews,
                }
                for r in diagnostics
            ]
        else:
            print("  (no sentiment models available for the diagnostic)")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(f"\n  written -> {METADATA_DIR / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
