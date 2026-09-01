"""Select the mixed-review upweighting factor, on dev, then report test once.

Only 8.9% of training reviews carry different polarities across aspects, so a
sentiment model can score well by reading overall tone and ignoring the aspect
entirely -- which both models were measured doing. ``--mixed-weight`` upweights
those pairs in the loss. This script picks the factor.

**The selection is made on dev.** Test is computed only for the chosen model, and
only so the final number can be reported. Choosing the weight by looking at test
scores would be test-set tuning, which is the failure this project exists to
avoid; the temptation is real here because the test diagnostic is the headline
number.

The objective is stated up front, before looking at anything:

    score = mixed_accuracy - lambda * max(0, macro_f1_at_w1 - macro_f1)

i.e. maximise accuracy on mixed reviews, penalised for any macro-F1 regression
against the unweighted model. ``lambda`` is how many points of mixed accuracy one
point of macro F1 is worth; the default of 1.0 treats them as equal.

Usage::

    python scripts/compare_mixed_weight.py \\
        --model models/sentiment_classifier \\
        --model models/_experiments/asc_mixed3 \\
        --model models/_experiments/asc_mixed8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ml.evaluation.mixed_reviews import evaluate_mixed, find_mixed_reviews
from ml.inference.calibration import confidently_wrong_rate, softmax
from ml.preprocessing.transform import load_taxonomy
from ml.training.transformer import SentencePairDataset, make_collate

PROCESSED = REPO_ROOT / "data" / "processed"


def probabilities_for(directory: Path, frame: pd.DataFrame, taxonomy) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
    aspects = [taxonomy.descriptions[a] for a in frame["aspect"]]

    dataset = SentencePairDataset(
        frame["text"], aspects, frame["label"].to_numpy(), tokenizer, 128
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=make_collate(tokenizer)
    )
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            outputs.append(model(**batch).logits.numpy())
    return softmax(np.concatenate(outputs))


def measure(probabilities: np.ndarray, frame: pd.DataFrame, name: str, split: str) -> dict:
    labels = frame["label"].to_numpy()
    predictions = probabilities.argmax(1)
    mixed_mask = frame["review_id"].isin(find_mixed_reviews(frame)).to_numpy()
    result = evaluate_mixed(frame, predictions, model=name, split=split)

    return {
        "model": name,
        "split": split,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float((predictions == labels).mean()),
        "mixed_accuracy": result.mixed_accuracy,
        "uniform_accuracy": result.uniform_accuracy,
        "collapsed_rate": result.collapsed_rate,
        "confidently_wrong_mixed": confidently_wrong_rate(
            probabilities[mixed_mask], labels[mixed_mask]
        ),
    }


def show(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    header = (
        f"  {'model':<22}{'macroF1':>9}{'acc':>8}{'MIXED':>9}"
        f"{'uniform':>9}{'collapsed':>11}{'cw-mixed':>10}"
    )
    print(header)
    print("  " + "-" * 78)
    for row in rows:
        print(
            f"  {row['model']:<22}{row['macro_f1']:>9.4f}{row['accuracy']:>8.4f}"
            f"{row['mixed_accuracy']:>9.4f}{row['uniform_accuracy']:>9.4f}"
            f"{row['collapsed_rate']:>11.4f}{row['confidently_wrong_mixed']:>9.2%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Points of mixed accuracy one point of macro F1 is worth.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "models" / "metadata" / "mixed_weight_selection.json",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    dev = pd.read_csv(PROCESSED / "asc_dev.csv")
    test = pd.read_csv(PROCESSED / "asc_test.csv")

    models = []
    for directory in args.model:
        if not (directory / "model.safetensors").exists():
            print(f"  skipping {directory} (no weights)", file=sys.stderr)
            continue
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        models.append((f"w={metadata.get('mixed_weight', 1.0):g}", directory))

    if not models:
        print("No models with weights found.", file=sys.stderr)
        return 1

    print("=" * 82)
    print("SELECTION on dev  (test is not consulted until a winner is chosen)")
    print("=" * 82)

    dev_rows = []
    for name, directory in models:
        dev_rows.append(measure(probabilities_for(directory, dev, taxonomy), dev, name, "dev"))
    show(dev_rows, "dev")

    baseline = next((r for r in dev_rows if r["model"] == "w=1"), dev_rows[0])
    print(f"\n  objective: mixed_accuracy - {args.lam:g} * max(0, macroF1(w=1) - macroF1)")
    print(f"  {'model':<22}{'mixed':>9}{'macroF1 drop':>15}{'score':>10}")
    print("  " + "-" * 58)
    for row in dev_rows:
        drop = max(0.0, baseline["macro_f1"] - row["macro_f1"])
        row["score"] = row["mixed_accuracy"] - args.lam * drop
        print(f"  {row['model']:<22}{row['mixed_accuracy']:>9.4f}{drop:>15.4f}{row['score']:>10.4f}")

    winner = max(dev_rows, key=lambda r: r["score"])
    print(f"\n  SELECTED on dev: {winner['model']}  (score {winner['score']:.4f})")

    print("\n" + "=" * 82)
    print(f"TEST -- reported once, for the dev-selected model ({winner['model']}) and w=1 for reference")
    print("=" * 82)
    test_rows = []
    for name, directory in models:
        if name not in (winner["model"], "w=1"):
            continue
        test_rows.append(measure(probabilities_for(directory, test, taxonomy), test, name, "test"))
    show(test_rows, "test")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "objective": f"mixed_accuracy - {args.lam} * max(0, macro_f1(w=1) - macro_f1)",
                "selected_on": "dev",
                "selected": winner["model"],
                "dev": dev_rows,
                "test": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
