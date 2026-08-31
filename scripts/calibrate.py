"""Fit a temperature for the sentiment classifier and report the improvement.

Temperature is fitted on **dev** and evaluated on **test**, so the reported
improvement is measured on data the temperature never saw. The scaling is
monotonic, so accuracy and macro F1 are provably unchanged -- this only makes the
confidence figure honest.

The fitted value is written into the model's ``metadata.json`` as
``"temperature"``; inference picks it up automatically.

Usage::

    python scripts/calibrate.py
    python scripts/calibrate.py --model models/sentiment_classifier
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

from ml.evaluation.mixed_reviews import find_mixed_reviews
from ml.inference.calibration import (
    confidently_wrong_rate,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from ml.preprocessing.transform import load_taxonomy

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def transformer_logits(directory: Path, frame: pd.DataFrame, taxonomy) -> np.ndarray:
    """Raw (uncalibrated) logits for every row of `frame`."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).eval()

    aspect_texts = [taxonomy.descriptions[aspect] for aspect in frame["aspect"]]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(frame), 64):
            encoded = tokenizer(
                list(frame["text"][start : start + 64]),
                aspect_texts[start : start + 64],
                truncation="only_first",
                max_length=metadata.get("max_length", 128),
                padding=True,
                return_tensors="pt",
            )
            outputs.append(model(**encoded).logits.numpy())
    return np.concatenate(outputs)


def report(name: str, probabilities: np.ndarray, labels: np.ndarray, mixed_mask: np.ndarray) -> None:
    accuracy = (probabilities.argmax(1) == labels).mean()
    print(
        f"  {name:<14}"
        f"acc {accuracy:>6.4f}   "
        f"ECE {expected_calibration_error(probabilities, labels):>6.4f}   "
        f"mean conf {probabilities.max(1).mean():>6.4f}   "
        f"conf-wrong all {confidently_wrong_rate(probabilities, labels):>6.2%}   "
        f"mixed {confidently_wrong_rate(probabilities[mixed_mask], labels[mixed_mask]):>6.2%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "sentiment_classifier")
    parser.add_argument("--dry-run", action="store_true", help="Do not write metadata.json.")
    args = parser.parse_args()

    if not (args.model / "metadata.json").exists():
        print(f"No model at {args.model}. Run scripts/train_transformer.py --stage asc.", file=sys.stderr)
        return 1

    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    dev = pd.read_csv(PROCESSED_DIR / "asc_dev.csv")
    test = pd.read_csv(PROCESSED_DIR / "asc_test.csv")

    print("Computing logits (dev, then test)...")
    dev_logits = transformer_logits(args.model, dev, taxonomy)
    test_logits = transformer_logits(args.model, test, taxonomy)
    dev_labels = dev["label"].to_numpy()
    test_labels = test["label"].to_numpy()

    temperature = fit_temperature(dev_logits, dev_labels)
    print(f"\nFitted temperature on dev: T = {temperature:.4f}")
    print(
        "  T > 1 means the model was over-confident; scaling softens the "
        "distribution\n  without moving any argmax."
    )

    mixed_mask = test["review_id"].isin(find_mixed_reviews(test)).to_numpy()
    before = softmax(test_logits, 1.0)
    after = softmax(test_logits, temperature)

    print("\nTest split (temperature fitted on dev only):")
    report("before", before, test_labels, mixed_mask)
    report("after", after, test_labels, mixed_mask)

    # The whole argument for temperature scaling rests on this being exact.
    unchanged = (before.argmax(1) == after.argmax(1)).all()
    print(f"\n  argmax unchanged for every row: {unchanged}")
    if not unchanged:
        print("  ERROR: temperature scaling must be monotonic.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n--dry-run: metadata.json not written.")
        return 0

    metadata_path = args.model / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["temperature"] = round(temperature, 4)
    metadata["calibration"] = {
        "method": "temperature_scaling",
        "fitted_on": "dev",
        "ece_before": round(expected_calibration_error(before, test_labels), 4),
        "ece_after": round(expected_calibration_error(after, test_labels), 4),
        "confidently_wrong_before": round(confidently_wrong_rate(before, test_labels), 4),
        "confidently_wrong_after": round(confidently_wrong_rate(after, test_labels), 4),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n  wrote temperature to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
