"""Fine-tune a transformer for either ABSA stage.

Runs on a Colab T4 or CPU-only; the device is detected. Both stages follow the
same protocol as the baselines -- train on ``train``, tune on ``dev``, touch
``test`` once.

Usage::

    # Stage A -- aspect detection
    python scripts/train_transformer.py --stage acd --model distilbert-base-uncased

    # Stage B -- sentiment
    python scripts/train_transformer.py --stage asc --model distilbert-base-uncased

    # Colab, stronger model
    python scripts/train_transformer.py --stage asc --model microsoft/deberta-v3-base --epochs 4

    # Fast smoke test
    python scripts/train_transformer.py --stage asc --limit 200 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ml.evaluation.metrics import (  # noqa: E402
    evaluate_multi_label,
    evaluate_single_label,
    save_results,
    tune_threshold,
)
from ml.preprocessing.transform import load_taxonomy  # noqa: E402
from ml.training.transformer import (  # noqa: E402
    SentencePairDataset,
    SingleTextDataset,
    TrainConfig,
    class_weights,
    multilabel_pos_weight,
    predict_logits,
    resolve_device,
    set_seed,
    sigmoid,
    softmax,
    train_model,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
TAXONOMY_PATH = REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml"


def load_split(name: str, prefix: str, limit: int | None) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{prefix}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/build_dataset.py first.")
    frame = pd.read_csv(path)
    return frame.head(limit) if limit else frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["acd", "asc"], required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Truncate splits (smoke test).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(args.seed)
    device = resolve_device(args.device)
    config = TrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        device=args.device,
    )

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    aspects = list(taxonomy.aspect_ids)
    polarities = list(taxonomy.polarities)

    print("=" * 62)
    print(f"{'STAGE A  Aspect Detection' if args.stage == 'acd' else 'STAGE B  Sentiment'}  --  {args.model}")
    print("=" * 62)
    print(f"  device={device}  fp16={config.resolved_fp16(device)}  epochs={args.epochs}  bs={args.batch_size}")
    if device.type == "cpu":
        print("  NOTE: running on CPU. Fine for DistilBERT; use a Colab GPU for larger models.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefix = "acd" if args.stage == "acd" else "asc"
    splits = {name: load_split(name, prefix, args.limit) for name in ("train", "dev", "test")}
    print(f"  train {len(splits['train'])}  dev {len(splits['dev'])}  test {len(splits['test'])}")

    # ---------------------------------------------------------------- build --
    if args.stage == "acd":
        labels = aspects
        y = {name: frame[aspects].to_numpy() for name, frame in splits.items()}
        datasets = {
            name: SingleTextDataset(frame["text"], y[name], tokenizer, config.max_length)
            for name, frame in splits.items()
        }
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(labels), problem_type="multi_label_classification"
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=multilabel_pos_weight(y["train"]).to(device))
    else:
        labels = polarities
        y = {name: frame["label"].to_numpy() for name, frame in splits.items()}
        datasets = {
            name: SentencePairDataset(
                frame["text"],
                [taxonomy.descriptions[a] for a in frame["aspect"]],
                y[name],
                tokenizer,
                config.max_length,
            )
            for name, frame in splits.items()
        }
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(labels)
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(y["train"], len(labels)).to(device))

    # ---------------------------------------------------------------- train --
    started = time.perf_counter()
    history = train_model(model, datasets["train"], config, loss_fn=loss_fn, device=device)
    fit_seconds = time.perf_counter() - started
    print(f"  trained in {fit_seconds / 60:.1f} min")

    # ----------------------------------------------------------- evaluate ----
    results = []
    tag = f"{args.model.split('/')[-1]}"

    if args.stage == "acd":
        dev_scores = sigmoid(predict_logits(model, datasets["dev"], config, device))
        threshold, dev_micro = tune_threshold(y["dev"], dev_scores)
        print(f"  tuned threshold={threshold:.2f} on dev (micro F1 {dev_micro:.4f})")

        dev_result = evaluate_multi_label(
            y["dev"], (dev_scores >= threshold).astype(int),
            labels=labels, model=tag, split="dev", threshold=threshold,
            fit_seconds=round(fit_seconds, 1), epochs=args.epochs,
        )
        print("\n" + dev_result.summary())
        results.append(dev_result)

        if not args.skip_test:
            test_scores = sigmoid(predict_logits(model, datasets["test"], config, device))
            test_result = evaluate_multi_label(
                y["test"], (test_scores >= threshold).astype(int),
                labels=labels, model=tag, split="test", threshold=threshold,
                fit_seconds=round(fit_seconds, 1), epochs=args.epochs,
            )
            print("\n" + test_result.summary())
            results.append(test_result)
    else:
        threshold = None
        dev_probs = softmax(predict_logits(model, datasets["dev"], config, device))
        dev_result = evaluate_single_label(
            y["dev"], dev_probs.argmax(1),
            labels=labels, model=tag, split="dev",
            fit_seconds=round(fit_seconds, 1), epochs=args.epochs,
        )
        print("\n" + dev_result.summary())
        results.append(dev_result)

        if not args.skip_test:
            test_probs = softmax(predict_logits(model, datasets["test"], config, device))
            test_result = evaluate_single_label(
                y["test"], test_probs.argmax(1),
                labels=labels, model=tag, split="test",
                fit_seconds=round(fit_seconds, 1), epochs=args.epochs,
            )
            print("\n" + test_result.summary())
            results.append(test_result)

    # ------------------------------------------------------------- export ----
    out_dir = args.out or (
        MODELS_DIR / ("aspect_detector" if args.stage == "acd" else "sentiment_classifier")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metadata = {
        "task": "aspect_category_detection" if args.stage == "acd" else "aspect_sentiment_classification",
        "base_model": args.model,
        "labels": labels,
        "max_length": config.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "fit_seconds": round(fit_seconds, 1),
        "device": str(device),
        "final_train_loss": history["epoch_loss"][-1],
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }
    if args.stage == "acd":
        metadata["threshold"] = threshold
        metadata["input_format"] = "raw review text"
    else:
        metadata["label_to_id"] = taxonomy.polarity_to_id
        metadata["input_format"] = "sentence pair: (review, aspect_description)"
        metadata["aspect_descriptions"] = taxonomy.descriptions
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results_path = MODELS_DIR / "metadata" / f"{args.stage}_{tag}_results.json"
    save_results(results, results_path)
    print(f"\n  model   -> {out_dir}")
    print(f"  metrics -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
