"""Train and evaluate the classical baselines for both ABSA stages.

Runs on CPU in seconds. Establishes the number a transformer has to beat before
it earns the extra complexity, GPU dependency and inference latency.

Protocol
--------
* Train on ``train``.
* Tune hyperparameters and the multi-label threshold on ``dev``.
* Touch ``test`` exactly once, at the end, for the reported number.

Usage::

    python scripts/train_baseline.py
    python scripts/train_baseline.py --skip-test     # while iterating
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.evaluation.metrics import (  # noqa: E402
    evaluate_multi_label,
    evaluate_single_label,
    save_results,
    tune_threshold,
)
from ml.preprocessing.transform import load_taxonomy  # noqa: E402
from ml.training.baseline import (  # noqa: E402
    AspectPrefixEncoder,
    build_acd_baseline,
    build_asc_baseline,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
TAXONOMY_PATH = REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml"


def load_split(name: str, prefix: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{prefix}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/build_dataset.py first.")
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-test", action="store_true", help="Evaluate on dev only.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    aspects = list(taxonomy.aspect_ids)
    polarities = list(taxonomy.polarities)
    results = []

    # =====================================================================
    # Stage A -- aspect category detection (multi-label)
    # =====================================================================
    print("=" * 62)
    print("STAGE A  Aspect Category Detection  (multi-label, 12 aspects)")
    print("=" * 62)

    acd = {name: load_split(name, "acd") for name in ("train", "dev", "test")}
    y = {name: frame[aspects].to_numpy() for name, frame in acd.items()}

    print(f"  train {len(acd['train']):>5} reviews   dev {len(acd['dev']):>4}   test {len(acd['test']):>5}")

    model_acd = build_acd_baseline()
    started = time.perf_counter()
    model_acd.fit(acd["train"]["text"], y["train"])
    fit_seconds = time.perf_counter() - started
    print(f"  fitted in {fit_seconds:.1f}s")

    # Threshold is tuned on dev only -- never on test.
    dev_scores = model_acd.predict_proba(acd["dev"]["text"])
    threshold, dev_micro = tune_threshold(y["dev"], dev_scores)
    print(f"  tuned threshold={threshold:.2f}  (dev micro F1 {dev_micro:.4f})")

    dev_acd = evaluate_multi_label(
        y["dev"], (dev_scores >= threshold).astype(int),
        labels=aspects, model="tfidf+logreg(ovr)", split="dev",
        threshold=threshold, fit_seconds=round(fit_seconds, 2),
    )
    print("\n" + dev_acd.summary())
    results.append(dev_acd)

    if not args.skip_test:
        test_scores = model_acd.predict_proba(acd["test"]["text"])
        test_acd = evaluate_multi_label(
            y["test"], (test_scores >= threshold).astype(int),
            labels=aspects, model="tfidf+logreg(ovr)", split="test",
            threshold=threshold, fit_seconds=round(fit_seconds, 2),
        )
        print("\n" + test_acd.summary())
        results.append(test_acd)

    # =====================================================================
    # Stage B -- aspect sentiment classification (3 classes)
    # =====================================================================
    print("\n" + "=" * 62)
    print("STAGE B  Aspect Sentiment Classification  (3 classes)")
    print("=" * 62)

    asc = {name: load_split(name, "asc") for name in ("train", "dev", "test")}
    encoder = AspectPrefixEncoder(taxonomy.descriptions)
    X = {name: encoder(frame) for name, frame in asc.items()}
    y_asc = {name: frame["label"].to_numpy() for name, frame in asc.items()}

    print(f"  train {len(asc['train']):>5} pairs   dev {len(asc['dev']):>4}   test {len(asc['test']):>5}")
    print(f"  input form: '<aspect description> | <review>'  (mirrors the transformer pair)")

    best = None
    for kind in ("logreg", "svc"):
        model = build_asc_baseline(kind)
        started = time.perf_counter()
        model.fit(X["train"], y_asc["train"])
        fit_seconds = time.perf_counter() - started

        dev_result = evaluate_single_label(
            y_asc["dev"], model.predict(X["dev"]),
            labels=polarities, model=f"tfidf+{kind}", split="dev",
            fit_seconds=round(fit_seconds, 2),
        )
        print(f"\n{dev_result.summary()}")
        results.append(dev_result)

        if best is None or dev_result.macro_f1 > best[1].macro_f1:
            best = (model, dev_result, kind)

    model_asc, best_dev, best_kind = best
    print(f"\n  --> selected '{best_kind}' on dev macro F1 {best_dev.macro_f1:.4f}")

    if not args.skip_test:
        test_asc = evaluate_single_label(
            y_asc["test"], model_asc.predict(X["test"]),
            labels=polarities, model=f"tfidf+{best_kind}", split="test",
        )
        print("\n" + test_asc.summary())
        results.append(test_asc)

    # =====================================================================
    # Export
    # =====================================================================
    acd_dir = args.models_dir / "baseline_aspect_detector"
    asc_dir = args.models_dir / "baseline_sentiment_classifier"
    acd_dir.mkdir(parents=True, exist_ok=True)
    asc_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_acd, acd_dir / "model.joblib")
    joblib.dump(model_asc, asc_dir / "model.joblib")

    # Metadata travels with the model so inference can never drift from training:
    # the label order, the threshold, and the exact input format.
    (acd_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": "aspect_category_detection",
                "model": "tfidf(word+char) + OneVsRest(LogisticRegression)",
                "labels": aspects,
                "threshold": threshold,
                "input_format": "raw review text",
                "dev_micro_f1": dev_acd.micro_f1,
                "sklearn_version": __import__("sklearn").__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (asc_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": "aspect_sentiment_classification",
                "model": f"tfidf(word+char) + {best_kind}",
                "labels": polarities,
                "label_to_id": taxonomy.polarity_to_id,
                "input_format": "'<aspect description> | <review text>'",
                "aspect_descriptions": taxonomy.descriptions,
                "dev_macro_f1": best_dev.macro_f1,
                "sklearn_version": __import__("sklearn").__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_results(results, args.models_dir / "metadata" / "baseline_results.json")
    print(f"\n  models  -> {args.models_dir}")
    print(f"  metrics -> {args.models_dir / 'metadata' / 'baseline_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
