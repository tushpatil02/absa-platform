"""Evaluation metrics for both ABSA stages.

Accuracy is deliberately never the headline. `neutral` is 5.3% of the sentiment
data, so a model that never predicts it still scores ~95% on that class by
abstaining. **Macro F1 is the selection metric**, and per-class numbers are
always reported alongside so a weak class cannot hide inside an average.

Two tasks, two shapes:

* **ASC** -- single-label, 3 classes. Macro F1 over {negative, neutral, positive}.
* **ACD** -- multi-label, 12 aspects. Micro F1 is the headline (it weights by
  support, which matches "how many aspect mentions did we catch"), with macro F1
  reported next to it because the tail aspects are small and micro F1 would hide
  them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class ClassMetrics:
    """Per-class precision/recall/F1 and how many examples backed them."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class SingleLabelResult:
    """Metrics for the sentiment task (ASC)."""

    model: str
    split: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class: list[ClassMetrics]
    confusion: list[list[int]]
    labels: list[str]
    n_examples: int
    extra: dict = field(default_factory=dict)

    @property
    def headline(self) -> float:
        """The number model selection is based on."""
        return self.macro_f1

    def summary(self) -> str:
        lines = [
            f"{self.model}  [{self.split}]  n={self.n_examples}",
            f"  accuracy    {self.accuracy:.4f}",
            f"  MACRO F1    {self.macro_f1:.4f}   <- selection metric",
            f"  weighted F1 {self.weighted_f1:.4f}",
            "",
            f"  {'class':<10}{'prec':>8}{'rec':>8}{'F1':>8}{'support':>9}",
        ]
        for entry in self.per_class:
            lines.append(
                f"  {entry.label:<10}{entry.precision:>8.3f}{entry.recall:>8.3f}"
                f"{entry.f1:>8.3f}{entry.support:>9}"
            )
        lines.append("")
        lines.append(f"  confusion (rows=true, cols=pred) {self.labels}")
        for label, row in zip(self.labels, self.confusion, strict=True):
            lines.append(f"    {label:<10}{row}")
        return "\n".join(lines)


@dataclass
class MultiLabelResult:
    """Metrics for the aspect-detection task (ACD)."""

    model: str
    split: str
    micro_f1: float
    macro_f1: float
    samples_f1: float
    subset_accuracy: float
    micro_precision: float
    micro_recall: float
    per_class: list[ClassMetrics]
    labels: list[str]
    n_examples: int
    threshold: float = 0.5
    extra: dict = field(default_factory=dict)

    @property
    def headline(self) -> float:
        return self.micro_f1

    def summary(self) -> str:
        lines = [
            f"{self.model}  [{self.split}]  n={self.n_examples}  threshold={self.threshold:.2f}",
            f"  MICRO F1    {self.micro_f1:.4f}   <- selection metric",
            f"  macro F1    {self.macro_f1:.4f}",
            f"  samples F1  {self.samples_f1:.4f}",
            f"  subset acc  {self.subset_accuracy:.4f}   (all 12 labels exactly right)",
            f"  micro P/R   {self.micro_precision:.3f} / {self.micro_recall:.3f}",
            "",
            f"  {'aspect':<18}{'prec':>8}{'rec':>8}{'F1':>8}{'support':>9}",
        ]
        for entry in sorted(self.per_class, key=lambda c: -c.support):
            lines.append(
                f"  {entry.label:<18}{entry.precision:>8.3f}{entry.recall:>8.3f}"
                f"{entry.f1:>8.3f}{entry.support:>9}"
            )
        return "\n".join(lines)


def evaluate_single_label(
    y_true, y_pred, *, labels: list[str], model: str, split: str, **extra
) -> SingleLabelResult:
    """Evaluate the 3-class sentiment task.

    `labels` is the label *names* in class-index order, so index i in the
    confusion matrix always means labels[i]. Passing them explicitly (rather
    than letting sklearn infer) keeps the matrix aligned even when a split
    happens to contain no examples of some class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    indices = list(range(len(labels)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    macro_p, macro_r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average="macro", zero_division=0
    )

    return SingleLabelResult(
        model=model,
        split=split,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=indices, average="weighted", zero_division=0)
        ),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        per_class=[
            ClassMetrics(labels[i], float(precision[i]), float(recall[i]), float(f1[i]), int(support[i]))
            for i in indices
        ],
        confusion=confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        labels=list(labels),
        n_examples=len(y_true),
        extra=extra,
    )


def evaluate_multi_label(
    y_true, y_pred, *, labels: list[str], model: str, split: str, threshold: float = 0.5, **extra
) -> MultiLabelResult:
    """Evaluate the 12-label aspect-detection task.

    `y_true` / `y_pred` are binary indicator matrices of shape (n, len(labels)).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    micro_p, micro_r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    return MultiLabelResult(
        model=model,
        split=split,
        micro_f1=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        samples_f1=float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        # Subset accuracy is brutal (all 12 labels must be exactly right) but it
        # is the honest "did we get the whole review right" number.
        subset_accuracy=float((y_true == y_pred).all(axis=1).mean()),
        micro_precision=float(micro_p),
        micro_recall=float(micro_r),
        per_class=[
            ClassMetrics(labels[i], float(precision[i]), float(recall[i]), float(f1[i]), int(support[i]))
            for i in range(len(labels))
        ],
        labels=list(labels),
        n_examples=int(y_true.shape[0]),
        threshold=threshold,
        extra=extra,
    )


def tune_threshold(y_true, y_scores, *, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Pick the multi-label decision threshold that maximises micro F1.

    Tuned on the **dev** split only. Using test here would be test-set tuning,
    which is one of the ways a reported number stops meaning anything.

    Returns:
        ``(best_threshold, best_micro_f1)``.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.02)

    best_threshold, best_score = 0.5, -1.0
    for threshold in grid:
        score = f1_score(y_true, (y_scores >= threshold).astype(int), average="micro", zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def save_results(results: list, path: Path) -> None:
    """Write results to JSON for the comparison table and the docs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8"
    )


def sklearn_report(y_true, y_pred, labels: list[str]) -> str:
    """Plain sklearn text report, handy inside notebooks."""
    return classification_report(
        y_true, y_pred, labels=list(range(len(labels))),
        target_names=labels, zero_division=0, digits=3,
    )
