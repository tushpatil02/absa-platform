"""Diagnostic: can the model do *aspect-conditional* sentiment?

Overall macro F1 hides the capability this product actually sells. Most reviews
are uniformly positive or uniformly negative, so a model that ignores the aspect
entirely and predicts the review's overall tone still scores respectably.

The interesting case is a **mixed review** -- one where the same text carries
different polarities for different aspects:

    "The camera is excellent, but the battery drains very quickly."

Here the aspect *must* change the prediction. A model that reads only overall
tone gets exactly one of the two right, by construction.

This module slices the test set down to those reviews and reports accuracy on
them specifically. The gap between overall accuracy and mixed-review accuracy is
a direct measure of how much the model is actually conditioning on the aspect,
and it is the number that justifies (or refutes) paying for a transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MixedReviewResult:
    """Accuracy on mixed vs uniform reviews, and the gap between them."""

    model: str
    split: str
    n_mixed_reviews: int
    n_mixed_pairs: int
    n_uniform_pairs: int
    mixed_accuracy: float
    uniform_accuracy: float
    overall_accuracy: float
    # Fraction of mixed reviews where every aspect got the same prediction,
    # i.e. the model ignored the aspect completely.
    collapsed_rate: float

    @property
    def gap(self) -> float:
        """How much worse mixed reviews are. Large gap = weak aspect conditioning."""
        return self.uniform_accuracy - self.mixed_accuracy

    def summary(self) -> str:
        return (
            f"{self.model}  [{self.split}]  aspect-conditioning diagnostic\n"
            f"  mixed reviews      {self.n_mixed_reviews} reviews / {self.n_mixed_pairs} pairs\n"
            f"  accuracy on mixed  {self.mixed_accuracy:.4f}\n"
            f"  accuracy on uniform{self.uniform_accuracy:>10.4f}   ({self.n_uniform_pairs} pairs)\n"
            f"  overall accuracy   {self.overall_accuracy:.4f}\n"
            f"  GAP                {self.gap:+.4f}   (uniform - mixed; large = ignores the aspect)\n"
            f"  collapsed          {self.collapsed_rate:.4f}   "
            f"(mixed reviews given one polarity for every aspect)"
        )


def find_mixed_reviews(frame: pd.DataFrame) -> set[str]:
    """Review ids whose gold labels are not all the same polarity."""
    per_review = frame.groupby("review_id")["polarity"].nunique()
    return set(per_review[per_review > 1].index)


def evaluate_mixed(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    model: str,
    split: str,
) -> MixedReviewResult:
    """Split accuracy by whether the source review was mixed.

    Args:
        frame: The ASC split, with ``review_id``, ``label`` and ``polarity``.
        y_pred: Predicted class indices, aligned row-for-row with `frame`.
        model: Model name for the report.
        split: Split name.
    """
    frame = frame.reset_index(drop=True).copy()
    frame["pred"] = np.asarray(y_pred)
    frame["correct"] = frame["pred"] == frame["label"]

    mixed_ids = find_mixed_reviews(frame)
    is_mixed = frame["review_id"].isin(mixed_ids)
    mixed, uniform = frame[is_mixed], frame[~is_mixed]

    # A review is "collapsed" when the model gave every aspect the same answer,
    # despite the gold labels differing -- the signature of ignoring the aspect.
    collapsed = 0
    for _, group in mixed.groupby("review_id"):
        if group["pred"].nunique() == 1:
            collapsed += 1

    return MixedReviewResult(
        model=model,
        split=split,
        n_mixed_reviews=len(mixed_ids),
        n_mixed_pairs=int(len(mixed)),
        n_uniform_pairs=int(len(uniform)),
        mixed_accuracy=float(mixed["correct"].mean()) if len(mixed) else float("nan"),
        uniform_accuracy=float(uniform["correct"].mean()) if len(uniform) else float("nan"),
        overall_accuracy=float(frame["correct"].mean()),
        collapsed_rate=collapsed / max(len(mixed_ids), 1),
    )
