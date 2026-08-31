"""Calibration measurement and temperature scaling.

Confidence matters here more than usual because the UI *shows* it to the user.
"Delivery: Negative, 88% confident" on a review praising the delivery is worse
than the same error reported at 45%.

Temperature scaling (Guo et al., 2017) divides the logits by a scalar ``T``
fitted to minimise NLL on **dev**. It is monotonic, so it cannot move any
argmax -- accuracy and macro F1 are provably unchanged, and model selection is
untouched.

**It was tried on this model and rejected.** Global scaling fitted T=1.0226 and
left test ECE slightly worse (0.0554 -> 0.0604). The reason is that the
miscalibration is *conditional*, not global:

=========  =========  ==========  =====  ======
slice      mean conf  accuracy    gap    ECE
=========  =========  ==========  =====  ======
uniform    0.8760     0.8753      +0.00  0.0525
mixed      0.7706     0.5414      +0.23  0.2307
=========  =========  ==========  =====  ======

Uniform reviews are already near-perfectly calibrated; mixed reviews are wildly
over-confident. Fitted separately the two slices want *opposite* corrections
(T=2.097 and T=0.885), which cancel to T~1. Applied per slice the effect is
dramatic -- mixed-review confidently-wrong falls 30.1% -> 3.8% -- but "is this
review mixed?" is not knowable at inference time. Conditioning on aspect count
was tested as a proxy and does not work: the confidence gap is flat across
counts, including 1-aspect reviews that are 0% mixed.

So the honest position is that this model's confidence is trustworthy on
single-polarity reviews and inflated on mixed ones, and no post-hoc scaling
fixes that. The functions below are kept because ECE and confidently-wrong-rate
are the right diagnostics to track, and because the fix becomes available as
soon as a model can actually detect mixed reviews. See docs/model.md.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with an optional temperature."""
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean NLL of `labels` under the temperature-scaled distribution."""
    probabilities = softmax(logits, temperature)
    picked = probabilities[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    low: float = 0.05,
    high: float = 10.0,
    tolerance: float = 1e-4,
) -> float:
    """Find the temperature minimising NLL, by golden-section search.

    NLL as a function of temperature is smooth and unimodal, so a derivative-free
    line search is sufficient and avoids depending on an optimiser.

    Args:
        logits: Raw model outputs, shape ``(n, n_classes)``.
        labels: True class indices, shape ``(n,)``.
        low, high: Search bracket.
        tolerance: Stop when the bracket is narrower than this.

    Returns:
        The fitted temperature. ``> 1`` means the model was over-confident.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2-D logits, got shape {logits.shape}")
    if len(logits) != len(labels):
        raise ValueError(f"{len(logits)} logits vs {len(labels)} labels")

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = low, high
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = negative_log_likelihood(logits, labels, x1)
    f2 = negative_log_likelihood(logits, labels, x2)

    while right - left > tolerance:
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = negative_log_likelihood(logits, labels, x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = negative_log_likelihood(logits, labels, x2)

    return float((left + right) / 2.0)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> float:
    """Expected Calibration Error: mean |confidence - accuracy| across bins.

    Predictions are bucketed by confidence; within each bucket the gap between
    average confidence and actual accuracy is measured, then averaged weighted by
    bucket size. 0 is perfect; a well-calibrated model reporting 80% is right 80%
    of the time.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)

    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in pairwise(edges):
        # Lower-exclusive except for the first bin, so every point lands once.
        in_bin = (confidence > lower) & (confidence <= upper)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def confidently_wrong_rate(
    probabilities: np.ndarray, labels: np.ndarray, *, threshold: float = 0.7
) -> float:
    """Share of predictions that are both wrong and reported above `threshold`.

    The user-facing failure mode: a confident, incorrect answer is worse than a
    hedged one, because the interface invites the reader to trust it.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    wrong = probabilities.argmax(axis=1) != labels
    confident = probabilities.max(axis=1) > threshold
    return float((wrong & confident).mean())
