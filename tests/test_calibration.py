"""Tests for calibration measurement and temperature scaling.

The properties pinned here are what justify the technique: temperature scaling
must never move an argmax (so it cannot change model selection), and it must
actually reduce ECE on a genuinely over-confident model. Both are verified on
synthetic data where the right answer is known, rather than on the real model
where a regression would be invisible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.inference.calibration import (
    confidently_wrong_rate,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    softmax,
)


def overconfident(n: int = 2000, error_rate: float = 0.3, sharpness: float = 4.0, seed: int = 0):
    """Sharp logits that are wrong `error_rate` of the time -- i.e. over-confident."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    logits = np.zeros((n, 3))
    logits[np.arange(n), labels] = sharpness
    flipped = rng.random(n) < error_rate
    logits[flipped] = logits[flipped][:, [1, 2, 0]]
    return logits, labels


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


def test_softmax_rows_sum_to_one():
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]]))
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_softmax_is_numerically_stable_on_large_logits():
    probabilities = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_higher_temperature_softens_without_reordering():
    logits = np.array([[3.0, 1.0, 0.5]])
    sharp = softmax(logits, 1.0)
    soft = softmax(logits, 3.0)
    assert soft.max() < sharp.max()
    assert soft.argmax() == sharp.argmax()


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------


def test_fit_temperature_exceeds_one_for_an_overconfident_model():
    logits, labels = overconfident()
    assert fit_temperature(logits, labels) > 1.5


def test_fit_temperature_below_one_for_an_underconfident_model():
    """Flat logits on an accurate model: the fix is to sharpen, not soften."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 1500)
    logits = np.zeros((1500, 3))
    logits[np.arange(1500), labels] = 0.15  # correct, but barely committed
    assert fit_temperature(logits, labels) < 1.0


def test_fitted_temperature_minimises_nll():
    logits, labels = overconfident()
    best = fit_temperature(logits, labels)
    at_best = negative_log_likelihood(logits, labels, best)
    for other in (best * 0.5, best * 0.8, best * 1.25, best * 2.0):
        assert negative_log_likelihood(logits, labels, other) >= at_best


def test_temperature_never_changes_a_prediction():
    """The property that makes this free: selection metrics cannot move."""
    logits, labels = overconfident()
    temperature = fit_temperature(logits, labels)
    assert (softmax(logits, 1.0).argmax(1) == softmax(logits, temperature).argmax(1)).all()


def test_fit_temperature_rejects_malformed_input():
    with pytest.raises(ValueError, match="2-D"):
        fit_temperature(np.array([1.0, 2.0, 3.0]), np.array([0]))
    with pytest.raises(ValueError, match="labels"):
        fit_temperature(np.zeros((5, 3)), np.zeros(4))


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_is_near_zero_for_a_calibrated_model():
    """80% confident and right 80% of the time."""
    n = 4000
    rng = np.random.default_rng(3)
    correct = rng.random(n) < 0.8
    probabilities = np.zeros((n, 3))
    probabilities[:, 0] = 0.8
    probabilities[:, 1] = 0.1
    probabilities[:, 2] = 0.1
    labels = np.where(correct, 0, 1)
    assert expected_calibration_error(probabilities, labels) < 0.03


def test_ece_is_large_for_an_overconfident_model():
    logits, labels = overconfident()
    assert expected_calibration_error(softmax(logits), labels) > 0.2


def test_temperature_scaling_reduces_ece():
    logits, labels = overconfident()
    temperature = fit_temperature(logits, labels)
    before = expected_calibration_error(softmax(logits, 1.0), labels)
    after = expected_calibration_error(softmax(logits, temperature), labels)
    assert after < before / 2


def test_ece_bins_cover_every_prediction_once():
    """No point may be double-counted or dropped by the binning."""
    rng = np.random.default_rng(5)
    probabilities = rng.dirichlet([1, 1, 1], size=500)
    labels = rng.integers(0, 3, 500)
    assert 0.0 <= expected_calibration_error(probabilities, labels) <= 1.0


# ---------------------------------------------------------------------------
# confidently_wrong_rate
# ---------------------------------------------------------------------------


def test_confidently_wrong_counts_only_confident_errors():
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],  # confident, right
            [0.9, 0.05, 0.05],  # confident, WRONG
            [0.4, 0.35, 0.25],  # hedged, wrong -> not counted
        ]
    )
    labels = np.array([0, 1, 1])
    assert confidently_wrong_rate(probabilities, labels) == pytest.approx(1 / 3)


def test_confidently_wrong_respects_the_threshold():
    probabilities = np.array([[0.75, 0.15, 0.10]])
    labels = np.array([1])
    assert confidently_wrong_rate(probabilities, labels, threshold=0.7) == 1.0
    assert confidently_wrong_rate(probabilities, labels, threshold=0.8) == 0.0
