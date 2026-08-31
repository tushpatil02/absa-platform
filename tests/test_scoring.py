"""Tests for the 1-10 sentiment score.

The properties matter more than the numbers: the score must be continuous,
monotonic, exactly anchored at its endpoints, and never conflated with
confidence. Those are the claims `docs/scoring.md` makes to a reader, so they
are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.inference.scoring import (
    SCORE_MAX,
    SCORE_MIN,
    aggregate_scores,
    build_score,
    positivity,
    score_label,
    to_score,
)

LABELS = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_confident_negative_scores_one():
    assert to_score(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_confident_positive_scores_ten():
    assert to_score(np.array([0.0, 0.0, 1.0])) == pytest.approx(10.0)


def test_pure_neutral_scores_midpoint():
    """5.5 is the midpoint of 1-10, not 5 -- the scale has no zero."""
    assert to_score(np.array([0.0, 1.0, 0.0])) == pytest.approx(5.5)


def test_perfectly_uncertain_also_scores_midpoint():
    """An even three-way split is, in expectation, neutral."""
    assert to_score(np.array([1 / 3, 1 / 3, 1 / 3])) == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_score_is_monotonic_in_positive_probability():
    previous = -np.inf
    for p in np.linspace(0, 1, 25):
        score = float(to_score(np.array([1 - p, 0.0, p])))
        assert score > previous
        previous = score


def test_score_always_within_bounds():
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet([1, 1, 1], size=2000)
    scores = to_score(probabilities)
    assert scores.min() >= SCORE_MIN - 1e-9
    assert scores.max() <= SCORE_MAX + 1e-9


def test_score_is_continuous_not_bucketed():
    """A barely-positive review must not jump to the top of the positive band."""
    barely = float(to_score(np.array([0.45, 0.10, 0.45])))
    strongly = float(to_score(np.array([0.02, 0.03, 0.95])))
    assert barely == pytest.approx(5.5)
    assert strongly > 9.0


def test_positivity_rejects_wrong_class_count():
    with pytest.raises(ValueError, match="Expected 3 classes"):
        positivity(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# build_score
# ---------------------------------------------------------------------------


def test_build_score_reports_argmax_polarity_and_its_probability():
    result = build_score(np.array([0.10, 0.15, 0.75]), LABELS)
    assert result.polarity == "positive"
    assert result.confidence == pytest.approx(0.75)
    assert result.score == pytest.approx(1 + 9 * (0.75 + 0.5 * 0.15), abs=0.01)


def test_confidence_is_independent_of_score():
    """Same score, very different certainty -- the two fields must not track."""
    confident = build_score(np.array([0.0, 1.0, 0.0]), LABELS)
    unsure = build_score(np.array([1 / 3, 1 / 3, 1 / 3]), LABELS)
    assert confident.score == pytest.approx(unsure.score)      # both 5.5
    assert confident.confidence > unsure.confidence            # 1.00 vs 0.33


def test_build_score_normalises_float_drift():
    result = build_score(np.array([0.1, 0.2, 0.7000001]), LABELS)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("bad", [np.array([0.0, 0.0, 0.0]), np.array([np.nan, 0.5, 0.5])])
def test_build_score_rejects_degenerate_input(bad):
    with pytest.raises(ValueError):
        build_score(bad, LABELS)


def test_build_score_rejects_label_mismatch():
    with pytest.raises(ValueError, match="probabilities for"):
        build_score(np.array([0.5, 0.5]), LABELS)


# ---------------------------------------------------------------------------
# Bands and aggregation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (1.0, "Extremely Negative"),
        (2.5, "Negative"),
        (4.0, "Slightly Negative"),
        (5.5, "Neutral"),
        (7.0, "Slightly Positive"),
        (8.5, "Positive"),
        (10.0, "Extremely Positive"),
    ],
)
def test_score_label_bands(score, expected):
    assert score_label(score) == expected


def test_pure_neutral_score_is_labelled_neutral():
    """Regression: 5.5 previously fell into 'Slightly Positive'."""
    assert score_label(float(to_score(np.array([0.0, 1.0, 0.0])))) == "Neutral"


def test_bands_are_symmetric_about_the_midpoint():
    """score s and its reflection (11 - s) must get mirrored labels."""
    mirror = {
        "Extremely Negative": "Extremely Positive",
        "Negative": "Positive",
        "Slightly Negative": "Slightly Positive",
        "Neutral": "Neutral",
    }
    for score in np.linspace(1.0, 5.5, 60):
        low = score_label(float(score))
        high = score_label(float(11.0 - score))
        assert mirror[low] == high, f"{score:.2f} -> {low} but {11 - score:.2f} -> {high}"


def test_every_score_in_range_gets_a_label():
    for score in np.linspace(1.0, 10.0, 200):
        assert score_label(float(score))


def test_aggregate_scores_matches_pooled_expectation():
    """Mean of per-review scores == score of the pooled distribution."""
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]])
    per_review = [float(to_score(row)) for row in probabilities]
    assert aggregate_scores(per_review) == pytest.approx(
        float(to_score(probabilities.mean(axis=0))), abs=0.01
    )


def test_aggregate_scores_handles_empty():
    assert aggregate_scores([]) is None
