"""Tests for the reliability gate.

A gate is only useful if it fails when it should. Most of these tests construct
data with a *known* answer -- signal, or none -- and check that the measurement
says so. The null tests matter most: a reliability check that reports "strong"
on random data would wave through exactly the result it exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.reliability import (
    compare_to_null,
    null_spread,
    split_half_reliability,
    star_baseline,
)


def make_reviews(
    phone_means: dict[str, float],
    *,
    n_per_phone: int = 40,
    noise: float = 1.0,
    aspect: str = "battery",
    seed: int = 0,
) -> pd.DataFrame:
    """Reviews whose per-phone means are known by construction."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, mean in phone_means.items():
        for index in range(n_per_phone):
            rows.append(
                {
                    "model_key": key,
                    "review_id": f"{key}-{index}",
                    "aspect": aspect,
                    "score": float(np.clip(rng.normal(mean, noise), 1, 10)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split-half reliability
# ---------------------------------------------------------------------------


def test_strong_signal_is_reported_as_strong():
    """Phones genuinely differ and within-phone noise is small."""
    means = {f"p{i}": 1.0 + 9.0 * i / 19 for i in range(20)}
    result = split_half_reliability(make_reviews(means, noise=0.5), "battery")
    assert result.spearman_brown > 0.9
    assert result.verdict == "strong"


def test_no_signal_is_reported_as_noise():
    """The test that matters.

    Every phone is identical, so the halves correlate only by accident. If this
    ever returns "strong", the gate is broken and would pass a null result.
    """
    means = {f"p{i}": 5.5 for i in range(30)}
    result = split_half_reliability(make_reviews(means, noise=3.0), "battery")
    assert result.spearman_brown < 0.4
    assert result.verdict == "NOISE"


def test_noise_swamping_a_real_difference_lowers_reliability():
    means = {f"p{i}": 1.0 + 9.0 * i / 19 for i in range(20)}
    clean = split_half_reliability(make_reviews(means, noise=0.3), "battery")
    noisy = split_half_reliability(make_reviews(means, noise=6.0), "battery")
    assert noisy.spearman_brown < clean.spearman_brown


def test_more_reviews_per_phone_raises_reliability():
    """Halves are more stable when each holds more reviews."""
    means = {f"p{i}": 1.0 + 9.0 * i / 19 for i in range(20)}
    few = split_half_reliability(make_reviews(means, n_per_phone=12, noise=3.0), "battery")
    many = split_half_reliability(make_reviews(means, n_per_phone=200, noise=3.0), "battery")
    assert many.spearman_brown > few.spearman_brown


def test_spearman_brown_corrects_upward():
    """Each half holds half the data, so the raw figure understates."""
    means = {f"p{i}": 1.0 + 9.0 * i / 19 for i in range(20)}
    result = split_half_reliability(make_reviews(means, noise=2.0), "battery")
    assert result.spearman_brown > result.raw_spearman


def test_phones_below_the_mention_floor_are_excluded():
    """A 2-review phone cannot make two meaningful halves."""
    frame = pd.concat(
        [
            make_reviews({f"p{i}": 5.0 for i in range(10)}, n_per_phone=40),
            make_reviews({"thin": 5.0}, n_per_phone=3, seed=1),
        ]
    )
    result = split_half_reliability(frame, "battery", min_mentions=10)
    assert result.n_phones == 10


def test_reliability_is_reproducible_for_a_seed():
    """The gate must not be re-rollable until it passes."""
    frame = make_reviews({f"p{i}": float(i) for i in range(1, 11)})
    first = split_half_reliability(frame, "battery", seed=7)
    second = split_half_reliability(frame, "battery", seed=7)
    assert first.spearman_brown == second.spearman_brown


def test_missing_aspect_returns_empty_rather_than_crashing():
    result = split_half_reliability(make_reviews({"a": 5.0}), "camera")
    assert result.n_phones == 0
    assert np.isnan(result.spearman_brown)


@pytest.mark.parametrize(
    "corrected,expected",
    [(0.95, "strong"), (0.7, "usable"), (0.5, "weak"), (0.1, "NOISE")],
)
def test_verdict_thresholds(corrected, expected):
    from ml.evaluation.reliability import AspectReliability

    result = AspectReliability("battery", 10, corrected, corrected, corrected, 0.5)
    assert result.verdict == expected


# ---------------------------------------------------------------------------
# The star-rating baseline
# ---------------------------------------------------------------------------


def _phones(keys, ratings):
    return pd.DataFrame({"model_key": list(keys), "avg_rating": list(ratings)})


def test_an_aspect_that_mirrors_stars_is_called_redundant():
    """If the pipeline just re-derives the star column, say so."""
    means = {f"p{i}": 1.0 + 9.0 * i / 9 for i in range(10)}
    frame = make_reviews(means, noise=0.05, n_per_phone=60)
    phones = _phones(means, [m / 2 for m in means.values()])
    result = star_baseline(frame, phones, "battery")
    assert result.r_squared > 0.9
    assert "REDUNDANT" in result.verdict


def test_an_aspect_unrelated_to_stars_is_called_independent():
    means = {f"p{i}": 1.0 + 9.0 * i / 9 for i in range(10)}
    frame = make_reviews(means, noise=0.05, n_per_phone=60)
    # Ratings deliberately unrelated to the aspect ordering.
    phones = _phones(means, [3.0, 4.5, 3.2, 4.9, 3.1, 4.7, 3.3, 4.4, 3.9, 4.1])
    result = star_baseline(frame, phones, "battery")
    assert result.r_squared < 0.4


def test_star_baseline_handles_too_few_phones():
    frame = make_reviews({"a": 5.0}, n_per_phone=40)
    result = star_baseline(frame, _phones(["a"], [4.0]), "battery")
    assert np.isnan(result.r_squared)


# ---------------------------------------------------------------------------
# The null simulation
# ---------------------------------------------------------------------------


def test_null_spread_is_non_trivial():
    """Identical phones still produce a visible spread.

    This is the number any real result has to clear. It is reported so that a
    2-point range is not mistaken for a finding when noise alone gives 1.8.
    """
    result = null_spread(n_phones=200, reviews_per_phone=100, score_std=3.0, seed=0)
    assert result["range"] > 1.0
    assert result["std"] > 0.0


def test_null_spread_shrinks_with_more_reviews():
    """Sampling noise falls as 1/sqrt(n), so the apparent spread must too."""
    few = null_spread(n_phones=200, reviews_per_phone=25, seed=0)
    many = null_spread(n_phones=200, reviews_per_phone=400, seed=0)
    assert many["std"] < few["std"]


def test_null_spread_is_deterministic():
    assert null_spread(seed=3) == null_spread(seed=3)


# ---------------------------------------------------------------------------
# The matched null
# ---------------------------------------------------------------------------


def test_real_differences_clear_the_matched_null():
    means = {f"p{i}": 1.0 + 9.0 * i / 19 for i in range(20)}
    result = compare_to_null(make_reviews(means, noise=1.0, n_per_phone=40), "battery")
    assert result.ratio > 1.5
    assert result.verdict in ("above null", "well above null")


def test_identical_phones_do_not_clear_the_matched_null():
    """The test that gives the comparison its point.

    Every phone is the same, so the observed spread is sampling noise and the
    matched null reproduces it. Anything else would mean the null was drawn
    from the wrong distribution.
    """
    means = {f"p{i}": 5.5 for i in range(30)}
    result = compare_to_null(make_reviews(means, noise=3.0, n_per_phone=40), "battery")
    assert result.ratio < 1.5
    assert result.verdict == "NOT ABOVE NULL"


def test_the_null_is_matched_to_the_aspect_not_a_fixed_variance():
    """A quiet aspect must get a quiet null.

    Using one fixed standard deviation for every aspect answers a question
    about a different corpus while looking like a check.
    """
    quiet = compare_to_null(
        make_reviews({f"p{i}": 5.5 for i in range(20)}, noise=0.5, n_per_phone=40), "battery"
    )
    loud = compare_to_null(
        make_reviews({f"p{i}": 5.5 for i in range(20)}, noise=5.0, n_per_phone=40), "battery"
    )
    assert quiet.null_std < loud.null_std


def test_matched_null_reports_the_counts_it_used():
    result = compare_to_null(
        make_reviews({f"p{i}": float(i + 1) for i in range(10)}, n_per_phone=44), "battery"
    )
    assert result.n_phones == 10
    assert result.reviews_per_phone == 44


def test_matched_null_handles_a_missing_aspect():
    result = compare_to_null(make_reviews({"a": 5.0}), "camera")
    assert result.n_phones == 0
