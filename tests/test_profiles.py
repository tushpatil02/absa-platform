"""Tests for per-phone profile aggregation.

Shrinkage is the interesting part, and its job is to be *sceptical*: a phone
with four enthusiastic camera sentences must not outrank one with two hundred
merely good ones. The tests below check that thin evidence is pulled toward the
catalogue mean, that abundant evidence is left alone, and -- the case that
matters most -- that an aspect carrying no real signal collapses rather than
inventing a ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.catalog.profiles import (
    build_profiles,
    estimate_shrinkage,
    shrink,
    to_wide,
)


def reviews(spec: dict[str, tuple[float, int]], aspect: str = "battery", seed: int = 0):
    """``{phone: (mean, count)}`` -> per-review rows with that mean exactly."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, (mean, count) in spec.items():
        # Centre the draws so the sample mean is exactly `mean`, making the
        # arithmetic in these tests checkable by hand.
        draws = rng.normal(0, 1.0, size=count)
        draws = draws - draws.mean() + mean
        for index, value in enumerate(draws):
            rows.append(
                {
                    "model_key": key,
                    "review_id": f"{key}-{index}",
                    "aspect": aspect,
                    "score": float(value),
                    "confidence": 0.8,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# shrink()
# ---------------------------------------------------------------------------


def test_shrinkage_pulls_thin_evidence_toward_the_mean():
    thin = shrink(np.array([10.0]), np.array([2.0]), grand_mean=5.0, k=20.0)
    assert 5.0 < thin[0] < 6.0


def test_shrinkage_leaves_abundant_evidence_alone():
    thick = shrink(np.array([10.0]), np.array([2000.0]), grand_mean=5.0, k=20.0)
    assert thick[0] == pytest.approx(10.0, abs=0.1)


def test_shrinkage_is_monotone_in_evidence():
    values = [shrink(np.array([9.0]), np.array([n]), 5.0, 20.0)[0] for n in (1, 5, 20, 100)]
    assert values == sorted(values)


def test_infinite_k_collapses_everything_to_the_mean():
    """No between-phone signal means no phone's own mean is informative."""
    result = shrink(np.array([1.0, 10.0]), np.array([50.0, 50.0]), 5.5, float("inf"))
    assert result.tolist() == [5.5, 5.5]


# ---------------------------------------------------------------------------
# estimate_shrinkage()
# ---------------------------------------------------------------------------


def test_real_differences_give_a_small_k():
    """Phones far apart relative to their noise should keep their own means."""
    spec = {f"p{i}": (1.0 + i, 60) for i in range(9)}
    params = estimate_shrinkage(reviews(spec), "battery")
    assert params.between_variance > 0
    assert params.k < 30


def test_identical_phones_give_an_infinite_k():
    """The case that must not invent a ranking.

    Every phone has the same true mean, so all observed spread is sampling
    noise. between_variance floors at 0 and k goes infinite, collapsing the
    scores rather than dressing noise up as discrimination.
    """
    spec = {f"p{i}": (5.5, 40) for i in range(15)}
    params = estimate_shrinkage(reviews(spec), "battery")
    assert params.between_variance == 0.0
    assert not np.isfinite(params.k)


def test_grand_mean_is_over_reviews_not_phones():
    spec = {"big": (2.0, 100), "small": (8.0, 10)}
    params = estimate_shrinkage(reviews(spec), "battery")
    # Weighted toward the phone with more reviews.
    assert params.grand_mean < 5.0


def test_missing_aspect_is_handled():
    params = estimate_shrinkage(reviews({"a": (5.0, 10)}), "camera")
    assert params.n_phones == 0
    assert not np.isfinite(params.k)


# ---------------------------------------------------------------------------
# build_profiles()
# ---------------------------------------------------------------------------


def test_profiles_carry_both_raw_and_shrunk_scores():
    """Shrinkage must stay inspectable rather than baked in silently."""
    frame = reviews({f"p{i}": (1.0 + i, 40) for i in range(8)})
    profiles, _ = build_profiles(frame)
    assert {"score", "raw_score", "mentions"} <= set(profiles.columns)
    assert (profiles["raw_score"] != profiles["score"]).any()


def test_phones_below_the_mention_floor_get_no_score():
    """An estimate that is almost entirely prior is not a measurement."""
    frame = pd.concat([reviews({f"p{i}": (5.0, 40) for i in range(5)}), reviews({"thin": (9.0, 2)}, seed=3)])
    profiles, _ = build_profiles(frame, min_mentions=5)
    assert "thin" not in set(profiles["model_key"])


def test_shrunk_scores_are_less_spread_than_raw():
    """The compression is the uncertainty; it must actually happen."""
    frame = reviews({f"p{i}": (1.0 + i, 8) for i in range(9)})
    profiles, _ = build_profiles(frame)
    assert profiles["score"].std() < profiles["raw_score"].std()


def test_no_rescaling_is_applied_after_shrinkage():
    """The regression this pins.

    Z-scoring or percentile-ranking shrunk scores puts back exactly the spread
    shrinkage removed. On a null where every phone is identical that produced a
    1.83-point range out of nothing, so the output must stay compressed.
    """
    frame = reviews({f"p{i}": (5.5, 30) for i in range(20)})
    profiles, params = build_profiles(frame)
    assert not np.isfinite(params[0].k)
    assert profiles["score"].std() == pytest.approx(0.0, abs=1e-9)


def test_mentions_are_counted_per_aspect():
    frame = pd.concat(
        [reviews({"a": (5.0, 30)}, aspect="battery"), reviews({"a": (7.0, 12)}, aspect="camera")]
    )
    profiles, _ = build_profiles(frame)
    by_aspect = profiles.set_index("aspect")["mentions"]
    assert by_aspect["battery"] == 30
    assert by_aspect["camera"] == 12


def test_build_profiles_is_deterministic():
    frame = reviews({f"p{i}": (1.0 + i, 20) for i in range(6)})
    first, _ = build_profiles(frame)
    second, _ = build_profiles(frame)
    assert first.equals(second)


# ---------------------------------------------------------------------------
# to_wide()
# ---------------------------------------------------------------------------


def test_to_wide_pivots_to_one_row_per_phone():
    frame = pd.concat(
        [
            reviews({f"p{i}": (5.0, 20) for i in range(4)}, aspect=aspect, seed=index)
            for index, aspect in enumerate(("battery", "camera"))
        ]
    )
    profiles, _ = build_profiles(frame)
    wide = to_wide(profiles, ("battery", "camera"))
    assert list(wide.columns) == ["battery", "camera"]
    assert len(wide) == 4


def test_a_phone_missing_an_axis_is_dropped_not_imputed():
    """Filling a gap with the mean would present a guess as a measurement."""
    frame = pd.concat(
        [
            reviews({"a": (5.0, 20), "b": (6.0, 20)}, aspect="battery"),
            reviews({"a": (7.0, 20)}, aspect="camera", seed=2),
        ]
    )
    profiles, _ = build_profiles(frame)
    wide = to_wide(profiles, ("battery", "camera"))
    assert set(wide.index) == {"a"}


def test_to_wide_raises_when_an_axis_is_entirely_absent():
    profiles, _ = build_profiles(reviews({"a": (5.0, 20)}))
    with pytest.raises(KeyError, match="No profiles for aspects"):
        to_wide(profiles, ("battery", "camera"))
