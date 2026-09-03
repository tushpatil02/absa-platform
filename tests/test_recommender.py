"""Tests for slider ranking and the price axis.

The properties worth pinning are the ones that make the sliders *mean*
something: exceeding a requirement is never punished, an axis set to 1 drops
out of the ranking entirely, and the match percentage is relative to what was
actually asked for. Those are the claims the UI makes to the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.recommender.price import price_to_score, score_to_price
from ml.recommender.similarity import AXES, as_vector, match_percent, rank, shortfall


def profile(battery=5.0, camera=5.0, price=5.0, display=5.0, performance=5.0):
    return {
        "battery": battery,
        "camera": camera,
        "price": price,
        "display": display,
        "performance": performance,
    }


# ---------------------------------------------------------------------------
# Asymmetry -- the decision that defines the model
# ---------------------------------------------------------------------------


def test_exceeding_a_requirement_is_never_punished():
    """A shopper asking for 6 does not want the 9 ranked below the 6."""
    wanted = as_vector(profile(battery=6.0))
    exact = as_vector(profile(battery=6.0))
    better = as_vector(profile(battery=9.0))
    assert match_percent(wanted, better) >= match_percent(wanted, exact)


def test_a_perfect_phone_and_an_exactly_adequate_one_both_match_fully():
    wanted = as_vector(profile(battery=6.0, camera=6.0))
    assert match_percent(wanted, as_vector(profile(battery=6.0, camera=6.0))) == 100.0
    assert match_percent(wanted, as_vector(profile(battery=10.0, camera=10.0))) == 100.0


def test_shortfall_is_one_sided():
    gaps = shortfall(np.array([5.0, 5.0]), np.array([3.0, 9.0]))
    assert gaps.tolist() == [2.0, 0.0]


def test_falling_short_reduces_the_match():
    wanted = as_vector(profile(camera=9.0))
    good = match_percent(wanted, as_vector(profile(camera=9.0)))
    poor = match_percent(wanted, as_vector(profile(camera=2.0)))
    assert poor < good


# ---------------------------------------------------------------------------
# Slider semantics
# ---------------------------------------------------------------------------


def test_an_axis_set_to_one_stops_affecting_the_ranking():
    """"I don't care about this" must actually mean it.

    Every phone clears a requirement of 1, so an axis parked at the minimum can
    never separate two phones.
    """
    wanted = profile(camera=9.0, battery=1.0, price=1.0, display=1.0, performance=1.0)
    phones = {
        "good-camera-bad-battery": profile(camera=9.0, battery=1.0),
        "good-camera-good-battery": profile(camera=9.0, battery=10.0),
    }
    results = rank(wanted, phones)
    assert {m.match_percent for m in results} == {100.0}


def test_only_the_requested_axis_orders_the_result():
    wanted = profile(camera=10.0, battery=1.0, price=1.0, display=1.0, performance=1.0)
    phones = {
        "weak-camera": profile(camera=2.0, battery=10.0, display=10.0),
        "strong-camera": profile(camera=9.0, battery=1.0, display=1.0),
    }
    assert rank(wanted, phones)[0].model_key == "strong-camera"


def test_all_sliders_at_minimum_match_everything():
    """Nothing was asked for, so nothing can fail -- and no divide by zero."""
    wanted = profile(1.0, 1.0, 1.0, 1.0, 1.0)
    phones = {"anything": profile(1.0, 1.0, 1.0, 1.0, 1.0)}
    assert rank(wanted, phones)[0].match_percent == 100.0


def test_match_is_relative_to_what_was_asked():
    """The same phone matches a modest query better than a demanding one."""
    phones = {"mid": profile(battery=5.0)}
    modest = rank(profile(battery=5.0), phones)[0].match_percent
    demanding = rank(profile(battery=10.0), phones)[0].match_percent
    assert modest > demanding


# ---------------------------------------------------------------------------
# Ranking contract
# ---------------------------------------------------------------------------


def test_results_are_sorted_by_match_descending():
    phones = {
        "a": profile(battery=9.0),
        "b": profile(battery=5.0),
        "c": profile(battery=1.0),
    }
    results = rank(profile(battery=8.0), phones)
    percentages = [m.match_percent for m in results]
    assert percentages == sorted(percentages, reverse=True)


def test_ties_break_on_overall_quality():
    """Two phones meeting every requirement are ordered by how good they are."""
    phones = {
        "adequate": profile(3.0, 3.0, 3.0, 3.0, 3.0),
        "excellent": profile(9.0, 9.0, 9.0, 9.0, 9.0),
    }
    results = rank(profile(2.0, 2.0, 2.0, 2.0, 2.0), phones)
    assert [m.match_percent for m in results] == [100.0, 100.0]
    assert results[0].model_key == "excellent"


def test_ranking_is_deterministic():
    phones = {name: profile(battery=5.0) for name in ("a", "b", "c")}
    assert [m.model_key for m in rank(profile(), phones)] == [
        m.model_key for m in rank(profile(), phones)
    ]


def test_limit_truncates():
    phones = {name: profile(battery=float(i)) for i, name in enumerate("abcde", start=1)}
    assert len(rank(profile(battery=5.0), phones, limit=2)) == 2


def test_worst_axis_names_the_biggest_gap():
    wanted = profile(camera=9.0, battery=8.0)
    result = rank(wanted, {"p": profile(camera=2.0, battery=7.0)})[0]
    assert result.worst_axis == "camera"


def test_worst_axis_is_none_when_everything_is_met():
    assert rank(profile(2.0, 2.0, 2.0, 2.0, 2.0), {"p": profile(9.0, 9.0, 9.0, 9.0, 9.0)})[
        0
    ].worst_axis is None


def test_match_percent_is_bounded():
    for requirement in (1.0, 5.0, 10.0):
        for value in (1.0, 5.0, 10.0):
            percent = match_percent(as_vector(profile(**{a: requirement for a in AXES})),
                                    as_vector(profile(**{a: value for a in AXES})))
            assert 0.0 <= percent <= 100.0


def test_a_missing_axis_raises_rather_than_defaulting():
    """Defaulting a missing axis to 0 would fail a requirement never measured."""
    with pytest.raises(KeyError, match="Missing axes"):
        as_vector({"battery": 5.0})


# ---------------------------------------------------------------------------
# Price axis
# ---------------------------------------------------------------------------


def test_cheapest_scores_ten_and_dearest_scores_one():
    assert price_to_score(100, 100, 1000) == pytest.approx(10.0)
    assert price_to_score(1000, 100, 1000) == pytest.approx(1.0)


def test_price_score_decreases_with_price():
    scores = [price_to_score(p, 100, 1000) for p in (100, 200, 400, 800, 1000)]
    assert scores == sorted(scores, reverse=True)


def test_price_is_logarithmic_not_linear():
    """Equal ratios must give equal steps.

    $100 to $200 should move the score as much as $400 to $800; a linear
    mapping would squash the whole upper catalogue into the top of the scale.
    """
    step_low = price_to_score(100, 100, 1600) - price_to_score(200, 100, 1600)
    step_high = price_to_score(400, 100, 1600) - price_to_score(800, 100, 1600)
    assert step_low == pytest.approx(step_high, abs=1e-9)


def test_prices_outside_the_range_are_clipped_not_extrapolated():
    assert price_to_score(50, 100, 1000) == pytest.approx(10.0)
    assert price_to_score(5000, 100, 1000) == pytest.approx(1.0)


def test_a_single_price_catalogue_returns_the_midpoint():
    """Claiming every phone is a bargain would be the dishonest answer."""
    assert price_to_score(200, 200, 200) == pytest.approx(5.5)


def test_price_round_trips():
    for price in (120.0, 250.0, 640.0):
        score = price_to_score(price, 100, 1000)
        assert score_to_price(score, 100, 1000) == pytest.approx(price, rel=1e-6)


@pytest.mark.parametrize("cheapest,dearest", [(0, 100), (-5, 100), (100, 50)])
def test_invalid_price_bounds_raise(cheapest, dearest):
    with pytest.raises(ValueError):
        price_to_score(50, cheapest, dearest)
