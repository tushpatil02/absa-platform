"""Tests for synthetic review generation and the paired bootstrap.

The generator exists because every dataset advertising "2025 phone reviews" that
could actually be downloaded turned out to be template output -- one ships
50,000 rows containing 110 distinct strings. So the test that matters most here
is the one that would have caught *that*: distinct-text share.

The bootstrap tests matter for a different reason. The sweep in
scripts/compare_synthetic.py reports a run-to-run standard deviation of exactly
0.0000 for the real-only row, because the solver is deterministic. Any positive
difference compared against that looks significant. These tests pin that the
bootstrap measures test-set sampling instead, and says "no effect" when there
is none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import random

from ml.evaluation.metrics import paired_bootstrap
from ml.synthetic.reviews import (
    ASPECTS,
    CLAIMS,
    POLARITIES,
    generate,
    generate_review,
    uniqueness,
)

# ---------------------------------------------------------------------------
# The check that would have caught the datasets this replaces
# ---------------------------------------------------------------------------


def test_generated_text_is_overwhelmingly_distinct():
    """The Kaggle "2025 Edition" scores 0.0022 here (110 unique of 50,000)."""
    reviews = generate(["Phone A", "Phone B"], 5000, seed=0)
    assert uniqueness(reviews) > 0.90


def test_uniqueness_detects_a_degenerate_generator():
    """The metric must actually fall when text repeats, or it guards nothing."""

    class Fake:
        text = "Great phone."

    assert uniqueness([Fake()] * 100) == pytest.approx(0.01)


def test_uniqueness_of_nothing_is_zero_not_a_crash():
    assert uniqueness([]) == 0.0


# ---------------------------------------------------------------------------
# Labels are exact by construction -- the one real advantage of synthetic data
# ---------------------------------------------------------------------------


def test_every_labelled_aspect_has_its_clause_in_the_text():
    """A label with no corresponding text would be a silent mislabel.

    Checked against the clause the generator actually emitted, not a keyword
    list: about 7% of battery clauses contain no battery word ("it drains
    overnight doing nothing"), so keyword matching would report false failures
    on exactly the realistic cases.
    """
    for review in generate(["Phone"], 500, seed=3):
        lowered = review.text.lower()
        assert set(review.clauses) == set(review.labels)
        for aspect, clause in review.clauses.items():
            assert clause.lower() in lowered, (
                f"clause for {aspect!r} missing from assembled text: {clause!r}"
            )


def test_labels_only_use_known_aspects_and_polarities():
    for review in generate(["Phone"], 300, seed=1):
        for aspect, polarity in review.labels.items():
            assert aspect in ASPECTS
            assert polarity in POLARITIES


def test_every_aspect_polarity_pair_has_templates():
    """A missing key would raise at generation time on a rare draw."""
    for aspect in ASPECTS:
        for polarity in POLARITIES:
            assert CLAIMS[(aspect, polarity)], f"no templates for {aspect}/{polarity}"


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


def test_most_multi_aspect_reviews_are_mixed_polarity():
    """Mixed reviews are the slice the model is worst at.

    A generator producing only uniform reviews would augment the easy half of
    the problem and leave the hard half untouched.
    """
    reviews = generate(["Phone"], 2000, seed=0)
    multi = [r for r in reviews if len(r.labels) > 1]
    assert multi
    assert sum(r.is_mixed for r in multi) / len(multi) > 0.5


def test_polarity_mix_resembles_the_real_corpus():
    """M-ABSA is 67.4% positive / 27.4% negative / 5.2% neutral.

    Drifting far from that would change the class balance the model trains on
    for a reason unrelated to the augmentation being tested.
    """
    reviews = generate(["Phone"], 3000, seed=0)
    labels = [p for r in reviews for p in r.labels.values()]
    positive = labels.count("positive") / len(labels)
    neutral = labels.count("neutral") / len(labels)
    assert 0.45 < positive < 0.75
    assert neutral < 0.15


def test_reviews_are_spread_across_phones():
    reviews = generate(["A", "B", "C", "D"], 400, seed=0)
    counts = {name: sum(r.phone == name for r in reviews) for name in "ABCD"}
    assert min(counts.values()) > 0
    assert max(counts.values()) - min(counts.values()) <= 1


# ---------------------------------------------------------------------------
# Determinism -- a generator that cannot be reproduced is not a dataset
# ---------------------------------------------------------------------------


def test_generation_is_deterministic_for_a_seed():
    assert [r.text for r in generate(["P"], 200, seed=7)] == [
        r.text for r in generate(["P"], 200, seed=7)
    ]


def test_different_seeds_give_different_text():
    assert [r.text for r in generate(["P"], 200, seed=0)] != [
        r.text for r in generate(["P"], 200, seed=1)
    ]


def test_single_review_is_well_formed():
    review = generate_review("Phone X", 0, random.Random(0))
    assert review.text
    assert review.text[0].isupper() or review.text[0].isdigit() or review.text.isupper()
    assert review.labels


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def test_identical_predictions_give_an_interval_containing_zero():
    y = np.array(["a", "b"] * 200)
    pred = np.array(["a", "b"] * 200)
    result = paired_bootstrap(y, pred, pred, accuracy, n_resamples=500, seed=0)
    assert result.point == 0.0
    assert result.includes_zero
    assert result.verdict == "not distinguishable from noise"


def test_a_clearly_better_model_gives_an_interval_excluding_zero():
    y = np.array(["a"] * 300)
    worse = np.array(["b"] * 300)
    better = np.array(["a"] * 300)
    result = paired_bootstrap(y, worse, better, accuracy, n_resamples=500, seed=0)
    assert result.point == pytest.approx(1.0)
    assert not result.includes_zero
    assert result.verdict == "improvement"


def test_a_worse_model_is_called_a_regression():
    y = np.array(["a"] * 300)
    result = paired_bootstrap(
        y, np.array(["a"] * 300), np.array(["b"] * 300), accuracy, n_resamples=500, seed=0
    )
    assert result.verdict == "regression"


def test_a_tiny_difference_is_not_called_significant():
    """The case the run-to-run sd got wrong.

    One extra correct answer out of 400 is not evidence of anything, and the
    interval has to say so.
    """
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b"], 400)
    baseline = y.copy()
    baseline[:80] = np.where(baseline[:80] == "a", "b", "a")  # 80 errors
    candidate = baseline.copy()
    candidate[0] = y[0]  # fix exactly one
    result = paired_bootstrap(y, baseline, candidate, accuracy, n_resamples=1000, seed=0)
    assert result.includes_zero


def test_bootstrap_is_deterministic_for_a_seed():
    y = np.array(["a", "b"] * 100)
    a = np.array(["a"] * 200)
    b = np.array(["b"] * 200)
    first = paired_bootstrap(y, a, b, accuracy, n_resamples=200, seed=5)
    second = paired_bootstrap(y, a, b, accuracy, n_resamples=200, seed=5)
    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="Length mismatch"):
        paired_bootstrap(np.array(["a", "b"]), np.array(["a"]), np.array(["a", "b"]), accuracy)
