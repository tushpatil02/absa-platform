"""Tests for the shared inference layer.

The critical property here is **train/serve parity**: the predictor must clean
text with the same function that built the training data. If those ever diverge
the model is served inputs it was never trained on, and no test on either side
would notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.inference.predictor import (
    AnalysisResult,
    AspectPrediction,
    EmptyReviewError,
    ReviewTooLongError,
    load_predictor,
    summarise_product,
)
from ml.inference.scoring import build_score
from ml.preprocessing.transform import load_taxonomy

TAXONOMY = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
POLARITIES = list(TAXONOMY.polarities)
MODELS_DIR = REPO_ROOT / "models"
HAS_MODELS = (MODELS_DIR / "baseline_sentiment_classifier" / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Product aggregation — pure, no model needed
# ---------------------------------------------------------------------------


def _prediction(aspect: str, probabilities: list[float]) -> AspectPrediction:
    return AspectPrediction(
        aspect=aspect,
        display_name=aspect.title(),
        detection_confidence=0.9,
        sentiment=build_score(np.array(probabilities), POLARITIES),
    )


def _result(*predictions: AspectPrediction) -> AnalysisResult:
    return AnalysisResult(
        review="r", cleaned="r", aspects=list(predictions),
        overall_score=None, model="test",
    )


def test_summarise_counts_mentions_per_aspect():
    results = [
        _result(_prediction("battery", [0.9, 0.05, 0.05])),
        _result(_prediction("battery", [0.1, 0.1, 0.8]), _prediction("camera", [0.05, 0.05, 0.9])),
    ]
    summary = summarise_product(results, POLARITIES)

    by_aspect = {entry["aspect"]: entry for entry in summary["aspects"]}
    assert by_aspect["battery"]["mentions"] == 2
    assert by_aspect["camera"]["mentions"] == 1
    assert summary["reviews_analyzed"] == 2


def test_summarise_shares_are_over_mentions_not_all_reviews():
    """'61% negative on battery' must mean 61% of those who discussed battery."""
    results = [
        _result(_prediction("battery", [0.9, 0.05, 0.05])),   # negative
        _result(_prediction("camera", [0.05, 0.05, 0.9])),    # different aspect
    ]
    summary = summarise_product(results, POLARITIES)
    battery = next(e for e in summary["aspects"] if e["aspect"] == "battery")

    # One mention, all of it negative -> 100%, not 50% of the two reviews.
    assert battery["mentions"] == 1
    assert battery["shares"]["negative"] == pytest.approx(1.0)
    assert battery["mention_share"] == pytest.approx(0.5)


def test_summarise_shares_sum_to_one():
    results = [
        _result(_prediction("battery", [0.9, 0.05, 0.05])),
        _result(_prediction("battery", [0.05, 0.9, 0.05])),
        _result(_prediction("battery", [0.05, 0.05, 0.9])),
    ]
    summary = summarise_product(results, POLARITIES)
    battery = summary["aspects"][0]

    # Counts are exact.
    assert sum(battery["counts"].values()) == battery["mentions"]

    # Shares are rounded to 4dp for payload size, so three 1/3 shares sum to
    # 0.9999, not 1.0. The tolerance is the maximum rounding error: one half-ulp
    # of the last decimal per class.
    tolerance = len(POLARITIES) * 0.00005
    assert sum(battery["shares"].values()) == pytest.approx(1.0, abs=tolerance)


def test_summarise_identifies_extremes():
    results = [
        _result(
            _prediction("camera", [0.02, 0.03, 0.95]),
            _prediction("battery", [0.95, 0.03, 0.02]),
        )
    ]
    summary = summarise_product(results, POLARITIES)
    assert summary["most_positive"]["aspect"] == "camera"
    assert summary["most_negative"]["aspect"] == "battery"


def test_summarise_sorts_by_mentions():
    results = [
        _result(_prediction("battery", [0.9, 0.05, 0.05])),
        _result(_prediction("battery", [0.9, 0.05, 0.05])),
        _result(_prediction("camera", [0.9, 0.05, 0.05])),
    ]
    summary = summarise_product(results, POLARITIES)
    mentions = [entry["mentions"] for entry in summary["aspects"]]
    assert mentions == sorted(mentions, reverse=True)


def test_summarise_handles_no_results():
    summary = summarise_product([], POLARITIES)
    assert summary["reviews_analyzed"] == 0
    assert summary["aspects"] == []
    assert summary["most_positive"] is None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_predictor_names_the_fix_when_nothing_is_trained(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baseline"):
        load_predictor(tmp_path, TAXONOMY)


def test_load_predictor_rejects_missing_transformer(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_transformer"):
        load_predictor(tmp_path, TAXONOMY, prefer="transformer")


def test_auto_ignores_a_half_finished_transformer(tmp_path):
    """Both transformer stages must exist before 'auto' picks it.

    This state occurs for real: training stage B writes sentiment_classifier/
    while stage A is still running. Pairing a transformer sentiment head with a
    missing aspect detector would fail at request time instead of at load time.
    """
    import json
    import shutil

    # A complete baseline plus only ONE half of the transformer.
    for name in ("baseline_aspect_detector", "baseline_sentiment_classifier"):
        source = MODELS_DIR / name
        if not (source / "metadata.json").exists():
            pytest.skip("Baseline models not trained")
        shutil.copytree(source, tmp_path / name)

    half = tmp_path / "sentiment_classifier"
    half.mkdir()
    (half / "metadata.json").write_text(json.dumps({"base_model": "x"}), encoding="utf-8")

    predictor = load_predictor(tmp_path, TAXONOMY, prefer="auto")
    assert predictor.model_name.startswith("baseline:")


# ---------------------------------------------------------------------------
# Behaviour against the real baseline artefacts
# ---------------------------------------------------------------------------

pytestmark_models = pytest.mark.skipif(
    not HAS_MODELS, reason="Baseline models not trained; run scripts/train_baseline.py"
)


@pytest.fixture(scope="module")
def predictor():
    if not HAS_MODELS:
        pytest.skip("Baseline models not trained")
    return load_predictor(MODELS_DIR, TAXONOMY, prefer="baseline")


@pytestmark_models
def test_predictor_uses_the_training_cleaner(predictor):
    """Parity check: the cleaned text must match clean_text() exactly."""
    from ml.preprocessing.clean import clean_text

    raw = "<b>Great</b> camera &amp; fast delivery!!!!!! See https://example.com"
    result = predictor.analyze(raw)
    assert result.cleaned == clean_text(raw)
    assert "<b>" not in result.cleaned
    assert "[URL]" in result.cleaned


@pytestmark_models
def test_predictor_returns_valid_scores(predictor):
    result = predictor.analyze("The camera is great but the battery is terrible.")
    assert result.aspects
    for prediction in result.aspects:
        sentiment = prediction.sentiment
        assert 1.0 <= sentiment.score <= 10.0
        assert 0.0 <= sentiment.confidence <= 1.0
        assert sentiment.polarity in POLARITIES
        assert sum(sentiment.probabilities.values()) == pytest.approx(1.0, abs=0.01)


@pytestmark_models
def test_predictor_orders_by_detection_confidence(predictor):
    result = predictor.analyze(
        "Great camera, poor battery, slow software, and the screen is lovely."
    )
    confidences = [p.detection_confidence for p in result.aspects]
    assert confidences == sorted(confidences, reverse=True)


@pytestmark_models
def test_predictor_respects_top_k(predictor):
    result = predictor.analyze(
        "Great camera, poor battery, slow software, lovely screen, cheap price.", top_k=2
    )
    assert len(result.aspects) <= 2


@pytestmark_models
def test_predictor_always_returns_an_aspect(predictor):
    """Short reviews may clear no threshold; an empty list reads as failure."""
    assert predictor.analyze("Love it").aspects


@pytestmark_models
@pytest.mark.parametrize("bad", ["", "   ", "\n\t ", "!!! ??? ..."])
def test_predictor_rejects_unusable_text(predictor, bad):
    with pytest.raises(EmptyReviewError):
        predictor.analyze(bad)


@pytestmark_models
def test_predictor_rejects_overlong_text(predictor):
    with pytest.raises(ReviewTooLongError, match="5000"):
        predictor.analyze("word " * 2000)


@pytestmark_models
def test_predictor_is_deterministic(predictor):
    """Same input, same output — required for reproducible demos and tests."""
    review = "The camera is excellent but the battery drains quickly."
    first = predictor.analyze(review)
    second = predictor.analyze(review)
    assert [p.sentiment.score for p in first.aspects] == [
        p.sentiment.score for p in second.aspects
    ]


@pytestmark_models
def test_predictor_probabilities_are_in_canonical_label_order(predictor):
    """Guards the classes_ remap: a mislabelled column would invert polarity."""
    result = predictor.analyze("Absolutely terrible battery, dies in an hour.")
    battery = next((p for p in result.aspects if p.aspect == "battery"), None)
    if battery is None:
        pytest.skip("battery not detected in this phrasing")
    # An emphatically negative sentence must not come back as confidently positive.
    assert battery.sentiment.probabilities["negative"] > battery.sentiment.probabilities["positive"]


@pytestmark_models
def test_batch_analysis_matches_individual_analysis(predictor):
    reviews = ["Great camera!", "Terrible battery."]
    batch = predictor.analyze_batch(reviews)
    individual = [predictor.analyze(review) for review in reviews]
    assert [len(r.aspects) for r in batch] == [len(r.aspects) for r in individual]
