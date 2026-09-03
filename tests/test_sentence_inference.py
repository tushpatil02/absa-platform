"""Tests for sentence-level inference and the multi-sentence benchmark.

The composition logic is tested against stub stages rather than trained
artefacts. That keeps these fast and, more importantly, deterministic: the
question here is whether the predictor *combines* per-sentence evidence
correctly, which should not depend on what a model happens to predict today.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.multi_sentence import build_benchmark, evaluate_mode
from ml.inference.predictor import Predictor
from ml.preprocessing.transform import load_taxonomy

TAXONOMY = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
ASPECTS = list(TAXONOMY.aspect_ids)
POLARITIES = list(TAXONOMY.polarities)  # negative, neutral, positive

NEGATIVE = [0.9, 0.05, 0.05]
POSITIVE = [0.05, 0.05, 0.9]
NEUTRAL = [0.1, 0.8, 0.1]


class KeywordDetector:
    """Fires on an aspect when its keyword appears in the unit."""

    name = "stub-detector"
    threshold = 0.5

    def __init__(self, keywords: dict[str, str], confidence: float = 0.9):
        self.keywords = keywords
        self.confidence = confidence

    def detect(self, text: str) -> np.ndarray:
        scores = np.zeros(len(ASPECTS))
        lowered = text.lower()
        for aspect, keyword in self.keywords.items():
            if keyword in lowered:
                scores[ASPECTS.index(aspect)] = self.confidence
        return scores


class KeywordClassifier:
    """Returns a fixed distribution based on a sentiment word in the unit."""

    name = "stub-classifier"

    def classify(self, text: str, aspect_descriptions: list[str]) -> np.ndarray:
        lowered = text.lower()
        if "terrible" in lowered or "awful" in lowered:
            row = NEGATIVE
        elif "great" in lowered or "excellent" in lowered:
            row = POSITIVE
        else:
            row = NEUTRAL
        return np.array([row for _ in aspect_descriptions], dtype=float)


@pytest.fixture
def predictor():
    detector = KeywordDetector({"battery": "battery", "camera": "camera"})
    return Predictor(detector, KeywordClassifier(), TAXONOMY)


# ---------------------------------------------------------------------------
# The core behaviour
# ---------------------------------------------------------------------------


def test_opposite_opinions_are_kept_apart(predictor):
    """The whole point.

    Scored as one unit, "great battery ... terrible camera" gives the classifier
    a single input carrying two opinions and it must pick one. Per sentence,
    each aspect gets the sentence that is actually about it.
    """
    review = "The battery is great. The camera is terrible."
    result = predictor.analyze(review)

    by_aspect = {p.aspect: p.sentiment.polarity for p in result.aspects}
    assert by_aspect["battery"] == "positive"
    assert by_aspect["camera"] == "negative"


def test_whole_review_mode_collapses_them(predictor):
    """The comparison case, so the test above is not measuring nothing.

    With one unit the stub classifier sees both sentiment words and returns one
    distribution for every aspect -- which is exactly how the real model behaves
    on mixed input, at 0.541 accuracy.
    """
    review = "The battery is great. The camera is terrible."
    result = predictor.analyze(review, by_sentence=False)

    polarities = {p.sentiment.polarity for p in result.aspects}
    assert len(polarities) == 1


def test_evidence_names_the_sentence_that_produced_the_score(predictor):
    result = predictor.analyze("The battery is great. The camera is terrible.")
    by_aspect = {p.aspect: p for p in result.aspects}

    assert by_aspect["battery"].evidence == ("The battery is great.",)
    assert by_aspect["camera"].evidence == ("The camera is terrible.",)
    assert by_aspect["battery"].mentions == 1


def test_repeated_mentions_all_become_evidence(predictor):
    result = predictor.analyze(
        "The battery is great. Battery life is excellent. The camera is terrible."
    )
    battery = next(p for p in result.aspects if p.aspect == "battery")
    assert battery.mentions == 2
    assert all("attery" in sentence for sentence in battery.evidence)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_conflicting_sentences_average_rather_than_last_write_wins(predictor):
    """Two opposite sentences about one aspect must land between them.

    Taking whichever sentence came last would make the score depend on writing
    order, and taking the most confident would throw away half the evidence.
    """
    result = predictor.analyze("The battery is great. The battery is terrible.")
    battery = next(p for p in result.aspects if p.aspect == "battery")

    assert battery.mentions == 2
    assert 3.5 < battery.sentiment.score < 7.5


def test_aggregation_is_weighted_by_detection_confidence():
    """A sentence the detector is surer about should count for more."""

    class GradedDetector(KeywordDetector):
        def detect(self, text: str) -> np.ndarray:
            scores = np.zeros(len(ASPECTS))
            if "battery" in text.lower():
                # Confident on the praise, barely above threshold on the complaint.
                scores[ASPECTS.index("battery")] = 0.99 if "great" in text else 0.51
            return scores

    predictor = Predictor(GradedDetector({}), KeywordClassifier(), TAXONOMY)
    result = predictor.analyze("The battery is great. The battery is terrible.")
    battery = next(p for p in result.aspects if p.aspect == "battery")

    # Weighted toward the high-confidence positive sentence, not the midpoint.
    assert battery.sentiment.score > 5.5


def test_detection_confidence_is_the_strongest_mention(predictor):
    result = predictor.analyze("The battery is great. The camera is terrible.")
    for prediction in result.aspects:
        assert prediction.detection_confidence == pytest.approx(0.9)


def test_aspects_are_ordered_by_detection_confidence(predictor):
    result = predictor.analyze("The battery is great. The camera is terrible.")
    confidences = [p.detection_confidence for p in result.aspects]
    assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Fallbacks and contract
# ---------------------------------------------------------------------------


def test_a_review_with_no_confident_aspect_still_returns_one(predictor):
    """Returning nothing reads as a failure to a user, so fall back."""
    result = predictor.analyze("It arrived on Tuesday and I opened the box.")
    assert result.aspects
    assert len(result.aspects) == 1


def test_unrelated_sentences_do_not_vote(predictor):
    """A sentence about nothing must not force its best-guess aspect through.

    Applying the whole-review argmax fallback per sentence would make every
    "Arrived Tuesday." add a spurious mention.
    """
    result = predictor.analyze(
        "The battery is great. It arrived Tuesday. The weather was fine."
    )
    assert [p.aspect for p in result.aspects] == ["battery"]
    assert result.aspects[0].mentions == 1


def test_top_k_limits_the_result(predictor):
    result = predictor.analyze("The battery is great. The camera is terrible.", top_k=1)
    assert len(result.aspects) == 1


def test_batch_matches_individual(predictor):
    reviews = ["The battery is great.", "The camera is terrible."]
    batch = predictor.analyze_batch(reviews)
    individual = [predictor.analyze(review) for review in reviews]
    assert [r.as_dict() for r in batch] == [r.as_dict() for r in individual]


def test_result_serialises_evidence(predictor):
    payload = predictor.analyze("The battery is great.").as_dict()
    aspect = payload["aspects"][0]
    assert aspect["mentions"] == 1
    assert aspect["evidence"] == ["The battery is great."]


def test_sentence_level_is_deterministic(predictor):
    review = "The battery is great. The camera is terrible."
    assert predictor.analyze(review).as_dict() == predictor.analyze(review).as_dict()


# ---------------------------------------------------------------------------
# The benchmark builder
# ---------------------------------------------------------------------------


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"review_id": "a", "text": "The battery is great.", "aspect": "battery", "polarity": "positive"},
            {"review_id": "b", "text": "The camera is terrible.", "aspect": "camera", "polarity": "negative"},
            {"review_id": "c", "text": "The screen is great.", "aspect": "display", "polarity": "positive"},
            {"review_id": "d", "text": "Shipping was slow.", "aspect": "delivery", "polarity": "negative"},
        ]
    )


def test_benchmark_components_never_contradict_each_other():
    """Gold labels must be unambiguous.

    Composing two rows that both annotate `battery` with different polarities
    would make the union self-contradictory, and any accuracy computed against
    it meaningless.
    """
    reviews = build_benchmark(_pairs(), n_reviews=30, seed=0)
    assert reviews
    for review in reviews:
        assert len(review.gold) == len(set(review.gold))
        assert len(review.gold) >= 2


def test_benchmark_gold_is_the_union_of_its_parts():
    reviews = build_benchmark(_pairs(), n_reviews=20, seed=1)
    for review in reviews:
        assert len(review.source_ids) == len(set(review.source_ids))
        for part in review.parts:
            assert part.rstrip(".") in review.text


def test_benchmark_is_deterministic_for_a_seed():
    first = build_benchmark(_pairs(), n_reviews=20, seed=7)
    second = build_benchmark(_pairs(), n_reviews=20, seed=7)
    assert [r.text for r in first] == [r.text for r in second]


def test_benchmark_marks_mixed_reviews():
    reviews = build_benchmark(_pairs(), n_reviews=40, seed=0)
    assert any(r.is_mixed for r in reviews)
    for review in reviews:
        assert review.is_mixed == (len(set(review.gold.values())) > 1)


def test_stripping_terminators_removes_the_boundary_cues():
    reviews = build_benchmark(_pairs(), n_reviews=20, seed=0, strip_terminators=True)
    # Only the final component may keep punctuation it never had; none is added.
    assert all(". " not in review.text for review in reviews)


def test_evaluate_mode_reports_the_collapse(predictor):
    """End-to-end: the metric must show sentence mode beating whole-review."""
    reviews = build_benchmark(_pairs(), n_reviews=40, seed=0)
    whole = evaluate_mode(predictor, reviews, by_sentence=False)
    sentence = evaluate_mode(predictor, reviews, by_sentence=True)

    assert whole.collapsed_rate > sentence.collapsed_rate
    assert sentence.mixed_sentiment_accuracy > whole.mixed_sentiment_accuracy
