"""Measure inference on multi-sentence reviews.

Why a constructed benchmark
---------------------------
The change under test is running inference per sentence rather than per review.
M-ABSA cannot measure it directly: every row is already a single sentence, so
splitting is a no-op and both modes score identically. The gap only appears on
the input the deployed system actually receives -- an Amazon review averaging
~50 words that praises one aspect and criticises another.

So the benchmark is built by *composing* held-out test rows into pseudo-reviews:

* components are drawn from the test split only, so no training text is touched;
* components must have **disjoint aspect sets**, so the gold labels of one can
  never contradict another's and the union is unambiguous;
* every label is a real human annotation. Nothing about the sentiment is
  synthesised -- only the concatenation is.

What it can and cannot show
---------------------------
It shows whether splitting recovers per-aspect opinions that whole-review
inference collapses. It does *not* show real-world accuracy on Amazon prose:
composed reviews have cleaner topic boundaries than genuine ones, and no
discourse connecting the parts. Treat the numbers as an upper bound on the
benefit, which is why :func:`build_benchmark` also emits a ``joined`` variant
with the sentence terminators stripped -- the adversarial case where the writer
supplied no boundaries at all.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

# Terminators that already end a sentence, so composing need not add one.
_TERMINATORS = ".!?"


@dataclass(frozen=True)
class PseudoReview:
    """Several test sentences concatenated into one review."""

    text: str
    gold: dict[str, str]
    parts: tuple[str, ...]
    source_ids: tuple[str, ...]

    @property
    def is_mixed(self) -> bool:
        """True when the review carries more than one polarity.

        This is the slice that matters: uniform reviews are the ones the model
        already handles at 0.875, and mixed ones are where it falls to 0.541.
        """
        return len(set(self.gold.values())) > 1


@dataclass
class ModeResult:
    """Scores for one inference mode."""

    mode: str
    detection_recall: float = 0.0
    sentiment_accuracy: float = 0.0
    mixed_sentiment_accuracy: float = 0.0
    uniform_sentiment_accuracy: float = 0.0
    collapsed_rate: float = 0.0
    n_reviews: int = 0
    n_gold_aspects: int = 0
    n_scored: int = 0

    def summary(self) -> str:
        return (
            f"  {self.mode:<16s}"
            f"  detect_recall={self.detection_recall:.4f}"
            f"  sent_acc={self.sentiment_accuracy:.4f}"
            f"  mixed={self.mixed_sentiment_accuracy:.4f}"
            f"  uniform={self.uniform_sentiment_accuracy:.4f}"
            f"  collapsed={self.collapsed_rate:.4f}"
        )


@dataclass
class _Accumulator:
    detected: int = 0
    gold_total: int = 0
    correct: int = 0
    scored: int = 0
    mixed_correct: int = 0
    mixed_scored: int = 0
    uniform_correct: int = 0
    uniform_scored: int = 0
    collapsed: int = 0
    mixed_reviews: int = 0
    reviews: int = 0
    _unused: list = field(default_factory=list)


def _terminate(sentence: str) -> str:
    """Give a component a sentence terminator if it lacks one."""
    stripped = sentence.strip()
    if not stripped:
        return stripped
    return stripped if stripped[-1] in _TERMINATORS else stripped + "."


def build_benchmark(
    pairs: pd.DataFrame,
    *,
    n_reviews: int = 400,
    sizes: tuple[int, ...] = (2, 3, 4),
    seed: int = 0,
    strip_terminators: bool = False,
) -> list[PseudoReview]:
    """Compose test rows into multi-aspect pseudo-reviews.

    Args:
        pairs: Test ASC rows, with ``review_id``, ``text``, ``aspect`` and
            ``polarity`` columns.
        n_reviews: How many pseudo-reviews to build.
        sizes: How many source sentences each may draw from.
        seed: Fixed, so a rerun compares like with like.
        strip_terminators: Join without adding punctuation, producing the
            adversarial case where nothing marks the sentence boundaries.

    Returns:
        Pseudo-reviews whose components have pairwise-disjoint aspect sets.
    """
    by_review: dict[str, dict[str, str]] = defaultdict(dict)
    text_of: dict[str, str] = {}
    for row in pairs.itertuples(index=False):
        by_review[row.review_id][row.aspect] = row.polarity
        text_of[row.review_id] = row.text

    review_ids = sorted(by_review)
    rng = random.Random(seed)

    built: list[PseudoReview] = []
    attempts = 0
    # Bounded so a pathological corpus cannot spin here; disjointness usually
    # succeeds within a couple of draws.
    max_attempts = n_reviews * 60

    while len(built) < n_reviews and attempts < max_attempts:
        attempts += 1
        size = rng.choice(sizes)
        candidates = rng.sample(review_ids, min(size, len(review_ids)))

        gold: dict[str, str] = {}
        conflict = False
        for review_id in candidates:
            aspects = by_review[review_id]
            if gold.keys() & aspects.keys():
                conflict = True
                break
            gold.update(aspects)
        if conflict or len(gold) < 2:
            continue

        parts = [text_of[review_id].strip() for review_id in candidates]
        if strip_terminators:
            joined = " ".join(part.rstrip(_TERMINATORS + " ") for part in parts)
        else:
            joined = " ".join(_terminate(part) for part in parts)

        built.append(
            PseudoReview(
                text=joined,
                gold=gold,
                parts=tuple(parts),
                source_ids=tuple(candidates),
            )
        )

    return built


def evaluate_mode(predictor, reviews: list[PseudoReview], *, by_sentence: bool) -> ModeResult:
    """Score one inference mode over the benchmark.

    Sentiment accuracy is computed only over gold aspects the detector actually
    found, so it measures the sentiment stage rather than silently mixing in
    detection failures -- those are reported separately as ``detection_recall``.
    """
    acc = _Accumulator()

    for review in reviews:
        acc.reviews += 1
        result = predictor.analyze(review.text, by_sentence=by_sentence)
        predicted = {p.aspect: p.sentiment.polarity for p in result.aspects}

        acc.gold_total += len(review.gold)
        found = review.gold.keys() & predicted.keys()
        acc.detected += len(found)

        for aspect in found:
            right = predicted[aspect] == review.gold[aspect]
            acc.scored += 1
            acc.correct += right
            if review.is_mixed:
                acc.mixed_scored += 1
                acc.mixed_correct += right
            else:
                acc.uniform_scored += 1
                acc.uniform_correct += right

        if review.is_mixed:
            acc.mixed_reviews += 1
            # "Collapsed" means the model gave every aspect the same polarity
            # for a review that genuinely carries several -- the specific
            # failure whole-review inference produces.
            if predicted and len(set(predicted.values())) == 1:
                acc.collapsed += 1

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return ModeResult(
        mode="sentence" if by_sentence else "whole-review",
        detection_recall=ratio(acc.detected, acc.gold_total),
        sentiment_accuracy=ratio(acc.correct, acc.scored),
        mixed_sentiment_accuracy=ratio(acc.mixed_correct, acc.mixed_scored),
        uniform_sentiment_accuracy=ratio(acc.uniform_correct, acc.uniform_scored),
        collapsed_rate=ratio(acc.collapsed, acc.mixed_reviews),
        n_reviews=acc.reviews,
        n_gold_aspects=acc.gold_total,
        n_scored=acc.scored,
    )
