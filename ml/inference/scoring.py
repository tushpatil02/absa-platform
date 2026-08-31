"""Turn model probabilities into the 1-10 sentiment score.

The original college project had a 1-10 slider whose values were essentially
decorative. This module defines the score so that every value is derived from
the model's actual output distribution and nothing is invented.

The mapping
-----------
The classifier emits a distribution over three ordered classes. Treat the
polarity axis as ordinal with fixed anchors:

    negative = 0.0     neutral = 0.5     positive = 1.0

and take the **expected value** under the predicted distribution::

    positivity = 1.0 * P(positive) + 0.5 * P(neutral) + 0.0 * P(negative)
    score      = 1 + 9 * positivity

Why expected value rather than bucketing the argmax:

* It is **continuous** -- a barely-positive review scores near the middle, not
  at the top of the positive band. Bucketing throws that away.
* It is **monotonic** in P(positive) and anti-monotonic in P(negative), so the
  slider always moves the direction the model moved.
* It has **exact endpoints**: a confident negative gives 1.0, a confident
  positive gives 10.0, and pure neutral gives 5.5 -- the midpoint of 1-10.
* It needs **no free parameters**, so there is nothing to tune toward a
  nicer-looking UI.

Confidence is reported **separately** as ``max(P)``. Blending confidence into
the score would conflate two different questions -- "how positive?" and "how
sure?" -- and make both unreadable. A score of 5.5 from a confident neutral and
a 5.5 from a three-way coin flip are genuinely different, and the UI shows that
difference through the confidence field.

Calibration caveat
------------------
These are softmax outputs, which are known to be over-confident. The score's
*ordering* is trustworthy; the absolute confidence is not a probability of
correctness. ``docs/scoring.md`` states this, and the UI labels it "confidence",
never "accuracy".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ordinal anchors on the polarity axis. Index order must match the label order
# in aspect_taxonomy.yaml (negative, neutral, positive).
POLARITY_ANCHORS: tuple[float, ...] = (0.0, 0.5, 1.0)

SCORE_MIN = 1.0
SCORE_MAX = 10.0

# Bands used only for the human-readable label beside the number. The score
# itself is continuous; these never alter it.
#
# The bands are symmetric about 5.5, the score of a pure neutral prediction:
# reflecting a score s to (11 - s) lands in the mirrored band. That symmetry is
# not cosmetic -- an asymmetric table would label the exact neutral midpoint as
# leaning one way, which is what an earlier version of this table did (5.5 came
# out "Slightly Positive"). There is a test pinning it.
SCORE_BANDS: tuple[tuple[float, float, str], ...] = (
    (1.00, 1.75, "Extremely Negative"),
    (1.75, 3.25, "Negative"),
    (3.25, 4.75, "Slightly Negative"),
    (4.75, 6.25, "Neutral"),          # centred on 5.5
    (6.25, 7.75, "Slightly Positive"),
    (7.75, 9.25, "Positive"),
    (9.25, 10.0001, "Extremely Positive"),
)


@dataclass(frozen=True)
class SentimentScore:
    """A scored prediction for one (review, aspect) pair."""

    polarity: str
    score: float
    confidence: float
    label: str
    probabilities: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "polarity": self.polarity,
            "score": self.score,
            "confidence": self.confidence,
            "label": self.label,
            "probabilities": self.probabilities,
        }


def positivity(probabilities: np.ndarray, anchors: tuple[float, ...] = POLARITY_ANCHORS) -> np.ndarray:
    """Expected position on the 0-1 polarity axis.

    Args:
        probabilities: Shape ``(..., n_classes)``, rows summing to 1.
        anchors: Ordinal anchor per class, ascending.

    Returns:
        Array of shape ``(...)`` in [0, 1].
    """
    probabilities = np.asarray(probabilities, dtype=float)
    anchor_vector = np.asarray(anchors, dtype=float)
    if probabilities.shape[-1] != anchor_vector.shape[0]:
        raise ValueError(
            f"Expected {anchor_vector.shape[0]} classes, got {probabilities.shape[-1]}"
        )
    return probabilities @ anchor_vector


def to_score(probabilities: np.ndarray, anchors: tuple[float, ...] = POLARITY_ANCHORS) -> np.ndarray:
    """Map probabilities to the 1-10 score."""
    return SCORE_MIN + (SCORE_MAX - SCORE_MIN) * positivity(probabilities, anchors)


def score_label(score: float) -> str:
    """Human-readable band for a score. Presentation only."""
    for low, high, name in SCORE_BANDS:
        if low <= score < high:
            return name
    # Guard against float drift at the exact endpoints.
    return SCORE_BANDS[0][2] if score < SCORE_MIN else SCORE_BANDS[-1][2]


def build_score(
    probabilities: np.ndarray,
    labels: list[str],
    *,
    anchors: tuple[float, ...] = POLARITY_ANCHORS,
    decimals: int = 2,
) -> SentimentScore:
    """Build the full scored result for a single prediction.

    Args:
        probabilities: 1-D array over classes, summing to ~1.
        labels: Class names in index order.
        anchors: Ordinal anchors.
        decimals: Rounding for the API response.
    """
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    if probabilities.shape[0] != len(labels):
        raise ValueError(f"Got {probabilities.shape[0]} probabilities for {len(labels)} labels")

    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Probabilities must be finite and sum to a positive value")
    probabilities = probabilities / total  # tolerate minor float drift

    index = int(probabilities.argmax())
    score = float(to_score(probabilities, anchors))

    return SentimentScore(
        polarity=labels[index],
        score=round(score, decimals),
        confidence=round(float(probabilities[index]), 4),
        label=score_label(score),
        probabilities={
            name: round(float(value), 4) for name, value in zip(labels, probabilities)
        },
    )


def aggregate_scores(scores: list[float]) -> float | None:
    """Mean score across many reviews for one aspect (product-level view).

    The mean of the per-review expected values equals the expected value of the
    pooled distribution, so aggregating this way stays consistent with the
    per-review definition instead of introducing a second, different rule.
    """
    if not scores:
        return None
    return round(float(np.mean(scores)), 2)
