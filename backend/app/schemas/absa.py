"""Request and response schemas.

Pydantic does the input validation, so the route handlers never hand unvalidated
text to the model. The constraints here are the API's error-handling contract:
an empty review, a 60,000-character paste, or a 10,000-item batch is rejected
with a 422 and a readable message rather than crashing a worker.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_REVIEW_CHARS = 5000
MAX_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Analyse one review."""

    review: str = Field(
        ...,
        min_length=1,
        max_length=MAX_REVIEW_CHARS,
        description="Raw review text.",
        json_schema_extra={
            "example": "The display is beautiful and the camera takes excellent "
            "photos, but the battery life is disappointing."
        },
    )
    top_k: int | None = Field(
        default=None, ge=1, le=12,
        description="Return at most this many aspects, strongest detection first.",
    )

    @field_validator("review")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Review must contain non-whitespace text.")
        return value


class BatchAnalyzeRequest(BaseModel):
    """Analyse many reviews and return a product-level summary."""

    reviews: list[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description="Review texts. Blank entries are skipped, not rejected.",
    )
    product_name: str | None = Field(default=None, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=12)

    @field_validator("reviews")
    @classmethod
    def _has_usable_review(cls, value: list[str]) -> list[str]:
        """Reject only if *every* entry is blank.

        A CSV upload with a few empty rows should still work; the response
        reports how many were skipped.
        """
        if not any(item and item.strip() for item in value):
            raise ValueError("At least one review must contain text.")
        too_long = [i for i, item in enumerate(value) if item and len(item) > MAX_REVIEW_CHARS]
        if too_long:
            preview = ", ".join(str(i) for i in too_long[:5])
            raise ValueError(
                f"{len(too_long)} review(s) exceed {MAX_REVIEW_CHARS} characters "
                f"(index {preview}{'...' if len(too_long) > 5 else ''})."
            )
        return value


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class AspectSentiment(BaseModel):
    """One detected aspect with its sentiment."""

    aspect: str = Field(description="Stable aspect id, e.g. 'battery'.")
    display_name: str = Field(description="Human-readable name, e.g. 'Battery'.")
    polarity: Literal["negative", "neutral", "positive"]
    score: float = Field(ge=1.0, le=10.0, description="1-10 sentiment score; see docs/scoring.md.")
    label: str = Field(description="Band name for the score, e.g. 'Slightly Positive'.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Probability of the predicted polarity. Reported separately "
        "from the score on purpose: 'how positive' and 'how sure' are "
        "different questions.",
    )
    detection_confidence: float = Field(
        ge=0.0, le=1.0, description="Probability that this aspect is discussed at all."
    )
    mentions: int = Field(
        default=0, description="How many sentences in this review discussed the aspect."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="The sentences that produced this score, in the order they "
        "appear. Inference runs per sentence, so every score can be traced "
        "back to the text that caused it instead of being taken on trust.",
    )
    probabilities: dict[str, float] = Field(description="Full distribution over polarities.")


class AnalyzeResponse(BaseModel):
    review: str
    aspects: list[AspectSentiment]
    overall_score: float | None = Field(
        default=None, description="Mean of per-aspect scores; null when no aspect was detected."
    )
    model: str = Field(description="Which model produced this, e.g. 'baseline:tfidf+logreg'.")


class AspectSummary(BaseModel):
    """Aggregated sentiment for one aspect across many reviews."""

    aspect: str
    display_name: str
    mentions: int = Field(description="How many analysed reviews discussed this aspect.")
    mention_share: float = Field(ge=0.0, le=1.0)
    average_score: float | None
    counts: dict[str, int]
    shares: dict[str, float] = Field(
        description="Polarity shares among the reviews that mention this aspect "
        "(not among all reviews)."
    )


class BatchAnalyzeResponse(BaseModel):
    product_name: str | None = None
    reviews_analyzed: int
    reviews_skipped: int = Field(description="Blank or unusable entries that were skipped.")
    overall_score: float | None
    aspects: list[AspectSummary]
    most_positive: AspectSummary | None = None
    most_negative: AspectSummary | None = None
    model: str


class AspectInfo(BaseModel):
    """One entry of the taxonomy, for populating UI filters and legends."""

    id: str
    display_name: str
    description: str


class AspectsResponse(BaseModel):
    aspects: list[AspectInfo]
    polarities: list[str]
    score_range: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    model: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error body so the frontend has one shape to render."""

    error: str = Field(description="Machine-readable code, e.g. 'empty_review'.")
    detail: str = Field(description="Human-readable explanation.")
