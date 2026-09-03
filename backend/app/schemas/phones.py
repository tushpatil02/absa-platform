"""Request and response models for the catalogue and recommender."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ml.recommender.similarity import AXES

SLIDER_MIN = 1.0
SLIDER_MAX = 10.0


class AspectScoreOut(BaseModel):
    """One axis of a phone's profile."""

    aspect: str
    display_name: str
    score: float = Field(ge=1.0, le=10.0)
    mentions: int = Field(
        description="Reviews that discussed this aspect. For `price` this is the "
        "review count, since the score comes from the listed price."
    )
    source: Literal["reviews", "listed_price"] = Field(
        description="Where the number came from. `price` is derived from the "
        "listed price, not from review sentiment: price opinions in the "
        "training data are 85.6% positive, so a sentiment-derived Price axis "
        "cannot separate phones. Everything else is learned from reviews."
    )


class PhoneSummary(BaseModel):
    """A catalogue row."""

    model_key: str
    name: str
    brand: str
    price: float | None
    image: str | None = None
    url: str | None = None
    reviews_total: int
    avg_rating: float | None = None
    aspects: list[AspectScoreOut] = Field(default_factory=list)
    rankable: bool = Field(
        description="Whether all five slider axes have scores. Phones missing "
        "an axis are excluded from recommendations rather than imputed."
    )


class PhoneListResponse(BaseModel):
    phones: list[PhoneSummary]
    total: int
    limit: int
    offset: int
    brands: list[str]


class EvidenceOut(BaseModel):
    """A sentence that drove one aspect's score, with its polarity."""

    aspect: str
    display_name: str
    polarity: Literal["negative", "neutral", "positive"]
    score: float
    sentence: str


class PhoneDetail(PhoneSummary):
    reviews_scored: int
    evidence: list[EvidenceOut] = Field(
        default_factory=list,
        description="Example sentences behind the scores, so a number can be "
        "traced to the text that produced it.",
    )


class RecommendRequest(BaseModel):
    """Slider settings.

    Each value is a *requirement*: "I need at least this much". A phone is
    penalised for falling short and never for exceeding, so an axis left at 1
    drops out of the ranking entirely.
    """

    battery: float = Field(default=5.0, ge=SLIDER_MIN, le=SLIDER_MAX)
    camera: float = Field(default=5.0, ge=SLIDER_MIN, le=SLIDER_MAX)
    price: float = Field(
        default=5.0,
        ge=SLIDER_MIN,
        le=SLIDER_MAX,
        description="Higher means cheaper, so every slider points the same way.",
    )
    display: float = Field(default=5.0, ge=SLIDER_MIN, le=SLIDER_MAX)
    performance: float = Field(
        default=5.0, ge=SLIDER_MIN, le=SLIDER_MAX, description="Shown as 'Processor'."
    )
    limit: int = Field(default=10, ge=1, le=50)

    def preferences(self) -> dict[str, float]:
        return {axis: float(getattr(self, axis)) for axis in AXES}


class MatchOut(BaseModel):
    """One ranked recommendation."""

    phone: PhoneSummary
    match_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Share of the requirement this phone meets, normalised by "
        "the worst score *this query* could produce. 100% when every slider is "
        "satisfied; 100% for everything when all sliders sit at 1, because "
        "nothing was asked for.",
    )
    shortfalls: dict[str, float] = Field(
        description="How far each axis falls below the requirement. Zero when met."
    )
    worst_axis: str | None = Field(
        default=None, description="The requirement missed by the most, if any."
    )


class RecommendResponse(BaseModel):
    matches: list[MatchOut]
    preferences: dict[str, float]
    considered: int = Field(description="Phones that had all five axes and were ranked.")
    price_target: float | None = Field(
        default=None,
        description="Listed price the Price slider position corresponds to, so "
        "the UI can show 'around $310' instead of a bare number.",
    )


class SubmitReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    rating: int | None = Field(default=None, ge=1, le=5)


class SubmitReviewResponse(BaseModel):
    """The submitted review, analysed immediately.

    The analysis is returned but the phone's published profile is *not*
    recalculated: one review cannot move a score built from hundreds, and
    letting it appear to would invite exactly the gaming this design avoids.
    Submitted reviews are stored and shown separately.
    """

    review_id: int
    phone: str
    aspects: list[AspectScoreOut]
    evidence: list[EvidenceOut]
    overall_score: float | None
    model: str
