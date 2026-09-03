"""Catalogue and recommender routes.

The catalogue is loaded once at startup. A missing catalogue is an operator
mistake, not a server error, so it returns 503 naming the script that builds
it -- the same convention the model loader uses.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.schemas.phones import (
    AspectScoreOut,
    EvidenceOut,
    MatchOut,
    PhoneDetail,
    PhoneListResponse,
    PhoneSummary,
    RecommendRequest,
    RecommendResponse,
    SubmitReviewRequest,
    SubmitReviewResponse,
)
from ml.inference.predictor import EmptyReviewError, ReviewTooLongError
from ml.recommender.similarity import AXES

logger = logging.getLogger(__name__)
router = APIRouter()

# Evidence sentences returned on a phone page. Enough to justify a score,
# few enough not to turn the response into the corpus.
MAX_EVIDENCE = 12


def _catalog(request: Request):
    catalog = getattr(request.app.state, "catalog", None)
    if catalog is None or not catalog.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                getattr(catalog, "error", None)
                if catalog
                else "Catalogue is not loaded."
            ),
        )
    return catalog


def _predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                getattr(request.app.state, "load_error", None)
                or "Model is not loaded. Run scripts/train_baseline.py, then restart the API."
            ),
        )
    return predictor


def _summary(phone) -> PhoneSummary:
    return PhoneSummary(
        model_key=phone.model_key,
        name=phone.name,
        brand=phone.brand,
        price=phone.price,
        image=phone.image,
        url=phone.url,
        reviews_total=phone.reviews_total,
        avg_rating=phone.avg_rating,
        rankable=phone.rankable,
        aspects=[
            AspectScoreOut(
                aspect=score.aspect,
                display_name=score.display_name,
                score=score.score,
                mentions=score.mentions,
                source=score.source,
            )
            # Slider order, so the UI never has to re-sort to match its layout.
            for axis in AXES
            if (score := phone.aspects.get(axis)) is not None
        ],
    )


@router.get("/phones", response_model=PhoneListResponse, tags=["catalog"])
def list_phones(
    request: Request,
    q: str | None = Query(default=None, description="Substring match on the phone name."),
    brand: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> PhoneListResponse:
    """The catalogue, most-reviewed first."""
    catalog = _catalog(request)
    page, total = catalog.search(query=q, brand=brand, limit=limit, offset=offset)
    return PhoneListResponse(
        phones=[_summary(phone) for phone in page],
        total=total,
        limit=limit,
        offset=offset,
        brands=catalog.brands(),
    )


@router.get("/phones/{model_key}", response_model=PhoneDetail, tags=["catalog"])
def get_phone(request: Request, model_key: str) -> PhoneDetail:
    """One phone, with example sentences behind its scores."""
    catalog = _catalog(request)
    phone = catalog.phones.get(model_key)
    if phone is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No phone {model_key!r}")

    evidence = [
        EvidenceOut(**item)
        for item in getattr(request.app.state, "evidence", {}).get(model_key, [])[:MAX_EVIDENCE]
    ]
    summary = _summary(phone)
    return PhoneDetail(
        **summary.model_dump(),
        reviews_scored=phone.reviews_scored,
        evidence=evidence,
    )


@router.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(request: Request, payload: RecommendRequest) -> RecommendResponse:
    """Rank phones against the five sliders.

    Sliders are requirements, not weights: exceeding one is never penalised, so
    an axis left at 1 stops affecting the ranking. See
    ``ml/recommender/similarity.py``.
    """
    catalog = _catalog(request)
    preferences = payload.preferences()
    rankable = catalog.rankable
    if not rankable:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No phone has scores on all five axes. Run scripts/score_catalog.py "
            "and scripts/build_profiles.py.",
        )

    matches = catalog.recommend(preferences, limit=payload.limit)
    return RecommendResponse(
        matches=[
            MatchOut(
                phone=_summary(catalog.phones[match.model_key]),
                match_percent=match.match_percent,
                shortfalls=match.shortfalls,
                worst_axis=match.worst_axis,
            )
            for match in matches
        ],
        preferences=preferences,
        considered=len(rankable),
        price_target=catalog.price_for_score(preferences["price"]),
    )


@router.post(
    "/phones/{model_key}/reviews",
    response_model=SubmitReviewResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["catalog"],
)
def submit_review(
    request: Request, model_key: str, payload: SubmitReviewRequest
) -> SubmitReviewResponse:
    """Analyse a submitted review and store it.

    The phone's published scores are deliberately *not* recalculated -- see
    ``app/core/storage.py``. The response is what the model makes of this one
    review, sentence by sentence.
    """
    catalog = _catalog(request)
    predictor = _predictor(request)
    phone = catalog.phones.get(model_key)
    if phone is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No phone {model_key!r}")

    try:
        result = predictor.analyze(payload.text)
    except EmptyReviewError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ReviewTooLongError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    aspects = [
        AspectScoreOut(
            aspect=prediction.aspect,
            display_name=prediction.display_name,
            score=prediction.sentiment.score,
            mentions=prediction.mentions,
            source="reviews",
        )
        for prediction in result.aspects
    ]
    evidence = [
        EvidenceOut(
            aspect=prediction.aspect,
            display_name=prediction.display_name,
            polarity=prediction.sentiment.polarity,
            score=prediction.sentiment.score,
            sentence=sentence,
        )
        for prediction in result.aspects
        for sentence in prediction.evidence
    ]

    store = request.app.state.review_store
    review_id = store.add(
        model_key=model_key,
        text=payload.text,
        rating=payload.rating,
        analysis=json.dumps(result.as_dict()),
    )

    return SubmitReviewResponse(
        review_id=review_id,
        phone=phone.name,
        aspects=aspects,
        evidence=evidence,
        overall_score=result.overall_score,
        model=result.model,
    )
