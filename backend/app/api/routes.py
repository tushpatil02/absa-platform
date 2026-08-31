"""API routes.

The model is loaded once at startup and held in application state -- loading it
per request would dominate latency and, for a transformer, exhaust memory.

Errors from the ML layer are translated into HTTP here rather than leaking
tracebacks: `EmptyReviewError` and `ReviewTooLongError` are user mistakes (422),
a missing model is an operator mistake (503), and anything else is a 500 with a
logged traceback and a generic body.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.absa import (
    AnalyzeRequest,
    AnalyzeResponse,
    AspectInfo,
    AspectsResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
)
from ml.inference.predictor import (
    EmptyReviewError,
    ReviewTooLongError,
    summarise_product,
)
from ml.inference.scoring import SCORE_MAX, SCORE_MIN

logger = logging.getLogger(__name__)
router = APIRouter()


def _predictor(request: Request):
    """Fetch the loaded predictor, or fail with a clear 503."""
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


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    """Liveness and model-readiness.

    Returns 200 even when the model failed to load, reporting ``degraded`` --
    a load failure is diagnosable information, not a reason to look dead to an
    orchestrator's health probe.
    """
    predictor = getattr(request.app.state, "predictor", None)
    settings = request.app.state.settings
    return HealthResponse(
        status="ok" if predictor else "degraded",
        version=settings.version,
        model_loaded=predictor is not None,
        model=getattr(predictor, "model_name", None),
        detail=getattr(request.app.state, "load_error", None),
    )


@router.get("/aspects", response_model=AspectsResponse, tags=["meta"])
def list_aspects(request: Request) -> AspectsResponse:
    """The aspect taxonomy, so the UI never hard-codes the label list."""
    taxonomy = request.app.state.taxonomy
    return AspectsResponse(
        aspects=[
            AspectInfo(
                id=aspect,
                display_name=taxonomy.display_names[aspect],
                description=taxonomy.descriptions[aspect],
            )
            for aspect in taxonomy.aspect_ids
        ],
        polarities=list(taxonomy.polarities),
        score_range={"min": SCORE_MIN, "max": SCORE_MAX},
    )


@router.post("/analyze", response_model=AnalyzeResponse, tags=["absa"])
def analyze(request: Request, payload: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse one review: detect aspects, then score sentiment for each."""
    predictor = _predictor(request)
    try:
        result = predictor.analyze(payload.review, top_k=payload.top_k)
    except EmptyReviewError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ReviewTooLongError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference failed."
        ) from exc

    return AnalyzeResponse(**result.as_dict())


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse, tags=["absa"])
def analyze_batch(request: Request, payload: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyse many reviews and aggregate them into a product-level summary.

    Individual unusable reviews are skipped and counted rather than failing the
    whole request -- a 500-row CSV with three blank rows should still produce a
    summary.
    """
    predictor = _predictor(request)
    taxonomy = request.app.state.taxonomy

    results, skipped = [], 0
    for review in payload.reviews:
        if not review or not review.strip():
            skipped += 1
            continue
        try:
            results.append(predictor.analyze(review, top_k=payload.top_k))
        except (EmptyReviewError, ReviewTooLongError):
            skipped += 1
        except Exception:
            logger.exception("Inference failed on one review; skipping it")
            skipped += 1

    if not results:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"None of the {len(payload.reviews)} submitted reviews could be analysed.",
        )

    summary = summarise_product(results, list(taxonomy.polarities))
    return BatchAnalyzeResponse(
        product_name=payload.product_name,
        reviews_analyzed=summary["reviews_analyzed"],
        reviews_skipped=skipped,
        overall_score=summary["overall_score"],
        aspects=summary["aspects"],
        most_positive=summary["most_positive"],
        most_negative=summary["most_negative"],
        model=results[0].model,
    )
