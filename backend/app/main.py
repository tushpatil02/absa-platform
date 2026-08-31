"""FastAPI application entry point.

Run locally::

    cd backend
    uvicorn app.main:app --reload --port 8000

Interactive docs at http://localhost:8000/docs.

The model is loaded once during startup. A load failure is caught and recorded
rather than raised: the API still starts, `/health` reports ``degraded`` with
the reason, and every inference route returns a 503 explaining what to run. An
API that refuses to boot with a stack trace is much harder to diagnose in a
container than one that boots and tells you what is missing.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# The backend imports the same `ml` package the training pipeline uses, so
# preprocessing and scoring cannot drift between training and serving.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from ml.inference.predictor import load_predictor
from ml.preprocessing.transform import load_taxonomy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s"
)
logger = logging.getLogger("absa.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.load_error = None
    app.state.predictor = None

    try:
        app.state.taxonomy = load_taxonomy(settings.taxonomy_path)
        logger.info("Taxonomy loaded: %d aspects", len(app.state.taxonomy.aspect_ids))
    except Exception as exc:
        logger.exception("Taxonomy failed to load")
        app.state.taxonomy = None
        app.state.load_error = f"Taxonomy failed to load: {exc}"
        yield
        return

    try:
        app.state.predictor = load_predictor(
            settings.models_dir,
            app.state.taxonomy,
            prefer=settings.predictor,
            prefer_aspect=settings.aspect_model,
            prefer_sentiment=settings.sentiment_model,
            device=settings.device,
        )
        logger.info("Predictor ready: %s", app.state.predictor.model_name)
    except FileNotFoundError as exc:
        app.state.load_error = str(exc)
        logger.error("No model artefacts: %s", exc)
    except Exception as exc:
        app.state.load_error = f"Model failed to load: {exc}"
        logger.exception("Model failed to load")

    yield
    logger.info("Shutting down")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Aspect-Based Sentiment Analysis for product reviews. Detects which "
        "aspects a review discusses (camera, battery, price, ...) and scores "
        "sentiment for each on a 1-10 scale.\n\n"
        "See `docs/scoring.md` for how the score is derived from model "
        "probabilities."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,  # No cookies or auth; keeps the origin check strict.
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback to the client; always log it server-side."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


@app.get("/", tags=["meta"], include_in_schema=False)
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/health",
    }
