"""API tests.

These exercise the real app against the real baseline artefacts, so they cover
the wiring between FastAPI, the schemas and the ML layer -- not a mock of it.
The whole suite is skipped (not failed) when the models have not been trained,
so a fresh clone can still run the rest of the tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

MODELS_DIR = REPO_ROOT / "models"
HAS_MODELS = (MODELS_DIR / "baseline_sentiment_classifier" / "metadata.json").exists()

pytestmark = pytest.mark.skipif(
    not HAS_MODELS, reason="Baseline models not trained; run scripts/train_baseline.py"
)

MIXED_REVIEW = (
    "The display is beautiful and the camera takes excellent photos, "
    "but the battery life is disappointing."
)


@pytest.fixture(scope="module")
def client():
    from app.main import app
    from fastapi.testclient import TestClient

    # TestClient as a context manager runs lifespan, so the model really loads.
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def test_health_reports_loaded_model(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]


def test_aspects_lists_the_full_taxonomy(client):
    body = client.get("/api/aspects").json()
    assert len(body["aspects"]) == 12
    assert body["polarities"] == ["negative", "neutral", "positive"]
    assert body["score_range"] == {"min": 1.0, "max": 10.0}
    assert {"id", "display_name", "description"} <= set(body["aspects"][0])


def test_openapi_schema_is_served(client):
    """The /docs page depends on this; it is also the API documentation."""
    schema = client.get("/openapi.json").json()
    assert "/api/analyze" in schema["paths"]
    assert "/api/analyze/batch" in schema["paths"]


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


def test_analyze_returns_scored_aspects(client):
    body = client.post("/api/analyze", json={"review": MIXED_REVIEW}).json()
    assert body["aspects"], "expected at least one detected aspect"

    for aspect in body["aspects"]:
        assert 1.0 <= aspect["score"] <= 10.0
        assert 0.0 <= aspect["confidence"] <= 1.0
        assert 0.0 <= aspect["detection_confidence"] <= 1.0
        assert aspect["polarity"] in {"negative", "neutral", "positive"}
        assert aspect["label"]
        assert set(aspect["probabilities"]) == {"negative", "neutral", "positive"}
        assert sum(aspect["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_analyze_aspects_are_ordered_by_detection_confidence(client):
    body = client.post("/api/analyze", json={"review": MIXED_REVIEW}).json()
    confidences = [a["detection_confidence"] for a in body["aspects"]]
    assert confidences == sorted(confidences, reverse=True)


def test_analyze_respects_top_k(client):
    body = client.post("/api/analyze", json={"review": MIXED_REVIEW, "top_k": 2}).json()
    assert len(body["aspects"]) <= 2


def test_analyze_always_returns_at_least_one_aspect(client):
    """Short reviews may clear no threshold; falling back beats an empty list."""
    body = client.post("/api/analyze", json={"review": "Love it"}).json()
    assert len(body["aspects"]) >= 1


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"review": ""}, 422),
        ({"review": "   "}, 422),
        ({}, 422),
        ({"review": "ok", "top_k": 0}, 422),
        ({"review": "ok", "top_k": 99}, 422),
        ({"review": "x" * 6000}, 422),   # caught by max_length before the model
        ({"review": 12345}, 422),
    ],
)
def test_analyze_rejects_invalid_input(client, payload, expected):
    assert client.post("/api/analyze", json=payload).status_code == expected


def test_analyze_rejects_text_with_no_letters(client):
    """Passes schema validation, then fails in the ML layer -- must be a 422."""
    response = client.post("/api/analyze", json={"review": "!!! ??? ..."})
    assert response.status_code == 422
    assert "usable text" in response.json()["detail"].lower()


def test_analyze_handles_unicode_and_emoji(client):
    response = client.post(
        "/api/analyze", json={"review": "Der Akku ist schlecht \U0001f600 — camera ok"}
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /analyze/batch
# ---------------------------------------------------------------------------


def test_batch_returns_product_summary(client):
    reviews = [
        "The camera is fantastic and photos look great.",
        "Battery dies after four hours. Very disappointing.",
        "Cheap price, fast delivery, works fine.",
        "Screen is gorgeous but it is far too expensive.",
    ]
    body = client.post(
        "/api/analyze/batch", json={"reviews": reviews, "product_name": "Test Phone"}
    ).json()

    assert body["product_name"] == "Test Phone"
    assert body["reviews_analyzed"] == 4
    assert body["reviews_skipped"] == 0
    assert body["aspects"]
    assert body["most_positive"] and body["most_negative"]

    for aspect in body["aspects"]:
        assert aspect["mentions"] >= 1
        assert sum(aspect["shares"].values()) == pytest.approx(1.0, abs=0.01)
        # Shares are over reviews that MENTION the aspect, not all reviews.
        assert sum(aspect["counts"].values()) == aspect["mentions"]


def test_batch_skips_blank_rows_instead_of_failing(client):
    """A CSV with a few empty rows should still produce a summary."""
    body = client.post(
        "/api/analyze/batch",
        json={"reviews": ["Great camera!", "", "   ", "Terrible battery."]},
    ).json()
    assert body["reviews_analyzed"] == 2
    assert body["reviews_skipped"] == 2


def test_batch_most_positive_and_negative_are_ordered(client):
    body = client.post(
        "/api/analyze/batch",
        json={"reviews": ["Amazing camera, superb photos!", "Battery is awful and dies fast."]},
    ).json()
    assert body["most_positive"]["average_score"] >= body["most_negative"]["average_score"]


@pytest.mark.parametrize(
    "payload",
    [
        {"reviews": []},
        {"reviews": ["", "  "]},
        {"reviews": ["ok"] * 501},
        {"reviews": ["x" * 6000]},
        {},
    ],
)
def test_batch_rejects_invalid_input(client, payload):
    assert client.post("/api/analyze/batch", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Degraded mode
# ---------------------------------------------------------------------------


def test_missing_model_yields_503_not_a_crash(monkeypatch):
    """When no artefacts exist the API must still boot and explain itself."""
    from fastapi.testclient import TestClient

    import ml.inference.predictor as predictor_module

    def fail(*args, **kwargs):
        raise FileNotFoundError("No model artefacts found in models/.")

    monkeypatch.setattr(predictor_module, "load_predictor", fail)
    monkeypatch.setattr("app.main.load_predictor", fail)

    from app.main import app

    with TestClient(app) as test_client:
        health = test_client.get("/api/health").json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False

        response = test_client.post("/api/analyze", json={"review": "Great camera"})
        assert response.status_code == 503
        assert "model artefacts" in response.json()["detail"].lower()
