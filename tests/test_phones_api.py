"""Tests for the catalogue and recommender routes.

These run against a small catalogue written to a temp directory rather than the
real one, so they neither depend on a ~70-minute scoring pass having been run
nor change meaning when the corpus is rebuilt. The recommender's arithmetic is
covered in test_recommender.py; what is checked here is the wiring, the
degradation behaviour, and the contract the frontend depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

MODELS_DIR = REPO_ROOT / "models"
HAS_MODELS = (MODELS_DIR / "baseline_sentiment_classifier" / "metadata.json").exists()

pytestmark = pytest.mark.skipif(
    not HAS_MODELS, reason="Baseline models not trained; run scripts/train_baseline.py"
)

AXES = ("battery", "camera", "price", "display", "performance")

PHONES = [
    # name,                key,        brand,      price, battery, camera, display, perf
    ("Alpha One", "alpha one", "Alpha", 150.0, 9.0, 4.0, 5.0, 5.0),
    ("Beta Two", "beta two", "Beta", 450.0, 4.0, 9.0, 8.0, 8.0),
    ("Gamma Three", "gamma three", "Gamma", 900.0, 7.0, 7.0, 9.0, 9.0),
]


@pytest.fixture(scope="module")
def catalog_dir(tmp_path_factory):
    """A three-phone catalogue in the on-disk format the API loads."""
    directory = tmp_path_factory.mktemp("catalog")

    pd.DataFrame(
        [
            {
                "model_key": key,
                "name": name,
                "brand": brand,
                "price": price,
                "image": "https://example.invalid/x.jpg",
                "url": "https://example.invalid/p",
                "listings": 1,
                "reviews_total": 120,
                "avg_rating": 4.0,
                "reviews_sampled": 120,
            }
            for name, key, brand, price, *_ in PHONES
        ]
    ).to_csv(directory / "phones.csv", index=False)

    rows = []
    for _, key, _, price, battery, camera, display, performance in PHONES:
        values = {
            "battery": battery,
            "camera": camera,
            "display": display,
            "performance": performance,
            # Cheapest -> 10, dearest -> 1, matching the real pipeline.
            "price": {150.0: 10.0, 450.0: 5.0, 900.0: 1.0}[price],
        }
        for aspect, score in values.items():
            rows.append(
                {
                    "model_key": key,
                    "aspect": aspect,
                    "score": score,
                    "raw_score": score,
                    "mentions": 30,
                    "mean_confidence": 0.8,
                }
            )
    pd.DataFrame(rows).to_csv(directory / "phone_profiles.csv", index=False)

    pd.DataFrame(
        [
            {
                "model_key": "alpha one",
                "aspect": "battery",
                "polarity": "positive",
                "score": 9.4,
                "evidence": "The battery easily lasts a day and a half of heavy use.",
            },
            {
                "model_key": "alpha one",
                "aspect": "camera",
                "polarity": "negative",
                "score": 2.1,
                "evidence": "Photos come out grainy whenever the light is low.",
            },
        ]
    ).to_csv(directory / "review_aspects.csv", index=False)

    return directory


@pytest.fixture(scope="module")
def client(catalog_dir, tmp_path_factory):
    from app.core.config import get_settings
    from fastapi.testclient import TestClient

    get_settings.cache_clear()
    settings = get_settings()
    settings.processed_dir = catalog_dir
    settings.database_path = tmp_path_factory.mktemp("db") / "reviews.db"

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_lists_phones_with_their_profiles(client):
    payload = client.get("/api/phones").json()
    assert payload["total"] == 3
    assert {phone["name"] for phone in payload["phones"]} == {
        "Alpha One",
        "Beta Two",
        "Gamma Three",
    }
    assert sorted(payload["brands"]) == ["Alpha", "Beta", "Gamma"]


def test_aspects_are_returned_in_slider_order(client):
    """The UI lays sliders out in a fixed order and must not have to re-sort."""
    phone = client.get("/api/phones").json()["phones"][0]
    assert [aspect["aspect"] for aspect in phone["aspects"]] == list(AXES)


def test_price_axis_declares_it_is_not_from_reviews(client):
    """Price sentiment is 85.6% positive, so this axis uses the listed price.

    The UI has to be able to say so rather than implying shoppers praised it.
    """
    phone = client.get("/api/phones").json()["phones"][0]
    sources = {aspect["aspect"]: aspect["source"] for aspect in phone["aspects"]}
    assert sources["price"] == "listed_price"
    assert sources["battery"] == "reviews"


def test_filters_by_brand_and_query(client):
    assert client.get("/api/phones", params={"brand": "Beta"}).json()["total"] == 1
    assert client.get("/api/phones", params={"q": "gamma"}).json()["total"] == 1
    assert client.get("/api/phones", params={"q": "nonexistent"}).json()["total"] == 0


def test_pagination_reports_the_full_total(client):
    payload = client.get("/api/phones", params={"limit": 2, "offset": 0}).json()
    assert len(payload["phones"]) == 2
    assert payload["total"] == 3


def test_phone_detail_includes_evidence(client):
    payload = client.get("/api/phones/alpha one").json()
    assert payload["name"] == "Alpha One"
    assert payload["evidence"]
    assert any("grainy" in item["sentence"] for item in payload["evidence"])


def test_evidence_shows_both_praise_and_criticism(client):
    """Quoting only the best sentences would make the page an advertisement."""
    evidence = client.get("/api/phones/alpha one").json()["evidence"]
    assert {item["polarity"] for item in evidence} == {"positive", "negative"}


def test_unknown_phone_is_404(client):
    assert client.get("/api/phones/no-such-phone").status_code == 404


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


def _recommend(client, **sliders):
    body = dict.fromkeys(AXES, 5.0) | sliders
    response = client.post("/api/recommend", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_recommend_ranks_by_the_requested_axis(client):
    """Camera 10, everything else 1: only the camera should matter."""
    payload = _recommend(client, camera=10.0, battery=1.0, price=1.0, display=1.0, performance=1.0)
    assert payload["matches"][0]["phone"]["name"] == "Beta Two"


def test_a_different_axis_gives_a_different_winner(client):
    """Guards against a ranking that ignores the sliders entirely."""
    battery = _recommend(
        client, battery=10.0, camera=1.0, price=1.0, display=1.0, performance=1.0
    )
    camera = _recommend(client, camera=10.0, battery=1.0, price=1.0, display=1.0, performance=1.0)
    assert battery["matches"][0]["phone"]["name"] == "Alpha One"
    assert camera["matches"][0]["phone"]["name"] == "Beta Two"


def test_price_slider_prefers_the_cheaper_phone(client):
    payload = _recommend(client, price=10.0, battery=1.0, camera=1.0, display=1.0, performance=1.0)
    assert payload["matches"][0]["phone"]["name"] == "Alpha One"


def test_match_percentages_are_bounded_and_ordered(client):
    matches = _recommend(client, camera=9.0, battery=9.0)["matches"]
    percentages = [match["match_percent"] for match in matches]
    assert percentages == sorted(percentages, reverse=True)
    assert all(0.0 <= value <= 100.0 for value in percentages)


def test_shortfalls_name_what_was_missed(client):
    match = _recommend(
        client, camera=10.0, battery=1.0, price=1.0, display=1.0, performance=1.0
    )["matches"][-1]
    assert match["shortfalls"]["camera"] > 0
    assert match["worst_axis"] == "camera"


def test_a_met_requirement_has_no_shortfall(client):
    match = _recommend(client, battery=1.0, camera=1.0, price=1.0, display=1.0, performance=1.0)[
        "matches"
    ][0]
    assert all(value == 0 for value in match["shortfalls"].values())
    assert match["worst_axis"] is None


def test_price_target_explains_the_slider(client):
    """A bare "7.5" means nothing; "around $260" does."""
    payload = _recommend(client, price=7.5)
    assert payload["price_target"] is not None
    assert 150.0 <= payload["price_target"] <= 900.0


def test_limit_is_respected(client):
    assert len(client.post("/api/recommend", json=dict.fromkeys(AXES, 5.0) | {"limit": 2}).json()["matches"]) == 2


@pytest.mark.parametrize("value", [0, 11, -3])
def test_out_of_range_sliders_are_rejected(client, value):
    body = dict.fromkeys(AXES, 5.0) | {"battery": value}
    assert client.post("/api/recommend", json=body).status_code == 422


def test_recommend_reports_how_many_phones_were_considered(client):
    assert _recommend(client)["considered"] == 3


# ---------------------------------------------------------------------------
# Review submission
# ---------------------------------------------------------------------------


def test_submitted_review_is_analysed_per_sentence(client):
    response = client.post(
        "/api/phones/alpha one/reviews",
        json={
            "text": "The battery lasts all day. The camera is terrible in low light.",
            "rating": 3,
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["review_id"] > 0
    assert payload["aspects"]
    # Every score must be traceable to the sentence that produced it.
    assert payload["evidence"]
    for item in payload["evidence"]:
        assert item["sentence"]


def test_submitting_does_not_move_the_published_scores(client):
    """A published score built from hundreds of reviews must not visibly move.

    Otherwise the page is gameable: type "best camera ever" ten times and watch
    the ranking change.
    """
    before = client.get("/api/phones/alpha one").json()["aspects"]
    for _ in range(3):
        client.post(
            "/api/phones/alpha one/reviews",
            json={"text": "Best camera ever. Amazing camera. Incredible camera."},
        )
    after = client.get("/api/phones/alpha one").json()["aspects"]
    assert before == after


def test_review_for_unknown_phone_is_404(client):
    response = client.post("/api/phones/nope/reviews", json={"text": "Great phone overall."})
    assert response.status_code == 404


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_review_is_rejected(client, text):
    response = client.post("/api/phones/alpha one/reviews", json={"text": text})
    assert response.status_code == 422


def test_overlong_review_is_rejected(client):
    response = client.post("/api/phones/alpha one/reviews", json={"text": "a" * 6000})
    assert response.status_code == 422
