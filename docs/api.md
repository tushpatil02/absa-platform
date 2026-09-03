# API reference

Interactive documentation is generated from the code and served at
**`http://localhost:8000/docs`** — that is the authority. This page is the
narrative version.

Base path: `/api`. No authentication (the service holds no user data).

---

## `GET /api/health`

Liveness and model readiness.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_loaded": true,
  "model": "baseline:tfidf(word+char) + logreg",
  "detail": null
}
```

Returns **200 even when the model failed to load**, with `status: "degraded"` and
the reason in `detail`. A load failure is diagnosable information, not a reason
to look dead to an orchestrator's health probe.

---

## `GET /api/aspects`

The taxonomy, so the UI never hard-codes the label list.

```json
{
  "aspects": [
    { "id": "battery", "display_name": "Battery",
      "description": "battery life, charging speed and power supply" }
  ],
  "polarities": ["negative", "neutral", "positive"],
  "score_range": { "min": 1.0, "max": 10.0 }
}
```

---

## `POST /api/analyze`

Analyse one review.

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `review` | string | yes | 1–5,000 characters, not blank |
| `top_k` | integer | no | 1–12; return only the strongest *k* aspects |

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"review": "The camera is excellent, but the battery drains very quickly."}'
```

**Response**

```json
{
  "review": "The camera is excellent, but the battery drains very quickly.",
  "aspects": [
    {
      "aspect": "camera",
      "display_name": "Camera",
      "polarity": "positive",
      "score": 9.15,
      "label": "Positive",
      "confidence": 0.85,
      "detection_confidence": 0.98,
      "probabilities": { "negative": 0.04, "neutral": 0.11, "positive": 0.85 }
    }
  ],
  "overall_score": 6.4,
  "model": "baseline:tfidf(word+char) + logreg"
}
```

Aspects are ordered by `detection_confidence`, strongest first.

**The two confidence fields are different questions:**

- `detection_confidence` — is this aspect discussed at all?
- `confidence` — given that it is, how sure is the polarity?

And `score` is separate from both: it is *how positive*, not *how sure*. See
[scoring.md](scoring.md).

If no aspect clears the detection threshold, the single best aspect is returned
with its (low) confidence attached, rather than an empty list.

**Errors**

| Status | Cause |
|---|---|
| 422 | Empty, blank, over 5,000 chars, bad `top_k`, or no letters after cleaning |
| 503 | No model loaded — body names the script to run |
| 500 | Unexpected failure; traceback is logged, not returned |

---

## `POST /api/analyze/batch`

Analyse many reviews and aggregate them into a product-level summary. This is the
endpoint that answers *what do customers actually like and dislike*, which an
overall star rating cannot.

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `reviews` | string[] | yes | 1–500 items; each ≤ 5,000 chars |
| `product_name` | string | no | ≤ 200 chars, echoed back |
| `top_k` | integer | no | 1–12, applied per review |

```bash
curl -X POST http://localhost:8000/api/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Camera is superb.", "Battery dies after four hours."],
       "product_name": "Example Phone"}'
```

**Response**

```json
{
  "product_name": "Example Phone",
  "reviews_analyzed": 2,
  "reviews_skipped": 0,
  "overall_score": 5.6,
  "aspects": [
    {
      "aspect": "battery",
      "display_name": "Battery",
      "mentions": 1,
      "mention_share": 0.5,
      "average_score": 2.1,
      "counts": { "negative": 1, "neutral": 0, "positive": 0 },
      "shares": { "negative": 1.0, "neutral": 0.0, "positive": 0.0 }
    }
  ],
  "most_positive": { "...": "same shape" },
  "most_negative": { "...": "same shape" },
  "model": "baseline:tfidf(word+char) + logreg"
}
```

`shares` is computed over the reviews that **mention** the aspect, not over all
reviews — "61% negative on battery" means 61% of the people who discussed
battery. Any other denominator makes the figure meaningless. `mention_share` is
the separate "how many of all reviews brought this up" number.

Blank or unusable rows are **skipped and counted** in `reviews_skipped` rather
than failing the request. A 500-row CSV with three bad rows still returns a
summary. Only if *every* row fails does the endpoint return 422.

---

## `GET /api/phones`

The catalogue, most-reviewed first.

| Query | Type | Default | Notes |
|---|---|---|---|
| `q` | string | — | Substring match on the phone name |
| `brand` | string | — | Exact, case-insensitive |
| `limit` | integer | 50 | 1–200 |
| `offset` | integer | 0 | |

```json
{
  "phones": [
    {
      "model_key": "samsung galaxy s7",
      "name": "Samsung Galaxy S7",
      "brand": "Samsung",
      "price": 237.0,
      "reviews_total": 1926,
      "avg_rating": 3.49,
      "rankable": true,
      "aspects": [
        { "aspect": "battery", "display_name": "Battery", "score": 6.4,
          "mentions": 118, "source": "reviews" },
        { "aspect": "price", "display_name": "Price", "score": 5.1,
          "mentions": 1926, "source": "listed_price" }
      ]
    }
  ],
  "total": 211, "limit": 50, "offset": 0,
  "brands": ["Apple", "Google", "HUAWEI", "..."]
}
```

`aspects` arrives in **slider order** (battery, camera, price, display,
performance), so the UI never has to re-sort to match its own layout.

`source` distinguishes `"reviews"` from `"listed_price"`. Price is the one axis
not derived from sentiment — price opinions in the training data are 85.6%
positive and cannot separate phones — and a client showing that score must be
able to say where it came from. See [recommender.md](recommender.md).

`rankable` is false when an axis has too few mentions to score. Such phones are
excluded from `/api/recommend` rather than having the gap imputed.

Returns **503** if the catalogue has not been built, naming the scripts that
build it. `/api/analyze` keeps working: the model and the catalogue fail
independently.

---

## `GET /api/phones/{model_key}`

One phone, plus the sentences behind its scores.

```json
{
  "model_key": "samsung galaxy s7",
  "name": "Samsung Galaxy S7",
  "reviews_scored": 400,
  "reviews_total": 1926,
  "aspects": [ "..." ],
  "evidence": [
    { "aspect": "battery", "display_name": "Battery", "polarity": "negative",
      "score": 1.4, "sentence": "Battery drains within a few hours of light use." },
    { "aspect": "battery", "display_name": "Battery", "polarity": "positive",
      "score": 9.6, "sentence": "The battery easily gets me through a full day." }
  ]
}
```

`evidence` carries the strongest praise **and** the strongest criticism for each
aspect. Returning only the best sentences would make the payload an
advertisement; only the worst, a complaint board. Both are selective quotation
of the model's own output.

404 for an unknown `model_key`.

---

## `POST /api/recommend`

Rank phones against the five sliders.

| Field | Type | Default | Notes |
|---|---|---|---|
| `battery` | float | 5.0 | 1–10 |
| `camera` | float | 5.0 | 1–10 |
| `price` | float | 5.0 | 1–10; **higher means cheaper** |
| `display` | float | 5.0 | 1–10 |
| `performance` | float | 5.0 | 1–10; shown as "Processor" |
| `limit` | integer | 10 | 1–50 |

Each value is a **requirement**, not a weight: a phone is penalised for falling
short and never for exceeding, so an axis left at 1 drops out of the ranking
entirely.

```json
{
  "matches": [
    {
      "phone": { "...": "as in /api/phones" },
      "match_percent": 87.4,
      "shortfalls": { "battery": 0.0, "camera": 1.6, "price": 0.0,
                      "display": 0.0, "performance": 0.0 },
      "worst_axis": "camera"
    }
  ],
  "preferences": { "battery": 8.0, "camera": 9.0, "price": 5.0,
                   "display": 5.0, "performance": 5.0 },
  "considered": 211,
  "price_target": 261.4
}
```

`match_percent` is normalised by the worst score **this query** could produce, so
it reads as "how much of what you asked for you get". Two consequences a client
must handle rather than hide:

- it is 100% whenever every requirement is met, however modest the request;
- it is 100% for *every* phone when all sliders sit at 1, because nothing was
  asked for.

`price_target` is the listed price the Price slider currently corresponds to, so
a client can render "around $261" instead of a bare number.

---

## `POST /api/phones/{model_key}/reviews`

Analyse a submitted review and store it. **201 Created.**

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | 1–5,000 characters |
| `rating` | integer | no | 1–5 |

```json
{
  "review_id": 17,
  "phone": "Samsung Galaxy S7",
  "overall_score": 5.4,
  "aspects": [
    { "aspect": "battery", "display_name": "Battery", "score": 9.1,
      "mentions": 1, "source": "reviews" }
  ],
  "evidence": [
    { "aspect": "battery", "display_name": "Battery", "polarity": "positive",
      "score": 9.1, "sentence": "The battery lasts all day." }
  ],
  "model": "aspect=tfidf-logreg · sentiment=distilbert-base-uncased"
}
```

The phone's published scores are **not** recalculated. A profile built from
hundreds of reviews should not visibly move because one person typed a sentence,
and an endpoint that let it would be trivially gameable. Submitted reviews are
stored in SQLite and surfaced separately.

---

## Error body

All errors share one shape, so the frontend has a single thing to render:

```json
{ "detail": "Review is 6000 characters; the limit is 5000." }
```

Pydantic validation errors put a list of field errors in `detail`; the frontend
client (`frontend/src/api/client.ts`) normalises both forms into one string.

---

## Configuration

Environment variables, all prefixed `ABSA_` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `ABSA_MODELS_DIR` | `./models` | Where artefacts are loaded from |
| `ABSA_PREDICTOR` | `auto` | `auto` \| `baseline` \| `transformer` |
| `ABSA_DEVICE` | auto-detect | e.g. `cpu`, `cuda` |
| `ABSA_CORS_ORIGINS` | localhost dev ports | Comma-separated origins |
| `ABSA_MAX_REVIEW_CHARS` | `5000` | Per-review limit |
| `ABSA_MAX_BATCH_SIZE` | `500` | Reviews per batch request |
| `ABSA_PROCESSED_DIR` | `./data/processed` | Catalogue CSVs |
| `ABSA_DATABASE_PATH` | `./data/reviews.db` | Submitted reviews |
| `ABSA_EVIDENCE_PER_PHONE` | `12` | Example sentences per phone page |

`auto` prefers a fine-tuned transformer when its artefacts exist and falls back
to the baseline otherwise.
