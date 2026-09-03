# Architecture

## System

```
┌──────────────┐   HTTP/JSON    ┌──────────────┐   imports   ┌──────────────┐
│  React SPA   │ ─────────────▶ │   FastAPI    │ ──────────▶ │  ml/ package │
│ Vite · TS    │ ◀───────────── │  Pydantic    │             │  (inference) │
│ Recharts     │                │  /docs       │             └──────┬───────┘
└──────────────┘                └──────────────┘                    │ loads
                                                                    ▼
                                                            ┌──────────────┐
                                                            │  models/     │
                                                            │  artefacts   │
                                                            └──────────────┘
```

The backend **imports the same `ml` package the training pipeline uses**. This is
the single most important structural decision in the project: if training and
serving each had their own copy of the text cleaning or the scoring formula, a
divergence between them would be a silent accuracy bug that no test on either
side would catch. There is exactly one `clean_text`, and exactly one
`build_score`.

## Two-stage ML design

```
raw review
   │
   ├─▶ clean_text()                 same function that built the training data
   │
   ├─▶ Stage A: aspect detection    multi-label · 12 sigmoid outputs
   │      └─ threshold (tuned on dev) ─▶ which aspects are present
   │
   ├─▶ Stage B: sentiment           one sentence-pair per detected aspect
   │      [CLS] review [SEP] aspect description [SEP] ─▶ 3-class softmax
   │
   ├─▶ build_score()                probabilities ─▶ 1–10 score + confidence
   │
   └─▶ AnalysisResult
```

### Why two stages rather than generative triplet extraction

Generative ABSA (a seq2seq model emitting `(aspect, opinion, polarity)` triplets)
is the more fashionable 2026 formulation, and it was rejected deliberately:

| | Two-stage classification | Generative triplets |
|---|---|---|
| Per-aspect confidence | Native — softmax over 3 classes | Not available without extra machinery |
| Fixed taxonomy | Guaranteed — 12 outputs | Model can emit anything, needs post-hoc mapping |
| Inference cost | One pass + N short pairs | Autoregressive decoding |
| Deployable on CPU | Yes | Marginal |
| Failure modes | Legible per stage | Opaque |

The 1–10 slider **requires** a real probability distribution per aspect. That
requirement alone settles the choice.

### Why the aspect is a second segment

Stage B follows the sentence-pair formulation of Sun et al. (2019). The aspect
enters as the second segment rather than as a separate categorical feature, so
attention can link the aspect's words to the relevant clause of the review.

This matters because the same review is positive for one aspect and negative for
another. A model given only the review text cannot represent that, and the
project measures exactly this: see the
[aspect-conditioning diagnostic](../ml/evaluation/mixed_reviews.py).

## Repository layout

```
absa-platform/
├── data/                     gitignored — fetched, never committed
│   ├── raw/mabsa/            M-ABSA English phone + laptop (labels)
│   ├── raw/amazon/           68k phone reviews, CC0 (product identity)
│   ├── processed/            asc_*.csv · acd_*.csv · phones.csv
│   │                         review_aspects.csv · phone_profiles.csv
│   └── reviews.db            SQLite — submitted reviews, written at runtime
│
├── ml/                       all ML logic, importable from anywhere
│   ├── config/aspect_taxonomy.yaml    ENTITY#ATTRIBUTE → 12 aspects
│   ├── preprocessing/        parse · clean · transform · split
│   ├── training/             baseline (sklearn) · transformer (torch)
│   ├── evaluation/           metrics · mixed_reviews · multi_sentence
│   │                         reliability          ← the recommender's gate
│   ├── inference/            predictor · scoring · sentences
│   │                                              ← shared with backend
│   ├── catalog/              normalise · build · profiles
│   ├── recommender/          similarity · price
│   └── eda.py
│
├── scripts/                  CLI entry points; notebooks drive these
│   ├── download_data.py  build_dataset.py  run_eda.py
│   ├── train_baseline.py  train_transformer.py  compare_models.py
│   ├── eval_sentence_level.py
│   └── download_phones.py  build_catalog.py  score_catalog.py
│       build_profiles.py   evaluate_recommender.py
│
├── notebooks/                orchestration only, no logic
│   ├── 01_eda.ipynb
│   └── absa_training.ipynb   Colab, GPU-aware with CPU fallback
│
├── models/                   metadata committed, weights gitignored
├── backend/app/              FastAPI: main · api · schemas · core
├── frontend/src/             React: components · api · types
├── tests/                    preprocessing · scoring · eda · api
│                             sentences · catalog · recommender · reliability
└── docs/
```

### Two corpora, two jobs

M-ABSA supplies **labels**: it is the only dataset found with aspect *categories*
and polarities, and it is what the models train on. It has no product identity,
so it cannot answer "which phone should I buy".

The Amazon corpus supplies **product identity**: 67,986 reviews attached to 720
listings, but no aspect labels at all. It is never trained on — only scored.

Neither dataset alone supports the application. The join between them is the
taxonomy plus the trained model.

### Where logic is *not*

- **Not in notebooks.** They import from `ml/` and shell out to `scripts/`. A
  notebook holding logic cannot be diffed, tested, or imported by the API.
- **Not duplicated in the backend.** `backend/app/` contains HTTP concerns only:
  routing, validation, error translation. It owns no ML code.

## The offline pipeline

Inference over the catalogue is expensive (~1 hour on CPU) and everything after
it is arithmetic, so the two are separate stages with a file between them:

```
download_phones ─→ build_catalog ─→ score_catalog ─→ build_profiles ─→ API
   9 MB, CC0        720→211 phones    37k reviews      shrinkage
   no credentials   seconds           ~1 h, resumable  seconds
                                            │
                                            └─→ evaluate_recommender  (the gate)
```

`score_catalog.py` writes **per-review** rows, not per-phone averages. Aggregation
has several open questions — shrinkage strength, mention floors, split-half
resampling — and none of them should require re-running inference to explore.

## Request flow

```
POST /api/analyze  {"review": "..."}
  │
  ├─ Pydantic validates          length, non-blank, top_k range   → 422 on failure
  ├─ predictor.analyze()
  │    ├─ _prepare()             clean + usability check          → 422 / 413
  │    ├─ Stage A                aspect probabilities
  │    ├─ _select_aspects()      threshold, or top-1 fallback
  │    ├─ Stage B                one pair per selected aspect
  │    └─ build_score()          1–10 + confidence per aspect
  └─ AnalyzeResponse             typed JSON
```

The **top-1 fallback** exists for UX: a short review like *"love it"* may clear no
per-aspect threshold, and an empty list reads as a failure. Returning the single
best aspect with its (low) detection confidence attached is more honest than
showing nothing.

## Error handling

| Condition | Status | Where |
|---|---|---|
| Empty / whitespace review | 422 | Pydantic |
| Review > 5,000 chars | 422 | Pydantic |
| Text with no letters after cleaning | 422 | `EmptyReviewError` |
| Batch > 500 reviews | 422 | Pydantic |
| Batch where every row is blank | 422 | Route |
| Model not loaded | 503 | Route, with the command to fix it |
| Anything else | 500 | Handler — logs the traceback, returns a generic body |

A model-load failure **does not prevent startup**. The API boots, `/health`
reports `degraded` with the reason, and inference routes return 503 naming the
script to run. A container that refuses to start with a stack trace is far harder
to diagnose than one that starts and explains itself.

Individual failures inside a batch are skipped and counted rather than failing
the whole request — a 500-row CSV with three bad rows should still produce a
summary.

## Technology choices

| Layer | Choice | Why |
|---|---|---|
| ML | scikit-learn + PyTorch/Transformers | Baseline must be trivially reproducible; transformer needs HF |
| API | **FastAPI** | Pydantic validation *is* the error-handling layer; `/docs` is generated, not written |
| Frontend | **Vite + React + TS** | No SSR or SEO need — a SPA against a JSON API. Next.js would add a server for nothing |
| Charts | **Recharts** | Declarative, React-native, no D3 wrangling |
| Styling | **Plain CSS + custom properties** | The palette is shared with the Python charts; a framework would add a build step and indirection for no gain |
| Storage | **None currently** | Nothing needs persisting yet. SQLite is a two-file change if saved analyses are added |

### Training / serving split

```
laptop (CPU)          Colab (T4)              Hugging Face Hub        API (CPU)
─────────────         ──────────              ────────────────        ─────────
build dataset    →    fine-tune          →    model weights      →    load once
baselines             DeBERTa/DistilBERT      (never in git)          at startup
```

Weights live on the Hub rather than in git: a portfolio repo carrying 500 MB of
`.safetensors` is unwieldy and exceeds GitHub's limits. Only `metadata.json` and
the metrics are committed — small, diffable, and the source for the README's
results table.

Training is the only GPU-dependent step, so the local environment and the serving
container are both CPU-only and stay light.
