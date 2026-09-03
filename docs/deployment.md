# Deployment

The app splits cleanly into a stateless JSON API and a static SPA, so each half
goes to whatever is cheapest for its shape.

```
frontend (static build)  ──▶  Vercel / Netlify / GitHub Pages
backend  (Docker)        ──▶  Hugging Face Spaces / Render / Fly.io
model weights            ──▶  Hugging Face Hub (pulled at build or runtime)
```

> **Not yet verified.** Docker is not installed on the development machine, so the
> `Dockerfile` and `docker-compose.yml` are written but have **not** been
> build-tested. Everything else on this page has been run.

---

## 1. Model artefacts

Weights never go in git. `.gitignore` excludes `*.joblib`, `*.safetensors`,
`*.bin` and `*.onnx`; only `metadata.json` and the metrics are committed.

**Baseline** (default, and what the Docker image expects): the artefacts are
small enough to rebuild anywhere in seconds.

```bash
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train_baseline.py     # writes models/baseline_*/
```

**Transformer**: push from Colab to the Hub (see
[`notebooks/absa_training.ipynb`](../notebooks/absa_training.ipynb) §7), then pull
at deploy time:

```bash
huggingface-cli download tushpatil02/absa-sentiment-classifier \
  --local-dir models/sentiment_classifier
huggingface-cli download tushpatil02/absa-aspect-detector \
  --local-dir models/aspect_detector
```

`ABSA_PREDICTOR=auto` uses a transformer when its artefacts are present and falls
back to the baseline otherwise, so the same image serves both.

---

## 1b. Catalogue artefacts

The recommender needs three files in `data/catalog/`. Unlike weights, these
**are** committed — together they are around 500 KB, and without them a clone
would need an hour of CPU inference before the app could answer anything.

| File | What it is |
|---|---|
| `phones.csv` | 211 phones: name, brand, listed price, image, review counts |
| `phone_profiles.csv` | one row per (phone, aspect): shrunk score, raw score, mentions |
| `phone_evidence.json` | example sentences per phone, distilled at build time |

They are copied into the Docker image. The ~45 MB `review_aspects.csv` they were
built from is **not** — it is an intermediate whose only purpose is letting
aggregation be re-tried without re-running the model.

To rebuild them from scratch:

```bash
python scripts/download_phones.py    # 9 MB, CC0, no credentials
python scripts/build_catalog.py      # seconds
python scripts/score_catalog.py      # ~1 hour on CPU; resumable
python scripts/build_profiles.py     # seconds
```

A missing catalogue is not fatal. `/api/health` stays 200, `/api/analyze` keeps
working, and the catalogue routes return 503 naming the scripts above. The model
and the catalogue fail independently on purpose.

---

## 1c. Submitted reviews (SQLite)

`ABSA_DATABASE_PATH` (default `data/reviews.db`) holds reviews submitted through
the site. It is the only thing the API writes.

**On an ephemeral container this is lost on every restart.** That is acceptable
for a portfolio deployment — submitted reviews never feed back into the
published scores, so losing them degrades nothing but the "recently submitted"
list. If they should persist, mount a volume:

```yaml
volumes:
  - absa-data:/app/data
```

and note that the image's `data/catalog/` must remain readable, so mount a
subpath (`/app/data/reviews.db`) rather than shadowing the whole directory.

---

## 2. Backend

### Docker

```bash
docker compose up --build          # http://localhost:8000
```

The image is deliberately **CPU-only and torch-free** — `requirements-serve.txt`
omits PyTorch, so it stays a few hundred MB instead of several GB. To serve a
fine-tuned transformer, add to that file:

```
torch --index-url https://download.pytorch.org/whl/cpu
transformers>=5.0,<6
```

The container runs as a non-root user and its `HEALTHCHECK` hits `/api/health`
and requires `model_loaded: true`, so a container with missing artefacts reports
unhealthy rather than merely "listening".

### Hugging Face Spaces (free, good fit for ML demos)

1. Create a Space → **Docker** SDK.
2. Push this repository to it.
3. Add a Space secret for `HF_TOKEN` if pulling private model repos.
4. Spaces expects port **7860**, so override the command:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Set `ABSA_CORS_ORIGINS` to the frontend's deployed origin.

### Render / Fly.io / Railway

Any container host works. Point it at the `Dockerfile`; set:

| Variable | Value |
|---|---|
| `ABSA_CORS_ORIGINS` | your frontend origin, comma-separated |
| `ABSA_PREDICTOR` | `auto` |
| `PORT` handling | pass `--port $PORT` if the platform injects one |

Free tiers usually sleep after inactivity. The first request then pays model-load
time (a second or two for the baseline, longer for a transformer) — worth noting
in a demo link so a cold start does not look like a bug.

---

## 3. Frontend

```bash
cd frontend
npm ci
VITE_API_URL=https://your-api.example.com npm run build   # → dist/
```

`VITE_API_URL` is baked in at build time. Leave it unset for a same-origin deploy
(reverse proxy in front of both halves), in which case the client uses relative
`/api` paths.

### Vercel

- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_URL`

Netlify and Cloudflare Pages take the same three settings. The output is fully
static — no server-side rendering anywhere in this app.

**CORS**: the deployed frontend origin must appear in `ABSA_CORS_ORIGINS` on the
backend, or the browser will block every request while the API itself looks
healthy. This is the most common deployment mistake here.

---

## 4. Configuration reference

All backend settings are prefixed `ABSA_`; see [`.env.example`](../.env.example).

| Variable | Default | Notes |
|---|---|---|
| `ABSA_MODELS_DIR` | `./models` | Artefact location |
| `ABSA_PREDICTOR` | `auto` | `auto` \| `baseline` \| `transformer` |
| `ABSA_DEVICE` | auto | `cpu` / `cuda` |
| `ABSA_CORS_ORIGINS` | localhost dev | **Must include the deployed frontend origin** |
| `ABSA_MAX_REVIEW_CHARS` | `5000` | Per review |
| `ABSA_MAX_BATCH_SIZE` | `500` | Per batch request |

Never commit a real `.env`; it is gitignored. In Colab use the secrets panel
rather than a file.

---

## 5. Sizing

Measured on the development machine (Ryzen 5 5600H, CPU only):

| | Baseline |
|---|---|
| Single review | ~40 ms |
| Batch of 500 | 5.5 s (11 ms/review) |
| Memory | ~200 MB resident |

A single small instance handles a portfolio demo comfortably. A transformer on
CPU is roughly 5–10× slower per review; if that matters, export to ONNX with int8
quantisation before scaling the box up.

Both halves are stateless — there is no database — so horizontal scaling is
just running more replicas.

---

## 6. Post-deploy checks

```bash
curl https://your-api.example.com/api/health
# {"status":"ok","model_loaded":true,...}

curl -X POST https://your-api.example.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"review":"Great camera but the battery is terrible."}'
```

Then open the frontend and confirm the header status dot is green — it reflects
`/api/health`, so a red dot means the SPA reached the API and the API has no
model, while a missing dot means the SPA could not reach the API at all. That
distinction is usually enough to tell a CORS problem from a model problem.
