# API container.
#
# Two stages so the runtime image carries no build toolchain. Deliberately
# CPU-only and torch-free: the deployed model is the scikit-learn baseline, so
# the image stays a few hundred MB rather than several GB. To serve a fine-tuned
# transformer, add torch (CPU wheel) and transformers to requirements-serve.txt.

FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-serve.txt


FROM python:3.12-slim

# Run as a non-root user: the container never needs write access to its own code.
RUN useradd --create-home --uid 1000 absa
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Only what serving needs. Training code, notebooks, tests and the large
# intermediate data stay out; the built catalogue comes in, because the
# recommender cannot answer anything without it.
COPY --chown=absa:absa ml/ ./ml/
COPY --chown=absa:absa backend/ ./backend/
COPY --chown=absa:absa models/ ./models/
COPY --chown=absa:absa data/catalog/ ./data/catalog/

USER absa
EXPOSE 8000

# Hits the real readiness endpoint, so a container with no model loaded is
# reported unhealthy rather than merely "listening".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,json,sys; \
r=json.load(urllib.request.urlopen('http://localhost:8000/api/health')); \
sys.exit(0 if r.get('model_loaded') else 1)"

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
