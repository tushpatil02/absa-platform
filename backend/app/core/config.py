"""Application settings.

Everything configurable comes from the environment (or a local ``.env``), so no
secret or host-specific path is ever committed. See ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="ABSA_"
    )

    app_name: str = "ABSA Platform API"
    version: str = "0.1.0"
    debug: bool = False

    models_dir: Path = Field(default=REPO_ROOT / "models")
    taxonomy_path: Path = Field(default=REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")

    # The deployable catalogue: phones.csv, phone_profiles.csv and
    # phone_evidence.json, all small. The ~45 MB review_aspects.csv that
    # produced them is an intermediate and is never read here.
    catalog_dir: Path = Field(default=REPO_ROOT / "data" / "catalog")
    # Reviews submitted through the site. The only thing written at runtime.
    database_path: Path = Field(default=REPO_ROOT / "data" / "reviews.db")

    # "auto" follows models/metadata/comparison.json, which is written from
    # held-out test metrics. The two stages are resolved independently because
    # the comparison selected different families for each: TF-IDF wins aspect
    # detection (micro F1 0.7755 vs 0.6192), DistilBERT wins sentiment
    # (macro F1 0.6538 vs 0.6088).
    predictor: str = "auto"
    # Per-stage overrides. Set ABSA_SENTIMENT_MODEL=baseline to trade macro F1
    # for better-calibrated confidence -- see docs/model.md.
    aspect_model: str | None = None
    sentiment_model: str | None = None
    device: str | None = None

    # Guardrails. Batch size is capped because each review is a model call and
    # an unbounded list would let one request occupy the worker indefinitely.
    max_review_chars: int = 5000
    max_batch_size: int = 500

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        """Allow ABSA_CORS_ORIGINS="https://a.com,https://b.com"."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("predictor", "aspect_model", "sentiment_model")
    @classmethod
    def _check_predictor(cls, value: str | None) -> str | None:
        allowed = {"auto", "baseline", "transformer"}
        if value is not None and value not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)}, got {value!r}")
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
