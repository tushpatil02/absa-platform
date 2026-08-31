"""End-to-end ABSA inference.

This module is the **single source of truth for prediction**, imported by the
FastAPI backend, by the notebooks and by the tests. Nothing re-implements
preprocessing or scoring elsewhere: if training and serving ever disagree about
how text is cleaned or how a score is computed, that is a silent accuracy bug
that no test on either side would catch.

Pipeline::

    raw review
      -> clean_text()            same function used to build the training data
      -> aspect detection        multi-label, tuned threshold -> aspects present
      -> sentiment per aspect    one sentence-pair per detected aspect
      -> build_score()           probabilities -> 1-10 score + confidence
      -> AnalysisResult

Two backends implement the same interface:

* :class:`BaselinePredictor` -- scikit-learn pipelines from joblib. No torch
  import, tiny memory footprint, sub-millisecond. This is the default for
  deployment.
* :class:`TransformerPredictor` -- HuggingFace models. Imported lazily so a
  torch-free serving container never pays for it.

Both are constructed by :func:`load_predictor`, which picks based on what is
actually on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ml.inference.scoring import SentimentScore, aggregate_scores, build_score
from ml.preprocessing.clean import clean_text, is_usable

# A review longer than this is rejected rather than silently truncated -- the
# API tells the caller instead of returning a prediction based on half the text.
MAX_INPUT_CHARS = 5000


class EmptyReviewError(ValueError):
    """Raised when a review has no usable text after cleaning."""


class ReviewTooLongError(ValueError):
    """Raised when a review exceeds :data:`MAX_INPUT_CHARS`."""


@dataclass(frozen=True)
class AspectPrediction:
    """One detected aspect and its sentiment."""

    aspect: str
    display_name: str
    detection_confidence: float
    sentiment: SentimentScore

    def as_dict(self) -> dict:
        return {
            "aspect": self.aspect,
            "display_name": self.display_name,
            "detection_confidence": self.detection_confidence,
            **self.sentiment.as_dict(),
        }


@dataclass(frozen=True)
class AnalysisResult:
    """The full result for one review."""

    review: str
    cleaned: str
    aspects: list[AspectPrediction]
    overall_score: float | None
    model: str

    def as_dict(self) -> dict:
        return {
            "review": self.review,
            "aspects": [a.as_dict() for a in self.aspects],
            "overall_score": self.overall_score,
            "model": self.model,
        }


class SupportsAnalyze(Protocol):
    """What the backend depends on. Both predictors satisfy it."""

    def analyze(self, review: str, *, top_k: int | None = ...) -> AnalysisResult: ...


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------


@dataclass
class _PredictorBase:
    """Validation, cleaning and assembly shared by both backends."""

    aspects: list[str]
    polarities: list[str]
    display_names: dict[str, str]
    descriptions: dict[str, str]
    threshold: float
    model_name: str
    metadata: dict = field(default_factory=dict)

    def _prepare(self, review: str) -> str:
        """Validate and clean, or raise a specific, catchable error."""
        if review is None or not str(review).strip():
            raise EmptyReviewError("Review text is empty.")
        if len(review) > MAX_INPUT_CHARS:
            raise ReviewTooLongError(
                f"Review is {len(review)} characters; the limit is {MAX_INPUT_CHARS}."
            )
        cleaned = clean_text(review)
        if not is_usable(cleaned):
            raise EmptyReviewError(
                "Review contains no usable text after cleaning (needs at least a few letters)."
            )
        return cleaned

    def _assemble(
        self,
        review: str,
        cleaned: str,
        aspect_scores: np.ndarray,
        sentiment_probabilities: dict[str, np.ndarray],
        selected: list[str],
    ) -> AnalysisResult:
        predictions = []
        for aspect in selected:
            index = self.aspects.index(aspect)
            predictions.append(
                AspectPrediction(
                    aspect=aspect,
                    display_name=self.display_names.get(aspect, aspect),
                    detection_confidence=round(float(aspect_scores[index]), 4),
                    sentiment=build_score(sentiment_probabilities[aspect], self.polarities),
                )
            )
        # Strongest detection first -- the aspect the model is surest about
        # is the one a reader should see at the top.
        predictions.sort(key=lambda p: -p.detection_confidence)

        return AnalysisResult(
            review=review,
            cleaned=cleaned,
            aspects=predictions,
            overall_score=aggregate_scores([p.sentiment.score for p in predictions]),
            model=self.model_name,
        )

    def _select_aspects(self, scores: np.ndarray, top_k: int | None) -> list[str]:
        """Aspects above threshold; falls back to the single best.

        The fallback matters for UX: a short review like "love it" may clear no
        per-aspect threshold, and returning an empty list looks like a failure.
        Returning the top aspect with its (low) confidence attached is more
        honest than showing nothing.
        """
        above = [self.aspects[i] for i in np.flatnonzero(scores >= self.threshold)]
        if not above:
            above = [self.aspects[int(scores.argmax())]]
        above.sort(key=lambda a: -scores[self.aspects.index(a)])
        return above[:top_k] if top_k else above

    def analyze_batch(self, reviews: list[str], *, top_k: int | None = None) -> list[AnalysisResult]:
        """Analyse many reviews. Overridden where a real batched path exists."""
        return [self.analyze(review, top_k=top_k) for review in reviews]


# ---------------------------------------------------------------------------
# Baseline (scikit-learn)
# ---------------------------------------------------------------------------


class BaselinePredictor(_PredictorBase):
    """TF-IDF + linear models loaded from joblib."""

    def __init__(self, acd_dir: Path, asc_dir: Path, taxonomy):
        import joblib

        acd_meta = json.loads((acd_dir / "metadata.json").read_text(encoding="utf-8"))
        asc_meta = json.loads((asc_dir / "metadata.json").read_text(encoding="utf-8"))

        super().__init__(
            aspects=list(taxonomy.aspect_ids),
            polarities=list(taxonomy.polarities),
            display_names=dict(taxonomy.display_names),
            descriptions=dict(taxonomy.descriptions),
            threshold=float(acd_meta.get("threshold", 0.5)),
            model_name=f"baseline:{asc_meta.get('model', 'tfidf')}",
            metadata={"acd": acd_meta, "asc": asc_meta},
        )
        self._acd = joblib.load(acd_dir / "model.joblib")
        self._asc = joblib.load(asc_dir / "model.joblib")

        # The label order the classifier was fitted with may not be 0,1,2 if a
        # class was absent from training. Map explicitly rather than assume.
        self._asc_classes = list(self._asc.classes_)

    def _sentiment_probabilities(self, texts: list[str]) -> np.ndarray:
        """Predict, reordered into canonical polarity index order."""
        raw = self._asc.predict_proba(texts)
        ordered = np.zeros((len(texts), len(self.polarities)), dtype=float)
        for column, class_id in enumerate(self._asc_classes):
            ordered[:, int(class_id)] = raw[:, column]
        return ordered

    def analyze(self, review: str, *, top_k: int | None = None) -> AnalysisResult:
        cleaned = self._prepare(review)
        scores = self._acd.predict_proba([cleaned])[0]
        selected = self._select_aspects(scores, top_k)

        pair_texts = [f"{self.descriptions[a]} | {cleaned}" for a in selected]
        probabilities = self._sentiment_probabilities(pair_texts)

        return self._assemble(
            review, cleaned, scores,
            {aspect: probabilities[i] for i, aspect in enumerate(selected)},
            selected,
        )


# ---------------------------------------------------------------------------
# Transformer (HuggingFace)
# ---------------------------------------------------------------------------


class TransformerPredictor(_PredictorBase):
    """Fine-tuned encoders. torch is imported lazily, inside ``__init__``."""

    def __init__(self, acd_dir: Path, asc_dir: Path, taxonomy, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        acd_meta = json.loads((acd_dir / "metadata.json").read_text(encoding="utf-8"))
        asc_meta = json.loads((asc_dir / "metadata.json").read_text(encoding="utf-8"))

        super().__init__(
            aspects=list(taxonomy.aspect_ids),
            polarities=list(taxonomy.polarities),
            display_names=dict(taxonomy.display_names),
            descriptions=dict(taxonomy.descriptions),
            threshold=float(acd_meta.get("threshold", 0.5)),
            model_name=f"transformer:{asc_meta.get('base_model', 'unknown')}",
            metadata={"acd": acd_meta, "asc": asc_meta},
        )

        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._max_length = int(asc_meta.get("max_length", 128))

        self._acd_tokenizer = AutoTokenizer.from_pretrained(acd_dir)
        self._acd_model = AutoModelForSequenceClassification.from_pretrained(acd_dir).to(self._device).eval()
        self._asc_tokenizer = AutoTokenizer.from_pretrained(asc_dir)
        self._asc_model = AutoModelForSequenceClassification.from_pretrained(asc_dir).to(self._device).eval()

    def analyze(self, review: str, *, top_k: int | None = None) -> AnalysisResult:
        torch = self._torch
        cleaned = self._prepare(review)

        with torch.no_grad():
            encoded = self._acd_tokenizer(
                cleaned, truncation=True, max_length=self._max_length,
                padding=True, return_tensors="pt",
            ).to(self._device)
            scores = torch.sigmoid(self._acd_model(**encoded).logits)[0].cpu().numpy()

        selected = self._select_aspects(scores, top_k)

        with torch.no_grad():
            encoded = self._asc_tokenizer(
                [cleaned] * len(selected),
                [self.descriptions[a] for a in selected],
                truncation="only_first", max_length=self._max_length,
                padding=True, return_tensors="pt",
            ).to(self._device)
            probabilities = torch.softmax(self._asc_model(**encoded).logits, dim=-1).cpu().numpy()

        return self._assemble(
            review, cleaned, scores,
            {aspect: probabilities[i] for i, aspect in enumerate(selected)},
            selected,
        )


# ---------------------------------------------------------------------------
# Product-level aggregation
# ---------------------------------------------------------------------------


def summarise_product(results: list[AnalysisResult], polarities: list[str]) -> dict:
    """Aggregate many analysed reviews into a per-aspect product summary.

    This answers "what do customers actually like and dislike about this
    product?" -- the question an overall star rating cannot.

    Shares are computed over the reviews that *mention* each aspect, not over
    all reviews, so "61% negative on battery" means 61% of the people who talked
    about battery, which is the only reading that makes sense.
    """
    by_aspect: dict[str, dict] = {}

    for result in results:
        for prediction in result.aspects:
            entry = by_aspect.setdefault(
                prediction.aspect,
                {
                    "aspect": prediction.aspect,
                    "display_name": prediction.display_name,
                    "mentions": 0,
                    "scores": [],
                    "counts": {name: 0 for name in polarities},
                },
            )
            entry["mentions"] += 1
            entry["scores"].append(prediction.sentiment.score)
            entry["counts"][prediction.sentiment.polarity] += 1

    summary = []
    for entry in by_aspect.values():
        mentions = entry["mentions"]
        summary.append(
            {
                "aspect": entry["aspect"],
                "display_name": entry["display_name"],
                "mentions": mentions,
                "mention_share": round(mentions / max(len(results), 1), 4),
                "average_score": aggregate_scores(entry["scores"]),
                "counts": entry["counts"],
                "shares": {
                    name: round(count / mentions, 4) for name, count in entry["counts"].items()
                },
            }
        )
    summary.sort(key=lambda item: -item["mentions"])

    all_scores = [p.sentiment.score for r in results for p in r.aspects]
    return {
        "reviews_analyzed": len(results),
        "overall_score": aggregate_scores(all_scores),
        "aspects": summary,
        "most_positive": max(summary, key=lambda i: i["average_score"], default=None),
        "most_negative": min(summary, key=lambda i: i["average_score"], default=None),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_predictor(
    models_dir: Path,
    taxonomy,
    *,
    prefer: str = "auto",
    device: str | None = None,
) -> SupportsAnalyze:
    """Load the best available predictor.

    Args:
        models_dir: Root ``models/`` directory.
        taxonomy: Loaded taxonomy.
        prefer: ``"auto"`` (transformer if present, else baseline),
            ``"baseline"``, or ``"transformer"``.
        device: Torch device override.

    Raises:
        FileNotFoundError: If the requested artefacts are not on disk. The
            message names the script that produces them.
    """
    transformer_acd = models_dir / "aspect_detector"
    transformer_asc = models_dir / "sentiment_classifier"
    baseline_acd = models_dir / "baseline_aspect_detector"
    baseline_asc = models_dir / "baseline_sentiment_classifier"

    def complete(*paths: Path) -> bool:
        return all((p / "metadata.json").exists() for p in paths)

    if prefer in ("auto", "transformer") and complete(transformer_acd, transformer_asc):
        return TransformerPredictor(transformer_acd, transformer_asc, taxonomy, device)

    if prefer == "transformer":
        raise FileNotFoundError(
            f"No transformer artefacts in {models_dir}. Run scripts/train_transformer.py."
        )

    if complete(baseline_acd, baseline_asc):
        return BaselinePredictor(baseline_acd, baseline_asc, taxonomy)

    raise FileNotFoundError(
        f"No model artefacts found in {models_dir}. "
        "Run scripts/train_baseline.py (fast, CPU) or scripts/train_transformer.py."
    )
