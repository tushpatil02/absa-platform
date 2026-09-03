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

**The two stages are selected independently**, because the objective comparison
picked different model families for each:

===========================  ==================  ==================
stage                        TF-IDF baseline     DistilBERT
===========================  ==================  ==================
aspect detection (micro F1)  **0.7755**          0.6192
sentiment (macro F1)         0.6088              **0.6538**
===========================  ==================  ==================

Detection is largely a lexical problem -- the word "battery" all but determines
the aspect -- which is exactly what char+word n-grams are good at, and DistilBERT
over-predicts badly on 2,298 training reviews (micro precision 0.525). Sentiment
needs context the bag of words cannot represent. Forcing one family across both
stages would ship a materially worse detector, so :class:`Predictor` composes one
detector and one classifier chosen per stage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ml.inference.scoring import SentimentScore, aggregate_scores, build_score
from ml.inference.sentences import split_sentences
from ml.preprocessing.clean import clean_text, is_usable

logger = logging.getLogger(__name__)

# A review longer than this is rejected rather than silently truncated -- the
# API tells the caller instead of returning a prediction based on half the text.
MAX_INPUT_CHARS = 5000

DEFAULT_THRESHOLD = 0.5


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
    # Which sentences produced this. Empty when the review was scored whole.
    # Carried through to the API so a score can always be traced back to the
    # text that caused it, rather than asking the reader to trust the number.
    evidence: tuple[str, ...] = ()

    @property
    def mentions(self) -> int:
        """How many sentences supported this aspect."""
        return len(self.evidence)

    def as_dict(self) -> dict:
        return {
            "aspect": self.aspect,
            "display_name": self.display_name,
            "detection_confidence": self.detection_confidence,
            "mentions": self.mentions,
            "evidence": list(self.evidence),
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


# ---------------------------------------------------------------------------
# Stage interfaces
# ---------------------------------------------------------------------------


class AspectDetector(Protocol):
    """Stage A: review text -> a probability per aspect."""

    name: str
    threshold: float

    def detect(self, text: str) -> np.ndarray: ...


class SentimentClassifier(Protocol):
    """Stage B: (review, aspects) -> a probability distribution per aspect."""

    name: str

    def classify(self, text: str, aspect_descriptions: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Baseline stages (scikit-learn)
# ---------------------------------------------------------------------------


class BaselineAspectDetector:
    """One-vs-rest logistic regression over word + char TF-IDF."""

    def __init__(self, directory: Path):
        import joblib

        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self._model = joblib.load(directory / "model.joblib")
        self.threshold = float(metadata.get("threshold", DEFAULT_THRESHOLD))
        self.name = "tfidf-logreg"
        self.metadata = metadata

    def detect(self, text: str) -> np.ndarray:
        return np.asarray(self._model.predict_proba([text])[0], dtype=float)


class BaselineSentimentClassifier:
    """TF-IDF over aspect-prefixed text -> 3-class probabilities."""

    def __init__(self, directory: Path, n_classes: int):
        import joblib

        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self._model = joblib.load(directory / "model.joblib")
        self.name = "tfidf-" + ("svc" if "svc" in str(metadata.get("model", "")).lower() else "logreg")
        self.metadata = metadata
        self._n_classes = n_classes
        # The fitted label order may not be 0,1,2 if a class was absent from
        # training, so map explicitly rather than assuming column order.
        self._classes = [int(c) for c in self._model.classes_]

    def classify(self, text: str, aspect_descriptions: list[str]) -> np.ndarray:
        pairs = [f"{description} | {text}" for description in aspect_descriptions]
        raw = self._model.predict_proba(pairs)

        ordered = np.zeros((len(pairs), self._n_classes), dtype=float)
        for column, class_id in enumerate(self._classes):
            ordered[:, class_id] = raw[:, column]
        return ordered


# ---------------------------------------------------------------------------
# Transformer stages (HuggingFace)
# ---------------------------------------------------------------------------


class _TransformerStage:
    """Shared loading for both transformer stages. torch is imported lazily."""

    def __init__(self, directory: Path, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.max_length = int(self.metadata.get("max_length", 128))
        self.name = str(self.metadata.get("base_model", directory.name)).split("/")[-1]

        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(directory)
            .to(self._device)
            .eval()
        )


class TransformerAspectDetector(_TransformerStage):
    """Fine-tuned encoder with 12 sigmoid outputs."""

    def __init__(self, directory: Path, device: str | None = None):
        super().__init__(directory, device)
        self.threshold = float(self.metadata.get("threshold", DEFAULT_THRESHOLD))

    def detect(self, text: str) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            encoded = self._tokenizer(
                text, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self._device)
            logits = self._model(**encoded).logits
            return torch.sigmoid(logits)[0].float().cpu().numpy()


class TransformerSentimentClassifier(_TransformerStage):
    """Sentence-pair encoder: [CLS] review [SEP] aspect description [SEP]."""

    def classify(self, text: str, aspect_descriptions: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            encoded = self._tokenizer(
                [text] * len(aspect_descriptions),
                aspect_descriptions,
                truncation="only_first",  # never truncate the aspect away
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            logits = self._model(**encoded).logits
            return torch.softmax(logits, dim=-1).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Composed predictor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Evidence:
    """One sentence's contribution to one aspect."""

    detection: float
    probabilities: np.ndarray
    sentence: str


class Predictor:
    """Composes one aspect detector and one sentiment classifier.

    The two may come from different model families -- see the module docstring
    for why that is the measured-correct configuration here.
    """

    def __init__(self, detector: AspectDetector, classifier: SentimentClassifier, taxonomy):
        self.detector = detector
        self.classifier = classifier
        self.aspects = list(taxonomy.aspect_ids)
        self.polarities = list(taxonomy.polarities)
        self.display_names = dict(taxonomy.display_names)
        self.descriptions = dict(taxonomy.descriptions)

    @property
    def threshold(self) -> float:
        return self.detector.threshold

    @property
    def model_name(self) -> str:
        """Both stages, because they are chosen independently."""
        return f"aspect={self.detector.name} · sentiment={self.classifier.name}"

    # -- validation ---------------------------------------------------------

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

    # -- inference ----------------------------------------------------------

    def _gather(self, units: list[str], *, force_best: bool) -> dict[str, list[_Evidence]]:
        """Run both stages over each unit and collect per-aspect evidence.

        Args:
            units: Sentences, or a single element for whole-review scoring.
            force_best: When no aspect clears the threshold in a unit, take that
                unit's highest-scoring aspect anyway. Used only for the
                whole-review fallback -- applying it per sentence would make
                every "Arrived Tuesday." contribute a spurious vote.
        """
        evidence: dict[str, list[_Evidence]] = {}

        for unit in units:
            scores = np.asarray(self.detector.detect(unit), dtype=float).ravel()

            present = [self.aspects[i] for i in np.flatnonzero(scores >= self.threshold)]
            if not present and force_best:
                present = [self.aspects[int(scores.argmax())]]
            if not present:
                continue

            probabilities = self.classifier.classify(
                unit, [self.descriptions[aspect] for aspect in present]
            )
            for index, aspect in enumerate(present):
                evidence.setdefault(aspect, []).append(
                    _Evidence(
                        detection=float(scores[self.aspects.index(aspect)]),
                        probabilities=np.asarray(probabilities[index], dtype=float).ravel(),
                        sentence=unit,
                    )
                )

        return evidence

    def _combine(self, evidence: dict[str, list[_Evidence]]) -> list[AspectPrediction]:
        """Reduce each aspect's sentences to one prediction.

        Class probabilities are averaged across sentences, weighted by how
        confidently the detector placed that sentence on the aspect: a sentence
        it is 0.95 sure is about the battery should outweigh one it barely
        admitted at 0.51.

        Averaging *probabilities* rather than the derived 1-10 scores keeps the
        arithmetic in one place, so build_score stays the only thing that turns
        a distribution into a number and serving cannot drift from training.
        """
        predictions: list[AspectPrediction] = []

        for aspect, items in evidence.items():
            # Floor the weight so a detection of exactly 0.0 cannot make the
            # denominator vanish.
            weights = np.array([max(item.detection, 1e-6) for item in items])
            stacked = np.vstack([item.probabilities for item in items])
            mean = (stacked * weights[:, None]).sum(axis=0) / weights.sum()

            predictions.append(
                AspectPrediction(
                    aspect=aspect,
                    display_name=self.display_names.get(aspect, aspect),
                    detection_confidence=round(max(item.detection for item in items), 4),
                    sentiment=build_score(mean, self.polarities),
                    evidence=tuple(item.sentence for item in items),
                )
            )

        # Strongest detection first -- the aspect the model is surest about is
        # the one a reader should see at the top.
        predictions.sort(key=lambda prediction: -prediction.detection_confidence)
        return predictions

    def analyze(
        self,
        review: str,
        *,
        top_k: int | None = None,
        by_sentence: bool = True,
    ) -> AnalysisResult:
        """Analyse one review, sentence by sentence.

        Sentence-level is the default because the sentiment model scores 0.875
        on single-opinion text and 0.541 on text mixing opinions, and real
        reviews are overwhelmingly the latter -- see ml/inference/sentences.py
        for the measurement. ``by_sentence=False`` scores the review as one
        unit, which is how the published per-stage metrics were computed.
        """
        cleaned = self._prepare(review)
        units = split_sentences(cleaned) if by_sentence else [cleaned]

        evidence = self._gather(units, force_best=not by_sentence)

        if not evidence:
            # No individual sentence cleared the threshold. Returning nothing
            # reads as a failure, so fall back to scoring the whole review and
            # surfacing its single best aspect with its (low) confidence
            # attached, which is more honest than showing an empty result.
            evidence = self._gather([cleaned], force_best=True)

        predictions = self._combine(evidence)
        if top_k:
            predictions = predictions[:top_k]

        return AnalysisResult(
            review=review,
            cleaned=cleaned,
            aspects=predictions,
            overall_score=aggregate_scores([p.sentiment.score for p in predictions]),
            model=self.model_name,
        )

    def analyze_batch(
        self,
        reviews: list[str],
        *,
        top_k: int | None = None,
        by_sentence: bool = True,
    ) -> list[AnalysisResult]:
        return [
            self.analyze(review, top_k=top_k, by_sentence=by_sentence) for review in reviews
        ]


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
                    "counts": dict.fromkeys(polarities, 0),
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

BASELINE_ACD = "baseline_aspect_detector"
BASELINE_ASC = "baseline_sentiment_classifier"
TRANSFORMER_ACD = "aspect_detector"
TRANSFORMER_ASC = "sentiment_classifier"


# Weight filenames each family actually needs. metadata.json alone is NOT
# enough: it is committed to git for reference while the weights are not, so a
# fresh clone has the descriptor and no model. Checking only the descriptor made
# load_predictor crash with an OSError from transformers instead of falling back.
BASELINE_WEIGHTS = ("model.joblib",)
TRANSFORMER_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _available(directory: Path, weights: tuple[str, ...]) -> bool:
    """True only when the descriptor AND loadable weights are present."""
    if not (directory / "metadata.json").exists():
        return False
    return any((directory / name).exists() for name in weights)


def _winner_from_comparison(models_dir: Path, key: str) -> str | None:
    """Which family won this stage, according to models/metadata/comparison.json.

    That file is written by ``scripts/compare_models.py`` from held-out test
    metrics, so it -- not a hard-coded preference -- decides what ``auto`` picks.
    Returns ``"transformer"``, ``"baseline"``, or ``None`` when unknown.
    """
    path = models_dir / "metadata" / "comparison.json"
    if not path.exists():
        return None
    try:
        comparison = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    selected = comparison.get(key, {}).get("selected")
    if not selected:
        return None
    return "baseline" if "tfidf" in selected.lower() else "transformer"


def _resolve(preference: str, stage_key: str, models_dir: Path, has_transformer: bool) -> str:
    """Decide which family to load for one stage."""
    if preference in ("baseline", "transformer"):
        return preference
    if not has_transformer:
        return "baseline"

    winner = _winner_from_comparison(models_dir, stage_key)
    if winner is None:
        # No measured comparison, so do not gamble on the bigger model.
        logger.info(
            "No comparison.json for %s; defaulting to the baseline. "
            "Run scripts/compare_models.py to select on measured metrics.",
            stage_key,
        )
        return "baseline"
    return winner


def load_predictor(
    models_dir: Path,
    taxonomy,
    *,
    prefer: str = "auto",
    device: str | None = None,
    prefer_aspect: str | None = None,
    prefer_sentiment: str | None = None,
) -> Predictor:
    """Load the best available predictor, choosing each stage independently.

    Args:
        models_dir: Root ``models/`` directory.
        taxonomy: Loaded taxonomy.
        prefer: Default for both stages -- ``"auto"``, ``"baseline"`` or
            ``"transformer"``. Under ``"auto"`` each stage follows
            ``models/metadata/comparison.json``, which is produced from held-out
            test metrics; with no such file the baseline is used, because
            defaulting to the larger model would have shipped a detector 15.6
            micro-F1 points worse than the baseline here.
        device: Torch device override.
        prefer_aspect: Override `prefer` for stage A only.
        prefer_sentiment: Override `prefer` for stage B only.

    Raises:
        FileNotFoundError: If the requested artefacts are missing. The message
            names the script that produces them.
    """
    baseline_acd, baseline_asc = models_dir / BASELINE_ACD, models_dir / BASELINE_ASC
    transformer_acd, transformer_asc = models_dir / TRANSFORMER_ACD, models_dir / TRANSFORMER_ASC

    aspect_choice = _resolve(
        prefer_aspect or prefer,
        "aspect_detection",
        models_dir,
        _available(transformer_acd, TRANSFORMER_WEIGHTS),
    )
    sentiment_choice = _resolve(
        prefer_sentiment or prefer,
        "sentiment",
        models_dir,
        _available(transformer_asc, TRANSFORMER_WEIGHTS),
    )

    for choice, directory, script in (
        (aspect_choice, transformer_acd if aspect_choice == "transformer" else baseline_acd,
         "train_transformer.py --stage acd" if aspect_choice == "transformer" else "train_baseline.py"),
        (sentiment_choice, transformer_asc if sentiment_choice == "transformer" else baseline_asc,
         "train_transformer.py --stage asc" if sentiment_choice == "transformer" else "train_baseline.py"),
    ):
        weights = TRANSFORMER_WEIGHTS if choice == "transformer" else BASELINE_WEIGHTS
        if not _available(directory, weights):
            detail = ""
            if (directory / "metadata.json").exists():
                detail = (
                    " The descriptor is present but the weights are not -- they are "
                    "kept out of git and pulled from the Hugging Face Hub; see docs/deployment.md."
                )
            raise FileNotFoundError(
                f"No {choice} artefacts at {directory}. Run scripts/{script}.{detail}"
            )

    detector: AspectDetector = (
        TransformerAspectDetector(transformer_acd, device)
        if aspect_choice == "transformer"
        else BaselineAspectDetector(baseline_acd)
    )
    classifier: SentimentClassifier = (
        TransformerSentimentClassifier(transformer_asc, device)
        if sentiment_choice == "transformer"
        else BaselineSentimentClassifier(baseline_asc, len(taxonomy.polarities))
    )

    predictor = Predictor(detector, classifier, taxonomy)
    logger.info("Predictor ready: %s", predictor.model_name)
    return predictor
