"""Classical TF-IDF baselines for both ABSA stages.

These exist to be beaten -- and to make the case that they *need* beating. If a
transformer only matches a linear model on TF-IDF features, the linear model
wins: it trains in seconds, needs no GPU, and serves in under a millisecond.

Two stages:

**ACD** (aspect detection) -- multi-label. One-vs-rest logistic regression over
word + character n-grams. Character n-grams matter here because review text is
full of misspellings ("batery", "camara") that word n-grams miss entirely.

**ASC** (sentiment) -- single-label, 3 classes. The important design point is
that the model must see *which aspect it is being asked about*: the same review
is positive for `camera` and negative for `battery`. A bag of words over the
review alone cannot represent that, so the aspect is injected as a feature. Two
ways to do that are provided:

* ``aspect_prefix`` -- prepend the aspect description to the text, mirroring the
  ``[CLS] review [SEP] aspect [SEP]`` form the transformer stage uses.
* ``aspect_onehot`` -- a separate one-hot block union'd onto the TF-IDF features.

The prefix variant is the honest comparison against the transformer, since it
encodes the aspect the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Shared feature blocks
# ---------------------------------------------------------------------------


def _word_tfidf(**kwargs) -> TfidfVectorizer:
    """Word n-grams.

    ``sublinear_tf`` damps repeated words, which matters because reviewers
    repeat themselves ("slow slow slow"). Lowercasing is on *here* even though
    the cleaning stage preserved case: case carries emphasis for a transformer
    that can model it, but for a bag of words it only splits the vocabulary.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents=None,
        **kwargs,
    )


def _char_tfidf(**kwargs) -> TfidfVectorizer:
    """Character n-grams inside word boundaries -- robust to misspellings."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        lowercase=True,
        **kwargs,
    )


def build_text_features() -> FeatureUnion:
    return FeatureUnion(
        [("word", _word_tfidf()), ("char", _char_tfidf())],
        # char_wb over 3-5 grams is the expensive half; give it its own worker.
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Stage A -- aspect category detection (multi-label)
# ---------------------------------------------------------------------------


def build_acd_baseline(C: float = 4.0) -> Pipeline:
    """One-vs-rest logistic regression over word + char TF-IDF.

    ``class_weight="balanced"`` matters a lot here: aspects like `audio` appear
    on ~2% of reviews, and an unweighted model maximises accuracy by never
    predicting them.

    Logistic regression (not LinearSVC) because ``predict_proba`` is needed --
    the API returns a real confidence per aspect, and the decision threshold is
    tuned on dev.
    """
    return Pipeline(
        [
            ("features", build_text_features()),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=C,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Stage B -- aspect sentiment classification (single-label, 3 classes)
# ---------------------------------------------------------------------------


@dataclass
class AspectPrefixEncoder:
    """Render ``(text, aspect)`` into one string the vectorizer can consume.

    Mirrors the transformer's sentence-pair input so the two are comparable:
    the aspect description, a separator, then the review.
    """

    descriptions: dict[str, str]

    def __call__(self, frame) -> list[str]:
        return [
            f"{self.descriptions.get(aspect, aspect)} | {text}"
            for text, aspect in zip(frame["text"], frame["aspect"])
        ]


def build_asc_baseline(
    kind: str = "logreg",
    *,
    C: float = 2.0,
    calibrate: bool = True,
) -> Pipeline:
    """TF-IDF over aspect-prefixed text -> 3-class classifier.

    Args:
        kind: ``"logreg"`` or ``"svc"``.
        C: Regularisation strength.
        calibrate: Wrap LinearSVC in Platt scaling so it can emit probabilities.
            LinearSVC alone has only ``decision_function``, and the 1-10
            sentiment score is defined over a probability distribution -- an
            uncalibrated margin would make that score meaningless.

    ``class_weight="balanced"`` again, because neutral is 5.3% of the data.
    """
    if kind == "logreg":
        classifier = LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )
    elif kind == "svc":
        svc = LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE)
        classifier = (
            CalibratedClassifierCV(svc, method="sigmoid", cv=3) if calibrate else svc
        )
    else:
        raise ValueError(f"Unknown baseline kind {kind!r}; expected 'logreg' or 'svc'")

    return Pipeline([("features", build_text_features()), ("clf", classifier)])


def build_asc_onehot_baseline(aspects: list[str], C: float = 2.0) -> Pipeline:
    """Alternative encoding: TF-IDF(text) union one-hot(aspect).

    Kept as a comparison point. It gives the model the aspect as an explicit
    categorical feature rather than as text, which is cleaner in principle but
    loses the lexical overlap between the aspect description and the review
    (the word "battery" appearing in both).
    """
    select_text = FunctionTransformer(
        lambda frame: frame["text"].tolist(), validate=False
    )
    select_aspect = FunctionTransformer(
        lambda frame: frame[["aspect"]], validate=False
    )

    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", Pipeline([("sel", select_text), ("tfidf", _word_tfidf())])),
                        ("char", Pipeline([("sel", select_text), ("tfidf", _char_tfidf())])),
                        (
                            "aspect",
                            Pipeline(
                                [
                                    ("sel", select_aspect),
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            categories=[aspects],
                                            handle_unknown="ignore",
                                        ),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def predict_multilabel(pipeline: Pipeline, texts, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(binary_predictions, probabilities)`` for a multi-label model."""
    probabilities = pipeline.predict_proba(texts)
    return (probabilities >= threshold).astype(int), probabilities
