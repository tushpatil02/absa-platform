"""Pick the sentences a phone page shows beneath its scores.

``score_catalog.py`` records, for every review and aspect, the sentences that
produced the score. That is far too much to serve -- tens of thousands of rows
per phone -- so this selects a small, *balanced* set.

Balance is the point. Showing the highest-scoring sentences for every aspect
would turn a phone page into an advertisement, and showing the lowest would
turn it into a complaint board. Both would be selective quotation of the
model's own output. For each aspect this takes the strongest positive **and**
the strongest negative sentence where both exist, so the page reflects the
disagreement that is actually in the reviews.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.recommender.similarity import AXES, AXIS_LABELS

logger = logging.getLogger(__name__)

# Sentences shorter than this carry no reasoning worth showing ("Good.").
MIN_SENTENCE_CHARS = 20
# One review can mention an aspect several times; only its first sentence is
# shown, so a single verbose reviewer cannot dominate a phone page.
SEPARATOR = " || "


def load_evidence(processed_dir: Path, *, per_phone: int = 12) -> dict[str, list[dict]]:
    """Build ``{model_key: [evidence, ...]}`` from the scored reviews.

    Returns an empty mapping when the scores are missing, so the API can start
    and serve the rest of the catalogue.
    """
    path = processed_dir / "review_aspects.csv"
    if not path.exists():
        logger.info("No review_aspects.csv; phone pages will show no evidence.")
        return {}

    try:
        frame = pd.read_csv(path, usecols=["model_key", "aspect", "polarity", "score", "evidence"])
    except (ValueError, pd.errors.EmptyDataError) as exc:
        logger.warning("Could not read evidence: %s", exc)
        return {}

    frame = frame[frame["aspect"].isin(AXES)]
    frame = frame[frame["evidence"].notna()]
    # regex=False is load-bearing. pandas treats a multi-character `pat` as a
    # regular expression, and " || " is then alternation over spaces and the
    # empty string -- it splits between every character and every sentence
    # comes back empty. Silently, with no error and no evidence on any page.
    frame["sentence"] = (
        frame["evidence"].astype(str).str.split(SEPARATOR, regex=False).str[0].str.strip()
    )
    frame = frame[frame["sentence"].str.len() >= MIN_SENTENCE_CHARS]
    if frame.empty:
        return {}

    # Per aspect, the most extreme sentence in each direction. Sorting by score
    # and taking from both ends is what keeps the selection balanced.
    per_aspect = max(1, per_phone // (2 * len(AXES)))
    evidence: dict[str, list[dict]] = {}

    for (model_key, aspect), group in frame.groupby(["model_key", "aspect"], sort=False):
        ordered = group.sort_values("score")
        picks = pd.concat(
            [
                ordered[ordered["polarity"] == "negative"].head(per_aspect),
                ordered[ordered["polarity"] == "positive"].tail(per_aspect),
            ]
        ).drop_duplicates(subset=["sentence"])

        for row in picks.itertuples(index=False):
            evidence.setdefault(model_key, []).append(
                {
                    "aspect": aspect,
                    "display_name": AXIS_LABELS.get(aspect, aspect.title()),
                    "polarity": row.polarity,
                    "score": round(float(row.score), 2),
                    "sentence": row.sentence,
                }
            )

    # Present in slider order, negative first within an aspect, so a reader sees
    # the criticism as readily as the praise.
    order = {axis: index for index, axis in enumerate(AXES)}
    for items in evidence.values():
        items.sort(key=lambda item: (order.get(item["aspect"], 99), item["score"]))

    logger.info("Evidence loaded for %d phones", len(evidence))
    return evidence
