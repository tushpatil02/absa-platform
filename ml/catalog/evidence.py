"""Choose the sentences a phone page shows beneath its scores.

``score_catalog.py`` records, for every review and aspect, the sentences that
produced the score -- around 45 MB by the end. That file is an intermediate: it
exists so aggregation can be re-tried without re-running an hour of inference,
and it is far too large either to ship or to parse at API startup.

This distils it once, at build time, into a few hundred kilobytes of JSON.

Balance is the point
--------------------
Showing the highest-scoring sentences for every aspect would turn a phone page
into an advertisement; showing the lowest would turn it into a complaint board.
Both are selective quotation of the model's own output. For each aspect this
takes the strongest positive **and** the strongest negative sentence where both
exist, so the page reflects the disagreement actually present in the reviews.
"""

from __future__ import annotations

import pandas as pd

from ml.recommender.similarity import AXES, AXIS_LABELS

# Sentences shorter than this carry no reasoning worth showing ("Good.").
MIN_SENTENCE_CHARS = 20

# score_catalog.py joins a review's sentences for one aspect with this.
SEPARATOR = " || "

DEFAULT_PER_PHONE = 12


def select_evidence(
    review_aspects: pd.DataFrame,
    *,
    per_phone: int = DEFAULT_PER_PHONE,
) -> dict[str, list[dict]]:
    """Build ``{model_key: [evidence, ...]}`` from the scored reviews."""
    frame = review_aspects
    frame = frame[frame["aspect"].isin(AXES)]
    frame = frame[frame["evidence"].notna()]
    if frame.empty:
        return {}

    # regex=False is load-bearing. pandas treats a multi-character `pat` as a
    # regular expression, and " || " is then alternation over spaces and the
    # empty string -- it splits between every character and every sentence
    # comes back empty. Silently, with no error and no evidence on any page.
    frame = frame.assign(
        sentence=frame["evidence"].astype(str).str.split(SEPARATOR, regex=False).str[0].str.strip()
    )
    frame = frame[frame["sentence"].str.len() >= MIN_SENTENCE_CHARS]
    if frame.empty:
        return {}

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

    # Slider order, negative first within an aspect, so a reader meets the
    # criticism as readily as the praise.
    order = {axis: index for index, axis in enumerate(AXES)}
    for items in evidence.values():
        items.sort(key=lambda item: (order.get(item["aspect"], 99), item["score"]))

    return evidence
