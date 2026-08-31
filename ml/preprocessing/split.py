"""Train/validation/test splitting with hard leakage guarantees.

This module exists because of one specific failure mode. A single review
produces ~1.4 ``(review, aspect)`` rows, so splitting at the *row* level puts
sentences from the same review on both sides of the train/test boundary. The
model then scores well by recognising sentences it has already read. That is
the most common way a portfolio ABSA project ends up reporting 99%.

Two defences, both enforced rather than assumed:

1. **Group by review.** Splits are grouped on ``review_id``; every row from one
   review lands in exactly one split.
2. **Group by normalised text.** M-ABSA's own splits already leak 3 reviews
   (``train n test = 1``, ``train n dev = 2``) -- the same text appears under
   different ids. Grouping on text as well catches those.

:func:`assert_no_leakage` re-checks the result and raises. It is called by
``scripts/build_dataset.py`` and by the test suite, so a regression here fails
the build instead of inflating a metric.
"""

from __future__ import annotations

import collections
import re
import unicodedata
from dataclasses import dataclass

from ml.preprocessing.transform import AspectPair

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_for_dedup(text: str) -> str:
    """Aggressive normalisation used *only* for duplicate detection.

    Casing and punctuation are stripped here so that "Great phone!" and
    "great phone" are recognised as the same review. This never touches the
    text used for training -- see :mod:`ml.preprocessing.clean` for that.
    """
    folded = unicodedata.normalize("NFKD", text).casefold()
    return _NON_ALNUM.sub(" ", folded).strip()


@dataclass
class SplitReport:
    """What splitting did, including any duplicates it had to resolve."""

    counts: dict[str, int]
    review_counts: dict[str, int]
    cross_split_duplicates_resolved: int = 0
    duplicate_groups: int = 0
    rows_dropped_as_duplicates: int = 0

    def summary(self) -> str:
        pairs = "  ".join(f"{name}={count}" for name, count in self.counts.items())
        reviews = "  ".join(f"{name}={count}" for name, count in self.review_counts.items())
        return (
            f"pairs   {pairs}\n"
            f"reviews {reviews}\n"
            f"duplicate text groups={self.duplicate_groups}  "
            f"cross-split duplicates resolved={self.cross_split_duplicates_resolved}  "
            f"rows dropped={self.rows_dropped_as_duplicates}"
        )


# Priority when the same review text appears in more than one split.
# Test wins so the evaluation set stays intact; dev next; train yields, because
# losing a handful of training rows is cheaper than a contaminated test set.
_SPLIT_PRIORITY = {"test": 0, "dev": 1, "train": 2}


def deduplicate_across_splits(
    pairs: list[AspectPair],
) -> tuple[list[AspectPair], SplitReport]:
    """Enforce that identical review text never spans two splits.

    Uses M-ABSA's own split assignment (which is review-grouped upstream) and
    repairs only the cases where the same text appears in more than one split.
    The whole duplicate group is reassigned to the highest-priority split, then
    collapsed to a single review id so the text appears exactly once.
    """
    by_text: dict[str, list[AspectPair]] = collections.defaultdict(list)
    for pair in pairs:
        by_text[normalise_for_dedup(pair.text)].append(pair)

    kept: list[AspectPair] = []
    duplicate_groups = 0
    cross_split_resolved = 0
    rows_dropped = 0

    for group in by_text.values():
        splits = {pair.split for pair in group}
        review_ids = {pair.review_id for pair in group}

        if len(review_ids) > 1:
            duplicate_groups += 1

        if len(splits) > 1:
            cross_split_resolved += 1

        # Winning split, and a single canonical review id for the whole group.
        target_split = min(splits, key=lambda name: _SPLIT_PRIORITY.get(name, 99))
        canonical_id = sorted(review_ids)[0]

        # Collapse to one row per aspect. Where duplicates disagree on polarity,
        # prefer the row from the winning split; otherwise take the first
        # deterministically so runs are reproducible.
        best_by_aspect: dict[str, AspectPair] = {}
        for pair in sorted(group, key=lambda p: (_SPLIT_PRIORITY.get(p.split, 99), p.review_id)):
            if pair.aspect in best_by_aspect:
                rows_dropped += 1
                continue
            best_by_aspect[pair.aspect] = pair

        for aspect, pair in best_by_aspect.items():
            kept.append(
                AspectPair(
                    review_id=canonical_id,
                    text=pair.text,
                    aspect=aspect,
                    polarity=pair.polarity,
                    domain=pair.domain,
                    split=target_split,
                    terms=pair.terms,
                )
            )

    counts = collections.Counter(pair.split for pair in kept)
    reviews_per_split = {
        split: len({pair.review_id for pair in kept if pair.split == split})
        for split in counts
    }

    report = SplitReport(
        counts=dict(counts),
        review_counts=reviews_per_split,
        cross_split_duplicates_resolved=cross_split_resolved,
        duplicate_groups=duplicate_groups,
        rows_dropped_as_duplicates=rows_dropped,
    )
    return kept, report


def assert_no_leakage(pairs: list[AspectPair]) -> None:
    """Raise if any review id or review text appears in more than one split.

    Called after splitting. This is an assertion, not a repair step: if it ever
    fires, the pipeline above it is wrong and the metrics would be invalid.
    """
    splits_by_id: dict[str, set[str]] = collections.defaultdict(set)
    splits_by_text: dict[str, set[str]] = collections.defaultdict(set)

    for pair in pairs:
        splits_by_id[pair.review_id].add(pair.split)
        splits_by_text[normalise_for_dedup(pair.text)].add(pair.split)

    leaked_ids = {key: sorted(value) for key, value in splits_by_id.items() if len(value) > 1}
    leaked_text = {
        key: sorted(value) for key, value in splits_by_text.items() if len(value) > 1
    }

    if leaked_ids or leaked_text:
        lines = ["Data leakage detected across splits."]
        if leaked_ids:
            lines.append(f"  {len(leaked_ids)} review_id(s) span splits, e.g.:")
            for key, value in list(leaked_ids.items())[:5]:
                lines.append(f"    {key} -> {value}")
        if leaked_text:
            lines.append(f"  {len(leaked_text)} review text(s) span splits, e.g.:")
            for key, value in list(leaked_text.items())[:5]:
                lines.append(f"    {key[:70]!r} -> {value}")
        raise AssertionError("\n".join(lines))
