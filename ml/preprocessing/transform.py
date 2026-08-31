"""Map raw M-ABSA annotations onto the unified 12-aspect taxonomy.

Two transformations happen here, and both are lossy on purpose.

**Category collapse.** M-ABSA's ``phone`` domain uses 86 fine-grained
categories and ``laptop`` uses 108, drawn from two *completely disjoint* label
schemes (phone is an e-commerce taxonomy, laptop is SemEval-2014 style -- they
share zero coarse labels). 41 of the phone categories have fewer than 30
examples. Both schemes are mapped onto 12 shopper-recognisable aspects by
``ml/config/aspect_taxonomy.yaml``.

**Polarity aggregation.** After collapsing, one review can carry several
triplets for the same aspect ("the screen is bright" + "the screen scratches").
They are reduced to one ``(review, aspect)`` row by majority vote, with ties
broken toward the more informative label -- see :func:`resolve_polarity`.

The output feeds two datasets:

* **ACD** (aspect detection) -- one row per review, multi-label aspect targets.
* **ASC** (sentiment) -- one row per ``(review, aspect)`` pair, 3-class target.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from ml.preprocessing.parse import RawReview

# Tie-break order, applied ONLY among labels that were actually annotated.
#
# This deliberately never invents a label. A review with one positive and one
# negative triplet about the battery resolves to "negative", not "neutral" --
# no annotator called it neutral, and manufacturing that label would be exactly
# the kind of fabrication that makes a headline metric meaningless.
#
# Given a genuine tie, the order below prefers the more informative and scarcer
# label: neutral (if annotated) > negative > positive.
TIE_BREAK_PRIORITY = ("neutral", "negative", "positive")


@dataclass(frozen=True)
class Taxonomy:
    """The aspect taxonomy loaded from YAML."""

    aspect_ids: tuple[str, ...]
    display_names: dict[str, str]
    descriptions: dict[str, str]
    phone_map: dict[str, str]
    laptop_map: dict[str, str]
    drop: frozenset[str]
    polarities: tuple[str, ...]

    def map_category(self, category: str, domain: str) -> str | None:
        """Map a raw category to an aspect id.

        Returns ``None`` when the label is on the explicit drop list. Raises for
        an unrecognised label, so a taxonomy that has drifted from the data
        fails loudly at build time instead of silently discarding rows.
        """
        coarse = category.split("#")[0].strip()
        if coarse in self.drop:
            return None

        mapping = self.phone_map if domain == "phone" else self.laptop_map
        if coarse in mapping:
            return mapping[coarse]

        raise KeyError(
            f"Unmapped category {coarse!r} (from {category!r}) in domain {domain!r}. "
            f"Add it to {domain}_map or to `drop` in aspect_taxonomy.yaml."
        )

    @property
    def polarity_to_id(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.polarities)}


def load_taxonomy(path: Path) -> Taxonomy:
    """Load and validate ``aspect_taxonomy.yaml``."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    aspects = config["aspects"]
    aspect_ids = tuple(entry["id"] for entry in aspects)
    if len(set(aspect_ids)) != len(aspect_ids):
        raise ValueError("Duplicate aspect id in aspect_taxonomy.yaml")

    phone_map = dict(config["phone_map"])
    laptop_map = dict(config["laptop_map"])

    # Every mapping target must be a declared aspect, or the dataset would gain
    # a phantom label that the model head has no slot for.
    known = set(aspect_ids)
    for name, mapping in (("phone_map", phone_map), ("laptop_map", laptop_map)):
        unknown = {target for target in mapping.values() if target not in known}
        if unknown:
            raise ValueError(f"{name} points at undeclared aspects: {sorted(unknown)}")

    return Taxonomy(
        aspect_ids=aspect_ids,
        display_names={entry["id"]: entry["display_name"] for entry in aspects},
        descriptions={entry["id"]: entry["description"] for entry in aspects},
        phone_map=phone_map,
        laptop_map=laptop_map,
        drop=frozenset(config.get("drop", [])),
        polarities=tuple(config["polarities"]),
    )


def resolve_polarity(counts: collections.Counter[str]) -> str:
    """Reduce several polarity votes for one aspect to a single label.

    Majority wins. On a tie, :data:`TIE_BREAK_PRIORITY` picks among *the tied
    labels only*, so the result is always a label some annotator actually
    assigned. Deterministic, so the dataset is reproducible across machines.

    >>> resolve_polarity(collections.Counter({"positive": 1, "negative": 1}))
    'negative'
    """
    if not counts:
        raise ValueError("resolve_polarity called with no votes")

    highest = max(counts.values())
    tied = [polarity for polarity, count in counts.items() if count == highest]
    if len(tied) == 1:
        return tied[0]

    for polarity in TIE_BREAK_PRIORITY:
        if polarity in tied:
            return polarity
    return sorted(tied)[0]  # Unreachable with valid polarities; kept for safety.


@dataclass
class TransformReport:
    """What the mapping did, for the build log and the docs."""

    reviews_in: int = 0
    triplets_in: int = 0
    triplets_dropped_by_taxonomy: int = 0
    duplicate_triplets_removed: int = 0
    pairs_out: int = 0
    reviews_without_aspects: int = 0
    ties_resolved: int = 0

    def summary(self) -> str:
        return (
            f"reviews_in={self.reviews_in}  triplets_in={self.triplets_in}  "
            f"dup_triplets_removed={self.duplicate_triplets_removed}  "
            f"dropped_by_taxonomy={self.triplets_dropped_by_taxonomy}  "
            f"pairs_out={self.pairs_out}  "
            f"reviews_without_aspects={self.reviews_without_aspects}  "
            f"ties={self.ties_resolved}"
        )


@dataclass(frozen=True)
class AspectPair:
    """One ``(review, aspect) -> polarity`` training row."""

    review_id: str
    text: str
    aspect: str
    polarity: str
    domain: str
    split: str
    terms: tuple[str, ...]


def transform(
    reviews: Iterable[RawReview],
    taxonomy: Taxonomy,
    report: TransformReport,
    *,
    cleaned_text: dict[int, str] | None = None,
) -> list[AspectPair]:
    """Turn parsed reviews into ``(review, aspect, polarity)`` rows.

    Args:
        reviews: Parsed raw reviews.
        taxonomy: Loaded taxonomy.
        report: Mutated in place with counts.
        cleaned_text: Optional ``id(review) -> cleaned text`` override, so
            cleaning can run before mapping without re-parsing.

    Returns:
        One :class:`AspectPair` per ``(review, aspect)``.
    """
    pairs: list[AspectPair] = []

    for index, review in enumerate(reviews):
        report.reviews_in += 1
        text = (cleaned_text or {}).get(id(review), review.text)

        # A stable id that survives shuffling and identifies the *source* review.
        # This is the grouping key that keeps splits leak-free.
        review_id = f"{review.source_domain}-{review.source_split}-{review.line_number}"

        votes: dict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        terms: dict[str, list[str]] = collections.defaultdict(list)

        # M-ABSA repeats identical triplets on ~3.5% of phone rows. Counting a
        # duplicate twice would let one annotation quietly outvote another.
        seen: set[tuple[str, str, str]] = set()

        for triplet in review.triplets:
            report.triplets_in += 1

            key = (triplet.term, triplet.category, triplet.polarity)
            if key in seen:
                report.duplicate_triplets_removed += 1
                continue
            seen.add(key)

            aspect = taxonomy.map_category(triplet.category, review.source_domain)
            if aspect is None:
                report.triplets_dropped_by_taxonomy += 1
                continue

            votes[aspect][triplet.polarity] += 1
            terms[aspect].append(triplet.term)

        if not votes:
            report.reviews_without_aspects += 1
            continue

        for aspect, counts in votes.items():
            if len(counts) > 1 and len(set(counts.values())) == 1:
                report.ties_resolved += 1
            pairs.append(
                AspectPair(
                    review_id=review_id,
                    text=text,
                    aspect=aspect,
                    polarity=resolve_polarity(counts),
                    domain=review.source_domain,
                    split=review.source_split,
                    terms=tuple(terms[aspect]),
                )
            )
            report.pairs_out += 1

    return pairs
