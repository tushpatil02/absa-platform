"""Parse the raw M-ABSA annotation format.

M-ABSA stores one review per line as::

    <sentence>####[['camera quality', 'Camera#General', 'negative'], ...]

The right-hand side is a Python literal list of ``[aspect_term, category,
polarity]`` triplets. It is parsed with :func:`ast.literal_eval`, never
``eval``, so a malformed or hostile line can't execute anything.

Lines that cannot be parsed are collected into :class:`ParseReport` rather than
raised, so one bad line never kills a run of several thousand. The report is
printed by ``scripts/build_dataset.py`` and asserted on in the tests.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

SEPARATOR = "####"
VALID_POLARITIES = frozenset({"positive", "negative", "neutral"})


@dataclass(frozen=True)
class Triplet:
    """A single annotated opinion: which term, which category, what polarity."""

    term: str
    category: str
    polarity: str


@dataclass(frozen=True)
class RawReview:
    """One raw line: the review text plus every triplet annotated on it."""

    text: str
    triplets: tuple[Triplet, ...]
    source_domain: str
    source_split: str
    line_number: int


@dataclass
class ParseReport:
    """Counts of everything that was skipped, and why."""

    total_lines: int = 0
    parsed: int = 0
    blank_lines: int = 0
    missing_separator: list[int] = field(default_factory=list)
    malformed_payload: list[int] = field(default_factory=list)
    empty_text: list[int] = field(default_factory=list)
    bad_triplet_shape: list[int] = field(default_factory=list)
    invalid_polarity: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return (
            len(self.missing_separator)
            + len(self.malformed_payload)
            + len(self.empty_text)
            + len(self.bad_triplet_shape)
        )

    def summary(self) -> str:
        parts = [
            f"lines={self.total_lines}",
            f"parsed={self.parsed}",
            f"blank={self.blank_lines}",
            f"skipped={self.skipped}",
        ]
        if self.missing_separator:
            parts.append(f"no_separator={len(self.missing_separator)}")
        if self.malformed_payload:
            parts.append(f"malformed={len(self.malformed_payload)}")
        if self.empty_text:
            parts.append(f"empty_text={len(self.empty_text)}")
        if self.bad_triplet_shape:
            parts.append(f"bad_shape={len(self.bad_triplet_shape)}")
        if self.invalid_polarity:
            parts.append(f"invalid_polarity={len(self.invalid_polarity)}")
        return "  ".join(parts)


def parse_line(line: str) -> tuple[str, list[Triplet]] | None:
    """Parse one ``text####[[...]]`` line, or return ``None`` if unusable."""
    if SEPARATOR not in line:
        return None

    text, _, payload = line.partition(SEPARATOR)
    text = text.strip()
    if not text:
        return None

    try:
        raw = ast.literal_eval(payload.strip())
    except (ValueError, SyntaxError):
        return None

    if not isinstance(raw, list):
        return None

    triplets: list[Triplet] = []
    for entry in raw:
        # Expected shape is exactly [term, category, polarity], all strings.
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            return None
        term, category, polarity = (str(x).strip() for x in entry)
        triplets.append(Triplet(term=term, category=category, polarity=polarity.lower()))

    return text, triplets


def parse_file(path: Path, domain: str, split: str, report: ParseReport) -> list[RawReview]:
    """Parse one ``{domain}/{split}.txt`` file, accumulating stats into `report`.

    UTF-8 is read with ``errors="replace"`` so a stray invalid byte degrades to
    U+FFFD (which :mod:`ml.preprocessing.clean` then strips) instead of aborting
    the whole build.
    """
    reviews: list[RawReview] = []
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        report.total_lines += 1

        if not line.strip():
            report.blank_lines += 1
            continue

        if SEPARATOR not in line:
            report.missing_separator.append(line_number)
            continue

        parsed = parse_line(line)
        if parsed is None:
            # Distinguish the failure modes so the report is actionable.
            text_part = line.partition(SEPARATOR)[0].strip()
            if not text_part:
                report.empty_text.append(line_number)
            else:
                try:
                    value = ast.literal_eval(line.partition(SEPARATOR)[2].strip())
                    if isinstance(value, list):
                        report.bad_triplet_shape.append(line_number)
                    else:
                        report.malformed_payload.append(line_number)
                except (ValueError, SyntaxError):
                    report.malformed_payload.append(line_number)
            continue

        text, triplets = parsed

        kept: list[Triplet] = []
        for triplet in triplets:
            if triplet.polarity not in VALID_POLARITIES:
                report.invalid_polarity.append(
                    f"{domain}/{split}:{line_number} '{triplet.polarity}'"
                )
                continue
            kept.append(triplet)

        report.parsed += 1
        reviews.append(
            RawReview(
                text=text,
                triplets=tuple(kept),
                source_domain=domain,
                source_split=split,
                line_number=line_number,
            )
        )

    return reviews
