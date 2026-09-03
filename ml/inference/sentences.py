"""Split a review into sentence-sized units for inference.

Why this exists
---------------
The sentiment model is accurate on text carrying a *single* opinion and poor on
text carrying several. Measured on the held-out test set:

===============================  ==========  ==========
slice                            mean conf.  accuracy
===============================  ==========  ==========
one polarity across all aspects  0.876       **0.875**
different polarities per aspect  0.771       **0.541**
===============================  ==========  ==========

That is not a calibration problem -- temperature scaling was fitted and rejected
because it cannot fix a gap that is conditional on the input. It is a *unit of
inference* problem. The model was trained on M-ABSA, whose rows are single
sentences, so a single sentence is the input distribution it actually learned.

Amazon reviews are not single sentences. They average around 50 words and
routinely praise one aspect while criticising another -- "battery lasts all day
but the camera is grainy" -- which puts essentially the whole corpus in the
0.541 slice, below the 0.674 majority-class baseline. Running inference per
sentence puts each input back in the 0.875 slice.

What this module does *not* do
------------------------------
It does not split on contrastive conjunctions ("great camera but poor battery"
stays one unit). That would be the natural next step, but it manufactures
fragments unlike anything in training, so it is a measured change rather than an
assumed one. Sentence boundaries keep the input distribution matched to M-ABSA.
"""

from __future__ import annotations

import re

# Words that take a trailing period without ending a sentence. Deliberately
# short: every entry is a guess about the corpus, and a wrong guess silently
# glues two real sentences together. These are the ones that actually occur in
# consumer electronics reviews.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "e.g", "i.e", "etc", "vs", "approx", "est", "dept",
        "inc", "ltd", "co", "corp", "fig", "no", "ca",
    }
)

# A terminator, any closing quotes or brackets, then whitespace.
#
# Requiring the whitespace is what protects decimals for free: "6.1 inch" and
# "$299.99" have no space after the period, so they are never candidates. That
# is load-bearing -- screen sizes and prices are everywhere in this corpus.
_CANDIDATE = re.compile(r"[.!?]+[\"')\]]*\s+")

# Trailing run of letters and periods, used to test the word before a boundary.
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")

# Line breaks are hard boundaries regardless of punctuation; reviewers
# frequently write a "Pros:" heading followed by bulleted lines that carry
# no sentence terminators at all.
_HARD_BREAK = re.compile(r"[\r\n]+")

# A bullet marker counts only at the start of a line. Treating " - " as a
# boundary anywhere would cut "battery lasts 5 - 6 hours" in half.
_LEADING_BULLET = re.compile(r"^[-*•–—]+\s*")

DEFAULT_MIN_WORDS = 3
DEFAULT_MAX_WORDS = 60


def _is_abbreviation(text_before: str) -> bool:
    """True when the candidate period belongs to a known abbreviation."""
    match = _TRAILING_WORD.search(text_before)
    if not match:
        return False
    return match.group(1).lower().rstrip(".") in ABBREVIATIONS


def _split_on_terminators(text: str) -> list[str]:
    """Split one line on sentence terminators, skipping abbreviations."""
    parts: list[str] = []
    start = 0
    for match in _CANDIDATE.finditer(text):
        if _is_abbreviation(text[: match.start()]):
            continue
        parts.append(text[start : match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def _chunk_long(sentence: str, max_words: int) -> list[str]:
    """Break an over-long unit that carries no usable punctuation.

    Reviews written as one unpunctuated paragraph would otherwise reach the
    classifier as a single 400-word input and be truncated at 128 tokens, which
    silently discards most of the opinion. Splitting on commas first keeps the
    pieces linguistically sensible; a hard word-count cut is the last resort.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    pieces: list[str] = []
    current: list[str] = []
    for clause in re.split(r"(?<=,)\s+", sentence):
        clause_words = clause.split()
        if current and len(current) + len(clause_words) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.extend(clause_words)
        while len(current) > max_words:
            pieces.append(" ".join(current[:max_words]))
            current = current[max_words:]
    if current:
        pieces.append(" ".join(current))
    return pieces


def _merge_short(sentences: list[str], min_words: int) -> list[str]:
    """Attach fragments too short to carry an opinion to a neighbour.

    "Wow." or "Would buy again." on its own gives the aspect detector nothing to
    work with, and letting it through as a unit adds a low-confidence vote to
    the aggregate. Merging preserves the text while keeping the unit useful.
    """
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(sentence.split()) < min_words:
            merged[-1] = f"{merged[-1]} {sentence}".strip()
        else:
            merged.append(sentence)

    # The first unit can still be short if nothing preceded it to absorb it.
    if len(merged) > 1 and len(merged[0].split()) < min_words:
        merged[1] = f"{merged[0]} {merged[1]}".strip()
        merged.pop(0)
    return merged


def split_sentences(
    text: str,
    *,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[str]:
    """Split ``text`` into sentence-sized inference units.

    Always returns at least one unit for non-empty input, so callers never have
    to handle an empty list as a special case.

    >>> split_sentences("Battery lasts all day. The camera is grainy.")
    ['Battery lasts all day.', 'The camera is grainy.']
    >>> split_sentences("It has a 6.1 inch screen for $299.99 which is fine.")
    ['It has a 6.1 inch screen for $299.99 which is fine.']
    """
    if not text or not text.strip():
        return []

    units: list[str] = []
    for line in _HARD_BREAK.split(text):
        line = _LEADING_BULLET.sub("", line.strip())
        if not line:
            continue
        # Merge *within* the line only. A line break is a boundary the writer
        # put there deliberately, and merging across it would glue "great
        # screen" to "weak battery" in a bulleted pros-and-cons list -- which is
        # precisely the mixed-polarity input this module exists to avoid.
        units.extend(_merge_short(_split_on_terminators(line), min_words))

    if not units:
        return [text.strip()]

    chunked: list[str] = []
    for unit in units:
        chunked.extend(_chunk_long(unit, max_words))
    return chunked
