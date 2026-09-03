"""Tests for the sentence splitter.

Splitting is what makes sentence-level inference possible, and its failures are
asymmetric. An over-eager split produces a fragment with no opinion in it, which
is merely noisy. A *missed* split glues two opposite opinions into one unit and
puts the model back in the slice where it scores 0.541 -- the exact failure the
splitter exists to prevent. The merge tests below matter most for that reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.inference.sentences import split_sentences

# ---------------------------------------------------------------------------
# Boundaries the splitter must find
# ---------------------------------------------------------------------------


def test_splits_on_sentence_terminators():
    assert split_sentences("Battery lasts all day. The camera is grainy.") == [
        "Battery lasts all day.",
        "The camera is grainy.",
    ]


@pytest.mark.parametrize("terminator", [".", "!", "?", "...", "!!"])
def test_splits_on_each_terminator(terminator):
    text = f"The screen is bright{terminator} The battery is weak."
    assert len(split_sentences(text)) == 2


def test_splits_on_line_breaks_without_punctuation():
    """Bulleted pros-and-cons lists carry no terminators at all."""
    assert split_sentences("Pros:\n- great screen\n- weak battery") == [
        "Pros:",
        "great screen",
        "weak battery",
    ]


def test_closing_quotes_stay_with_their_sentence():
    parts = split_sentences('He said "it is great." The battery is weak.')
    assert len(parts) == 2
    assert parts[0].endswith('"')


# ---------------------------------------------------------------------------
# Boundaries the splitter must NOT find
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "It has a 6.1 inch screen which is fine.",
        "It costs $299.99 delivered.",
        "Battery is rated at 4.5 out of 5.",
    ],
)
def test_decimals_are_not_boundaries(text):
    """Requiring whitespace after the period protects prices and screen sizes.

    This corpus is full of both, so a decimal split would fragment most of it.
    """
    assert split_sentences(text) == [text]


@pytest.mark.parametrize("abbreviation", ["e.g.", "i.e.", "etc.", "vs.", "approx."])
def test_abbreviations_are_not_boundaries(abbreviation):
    text = f"Good value, {abbreviation} the display is superb and bright."
    assert split_sentences(text) == [text]


def test_hyphen_ranges_are_not_boundaries():
    """A bullet counts only at the start of a line.

    Treating " - " as a boundary anywhere cut "5 - 6 hours" in half, which
    destroys exactly the battery-life claims the recommender depends on.
    """
    text = "battery lasts 5 - 6 hours which is great"
    assert split_sentences(text) == [text]


# ---------------------------------------------------------------------------
# Merging -- the asymmetric-risk cases
# ---------------------------------------------------------------------------


def test_short_fragments_merge_within_a_line():
    """"Wow." carries no aspect, so it should not become its own unit."""
    assert split_sentences("The screen is nice. Wow. Would buy again.") == [
        "The screen is nice. Wow.",
        "Would buy again.",
    ]


def test_short_fragments_never_merge_across_a_line_break():
    """The regression this pins.

    Merging short fragments globally glued "great screen" to "weak battery" in
    a bulleted list -- recreating the mixed-polarity input that sentence-level
    inference exists to avoid.
    """
    parts = split_sentences("Camera:\n- sharp photos\n- awful zoom")
    assert "sharp photos" in parts
    assert "awful zoom" in parts
    assert not any("sharp photos awful zoom" in part for part in parts)


def test_leading_fragment_merges_forward():
    parts = split_sentences("Ok. The battery lasts all day and charges fast.")
    assert len(parts) == 1
    assert parts[0].startswith("Ok.")


# ---------------------------------------------------------------------------
# Long unpunctuated text
# ---------------------------------------------------------------------------


def test_long_run_on_is_chunked():
    """A 140-word paragraph with no punctuation must not reach the model whole.

    The classifier truncates at 128 tokens, so an unchunked run-on silently
    discards most of the opinion rather than failing visibly.
    """
    parts = split_sentences(" ".join(["word"] * 140))
    assert len(parts) > 1
    assert all(len(part.split()) <= 60 for part in parts)


def test_chunking_prefers_comma_boundaries():
    text = ", ".join(["clause " + " ".join(["filler"] * 9) for _ in range(8)])
    parts = split_sentences(text)
    assert len(parts) > 1
    # Pieces should not begin mid-clause when commas were available.
    assert all(part.strip() for part in parts)


def test_short_text_is_left_alone():
    assert split_sentences("love it") == ["love it"]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_input_returns_no_units(text):
    assert split_sentences(text) == []


def test_non_empty_input_always_returns_at_least_one_unit():
    """Callers must never have to handle an empty list as a special case."""
    for text in ["...", "???", "a", "!!!! ????"]:
        assert split_sentences(text), f"no units for {text!r}"


def test_no_text_is_lost():
    """Every word of the input must survive into some unit.

    Splitting is a reorganisation, not a filter: dropping text here would
    silently discard opinions and the scores would still look plausible.
    """
    text = "Battery lasts all day. Camera is grainy! Worth it? Yes, at 6.1 inches."
    words = text.replace(".", " ").replace("!", " ").replace("?", " ").split()
    joined = " ".join(split_sentences(text))
    for word in words:
        assert word.strip(",") in joined


def test_splitting_is_deterministic():
    text = "Battery lasts all day. The camera is grainy. Price is fair."
    assert split_sentences(text) == split_sentences(text)
