"""Tests for evidence selection.

Evidence is what makes a score checkable rather than something a reader has to
trust, so the property that matters is **balance**: praise and criticism must
both survive selection. A selector that quoted only the best sentences would
still produce plausible-looking output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.catalog.evidence import SEPARATOR, select_evidence

LONG_POSITIVE = "The battery easily lasts a day and a half of heavy use."
LONG_NEGATIVE = "Photos come out grainy whenever the light is low indoors."


def rows(*entries) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_key": key,
                "aspect": aspect,
                "polarity": polarity,
                "score": score,
                "evidence": text,
            }
            for key, aspect, polarity, score, text in entries
        ]
    )


def test_selects_evidence_per_phone():
    result = select_evidence(
        rows(("p1", "battery", "positive", 9.4, LONG_POSITIVE)),
    )
    assert list(result) == ["p1"]
    assert result["p1"][0]["sentence"] == LONG_POSITIVE


def test_keeps_criticism_alongside_praise():
    """The property the whole module exists for."""
    result = select_evidence(
        rows(
            ("p1", "battery", "positive", 9.4, LONG_POSITIVE),
            ("p1", "battery", "positive", 9.1, "Charges from empty in under an hour."),
            ("p1", "battery", "negative", 1.6, "It barely survives a morning of light use."),
        )
    )
    polarities = {item["polarity"] for item in result["p1"]}
    assert polarities == {"positive", "negative"}


def test_takes_the_most_extreme_sentence_in_each_direction():
    result = select_evidence(
        rows(
            ("p1", "battery", "positive", 9.9, "Outstanding battery, lasts two whole days."),
            ("p1", "battery", "positive", 6.2, "Battery is acceptable for the money."),
            ("p1", "battery", "negative", 1.2, "The battery died within a month of purchase."),
            ("p1", "battery", "negative", 4.4, "Battery drains faster than I expected."),
        )
    )
    sentences = {item["sentence"] for item in result["p1"]}
    assert "Outstanding battery, lasts two whole days." in sentences
    assert "The battery died within a month of purchase." in sentences


def test_only_the_first_sentence_of_a_review_is_used():
    """One verbose reviewer must not dominate a phone page."""
    joined = SEPARATOR.join([LONG_POSITIVE, "And another thing about the battery entirely."])
    result = select_evidence(rows(("p1", "battery", "positive", 9.0, joined)))
    assert result["p1"][0]["sentence"] == LONG_POSITIVE


def test_the_separator_is_split_literally_not_as_a_regex():
    """The regression this pins.

    pandas treats a multi-character split pattern as a regular expression, so
    " || " parsed as alternation over spaces and the empty string, split
    between every character, and left every sentence empty -- silently, with no
    error and no evidence on any page.
    """
    result = select_evidence(rows(("p1", "battery", "positive", 9.0, LONG_POSITIVE)))
    assert result["p1"][0]["sentence"] == LONG_POSITIVE
    assert result["p1"][0]["sentence"] != ""


def test_short_sentences_are_dropped():
    """"Good." carries no reasoning worth showing."""
    result = select_evidence(rows(("p1", "battery", "positive", 9.0, "Good.")))
    assert result == {}


def test_aspects_outside_the_five_axes_are_ignored():
    result = select_evidence(
        rows(
            ("p1", "delivery", "positive", 9.0, "Arrived two days earlier than promised."),
            ("p1", "battery", "positive", 9.0, LONG_POSITIVE),
        )
    )
    assert {item["aspect"] for item in result["p1"]} == {"battery"}


def test_evidence_is_ordered_by_axis_then_score():
    result = select_evidence(
        rows(
            ("p1", "camera", "negative", 2.0, LONG_NEGATIVE),
            ("p1", "battery", "positive", 9.0, LONG_POSITIVE),
        )
    )
    # Battery precedes camera in slider order.
    assert [item["aspect"] for item in result["p1"]] == ["battery", "camera"]


def test_duplicate_sentences_are_not_repeated():
    result = select_evidence(
        rows(
            ("p1", "battery", "positive", 9.0, LONG_POSITIVE),
            ("p1", "battery", "positive", 9.0, LONG_POSITIVE),
        )
    )
    assert len(result["p1"]) == 1


def test_missing_evidence_column_values_are_skipped():
    frame = rows(("p1", "battery", "positive", 9.0, LONG_POSITIVE))
    frame.loc[len(frame)] = ["p1", "camera", "negative", 2.0, None]
    result = select_evidence(frame)
    assert {item["aspect"] for item in result["p1"]} == {"battery"}


def test_empty_input_returns_an_empty_mapping():
    empty = pd.DataFrame(columns=["model_key", "aspect", "polarity", "score", "evidence"])
    assert select_evidence(empty) == {}


def test_selection_is_deterministic():
    frame = rows(
        ("p1", "battery", "positive", 9.4, LONG_POSITIVE),
        ("p1", "camera", "negative", 2.1, LONG_NEGATIVE),
    )
    assert select_evidence(frame) == select_evidence(frame)
