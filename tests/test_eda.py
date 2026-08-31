"""Tests for the EDA module.

These use a small synthetic frame rather than the real dataset, so the suite
runs without `data/processed/` present (the raw data is not committed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.eda import (
    POLARITY_COLOURS,
    POLARITY_ORDER,
    compute_stats,
    load_processed,
    render_all,
)


@pytest.fixture
def asc() -> pd.DataFrame:
    """Three reviews: one with two aspects, two with one each."""
    return pd.DataFrame(
        [
            ("r1", "The camera is great but the battery dies fast", "camera", "positive", "train"),
            ("r1", "The camera is great but the battery dies fast", "battery", "negative", "train"),
            ("r2", "Cheap and fine", "price", "positive", "train"),
            ("r3", "Screen is okay I guess", "display", "neutral", "test"),
        ],
        columns=["review_id", "text", "aspect", "polarity", "split"],
    ).assign(domain="phone")


def test_compute_stats_counts_reviews_not_rows(asc):
    """n_reviews must count source reviews, not (review, aspect) pairs."""
    stats = compute_stats(asc)
    assert stats.n_pairs == 4
    assert stats.n_reviews == 3
    assert stats.n_aspects == 4


def test_compute_stats_review_length_deduplicates(asc):
    """A review with two aspects must not be counted twice in the length stats."""
    stats = compute_stats(asc)
    # Lengths of the three distinct reviews: 9, 3, 5 words -> median 5.
    assert stats.median_words == 5
    assert stats.max_words == 9


def test_compute_stats_aspects_per_review(asc):
    stats = compute_stats(asc)
    assert stats.max_aspects_per_review == 2
    assert stats.mean_aspects_per_review == pytest.approx(4 / 3)


def test_compute_stats_splits(asc):
    stats = compute_stats(asc)
    assert stats.split_pairs == {"train": 3, "test": 1}
    assert stats.split_reviews == {"train": 2, "test": 1}


def test_polarity_order_matches_colour_keys():
    """Order and colour map must stay in sync or the legend mislabels segments."""
    assert set(POLARITY_ORDER) == set(POLARITY_COLOURS)
    assert POLARITY_ORDER == ("negative", "neutral", "positive")


def test_render_all_writes_every_figure(asc, tmp_path):
    written = render_all(asc, tmp_path)
    assert len(written) == 4
    for path in written:
        assert path.exists(), path
        assert path.stat().st_size > 1000, f"{path.name} looks empty"


def test_load_processed_reports_missing_data_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        load_processed(tmp_path)


def test_as_markdown_renders_a_table(asc):
    markdown = compute_stats(asc).as_markdown()
    assert markdown.startswith("| Metric | Value |")
    assert "Reviews | 3" in markdown
