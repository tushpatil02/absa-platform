"""Exploratory analysis of the processed ABSA datasets.

Logic lives here rather than in the notebook so that the same code runs from
``scripts/run_eda.py``, from ``notebooks/01_eda.ipynb`` in Colab, and from the
test suite. Notebooks orchestrate; they do not hold logic.

Charts follow one palette, chosen by the job each encoding does:

* **Magnitude** (counts, lengths) -> one sequential blue, single series, no legend.
* **Polarity** (negative/neutral/positive) -> diverging red <-> blue with a neutral
  grey midpoint. Never red/green: that is the one pair colour-blind readers
  cannot separate, and polarity is exactly where it would matter.

The palette was checked numerically (OKLab dE, Vienot dichromat simulation, WCAG
contrast) rather than by eye:

===========================  =========  =======  =======
pair                         normal dE  protan   deutan
===========================  =========  =======  =======
negative vs neutral               18.7     11.3      9.2
negative vs positive              32.3     21.4     27.3
neutral vs positive               17.8     16.0     18.3
===========================  =========  =======  =======

All clear the >=15 normal-vision floor and the >=8 CVD target; all three sit above
3:1 contrast on the chart surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in Colab or CI.

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Palette. Values are the validated defaults; swap them for a brand palette and
# re-run the checks in the module docstring.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQUENTIAL = "#2a78d6"  # single-series magnitude
POLARITY_COLOURS = {
    "negative": "#e34948",  # diverging warm pole
    "neutral": "#898781",   # neutral midpoint
    "positive": "#2a78d6",  # diverging cool pole
}
POLARITY_ORDER = ("negative", "neutral", "positive")

FONT_STACK = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def _style_axes(ax, *, xgrid: bool = True) -> None:
    """Recessive chrome: hairline grid, no box, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)
    if xgrid:
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    for label in ax.get_yticklabels():
        label.set_color(INK_SECONDARY)


def _new_figure(width: float, height: float):
    figure, ax = plt.subplots(figsize=(width, height), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    return figure, ax


@dataclass
class DatasetStats:
    """Headline numbers used by both the charts and the written docs."""

    n_pairs: int
    n_reviews: int
    n_aspects: int
    polarity_counts: dict[str, int]
    aspect_counts: dict[str, int]
    split_pairs: dict[str, int]
    split_reviews: dict[str, int]
    domain_counts: dict[str, int]
    mean_words: float
    median_words: int
    p90_words: int
    p99_words: int
    max_words: int
    mean_aspects_per_review: float
    max_aspects_per_review: int

    def as_markdown(self) -> str:
        total = self.n_pairs
        rows = [
            f"| Reviews | {self.n_reviews:,} |",
            f"| (review, aspect) pairs | {total:,} |",
            f"| Aspects | {self.n_aspects} |",
            f"| Mean aspects per review | {self.mean_aspects_per_review:.2f} |",
            f"| Max aspects on one review | {self.max_aspects_per_review} |",
            f"| Review length (median words) | {self.median_words} |",
            f"| Review length (p90 / p99 / max) | {self.p90_words} / {self.p99_words} / {self.max_words} |",
        ]
        for name in POLARITY_ORDER:
            count = self.polarity_counts.get(name, 0)
            rows.append(f"| {name.capitalize()} pairs | {count:,} ({100 * count / total:.1f}%) |")
        return "| Metric | Value |\n| --- | --- |\n" + "\n".join(rows)


def load_processed(processed_dir: Path) -> pd.DataFrame:
    """Load all ASC splits into one frame with a ``split`` column."""
    frames = []
    for split in ("train", "dev", "test"):
        path = processed_dir / f"asc_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/build_dataset.py first."
            )
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def compute_stats(asc: pd.DataFrame) -> DatasetStats:
    """Compute every headline number in one pass."""
    words = asc.drop_duplicates("review_id")["text"].str.split().str.len()
    per_review = asc.groupby("review_id")["aspect"].nunique()

    return DatasetStats(
        n_pairs=len(asc),
        n_reviews=asc["review_id"].nunique(),
        n_aspects=asc["aspect"].nunique(),
        polarity_counts=asc["polarity"].value_counts().to_dict(),
        aspect_counts=asc["aspect"].value_counts().to_dict(),
        split_pairs=asc["split"].value_counts().to_dict(),
        split_reviews=asc.groupby("split")["review_id"].nunique().to_dict(),
        domain_counts=asc["domain"].value_counts().to_dict(),
        mean_words=float(words.mean()),
        median_words=int(words.median()),
        p90_words=int(words.quantile(0.90)),
        p99_words=int(words.quantile(0.99)),
        max_words=int(words.max()),
        mean_aspects_per_review=float(per_review.mean()),
        max_aspects_per_review=int(per_review.max()),
    )


def plot_aspect_distribution(asc: pd.DataFrame, out_path: Path) -> Path:
    """Horizontal bars: how many pairs each aspect has.

    Single series, so one sequential hue and no legend -- the title names it.
    Every bar is directly labelled because there are only twelve and the exact
    counts are the point.
    """
    counts = asc["aspect"].value_counts().sort_values()

    figure, ax = _new_figure(8, 5.2)
    positions = range(len(counts))
    ax.barh(list(positions), counts.values, color=SEQUENTIAL, height=0.68, zorder=2)

    ax.set_yticks(list(positions))
    ax.set_yticklabels([name.replace("_", " ") for name in counts.index])
    _style_axes(ax)

    span = counts.max()
    for y, value in zip(positions, counts.values):
        ax.text(
            value + span * 0.012, y, f"{value:,}",
            va="center", ha="left", fontsize=9, color=INK_SECONDARY,
        )
    ax.set_xlim(0, span * 1.10)
    ax.set_xlabel("(review, aspect) pairs", fontsize=9, color=INK_MUTED)
    ax.set_title(
        "Pairs per aspect", fontsize=12, color=INK_PRIMARY,
        loc="left", pad=12, fontweight="bold",
    )

    figure.tight_layout()
    figure.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_polarity_by_aspect(asc: pd.DataFrame, out_path: Path) -> Path:
    """Stacked 100% bars: polarity mix per aspect, sorted by share negative.

    Diverging encoding, ordered negative -> neutral -> positive so the bar reads
    left-to-right as a sentiment axis. Sorting by share-negative is what makes
    the chart answer the actual question ("which aspects do customers complain
    about?") instead of just listing aspects.
    """
    table = (
        asc.groupby(["aspect", "polarity"]).size().unstack(fill_value=0)
        .reindex(columns=list(POLARITY_ORDER), fill_value=0)
    )
    shares = table.div(table.sum(axis=1), axis=0) * 100
    shares = shares.sort_values("negative")

    figure, ax = _new_figure(8.6, 5.2)
    positions = range(len(shares))
    left = pd.Series(0.0, index=shares.index)

    for polarity in POLARITY_ORDER:
        values = shares[polarity]
        ax.barh(
            list(positions), values.values, left=left.values,
            color=POLARITY_COLOURS[polarity], height=0.68, zorder=2,
            # A hairline in the surface colour gives the 2px gap between
            # segments, so adjacent fills never bleed into one another.
            edgecolor=SURFACE, linewidth=1.4,
        )
        left = left + values

    ax.set_yticks(list(positions))
    ax.set_yticklabels([name.replace("_", " ") for name in shares.index])
    _style_axes(ax)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of pairs (%)", fontsize=9, color=INK_MUTED)
    ax.set_title(
        "Polarity mix by aspect", fontsize=12, color=INK_PRIMARY,
        loc="left", pad=28, fontweight="bold",
    )

    # Direct label the negative share -- the number the chart is sorted by.
    for y, aspect in zip(positions, shares.index):
        value = shares.loc[aspect, "negative"]
        if value > 7:
            ax.text(
                value / 2, y, f"{value:.0f}%",
                va="center", ha="center", fontsize=8.5, color="#ffffff",
            )

    # Legend is mandatory at >=2 series so identity is never colour-alone.
    ax.legend(
        handles=[
            Patch(facecolor=POLARITY_COLOURS[name], label=name.capitalize())
            for name in POLARITY_ORDER
        ],
        loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.1,
    )

    figure.tight_layout()
    figure.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_review_length(asc: pd.DataFrame, out_path: Path, *, token_budget: int = 128) -> Path:
    """Histogram of review length, with the chosen max_length marked.

    The annotation is the point of the chart: it justifies ``max_length=128``
    by showing how little is truncated.
    """
    words = asc.drop_duplicates("review_id")["text"].str.split().str.len()

    figure, ax = _new_figure(8, 4.0)
    ax.hist(words, bins=60, color=SEQUENTIAL, zorder=2)
    # Counts are read off the y axis here, so the grid runs horizontally.
    _style_axes(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    p99 = int(words.quantile(0.99))
    covered = 100 * (words <= token_budget).mean()
    ax.axvline(token_budget, color=INK_MUTED, linewidth=1.4, linestyle="--", zorder=3)
    ax.text(
        token_budget + 3, ax.get_ylim()[1] * 0.82,
        f"max_length={token_budget}\ncovers {covered:.1f}% of reviews",
        fontsize=8.5, color=INK_SECONDARY, va="top",
    )

    ax.set_xlabel("words per review", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("reviews", fontsize=9, color=INK_MUTED)
    ax.set_title(
        f"Review length  (median {int(words.median())}, p99 {p99} words)",
        fontsize=12, color=INK_PRIMARY, loc="left", pad=12, fontweight="bold",
    )

    figure.tight_layout()
    figure.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_aspects_per_review(asc: pd.DataFrame, out_path: Path) -> Path:
    """How many distinct aspects each review mentions.

    This is the chart that justifies grouped splitting: any bar to the right of
    1 is a review that produces several rows, which row-level splitting would
    scatter across train and test.
    """
    per_review = asc.groupby("review_id")["aspect"].nunique().value_counts().sort_index()

    figure, ax = _new_figure(7, 3.6)
    ax.bar(
        per_review.index.astype(int), per_review.values,
        color=SEQUENTIAL, width=0.62, zorder=2,
    )
    _style_axes(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    for x, value in zip(per_review.index.astype(int), per_review.values):
        ax.text(
            x, value + per_review.max() * 0.02, f"{value:,}",
            ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY,
        )

    ax.set_xticks(list(per_review.index.astype(int)))
    ax.set_xlabel("distinct aspects mentioned", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("reviews", fontsize=9, color=INK_MUTED)
    ax.set_title(
        "Aspects per review", fontsize=12, color=INK_PRIMARY,
        loc="left", pad=12, fontweight="bold",
    )

    figure.tight_layout()
    figure.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out_path


def render_all(asc: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Render every EDA figure. Returns the paths written."""
    plt.rcParams["font.family"] = FONT_STACK
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_aspect_distribution(asc, figures_dir / "aspect_distribution.png"),
        plot_polarity_by_aspect(asc, figures_dir / "polarity_by_aspect.png"),
        plot_review_length(asc, figures_dir / "review_length.png"),
        plot_aspects_per_review(asc, figures_dir / "aspects_per_review.png"),
    ]
