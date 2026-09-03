"""Is the per-phone signal real, or is it noise with a confident face?

This is a gate, not a report. Everything downstream -- the sliders, the match
percentages, the "Battery 8.2" on a phone page -- assumes that a phone's aspect
score measures something about the phone. That assumption is testable, and
until it is tested the rest is decoration.

Two questions, both of which can fail
-------------------------------------
**1. Split-half reliability.** Divide a phone's reviews at random into two
halves, build the profile twice, and correlate the halves across phones. If a
score reflects the phone, the halves agree. If it reflects which reviews landed
in which half, they do not. Reported with the Spearman-Brown correction, which
estimates reliability at the full review count from the half-length measurement
(each half has half the data, so the raw correlation understates it).

**2. Beat the stars.** Amazon already ships a per-product number: the average
star rating. If per-aspect scores track stars almost perfectly, the entire ABSA
pipeline reduces to an expensive re-derivation of something already in the CSV,
and the honest thing is to say so. The recommender is only worth its complexity
if the aspects separate phones that stars do not.

Why this exists in this form
----------------------------
An earlier design showed shrunk scores next to percentile ranks, to be
"transparent" about uncertainty. Simulated against a null where every phone is
genuinely identical, that presentation produced a confident 1.83-point spread
and a full 0th-to-100th percentile range out of pure noise -- dividing shrunk
scores by their own compressed standard deviation *is* stretching them. So the
question "is there signal at all" has to be answered before, not after, any
presentation choice. :func:`null_spread` reproduces that simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class AspectReliability:
    """Split-half reliability for one aspect."""

    aspect: str
    n_phones: int
    raw_spearman: float
    spearman_brown: float
    pearson: float
    mean_abs_gap: float

    @property
    def verdict(self) -> str:
        """A plain reading of the corrected correlation.

        The thresholds are the conventional ones for psychometric reliability.
        They are a guide to interpretation, not a pass mark invented to make
        this particular result look acceptable.
        """
        value = self.spearman_brown
        if value >= 0.80:
            return "strong"
        if value >= 0.60:
            return "usable"
        if value >= 0.40:
            return "weak"
        return "NOISE"

    def summary(self) -> str:
        return (
            f"  {self.aspect:<14}n={self.n_phones:>4}  "
            f"raw={self.raw_spearman:+.3f}  "
            f"corrected={self.spearman_brown:+.3f}  "
            f"gap={self.mean_abs_gap:.2f}  {self.verdict}"
        )


def _phone_aspect_means(
    frame: pd.DataFrame, aspect: str, min_mentions: int
) -> pd.core.groupby.DataFrameGroupBy:
    subset = frame[frame["aspect"] == aspect]
    counts = subset.groupby("model_key").size()
    keep = counts[counts >= min_mentions].index
    return subset[subset["model_key"].isin(keep)]


def split_half_reliability(
    review_aspects: pd.DataFrame,
    aspect: str,
    *,
    min_mentions: int = 10,
    repeats: int = 25,
    seed: int = 0,
) -> AspectReliability:
    """Correlate two independent halves of each phone's reviews.

    Args:
        review_aspects: Per-review rows with ``model_key``, ``aspect``, ``score``.
        aspect: Which aspect to test.
        min_mentions: Phones below this are excluded -- a two-review phone
            cannot produce two meaningful halves, and including it would
            depress the correlation for a reason unrelated to signal.
        repeats: Random splits to average over. One split is itself noisy.
        seed: Fixed, so the gate cannot be re-rolled until it passes.
    """
    subset = _phone_aspect_means(review_aspects, aspect, min_mentions)
    if subset.empty:
        return AspectReliability(aspect, 0, float("nan"), float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    groups = {key: group["score"].to_numpy() for key, group in subset.groupby("model_key")}

    spearmans, pearsons, gaps = [], [], []
    for _ in range(repeats):
        first, second = [], []
        for scores in groups.values():
            shuffled = rng.permutation(scores)
            midpoint = len(shuffled) // 2
            first.append(shuffled[:midpoint].mean())
            second.append(shuffled[midpoint:].mean())

        first_array, second_array = np.array(first), np.array(second)
        # Constant input makes a correlation undefined; skip rather than emit nan.
        if first_array.std() == 0 or second_array.std() == 0:
            continue
        spearmans.append(stats.spearmanr(first_array, second_array).statistic)
        pearsons.append(stats.pearsonr(first_array, second_array).statistic)
        gaps.append(np.abs(first_array - second_array).mean())

    if not spearmans:
        return AspectReliability(aspect, len(groups), float("nan"), float("nan"), float("nan"), float("nan"))

    raw = float(np.mean(spearmans))
    # Spearman-Brown: each half holds half the reviews, so the observed
    # correlation understates reliability at full length.
    corrected = 2 * raw / (1 + raw) if raw > -1 else float("nan")

    return AspectReliability(
        aspect=aspect,
        n_phones=len(groups),
        raw_spearman=raw,
        spearman_brown=float(corrected),
        pearson=float(np.mean(pearsons)),
        mean_abs_gap=float(np.mean(gaps)),
    )


@dataclass(frozen=True)
class StarBaseline:
    """How much an aspect adds over the average star rating."""

    aspect: str
    n_phones: int
    spearman_with_stars: float
    r_squared: float

    @property
    def verdict(self) -> str:
        if self.r_squared >= 0.90:
            return "REDUNDANT -- stars already say this"
        if self.r_squared >= 0.70:
            return "largely redundant"
        if self.r_squared >= 0.40:
            return "correlated but distinct"
        return "independent of stars"

    def summary(self) -> str:
        return (
            f"  {self.aspect:<14}n={self.n_phones:>4}  "
            f"rho={self.spearman_with_stars:+.3f}  "
            f"R2={self.r_squared:.3f}  {self.verdict}"
        )


def star_baseline(
    review_aspects: pd.DataFrame,
    phones: pd.DataFrame,
    aspect: str,
    *,
    min_mentions: int = 10,
) -> StarBaseline:
    """Compare per-phone aspect scores against average star rating.

    A high correlation is not a bug -- good phones do get good stars *and* good
    battery reviews. It is a warning: the closer this gets to 1, the less the
    aspect adds over a column that was already in the dataset.
    """
    subset = _phone_aspect_means(review_aspects, aspect, min_mentions)
    if subset.empty:
        return StarBaseline(aspect, 0, float("nan"), float("nan"))

    means = subset.groupby("model_key")["score"].mean()
    stars = phones.set_index("model_key")["avg_rating"]
    joined = pd.concat([means.rename("score"), stars.rename("stars")], axis=1).dropna()
    if len(joined) < 3 or joined["score"].std() == 0:
        return StarBaseline(aspect, len(joined), float("nan"), float("nan"))

    rho = stats.spearmanr(joined["score"], joined["stars"]).statistic
    r = stats.pearsonr(joined["score"], joined["stars"]).statistic
    return StarBaseline(aspect, len(joined), float(rho), float(r**2))


@dataclass(frozen=True)
class NullComparison:
    """An aspect's observed spread against a null matched to its own noise."""

    aspect: str
    n_phones: int
    reviews_per_phone: int
    observed_std: float
    observed_range: float
    null_std: float
    null_range: float

    @property
    def ratio(self) -> float:
        """How many times the null's spread the observed spread is."""
        return self.observed_std / self.null_std if self.null_std else float("inf")

    @property
    def verdict(self) -> str:
        if self.ratio >= 3.0:
            return "well above null"
        if self.ratio >= 1.5:
            return "above null"
        return "NOT ABOVE NULL"

    def summary(self) -> str:
        return (
            f"  {self.aspect:<14}n={self.n_phones:>4}  "
            f"observed sd={self.observed_std:.2f} range={self.observed_range:.2f}  "
            f"null sd={self.null_std:.2f} range={self.null_range:.2f}  "
            f"{self.ratio:.1f}x  {self.verdict}"
        )


def compare_to_null(
    review_aspects: pd.DataFrame,
    aspect: str,
    *,
    min_mentions: int = 10,
    seed: int = 0,
) -> NullComparison:
    """Compare an aspect's observed spread to a null matched to that aspect.

    The null must use *this* aspect's within-phone variance and *this* aspect's
    review counts. A null drawn with some fixed standard deviation is not a
    baseline for anything -- it answers a question about a different corpus, and
    comparing an observed spread to it is worse than not comparing at all,
    because it looks like a check.
    """
    subset = _phone_aspect_means(review_aspects, aspect, min_mentions)
    if subset.empty:
        return NullComparison(aspect, 0, 0, *(float("nan"),) * 4)

    grouped = subset.groupby("model_key")["score"]
    means, counts = grouped.mean(), grouped.size()
    # The noise a single review carries about one phone.
    within_std = float(np.sqrt(subset.groupby("model_key")["score"].var(ddof=1).mean()))
    per_phone = int(counts.median())

    null = null_spread(
        n_phones=len(means),
        reviews_per_phone=max(per_phone, 1),
        score_std=within_std,
        seed=seed,
    )
    return NullComparison(
        aspect=aspect,
        n_phones=len(means),
        reviews_per_phone=per_phone,
        observed_std=float(means.std()),
        observed_range=float(means.max() - means.min()),
        null_std=null["std"],
        null_range=null["range"],
    )


def null_spread(
    n_phones: int = 200,
    reviews_per_phone: int = 100,
    *,
    score_std: float = 3.0,
    seed: int = 0,
) -> dict[str, float]:
    """Spread produced when every phone is genuinely identical.

    Draws every review score from one distribution, so any spread in the phone
    means is sampling noise. Whatever this returns is the amount of apparent
    discrimination a null world gives for free -- a real result has to clear it.

    This is the simulation that killed the percentile-display idea: it returned
    a 1.83-point range and a full 0-100 percentile spread from noise alone.
    """
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.normal(5.5, score_std, size=reviews_per_phone).mean() for _ in range(n_phones)]
    )
    return {
        "range": float(means.max() - means.min()),
        "std": float(means.std()),
        "p90_minus_p10": float(np.percentile(means, 90) - np.percentile(means, 10)),
    }
