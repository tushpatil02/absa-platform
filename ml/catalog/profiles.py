"""Turn per-review aspect scores into one profile per phone.

The hard part is not the average. It is that phones have wildly different
amounts of evidence -- the catalogue ranges from 20 reviews to 400 -- and a
phone whose camera was mentioned four times should not be allowed to top the
camera ranking on the strength of four enthusiastic sentences.

Shrinkage
---------
Each phone's aspect mean is pulled toward the catalogue mean in proportion to
how little evidence supports it::

    shrunk = (n * phone_mean + k * grand_mean) / (n + k)

``k`` is estimated from the data (empirical Bayes): it is the within-phone
variance divided by the between-phone variance, which is the point at which
the two sources of information deserve equal weight. A phone with n >> k keeps
its own mean; a phone with n << k is mostly told what the average phone looks
like, which is the honest answer when its own evidence is thin.

What is deliberately *not* done
-------------------------------
The shrunk scores are not z-scored, percentile-ranked, or otherwise rescaled to
fill the 1-10 range. Shrinkage compresses the spread on purpose -- that
compression *is* the uncertainty -- and dividing by the compressed standard
deviation puts it straight back. Simulated on a null where every phone is
identical, that manoeuvre produced a confident 1.83-point spread and a full
0th-to-100th percentile range out of nothing. See
:func:`ml.evaluation.reliability.null_spread`.

So a flat catalogue is displayed as a flat catalogue. If the phones really do
not differ on Camera, the Camera slider should struggle to separate them, and
the reliability gate is what says whether that is the case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fewer mentions than this and no score is published for that aspect.
#
# Chosen by measurement, not taste: 15 is the smallest floor at which split-half
# reliability reaches "usable" on *every* axis. Sweeping it over the full
# catalogue (Spearman-Brown corrected, phones retained in brackets):
#
#   floor   battery        camera         display        performance
#       5   0.733 [193]    0.528 [168]    0.716 [176]    0.670 [209]
#      10   0.772 [160]    0.569 [133]    0.780 [146]    0.688 [194]
#      15   0.821 [140]    0.653 [103]    0.814 [122]    0.757 [175]
#      30   0.867 [ 90]    0.710 [ 63]    0.882 [ 83]    0.749 [134]
#
# Camera is the binding constraint -- below 15 it is "weak", meaning the score
# reflects which reviews were sampled more than it reflects the phone. Going
# higher buys little and costs phones: 15 leaves 97 rankable on all five axes,
# 30 leaves 54.
#
# This changes only which scores are *published*. It does not change the model,
# the scores themselves, or how they are computed.
MIN_MENTIONS = 15


@dataclass(frozen=True)
class ShrinkageParams:
    """Empirical-Bayes parameters for one aspect."""

    aspect: str
    grand_mean: float
    within_variance: float
    between_variance: float
    k: float
    n_phones: int

    def summary(self) -> str:
        return (
            f"  {self.aspect:<14}mean={self.grand_mean:5.2f}  "
            f"within={self.within_variance:6.2f}  between={self.between_variance:6.3f}  "
            f"k={self.k:7.1f}  phones={self.n_phones}"
        )


def estimate_shrinkage(frame: pd.DataFrame, aspect: str) -> ShrinkageParams:
    """Estimate how much to trust a single phone's mean for this aspect.

    A large ``k`` means the phones barely differ relative to how noisy each
    phone's own reviews are -- so individual means should be pulled hard toward
    the average. That is a finding, not a failure: it says the aspect does not
    discriminate.
    """
    subset = frame[frame["aspect"] == aspect]
    if subset.empty:
        return ShrinkageParams(aspect, float("nan"), float("nan"), float("nan"), float("inf"), 0)

    grouped = subset.groupby("model_key")["score"]
    means = grouped.mean()
    counts = grouped.size()

    grand_mean = float(subset["score"].mean())
    # Pooled within-phone variance: how noisy reviews are about one phone.
    within = float(subset.groupby("model_key")["score"].var(ddof=1).mean())
    # Observed spread of phone means, minus the part sampling noise explains.
    observed = float(means.var(ddof=1)) if len(means) > 1 else 0.0
    expected_from_noise = within / float(counts.mean()) if counts.mean() else 0.0
    between = max(observed - expected_from_noise, 0.0)

    # between == 0 means the observed spread is fully explained by noise, so no
    # phone's own mean carries information: k is infinite and every score
    # collapses to the grand mean.
    k = float(within / between) if between > 0 else float("inf")

    return ShrinkageParams(
        aspect=aspect,
        grand_mean=grand_mean,
        within_variance=within,
        between_variance=between,
        k=k,
        n_phones=len(means),
    )


def shrink(mean: np.ndarray, count: np.ndarray, grand_mean: float, k: float) -> np.ndarray:
    """Pull each mean toward ``grand_mean`` by the weight of its evidence."""
    mean = np.asarray(mean, dtype=float)
    count = np.asarray(count, dtype=float)
    if not np.isfinite(k):
        return np.full(mean.shape, grand_mean)
    return (count * mean + k * grand_mean) / (count + k)


def build_profiles(
    review_aspects: pd.DataFrame,
    *,
    aspects: tuple[str, ...] | None = None,
    min_mentions: int = MIN_MENTIONS,
) -> tuple[pd.DataFrame, list[ShrinkageParams]]:
    """Aggregate per-review scores into per-phone aspect profiles.

    Returns:
        ``(profiles, params)``. ``profiles`` is long-form with one row per
        ``(model_key, aspect)``, carrying both the raw and shrunk score so the
        effect of shrinkage stays inspectable rather than baked in.
    """
    if aspects is None:
        aspects = tuple(sorted(review_aspects["aspect"].unique()))

    rows = []
    all_params: list[ShrinkageParams] = []

    for aspect in aspects:
        params = estimate_shrinkage(review_aspects, aspect)
        all_params.append(params)

        subset = review_aspects[review_aspects["aspect"] == aspect]
        if subset.empty:
            continue

        grouped = subset.groupby("model_key")["score"]
        means = grouped.mean()
        counts = grouped.size()
        keep = counts >= min_mentions
        if not keep.any():
            continue

        shrunk = shrink(
            means[keep].to_numpy(), counts[keep].to_numpy(), params.grand_mean, params.k
        )
        confidence = subset.groupby("model_key")["confidence"].mean() if "confidence" in subset else None

        for index, model_key in enumerate(means[keep].index):
            rows.append(
                {
                    "model_key": model_key,
                    "aspect": aspect,
                    "score": round(float(shrunk[index]), 3),
                    "raw_score": round(float(means[keep].iloc[index]), 3),
                    "mentions": int(counts[keep].iloc[index]),
                    "mean_confidence": (
                        round(float(confidence[model_key]), 3) if confidence is not None else None
                    ),
                }
            )

    return pd.DataFrame(rows), all_params


def to_wide(profiles: pd.DataFrame, axes: tuple[str, ...]) -> pd.DataFrame:
    """Pivot to one row per phone with a column per axis.

    Phones missing any axis are dropped: the recommender ranks on all five, and
    filling a gap with the mean would present a guess as a measurement.
    """
    wide = profiles.pivot(index="model_key", columns="aspect", values="score")
    missing = [axis for axis in axes if axis not in wide.columns]
    if missing:
        raise KeyError(f"No profiles for aspects: {missing}")
    return wide[list(axes)].dropna()
