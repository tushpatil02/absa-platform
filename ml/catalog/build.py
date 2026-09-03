"""Assemble the phone catalogue from Amazon listings and reviews.

This turns 720 listings and 67,986 reviews into a set of phones, each with the
reviews that belong to it. It does **no** sentiment work: scoring is a separate,
much slower pass, and keeping them apart means the catalogue can be rebuilt and
inspected in seconds.

Inclusion is deliberately loose here. The final catalogue is decided *after*
scoring, by how many aspect mentions the model actually finds -- not by a
keyword list written by hand. A keyword prefilter would quietly make the author's
vocabulary the arbiter of which phones exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.catalog.normalise import canonical_model, is_usable, model_key


@dataclass(frozen=True)
class CatalogConfig:
    """Inclusion rules for the candidate catalogue."""

    # A phone needs enough reviews for a per-aspect estimate to mean anything.
    min_reviews: int = 20
    # The Price axis is driven by listed price, so a phone without one cannot
    # be ranked on all five sliders.
    require_price: bool = True
    # Reviews scored per phone. Uncapped this is ~2 hours of CPU inference for
    # little gain: the standard error of a per-aspect mean is already small at
    # a few hundred reviews, and the top phone has 2,686.
    review_cap: int = 400
    seed: int = 0
    # Below this a review carries no usable opinion.
    min_review_words: int = 3


@dataclass
class CatalogReport:
    """What the build did, for the log and the docs."""

    listings_in: int = 0
    listings_unnormalisable: int = 0
    reviews_in: int = 0
    reviews_empty: int = 0
    reviews_duplicated: int = 0
    reviews_orphaned: int = 0
    models_found: int = 0
    models_too_few_reviews: int = 0
    models_no_price: int = 0
    phones_out: int = 0
    reviews_sampled: int = 0

    def summary(self) -> str:
        return (
            f"listings={self.listings_in} unnormalisable={self.listings_unnormalisable}  "
            f"reviews={self.reviews_in} empty={self.reviews_empty} "
            f"dup={self.reviews_duplicated} orphaned={self.reviews_orphaned}  "
            f"models={self.models_found} "
            f"(dropped: {self.models_too_few_reviews} thin, {self.models_no_price} unpriced)  "
            f"phones={self.phones_out} sampled_reviews={self.reviews_sampled}"
        )


def attach_models(items: pd.DataFrame) -> pd.DataFrame:
    """Add canonical model name, grouping key and usability to each listing."""
    frame = items.copy()
    frame["brand"] = frame["brand"].fillna("").astype(str)
    frame["canonical"] = [
        canonical_model(title, brand)
        for title, brand in zip(frame["title"], frame["brand"], strict=True)
    ]
    frame["usable"] = [
        is_usable(name, brand)
        for name, brand in zip(frame["canonical"], frame["brand"], strict=True)
    ]
    frame["model_key"] = frame["canonical"].map(model_key)
    return frame


def deduplicate_reviews(reviews: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop the same review appearing under several listings of one phone.

    Keyed on ``(model, reviewer name, body)``. Body alone is too aggressive:
    "Great phone" occurs 222 times from 186 different reviewers, and collapsing
    those would discard real opinions. Adding the name keeps them -- two people
    are unlikely to share a name *and* an exact review -- while still removing
    the 1,515 long reviews duplicated across listings of the same model.
    """
    before = len(reviews)
    deduped = reviews.drop_duplicates(subset=["model_key", "name", "body"], keep="first")
    return deduped, before - len(deduped)


def build_candidates(
    items: pd.DataFrame,
    reviews: pd.DataFrame,
    config: CatalogConfig | None = None,
    report: CatalogReport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the candidate phone list and the reviews to score.

    Returns:
        ``(phones, reviews)``. ``phones`` has one row per model; ``reviews`` has
        the sampled reviews, each carrying its ``model_key``.
    """
    config = config or CatalogConfig()
    report = report or CatalogReport()

    listings = attach_models(items)
    report.listings_in = len(listings)
    report.listings_unnormalisable = int((~listings["usable"]).sum())
    listings = listings[listings["usable"]]

    frame = reviews.copy()
    report.reviews_in = len(frame)
    frame["body"] = frame["body"].fillna("").astype(str)
    # A stable identity for each review, assigned before any filtering or
    # shuffling so it survives every later step. Split-half reliability needs
    # to tell two reviews apart; keying on the ASIN would give 24 ids for 800
    # reviews and silently make the whole test meaningless.
    frame["review_id"] = [f"r{index}" for index in range(len(frame))]

    before = len(frame)
    frame = frame[frame["body"].str.split().str.len() >= config.min_review_words]
    report.reviews_empty = before - len(frame)

    before = len(frame)
    frame = frame.merge(listings[["asin", "model_key"]], on="asin", how="inner")
    report.reviews_orphaned = before - len(frame)

    frame, removed = deduplicate_reviews(frame)
    report.reviews_duplicated = removed

    counts = frame.groupby("model_key").size()
    report.models_found = len(counts)

    # Listed price: median across the listings of this model that carry one.
    # Median rather than mean because a single mispriced accessory listing
    # would drag a mean badly.
    priced = listings[listings["price"] > 0].groupby("model_key")["price"].median()

    keep = counts[counts >= config.min_reviews].index
    report.models_too_few_reviews = len(counts) - len(keep)
    if config.require_price:
        before_price = len(keep)
        keep = [key for key in keep if key in priced.index]
        report.models_no_price = before_price - len(keep)

    listings = listings[listings["model_key"].isin(keep)]
    frame = frame[frame["model_key"].isin(keep)]

    # Display name and imagery come from the listing with the most reviews --
    # the one a shopper is most likely to have seen.
    listing_reviews = frame.groupby("asin").size()
    listings = listings.assign(n_reviews=listings["asin"].map(listing_reviews).fillna(0))
    primary = listings.sort_values("n_reviews", ascending=False).groupby("model_key").first()

    phones = pd.DataFrame(
        {
            "model_key": primary.index,
            "name": primary["canonical"].values,
            "brand": primary["brand"].values,
            "price": [priced.get(key) for key in primary.index],
            "image": primary["image"].values,
            "url": primary["url"].values,
            "listings": listings.groupby("model_key").size().reindex(primary.index).values,
            "reviews_total": counts.reindex(primary.index).values,
        }
    )
    phones["avg_rating"] = (
        frame.groupby("model_key")["rating"].mean().reindex(primary.index).round(3).values
    )

    # Shuffle once, then take the first N per phone: equivalent to sampling
    # each group, without groupby.apply (removed in pandas 3).
    sampled = (
        frame.sample(frac=1.0, random_state=config.seed)
        .groupby("model_key", sort=False)
        .head(config.review_cap)
        .reset_index(drop=True)
    )
    phones["reviews_sampled"] = (
        sampled.groupby("model_key").size().reindex(phones["model_key"]).values
    )

    phones = phones.sort_values("reviews_total", ascending=False).reset_index(drop=True)
    report.phones_out = len(phones)
    report.reviews_sampled = len(sampled)

    return phones, sampled
