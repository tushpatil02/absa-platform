"""Turn a listed price into a 1-10 Price axis.

Why this axis is not sentiment-derived
--------------------------------------
Every other axis reads the model's aspect sentiment. Price does not, and the
reason is measured rather than aesthetic.

In the training data, price opinions are **85.6% positive** (they were 90.8%
before the taxonomy fix, which added laptop price rows). Only 40 of 341 price
pairs are negative. Shoppers record price complaints far less often than price
praise, so a model trained on this can score well by answering "positive" and
almost every phone's price sentiment lands in a narrow band near the top.

A Price slider driven by that would barely reorder the results: it would look
functional and do nothing. Since the corpus carries an actual listed price,
this axis uses it, and the UI labels the axis as coming from listed price
rather than from reviews. Sentiment about price is still computed and shown on
the phone page -- it is simply not what the slider ranks on.

Direction and scale
-------------------
Higher is cheaper: 10 is the least expensive phone in the catalogue, 1 the most
expensive. That keeps every slider pointing the same way -- "more is better" --
so "Price 9" reads as "I want it cheap", consistent with "Camera 9".

The mapping is linear in **log** price. Price is perceived in ratios: the gap
between $100 and $200 feels like the gap between $400 and $800, and a linear
mapping would squash three quarters of the catalogue (median $200, max $948)
into the top of the scale. Log spacing also keeps genuinely similar prices
genuinely close, which a percentile rank would not -- percentiles would push
$200 and $210 apart whenever the catalogue happens to be dense between them.
"""

from __future__ import annotations

import numpy as np

SCORE_MIN = 1.0
SCORE_MAX = 10.0


def price_to_score(
    price: float | np.ndarray,
    cheapest: float,
    dearest: float,
) -> float | np.ndarray:
    """Map a listed price onto 1-10, where 10 is cheapest.

    Args:
        price: Listed price, same currency as the bounds.
        cheapest: Lowest price in the catalogue -> 10.
        dearest: Highest price in the catalogue -> 1.

    >>> round(price_to_score(100, 100, 1000), 2)
    10.0
    >>> round(price_to_score(1000, 100, 1000), 2)
    1.0
    >>> round(price_to_score(316.23, 100, 1000), 1)  # geometric midpoint
    5.5
    """
    if cheapest <= 0 or dearest <= 0:
        raise ValueError("Prices must be positive to take logarithms")
    if dearest < cheapest:
        raise ValueError(f"dearest ({dearest}) is below cheapest ({cheapest})")

    values = np.asarray(price, dtype=float)
    if dearest == cheapest:
        # A one-price catalogue has no price information to convey. Returning
        # the midpoint is honest; returning 10 would claim every phone is a
        # bargain.
        result = np.full(values.shape, (SCORE_MIN + SCORE_MAX) / 2)
    else:
        span = np.log(dearest) - np.log(cheapest)
        position = (np.log(np.clip(values, cheapest, dearest)) - np.log(cheapest)) / span
        result = SCORE_MAX - position * (SCORE_MAX - SCORE_MIN)

    return float(result) if np.isscalar(price) or np.ndim(price) == 0 else result


def score_to_price(score: float, cheapest: float, dearest: float) -> float:
    """Inverse of :func:`price_to_score`, for labelling a slider position.

    Lets the UI say "around $310" beside a Price slider at 5.5 instead of
    leaving the number unexplained.

    >>> round(score_to_price(10.0, 100, 1000))
    100
    >>> round(score_to_price(1.0, 100, 1000))
    1000
    """
    if dearest == cheapest:
        return float(cheapest)
    position = (SCORE_MAX - float(score)) / (SCORE_MAX - SCORE_MIN)
    return float(np.exp(np.log(cheapest) + position * (np.log(dearest) - np.log(cheapest))))
