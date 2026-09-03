"""Rank phones against a shopper's slider settings.

The five sliders are *requirements*, not weights
------------------------------------------------
Each slider says "I need at least this much" on a 1-10 scale, and a phone is
penalised only for **falling short** -- never for exceeding. Setting Camera to
9 and everything else to 1 means "I care about the camera and nothing else",
and the ranking follows: every phone trivially satisfies a requirement of 1, so
only the camera separates them.

The alternative reading -- sliders as importance weights on a weighted mean --
was rejected because it makes a phone's rank depend on aspects the shopper said
nothing about. Under a weighted mean, moving Camera from 5 to 9 also changes how
much Battery matters, since the weights renormalise. Requirements do not
interact that way.

Symmetric distance was rejected for a simpler reason: it punishes a phone for
being *better* than asked. A shopper who sets Battery to 6 does not want the
9-battery phone ranked below the 6-battery one.

Match percentage
----------------
Distance is normalised by the worst distance *this query* could produce -- a
hypothetical phone scoring 1 on everything::

    shortfall_i = max(0, requirement_i - profile_i)
    match       = 100 * (1 - ||shortfall|| / ||requirement - 1||)

So the number means "how much of what you asked for you actually get". It is
100% when every requirement is met, and 100% for *every* phone when all sliders
sit at 1 -- which is correct: nothing was asked for. Normalising by a fixed
constant instead would make an undemanding query report suspiciously high
matches for reasons the shopper cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The five axes the recommender ranks on, in slider order.
AXES: tuple[str, ...] = ("battery", "camera", "price", "display", "performance")

# Display names, and the aspect id each axis reads. `price` is the exception:
# its value comes from the listed price, not from sentiment -- see
# ml/recommender/price.py for why.
AXIS_LABELS: dict[str, str] = {
    "battery": "Battery",
    "camera": "Camera",
    "price": "Price",
    "display": "Display",
    "performance": "Processor",
}

SCORE_MIN = 1.0
SCORE_MAX = 10.0


@dataclass(frozen=True)
class Match:
    """One ranked phone."""

    model_key: str
    match_percent: float
    shortfalls: dict[str, float]
    profile: dict[str, float]

    @property
    def worst_axis(self) -> str | None:
        """The requirement this phone misses by the most, if any.

        Drives the "falls short on Camera" line in the UI: a rank is more
        useful when it says what it gave up.
        """
        missed = {axis: gap for axis, gap in self.shortfalls.items() if gap > 0}
        return max(missed, key=missed.get) if missed else None


def as_vector(values: dict[str, float], axes: tuple[str, ...] = AXES) -> np.ndarray:
    """Order a mapping into an axis vector, raising on anything missing.

    Raises rather than defaulting: a missing axis silently treated as 0 would
    make a phone look like it fails a requirement it was never measured on.
    """
    missing = [axis for axis in axes if axis not in values]
    if missing:
        raise KeyError(f"Missing axes: {missing}")
    return np.array([float(values[axis]) for axis in axes], dtype=float)


def shortfall(requirement: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """How far each axis falls below what was asked. Never negative."""
    return np.maximum(0.0, requirement - profile)


def match_percent(requirement: np.ndarray, profile: np.ndarray) -> float:
    """Share of the requirement that the profile meets, as 0-100.

    >>> import numpy as np
    >>> round(match_percent(np.array([5.0]), np.array([9.0])), 1)
    100.0
    >>> round(match_percent(np.array([10.0]), np.array([1.0])), 1)
    0.0
    """
    requirement = np.asarray(requirement, dtype=float).ravel()
    profile = np.asarray(profile, dtype=float).ravel()
    if requirement.shape != profile.shape:
        raise ValueError(f"Shape mismatch: {requirement.shape} vs {profile.shape}")

    gaps = shortfall(requirement, profile)
    # The worst any phone could do against this query: score 1 everywhere.
    worst = np.linalg.norm(requirement - SCORE_MIN)
    if worst <= 0:
        # Every slider at the minimum: nothing was required, so everything
        # matches. Guards the division as well.
        return 100.0
    return float(100.0 * (1.0 - np.linalg.norm(gaps) / worst))


def rank(
    requirement: dict[str, float],
    profiles: dict[str, dict[str, float]],
    *,
    axes: tuple[str, ...] = AXES,
    limit: int | None = None,
) -> list[Match]:
    """Rank phones by match, best first.

    Ties are broken by the mean profile score. Under a requirements model every
    phone that clears every slider matches 100%, so without a tiebreak the
    ordering among them would be arbitrary -- and "these all meet your needs,
    this one is best overall" is the sensible thing to say next.
    """
    wanted = as_vector(requirement, axes)

    matches: list[Match] = []
    for model_key, profile in profiles.items():
        vector = as_vector(profile, axes)
        gaps = shortfall(wanted, vector)
        matches.append(
            Match(
                model_key=model_key,
                match_percent=round(match_percent(wanted, vector), 1),
                shortfalls={axis: round(float(gap), 2) for axis, gap in zip(axes, gaps, strict=True)},
                profile={axis: round(float(value), 2) for axis, value in zip(axes, vector, strict=True)},
            )
        )

    matches.sort(
        key=lambda match: (
            -match.match_percent,
            -float(np.mean(list(match.profile.values()))),
            match.model_key,
        )
    )
    return matches[:limit] if limit else matches
