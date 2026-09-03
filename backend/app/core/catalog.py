"""The phone catalogue, held in memory and served to the API.

Loaded once at startup from the CSVs that ``build_catalog.py`` and
``build_profiles.py`` produce. The whole catalogue is a few hundred rows, so
there is nothing to gain from a database here -- and keeping the derived
artefacts as plain files means a reviewer can open them and check the numbers
against the pipeline that made them.

User-submitted reviews go to SQLite instead, because those are written at
runtime. See ``app.core.storage``.

A missing or half-built catalogue is not an exception. It is the normal state
of a fresh clone, and :meth:`Catalog.load` reports it the same way the model
loader does: the API starts, says what is missing and which script produces it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.recommender.price import score_to_price
from ml.recommender.similarity import AXES, AXIS_LABELS, Match, rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AspectScore:
    """One axis of a phone's profile."""

    aspect: str
    display_name: str
    score: float
    mentions: int
    source: str  # "reviews" or "listed_price"


@dataclass(frozen=True)
class Phone:
    """A catalogue entry."""

    model_key: str
    name: str
    brand: str
    price: float | None
    image: str | None
    url: str | None
    reviews_total: int
    reviews_scored: int
    avg_rating: float | None
    aspects: dict[str, AspectScore]

    @property
    def rankable(self) -> bool:
        """Whether every slider axis has a score.

        A phone missing an axis is excluded from recommendations rather than
        imputed: filling the gap with the catalogue mean would let a phone be
        ranked on a measurement that was never taken.
        """
        return all(axis in self.aspects for axis in AXES)

    def profile_vector(self) -> dict[str, float]:
        return {axis: self.aspects[axis].score for axis in AXES}


@dataclass
class Catalog:
    """Phones plus their aspect profiles."""

    phones: dict[str, Phone] = field(default_factory=dict)
    price_bounds: tuple[float, float] | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self.phones)

    @property
    def rankable(self) -> list[Phone]:
        return [phone for phone in self.phones.values() if phone.rankable]

    @classmethod
    def load(cls, processed_dir: Path) -> Catalog:
        """Load the catalogue, or return one carrying an explanatory error."""
        phones_path = processed_dir / "phones.csv"
        profiles_path = processed_dir / "phone_profiles.csv"

        missing = [path.name for path in (phones_path, profiles_path) if not path.exists()]
        if missing:
            return cls(
                error=(
                    f"Catalogue not built ({', '.join(missing)} missing). Run: "
                    "scripts/download_phones.py, scripts/build_catalog.py, "
                    "scripts/score_catalog.py, scripts/build_profiles.py"
                )
            )

        try:
            phones_frame = pd.read_csv(phones_path)
            profiles_frame = pd.read_csv(profiles_path)
        except Exception as exc:  # pragma: no cover -- corrupt file
            logger.exception("Catalogue failed to load")
            return cls(error=f"Catalogue failed to load: {exc}")

        by_phone: dict[str, dict[str, AspectScore]] = {}
        for row in profiles_frame.itertuples(index=False):
            by_phone.setdefault(row.model_key, {})[row.aspect] = AspectScore(
                aspect=row.aspect,
                display_name=AXIS_LABELS.get(row.aspect, row.aspect.replace("_", " ").title()),
                score=float(row.score),
                mentions=int(row.mentions),
                # Price is the one axis not derived from review sentiment, and
                # the UI must be able to say so rather than implying shoppers
                # praised the price.
                source="listed_price" if row.aspect == "price" else "reviews",
            )

        phones: dict[str, Phone] = {}
        for row in phones_frame.itertuples(index=False):
            phones[row.model_key] = Phone(
                model_key=row.model_key,
                name=row.name,
                brand=str(row.brand) if pd.notna(row.brand) else "",
                price=float(row.price) if pd.notna(row.price) else None,
                image=row.image if pd.notna(row.image) else None,
                url=row.url if pd.notna(row.url) else None,
                reviews_total=int(row.reviews_total),
                reviews_scored=int(getattr(row, "reviews_sampled", row.reviews_total)),
                avg_rating=float(row.avg_rating) if pd.notna(row.avg_rating) else None,
                aspects=by_phone.get(row.model_key, {}),
            )

        priced = phones_frame[phones_frame["price"] > 0]["price"]
        bounds = (float(priced.min()), float(priced.max())) if len(priced) else None

        catalog = cls(phones=phones, price_bounds=bounds)
        logger.info(
            "Catalogue loaded: %d phones, %d rankable on all five axes",
            len(catalog.phones),
            len(catalog.rankable),
        )
        return catalog

    # -- queries ------------------------------------------------------------

    def search(
        self,
        *,
        query: str | None = None,
        brand: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Phone], int]:
        """Filter the catalogue. Returns ``(page, total_matched)``."""
        results = list(self.phones.values())
        if brand:
            results = [p for p in results if p.brand.lower() == brand.lower()]
        if query:
            needle = query.lower().strip()
            results = [p for p in results if needle in p.name.lower()]

        results.sort(key=lambda phone: -phone.reviews_total)
        return results[offset : offset + limit], len(results)

    def brands(self) -> list[str]:
        return sorted({phone.brand for phone in self.phones.values() if phone.brand})

    def recommend(self, preferences: dict[str, float], limit: int = 10) -> list[Match]:
        """Rank the phones that have all five axes."""
        profiles = {phone.model_key: phone.profile_vector() for phone in self.rankable}
        return rank(preferences, profiles, limit=limit)

    def price_for_score(self, score: float) -> float | None:
        """The listed price a Price slider position corresponds to.

        Lets the UI put "around $310" beside the slider instead of leaving a
        bare number that the shopper has no way to interpret.
        """
        if not self.price_bounds:
            return None
        return round(score_to_price(score, *self.price_bounds), 2)
