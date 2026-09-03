"""SQLite storage for reviews submitted through the site.

The catalogue itself is read-only derived data and lives in CSVs. This holds
the one thing written at runtime: reviews a visitor types in.

Submitted reviews are stored and displayed, but they do **not** feed back into
a phone's published aspect scores. A score built from hundreds of reviews
should not visibly move because one person typed a sentence, and a page where
it did would be trivially gameable -- type "best camera ever" ten times and
watch the ranking change. They are shown as their own section, with their
analysis, which is the honest presentation: this is what the model makes of
what you wrote.

sqlite3 is used directly rather than through an ORM. There is one table.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS submitted_reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key   TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    rating      INTEGER,
    analysis    TEXT    NOT NULL,
    created_utc TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submitted_model ON submitted_reviews(model_key);
"""


@dataclass(frozen=True)
class StoredReview:
    id: int
    model_key: str
    text: str
    rating: int | None
    analysis: str
    created_utc: str


class ReviewStore:
    """Append-and-read store for submitted reviews."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        # check_same_thread=False because FastAPI serves requests from a thread
        # pool; each call opens and closes its own short-lived connection, so
        # no connection is ever shared between threads.
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def add(self, model_key: str, text: str, rating: int | None, analysis: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO submitted_reviews (model_key, text, rating, analysis, created_utc)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    model_key,
                    text,
                    rating,
                    analysis,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            return int(cursor.lastrowid)

    def for_phone(self, model_key: str, limit: int = 20) -> list[StoredReview]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM submitted_reviews WHERE model_key = ?"
                " ORDER BY id DESC LIMIT ?",
                (model_key, limit),
            ).fetchall()
        return [StoredReview(**dict(row)) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM submitted_reviews").fetchone()[0])
