"""Download the Amazon cell-phone review corpus.

This is the corpus the *recommender* runs on. M-ABSA supplies the aspect and
polarity labels that train the models; it has no product identity, so it cannot
answer "which phone should I buy". This one has 67,986 reviews attached to 720
Amazon listings, which is what makes per-phone sentiment profiles possible.

Dataset
-------
Amazon Cell Phones Reviews -- Griko Nibras
https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews
Source scraper: https://github.com/grikomsn/amazon-cell-phones-reviews

**Licence: CC0 1.0 Universal (public domain).** Verified directly from the
LICENSE file in the source repository rather than taken from the Kaggle page.
CC0 would permit vendoring the data, but it is fetched on demand anyway to keep
the repository small and to match how M-ABSA is handled.

Access
------
Kaggle's download endpoint serves this dataset **without authentication** -- no
account, no API token, no `kaggle.json`. That was verified, not assumed: the
endpoint returns 200 with a signed storage redirect. It matters because a
credential-gated dataset cannot run in CI or be reproduced by a reader.

Usage
-----
    python scripts/download_phones.py
    python scripts/download_phones.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/grikomsn/amazon-cell-phones-reviews"
)

# parents[1] because this file is scripts/download_phones.py -- see
# tests/test_scripts.py, which pins this for every script after a parents[2]
# bug wrote a dataset above the repository and still exited 0.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "amazon"

ITEMS_FILE = "20191226-items.csv"
REVIEWS_FILE = "20191226-reviews.csv"

# Expected shapes, asserted after extraction. A silently truncated or swapped
# archive would otherwise flow straight into the catalogue build and produce a
# smaller recommender that still looks fine.
EXPECTED = {ITEMS_FILE: 720, REVIEWS_FILE: 67_986}

ATTRIBUTION = """\
Amazon Cell Phones Reviews
Griko Nibras -- https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews
Licence: CC0 1.0 Universal (public domain dedication)
Scraped 2019-12-26. Reviews are Amazon customer content; product and brand
names are trademarks of their respective owners.
"""


def download(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "absa-platform/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def count_rows(payload: bytes) -> int:
    """Count CSV data rows without parsing them.

    Quoted newlines inside review bodies make a naive line count wrong, so this
    uses the csv module -- the whole point is to catch a truncated file.
    """
    import csv

    text = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8", newline="")
    reader = csv.reader(text)
    next(reader, None)  # header
    return sum(1 for _ in reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    targets = [args.out / ITEMS_FILE, args.out / REVIEWS_FILE]
    if all(path.exists() for path in targets) and not args.force:
        print(f"Already present in {args.out} -- use --force to re-download.")
        for path in targets:
            print(f"  {path.name}  {path.stat().st_size:,} bytes")
        return 0

    print(f"Downloading {DATASET_URL}")
    try:
        payload = download(DATASET_URL)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"FAILED: {exc}")
        print("The Kaggle endpoint serves this dataset without credentials; if it now")
        print("refuses, check https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews")
        return 1

    digest = hashlib.sha256(payload).hexdigest()
    print(f"  {len(payload):,} bytes  sha256={digest[:16]}...")

    args.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict] = {}

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
        missing = set(EXPECTED) - names
        if missing:
            print(f"FAILED: archive is missing {sorted(missing)}; it contains {sorted(names)}")
            return 1

        for name, expected_rows in EXPECTED.items():
            content = archive.read(name)
            rows = count_rows(content)
            if rows != expected_rows:
                # Not fatal: the upstream file could legitimately be revised.
                # But it must be visible, because every downstream count changes.
                print(f"  WARNING {name}: {rows:,} rows, expected {expected_rows:,}")
            (args.out / name).write_bytes(content)
            written[name] = {
                "bytes": len(content),
                "rows": rows,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            print(f"  {name}  {len(content):,} bytes  {rows:,} rows")

    (args.out / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "url": DATASET_URL,
                "licence": "CC0-1.0",
                "licence_verified_at": (
                    "https://github.com/grikomsn/amazon-cell-phones-reviews/blob/master/LICENSE"
                ),
                "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "archive_sha256": digest,
                "archive_bytes": len(payload),
                "files": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {args.out}")
    print("Next: python scripts/build_catalog.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
