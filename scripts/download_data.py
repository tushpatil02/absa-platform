"""Download the English M-ABSA splits used by this project.

The M-ABSA repository does not ship a LICENSE file, so the raw data is *not*
vendored into this repository. This script fetches it on demand into
`data/raw/`, which is gitignored.

Dataset
-------
M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis
Wu et al., EMNLP 2025. https://aclanthology.org/2025.emnlp-main.128/
Repository: https://github.com/swaggy66/M-ABSA

Usage
-----
    python ml/scripts/download_data.py
    python ml/scripts/download_data.py --domains phone laptop restaurant
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/swaggy66/M-ABSA/main/data"

# Only the two electronics domains are needed for the unified taxonomy.
# The others are downloadable for exploration / future cross-domain work.
DEFAULT_DOMAINS = ["phone", "laptop"]
ALL_DOMAINS = ["phone", "laptop", "restaurant", "hotel", "coursera", "food", "sight"]
SPLITS = ["train", "dev", "test"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "mabsa"

CITATION = """\
@inproceedings{wu-etal-2025-mabsa,
    title     = {{M-ABSA}: A Multilingual Dataset for Aspect-Based Sentiment Analysis},
    author    = {Wu, ChengYan and Ma, Bolei and Liu, Yihong and Zhang, Zheyu and
                 Deng, Ningyuan and Li, Yanshu and Chen, Baolan and Zhang, Yi and
                 Xue, Yun and Plank, Barbara},
    booktitle = {Proceedings of EMNLP 2025},
    year      = {2025},
    url       = {https://aclanthology.org/2025.emnlp-main.128/}
}
"""


def download_file(url: str, dest: Path, timeout: int = 30) -> int:
    """Fetch `url` into `dest`. Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "absa-platform/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        choices=ALL_DOMAINS,
        help=f"M-ABSA domains to download (default: {' '.join(DEFAULT_DOMAINS)})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist locally.",
    )
    args = parser.parse_args()

    print(f"Destination: {RAW_DIR}")
    downloaded = skipped = 0
    failures: list[str] = []

    for domain in args.domains:
        for split in SPLITS:
            dest = RAW_DIR / domain / f"{split}.txt"
            if dest.exists() and not args.force:
                skipped += 1
                print(f"  skip     {domain}/{split}.txt (exists)")
                continue

            url = f"{RAW_BASE}/{domain}/en/{split}.txt"
            try:
                size = download_file(url, dest)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                failures.append(f"{domain}/{split}: {exc}")
                print(f"  FAILED   {domain}/{split}.txt -> {exc}")
                continue

            downloaded += 1
            print(f"  ok       {domain}/{split}.txt ({size:,} bytes)")

    # Keep the citation next to the data so provenance travels with it.
    (RAW_DIR / "CITATION.bib").write_text(CITATION, encoding="utf-8")

    print(f"\nDownloaded {downloaded}, skipped {skipped}, failed {len(failures)}.")
    if failures:
        print("\nFailures:", *failures, sep="\n  ")
        print(
            "\nIf this persists the upstream repository layout may have changed; "
            "check https://github.com/swaggy66/M-ABSA/tree/main/data",
            file=sys.stderr,
        )
        return 1

    print(f"Citation written to {RAW_DIR / 'CITATION.bib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
