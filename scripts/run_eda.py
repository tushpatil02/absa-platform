"""Run exploratory analysis and render figures.

Reads ``data/processed/`` (produced by ``scripts/build_dataset.py``), prints the
headline statistics, and writes chart PNGs into ``docs/figures/``.

Usage::

    python scripts/run_eda.py
    python scripts/run_eda.py --figures-dir docs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.eda import (  # noqa: E402
    POLARITY_ORDER,
    compute_stats,
    load_processed,
    render_all,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    try:
        asc = load_processed(args.processed_dir)
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    stats = compute_stats(asc)

    print("=" * 58)
    print("DATASET OVERVIEW")
    print("=" * 58)
    print(f"  reviews                {stats.n_reviews:,}")
    print(f"  (review, aspect) pairs {stats.n_pairs:,}")
    print(f"  aspects                {stats.n_aspects}")
    print(f"  domains                {stats.domain_counts}")

    print("\n  splits (pairs / reviews)")
    for split in ("train", "dev", "test"):
        print(
            f"    {split:<6} {stats.split_pairs.get(split, 0):>6,} pairs   "
            f"{stats.split_reviews.get(split, 0):>6,} reviews"
        )

    print("\n  polarity")
    for name in POLARITY_ORDER:
        count = stats.polarity_counts.get(name, 0)
        share = 100 * count / stats.n_pairs
        bar = "#" * round(share / 2)
        print(f"    {name:<9} {count:>6,}  {share:>5.1f}%  {bar}")

    print("\n  review length (words)")
    print(
        f"    mean {stats.mean_words:.1f}   median {stats.median_words}   "
        f"p90 {stats.p90_words}   p99 {stats.p99_words}   max {stats.max_words}"
    )
    print(
        f"  aspects per review: mean {stats.mean_aspects_per_review:.2f}, "
        f"max {stats.max_aspects_per_review}"
    )

    print("\n  pairs per aspect")
    for aspect, count in sorted(stats.aspect_counts.items(), key=lambda kv: -kv[1]):
        subset = asc[asc["aspect"] == aspect]
        negative = 100 * (subset["polarity"] == "negative").mean()
        print(f"    {aspect:<18}{count:>6,}   {negative:>5.1f}% negative")

    # Class imbalance is the headline risk for this dataset; make it loud.
    rarest = min(stats.polarity_counts, key=stats.polarity_counts.get)
    rarest_n = stats.polarity_counts[rarest]
    print(
        f"\n  IMBALANCE: '{rarest}' is the minority class at {rarest_n:,} pairs "
        f"({100 * rarest_n / stats.n_pairs:.1f}%).\n"
        f"  Report macro F1 and per-class metrics; accuracy alone would be misleading."
    )

    written = render_all(asc, args.figures_dir)
    print(f"\n  figures -> {args.figures_dir}")
    for path in written:
        print(f"    {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
