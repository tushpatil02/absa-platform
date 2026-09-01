"""Fill in the GitHub/Hugging Face username placeholders.

The Colab notebooks clone the repo, and `docs/deployment.md` pulls model weights
from the Hub. Both ship with ``YOUR_USERNAME`` placeholders because the account
is not known until the repo exists. Run this once after creating it.

Usage::

    python scripts/set_repo_url.py --user tusharpatil
    python scripts/set_repo_url.py --user tusharpatil --hf-user someoneelse
    python scripts/set_repo_url.py --user tusharpatil --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files carrying a placeholder, and what the clone URL should look like.
TARGETS = (
    "notebooks/01_eda.ipynb",
    "notebooks/absa_training.ipynb",
    "docs/deployment.md",
    "README.md",
)

PLACEHOLDER = "YOUR_USERNAME"
CLONE_PLACEHOLDER = "git clone <this-repo> && cd absa-platform"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="GitHub username.")
    parser.add_argument(
        "--hf-user",
        default=None,
        help="Hugging Face username, if different from the GitHub one.",
    )
    parser.add_argument("--repo", default="absa-platform", help="Repository name.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", args.user):
        print(f"{args.user!r} is not a valid GitHub username.", file=sys.stderr)
        return 1

    hf_user = args.hf_user or args.user
    clone_line = f"git clone https://github.com/{args.user}/{args.repo}.git && cd {args.repo}"

    changed = 0
    for relative in TARGETS:
        path = REPO_ROOT / relative
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = original

        # Hub paths use the HF account; the clone URL uses the GitHub one. They
        # are usually the same person but need not be the same account.
        for artefact in ("absa-sentiment-classifier", "absa-aspect-detector"):
            updated = updated.replace(f"{PLACEHOLDER}/{artefact}", f"{hf_user}/{artefact}")
        updated = updated.replace(PLACEHOLDER, args.user)
        updated = updated.replace(CLONE_PLACEHOLDER, clone_line)

        if updated != original:
            changed += 1
            print(f"  {'would update' if args.dry_run else 'updated'}  {relative}")
            if not args.dry_run:
                path.write_text(updated, encoding="utf-8")

    if not changed:
        print("  nothing to change -- placeholders already filled in.")
        return 0

    # A notebook that stops being valid JSON breaks silently in Colab.
    if not args.dry_run:
        import json

        for relative in TARGETS:
            path = REPO_ROOT / relative
            if path.suffix == ".ipynb" and path.exists():
                json.loads(path.read_text(encoding="utf-8"))
        print("  notebooks still parse as valid JSON")

    print(f"\n  clone URL: https://github.com/{args.user}/{args.repo}.git")
    print(f"  hub repos: {hf_user}/absa-sentiment-classifier, {hf_user}/absa-aspect-detector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
