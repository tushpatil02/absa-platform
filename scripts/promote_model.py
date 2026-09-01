"""Promote an experiment checkpoint to be the served model.

Training experiments write to ``models/_experiments/<name>/``. The API serves
whatever sits in ``models/sentiment_classifier/`` or ``models/aspect_detector/``.
This script moves one to the other deliberately, rather than by hand-copying
directories, so that:

* the promotion is recorded in the model's ``metadata.json`` (what was promoted,
  from where, and on which selection);
* the previous checkpoint is kept as ``<stage>.previous/`` so a regression can be
  rolled back without retraining;
* the artefacts actually required to serve are verified present first, instead of
  the API discovering a half-copied directory at startup.

Usage::

    python scripts/promote_model.py --from models/_experiments/asc_mixed3 --stage sentiment
    python scripts/promote_model.py --from models/_experiments/asc_mixed3 --stage sentiment --dry-run
    python scripts/promote_model.py --rollback --stage sentiment
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

STAGE_DIRS = {"sentiment": "sentiment_classifier", "aspect": "aspect_detector"}

# What a transformer checkpoint must contain to be loadable. metadata.json alone
# is not enough -- that mistake previously let a weightless directory be selected
# and crash the API at startup.
REQUIRED = ("metadata.json", "config.json", "tokenizer_config.json")
WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def verify(directory: Path) -> list[str]:
    """Return a list of problems; empty means the checkpoint is servable."""
    problems = []
    if not directory.is_dir():
        return [f"{directory} does not exist"]
    for name in REQUIRED:
        if not (directory / name).exists():
            problems.append(f"missing {name}")
    if not any((directory / name).exists() for name in WEIGHTS):
        problems.append(f"no weights (need one of {', '.join(WEIGHTS)})")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", type=Path, help="Experiment directory.")
    parser.add_argument("--stage", choices=sorted(STAGE_DIRS), required=True)
    parser.add_argument("--reason", default="", help="Why, recorded in metadata.json.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rollback", action="store_true", help="Restore the previous checkpoint.")
    args = parser.parse_args()

    target = MODELS_DIR / STAGE_DIRS[args.stage]
    previous = target.with_name(target.name + ".previous")

    # ------------------------------------------------------------ rollback --
    if args.rollback:
        if not previous.is_dir():
            print(f"Nothing to roll back to: {previous} does not exist.", file=sys.stderr)
            return 1
        problems = verify(previous)
        if problems:
            print(f"Refusing to roll back, {previous} is incomplete: {problems}", file=sys.stderr)
            return 1
        if args.dry_run:
            print(f"--dry-run: would restore {previous} -> {target}")
            return 0
        swap = target.with_name(target.name + ".rolledback")
        if swap.exists():
            shutil.rmtree(swap)
        if target.exists():
            target.rename(swap)
        previous.rename(target)
        shutil.rmtree(swap, ignore_errors=True)
        print(f"Rolled back: {previous.name} is now {target.name}")
        return 0

    # ------------------------------------------------------------- promote --
    if args.source is None:
        parser.error("--from is required unless --rollback is given")

    problems = verify(args.source)
    if problems:
        print(f"Refusing to promote {args.source}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    metadata = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))
    current = None
    if (target / "metadata.json").exists():
        current = json.loads((target / "metadata.json").read_text(encoding="utf-8"))

    print(f"Promoting  {args.source}")
    print(f"       ->  {target}")
    print(f"  base model     {metadata.get('base_model', '?')}")
    print(f"  mixed_weight   {metadata.get('mixed_weight', 1.0)}")
    if current:
        print(
            f"  replacing      base={current.get('base_model', '?')} "
            f"mixed_weight={current.get('mixed_weight', 1.0)}"
        )

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if target.exists():
        if previous.exists():
            shutil.rmtree(previous)
        target.rename(previous)
        print(f"  kept previous  {previous.name}/")

    shutil.copytree(args.source, target)

    metadata["promoted_from"] = str(args.source.relative_to(REPO_ROOT)).replace("\\", "/")
    if args.reason:
        metadata["promoted_reason"] = args.reason
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    remaining = verify(target)
    if remaining:
        print(f"\nERROR: {target} is incomplete after copy: {remaining}", file=sys.stderr)
        return 1

    print(f"\n  {target.name} is now the served checkpoint.")
    print("  Restart the API to pick it up, then re-run scripts/compare_models.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
