"""Tests for the CLI scripts.

These exist because of a specific bug. ``download_data.py`` resolved its repo
root with ``parents[2]``, which was correct while the file lived at
``ml/scripts/`` and silently wrong after it moved to ``scripts/``. It then wrote
the dataset one directory *above* the repository, where ``build_dataset.py``
never looks.

Nothing caught it: the scripts still exited 0, the unit tests never ran them, and
locally ``data/raw/`` was already populated from before the move. It only
surfaced on a clean CI checkout, where the build step failed with "MISSING".

So the invariant is asserted directly: every script must resolve its repo root to
the real repository, and the paths they read and write must agree.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))

# A file that exists at the repo root and nowhere above it, used to prove a
# resolved path really is the repository.
ROOT_MARKER = "pyproject.toml"


def resolved_repo_root(script: Path) -> Path:
    """Evaluate the script's own REPO_ROOT expression without importing it.

    Importing would execute module-level side effects; parsing the assignment
    reproduces exactly what the script computes.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "REPO_ROOT" not in targets:
            continue
        # Expect: Path(__file__).resolve().parents[N]
        source = ast.unparse(node.value)
        if "parents[" not in source:
            pytest.fail(f"{script.name}: unexpected REPO_ROOT form {source!r}")
        index = int(source.split("parents[")[1].split("]")[0])
        return script.resolve().parents[index]
    pytest.fail(f"{script.name}: no REPO_ROOT assignment found")


def test_there_are_scripts_to_check():
    assert SCRIPTS, "no scripts found -- the glob is wrong"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_repo_root_points_at_the_repository(script: Path):
    """The regression: parents[N] must land on the repo, not above it."""
    resolved = resolved_repo_root(script)
    assert (resolved / ROOT_MARKER).exists(), (
        f"{script.name} resolves REPO_ROOT to {resolved}, which is not the "
        f"repository root (no {ROOT_MARKER} there). Check the parents[] index "
        f"against the file's depth."
    )
    assert resolved == REPO_ROOT


def test_download_and_build_agree_on_the_data_directory():
    """The two halves of the pipeline must point at the same place.

    This is the exact mismatch that broke CI: the downloader wrote to one
    directory and the builder read from another, each internally consistent.
    """
    download_root = resolved_repo_root(REPO_ROOT / "scripts" / "download_data.py")
    build_root = resolved_repo_root(REPO_ROOT / "scripts" / "build_dataset.py")
    assert download_root == build_root


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_help_runs(script: Path):
    """--help must work without side effects.

    Cheap smoke test: it catches import errors, bad argparse wiring and syntax
    problems in scripts the unit tests never otherwise execute.
    """
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"{script.name} --help exited {result.returncode}\n{result.stderr[-1500:]}"
    )
    assert "usage:" in result.stdout.lower()
