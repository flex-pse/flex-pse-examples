"""Repository-wide pytest configuration.

Puts the repository root on ``sys.path`` so tests can ``import tools.site.build``
and reuse the site build's validators rather than restating them, and exposes the
paths the example tests share.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
EXAMPLES_ROOT = REPO_ROOT / "examples"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def examples_root() -> Path:
    """The ``examples/`` directory."""
    return EXAMPLES_ROOT


def load_example_module(example_dir: Path, name: str = "model"):
    """Import one example's ``model.py`` under a name unique to that example.

    Thin delegate to :func:`tools.sweep.load_module`, which is where the real
    implementation lives and which ``tools/sweep.py`` needs for the same reason:
    every example's model module is called ``model``, so a bare import in a
    process that touches two examples silently returns the wrong one.
    """
    from tools.sweep import load_module

    return load_module(example_dir, name)


def example_dirs() -> list[Path]:
    """Every example directory carrying a site manifest.

    Used to parametrize the contract tests at collection time, so a new example
    is covered the moment its ``example.toml`` lands.
    """
    if not EXAMPLES_ROOT.is_dir():
        return []
    return sorted(p for p in EXAMPLES_ROOT.iterdir() if (p / "example.toml").exists())
