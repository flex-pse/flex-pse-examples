"""Programmatic access to the bundled flex-pse example models."""

import importlib
from pathlib import Path


def _examples_root() -> Path:
    # `examples` is remapped in pyproject.toml's [tool.setuptools.package-dir]
    # to a directory that isn't a physical child of this package (it's a
    # sibling at the repo root), and an editable install keeps it there
    # rather than copying it under this package on disk. So its location has
    # to come from the import system (which honors that remapping in both
    # editable and regular installs), not from `Path(__file__).parent`.
    return Path(importlib.import_module(f"{__name__}.examples").__path__[0])


def list_examples() -> list[str]:
    """Names of every bundled example."""
    return sorted(
        p.name for p in _examples_root().iterdir()
        if p.is_dir() and (p / "example.toml").exists()
    )


def example_dir(name: str) -> Path:
    """Filesystem path to a bundled example's directory (config.json, etc.)."""
    path = _examples_root() / name
    if not (path / "example.toml").exists():
        raise ValueError(f"no example named {name!r}; available: {list_examples()}")
    return path


def load_model(name: str):
    """Import and return an example's ``model`` module.

    Use its ``load_config()``/``build_model()``/``solve_model()`` to build and
    solve, e.g. ``m = load_model("pump_scheduling"); cfg = m.load_config();
    model = m.build_model(cfg); m.solve_model(model)``.
    """
    example_dir(name)
    return importlib.import_module(f"{__name__}.examples.{name}.model")
