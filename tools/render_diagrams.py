"""Re-render every example's flowsheet.svg from its diagram.py.

    python tools/render_diagrams.py           # write public/<name>/flowsheet.svg
    python tools/render_diagrams.py --check   # exit 1 if a committed SVG is stale

The WebAssembly pages cannot import the renderer (it is a sibling package, not
something Pyodide installs), so they show the committed SVG instead. This is
what keeps that file in step with diagram.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flex_pse_examples.diagrams import render  # noqa: E402


def load(path: Path):
    """Import an example's diagram.py by path."""
    spec = importlib.util.spec_from_file_location(f"diagram_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def targets():
    """(example_dir, flowsheet.svg path) for every example with a diagram.py."""
    for example in sorted((ROOT / "examples").iterdir()):
        if (example / "diagram.py").exists():
            yield example, example / "public" / example.name / "flowsheet.svg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if any SVG is stale")
    args = parser.parse_args(argv)
    stale = []
    for example, svg_path in targets():
        diagram = load(example / "diagram.py")
        svg = render(diagram.NODES, diagram.EDGES)
        if args.check:
            if not svg_path.exists() or svg_path.read_text() != svg:
                stale.append(svg_path.relative_to(ROOT))
            continue
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(svg)
        print(f"wrote {svg_path.relative_to(ROOT)}")
    if stale:
        print("stale (run python tools/render_diagrams.py):", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
