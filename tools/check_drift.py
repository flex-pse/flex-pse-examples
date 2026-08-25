"""Does an example's committed sweep still reproduce?

The site publishes numbers that were solved offline, sometimes months ago, from
a flex-pse installed off an unpinned git branch. Nothing about a green CI badge
says those numbers are still what the model produces -- so this re-runs one
example's sweep into a temporary directory and compares it against what is
committed.

Run by the weekly job in ``.github/workflows/examples.yml``, which opens an
issue rather than failing the build: drift is a thing to look at, not a thing
that should block every later pull request.

Usage::

    python tools/check_drift.py examples/pump_scheduling --rtol 1e-3

Exits 0 if the numbers still match, 1 if they moved.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Columns that legitimately differ between two runs of the same sweep and say
#: nothing about whether the answer changed.
VOLATILE = frozenset({"wall_seconds", "mip_gap", "termination", "solver"})


def compare(example_dir: Path, *, rtol: float) -> list[str]:
    """Re-solve ``example_dir``'s sweep and diff it against the committed one.

    Args:
        example_dir: An example directory under ``examples/``.
        rtol: Relative tolerance for the numeric comparison.

    Returns:
        A list of human-readable differences; empty means the sweep reproduces.
    """
    import numpy as np
    import pandas as pd

    from tools import sweep

    committed_path = example_dir / "public" / example_dir.name / "summary.csv"
    if not committed_path.exists():
        return [f"{committed_path} does not exist -- nothing to compare against"]
    committed = pd.read_csv(committed_path)

    with tempfile.TemporaryDirectory() as tmp:
        fresh = pd.read_csv(sweep.run(example_dir, out_dir=Path(tmp)) / "summary.csv")

    problems: list[str] = []

    if len(fresh) != len(committed):
        return [
            f"sweep length changed: committed has {len(committed)} point(s), "
            f"a fresh run produced {len(fresh)}"
        ]

    missing = set(committed.columns) - set(fresh.columns) - VOLATILE
    if missing:
        problems.append(f"columns disappeared from summary.csv: {sorted(missing)}")

    for column in sorted(set(committed.columns) & set(fresh.columns) - VOLATILE):
        old, new = committed[column], fresh[column]
        if old.dtype.kind in "fi" and new.dtype.kind in "fi":
            close = np.isclose(old, new, rtol=rtol, atol=0.0, equal_nan=True)
            for i in np.flatnonzero(~close):
                problems.append(
                    f"{committed.get('sweep_id', pd.Series(range(len(old))))[i]}"
                    f".{column}: committed {old[i]!r}, fresh {new[i]!r}"
                )
        elif not old.equals(new):
            for i in np.flatnonzero((old != new).to_numpy()):
                problems.append(
                    f"row {i} {column}: committed {old[i]!r}, fresh {new[i]!r}"
                )

    return problems


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("example", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    example_dir = args.example.resolve()
    problems = compare(example_dir, rtol=args.rtol)

    if not problems:
        print(f"{example_dir.name}: committed sweep still reproduces (rtol={args.rtol})")
        return 0

    print(
        f"\n{example_dir.name}: committed sweep no longer reproduces "
        f"({len(problems)} difference(s), rtol={args.rtol}):\n",
        file=sys.stderr,
    )
    for problem in problems[:40]:
        print(f"  - {problem}", file=sys.stderr)
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
    print(
        f"\nRe-run `python tools/sweep.py examples/{example_dir.name}` and review "
        f"the diff before committing it.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
