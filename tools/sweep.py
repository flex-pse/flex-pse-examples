"""Run an example's model across its sweep and write the site's data files.

Every example on the site ships a *sweep*: the same model solved at a range of
values of one knob, reduced to a few small CSVs that the WebAssembly notebook
replays in a browser. The optimization itself cannot run in a browser -- Pyomo,
IDAES and HiGHS/Gurobi have no WebAssembly build -- so this script is the bridge
between the two halves of an example, and its output is committed to the
repository.

This module is the *generic* half. Everything that is the same for every
example lives here: the loop, the timing, the provenance, the column pruning,
the view reduction, the CSV formatting and the size discipline. Everything that
is specific to an example lives in that example's ``sweep.py`` adapter, which
supplies three names and nothing else::

    points()               -> list of point dicts, each with a "label" key
    setup()                -> whatever solve_point() needs, built once
    solve_point(ctx, pt)   -> (results DataFrame, summary dict)

The knob is declared in Python rather than in ``example.toml`` because choosing
a sensible range usually means asking the model a question first -- the
desalination sweep spans fractions of ``model.max_product_af()``, which is not
knowable until the flowsheet is built.

Usage::

    python tools/sweep.py examples/pump_scheduling
    python tools/sweep.py examples/desalination_scheduling --smoke

The output layout, which :mod:`tools.site.build` validates and the WebAssembly
notebooks read::

    examples/<name>/public/<name>/
        provenance.csv     # key,value -- one row per fact about the run
        summary.csv        # one row per sweep point
        <view>/sNN.csv     # one file per point per view, wide: timestamp x column
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent


def _display(path: Path) -> str:
    """Return ``path`` relative to the repository when it is inside it.

    Sweeps normally write into the repository, but tests run them into a
    temporary directory, and ``Path.relative_to`` raises rather than falling
    back when the path is elsewhere.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


#: Written into every float in every view file. Five significant figures is well
#: past solver tolerance and roughly halves the committed CSV against repr().
FLOAT_FORMAT = "%.5g"

#: ISO 8601 without a timezone -- the horizons are naive local plant time, and a
#: "+00:00" suffix would claim a UTC they are not in.
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: A view file above this is a sign someone is dumping a raw results frame
#: rather than a reduced view; the browser pays for it on every selection.
#: ``tools/site/build.py`` refuses to publish one, so fail here first, where the
#: person who can fix it is still watching.
MAX_VIEW_BYTES = 2 * 1024 * 1024


# --------------------------------------------------------------------------
# Loading the example


def load_manifest(example_dir: Path) -> dict:
    """Return the parsed ``example.toml`` for ``example_dir``.

    Args:
        example_dir: An example directory under ``examples/``.

    Returns:
        The parsed manifest.

    Raises:
        FileNotFoundError: If the example has no ``example.toml``.
    """
    path = example_dir / "example.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- every example needs a manifest; see CONTRIBUTING.md"
        )
    return tomllib.loads(path.read_text())


def load_module(example_dir: Path, stem: str = "model"):
    """Import ``<example_dir>/<stem>.py`` under a name unique to that example.

    Every example has a ``model.py``, and every notebook and adapter imports it
    as the bare name ``model``. Inside one notebook that is fine -- one example
    per process. It is not fine anywhere that touches two examples: the first
    ``import model`` wins ``sys.modules`` and every later example silently gets
    the wrong flowsheet, which surfaces as ``AttributeError`` on whichever
    function the two models do not share.

    Args:
        example_dir: An example directory under ``examples/``.
        stem: The module file's name without ``.py``.

    Returns:
        The imported module, cached under ``<example>.<stem>``.

    Raises:
        FileNotFoundError: If the example has no such module.
    """
    path = example_dir / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    qualified = f"{example_dir.name}.{stem}"
    if qualified in sys.modules:
        return sys.modules[qualified]

    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))

    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def load_adapter(example_dir: Path):
    """Import an example's ``sweep.py`` adapter.

    Args:
        example_dir: An example directory under ``examples/``.

    Returns:
        The imported adapter module.

    Raises:
        FileNotFoundError: If the example has no ``sweep.py``.
        AttributeError: If the adapter is missing a required name.
    """
    path = example_dir / "sweep.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- an example needs a sweep adapter; see CONTRIBUTING.md"
        )
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))

    # Bind *this* example's flowsheet to the bare name the adapter imports,
    # immediately before executing it. Without this, sweeping two examples in one
    # process hands the second one the first one's model -- see load_module.
    sys.modules["model"] = load_module(example_dir, "model")

    spec = importlib.util.spec_from_file_location(
        f"{example_dir.name}.sweep", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name in ("points", "setup", "solve_point"):
        if not hasattr(module, name):
            raise AttributeError(
                f"{path} must define {name}() -- see the contract in tools/sweep.py"
            )
    return module


# --------------------------------------------------------------------------
# Reducing a results frame to a view


def reduce_view(frame, view: dict, columns: list[str] | None):
    """Reduce a full results frame to one of the views the site ships.

    A month at 15-minute resolution is ~2,976 rows and unreadable as a line
    chart; shipping it would cost megabytes per sweep point for detail no reader
    can see. Each view is a different honest answer to "what does the chart
    actually draw".

    Args:
        frame: A results DataFrame indexed by timestamp.
        view: A ``[[sweep.views]]`` table. ``kind`` is one of ``window`` (the
            first ``days`` days starting at ``start``, or a window centred in
            the horizon), ``diurnal`` (averaged onto a single day by time of
            day), or ``full`` (every row, for horizons short enough).
        columns: Columns to keep, or None for all of them. Names absent from
            the frame are ignored, so an example may declare one column list
            across views that do not all carry every column.

    Returns:
        The reduced DataFrame.

    Raises:
        ValueError: If ``kind`` is not a known view kind.
    """
    kind = view.get("kind", "full")

    if kind == "full":
        out = frame
    elif kind == "window":
        days = float(view.get("days", 3))
        span = frame.index[-1] - frame.index[0]
        start = view.get("start")
        if start is None:
            # Centre the window rather than taking the head: the first days of a
            # horizon are dominated by the initial condition, not by the tariff.
            offset = max((span - _days(days)) / 2, _days(0))
            start = frame.index[0] + offset
        else:
            import pandas as pd

            start = pd.Timestamp(start)
        out = frame.loc[start : start + _days(days)]
        if out.empty:
            raise ValueError(
                f"view {view.get('name')!r}: window starting {start} is outside the "
                f"horizon {frame.index[0]}..{frame.index[-1]}"
            )
    elif kind == "diurnal":
        keyed = frame.groupby(
            [frame.index.hour, frame.index.minute], sort=True
        ).mean(numeric_only=True)
        out = keyed.set_axis(
            [f"{h:02d}:{m:02d}" for h, m in keyed.index], axis=0
        ).rename_axis("time_of_day")
    else:
        raise ValueError(
            f"unknown view kind {kind!r} -- expected 'window', 'diurnal' or 'full'"
        )

    if columns:
        keep = [c for c in columns if c in out.columns]
        out = out[keep]
    return _shrink(out)


def _days(n: float):
    """Return ``n`` days as a pandas Timedelta."""
    import pandas as pd

    return pd.Timedelta(days=n)


def _shrink(frame):
    """Narrow dtypes so the CSV serializes tersely.

    Status and startup indicators are the bulk of a scheduling frame's columns
    and are 0/1 by construction, but arrive as floats from ``pyo.value``. Cast
    them so they land as ``0`` and ``1`` rather than ``0.0`` and ``1.0`` -- a
    third of the bytes, and it reads as the decision it is. Anything fractional
    is left alone, which is what happens under the LP relaxation.
    """
    import numpy as np

    out = frame.copy()
    for name in out.columns:
        col = out[name]
        if col.dtype.kind != "f" or col.isna().any():
            continue
        if np.array_equal(col.to_numpy(), col.to_numpy().astype(np.int8)):
            out[name] = col.astype(np.int8)
    return out


# --------------------------------------------------------------------------
# Provenance


def _flexpse_provenance() -> dict[str, str]:
    """Return the installed flex-pse version and, if resolvable, its commit.

    ``environment.yml`` installs flex-pse from ``git+...@main``, which pins
    nothing. Without the resolved commit a committed sweep cannot be reproduced
    or even dated against upstream, so dig it out of the ``direct_url.json`` pip
    leaves behind for a VCS install.
    """
    import importlib.metadata as md

    out = {"flexpse_version": "", "flexpse_commit": ""}
    try:
        dist = md.distribution("flex-pse")
    except md.PackageNotFoundError:
        return out
    out["flexpse_version"] = dist.version
    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            out["flexpse_commit"] = (
                json.loads(raw).get("vcs_info", {}).get("commit_id", "")
            )
    except (OSError, json.JSONDecodeError):
        pass
    return out


def _git(*args: str) -> str:
    """Return the output of a git command, or "" if git is unavailable."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_provenance(rows: list[dict], *, smoke: bool, wall: float) -> dict[str, str]:
    """Return the run-level facts written to ``provenance.csv``."""
    solvers = sorted({str(r.get("solver", "")) for r in rows} - {""})
    return {
        "generated_utc": datetime.now(timezone.utc).strftime(DATE_FORMAT + "Z"),
        "generator": "tools/sweep.py",
        "examples_commit": _git("rev-parse", "HEAD"),
        "examples_dirty": "yes" if _git("status", "--porcelain") else "no",
        **_flexpse_provenance(),
        "python_version": platform.python_version(),
        "platform": platform.platform(terse=True),
        "solver": ", ".join(solvers),
        "n_points": str(len(rows)),
        "total_wall_seconds": f"{wall:.1f}",
        "smoke": "yes" if smoke else "no",
        "notes": (
            "REDUCED INSTANCE -- not the published sweep"
            if smoke
            else "Solved at full size; see summary.csv for the per-point gap."
        ),
    }


# --------------------------------------------------------------------------
# The sweep


def run(example_dir: Path, *, smoke: bool = False, out_dir: Path | None = None) -> Path:
    """Solve one example across its sweep and write the site's data files.

    Args:
        example_dir: An example directory under ``examples/``.
        smoke: Pass ``smoke=True`` through to the adapter, which is expected to
            shrink the instance to something a CI runner can finish. The output
            is marked as reduced in ``provenance.csv`` so it can never be
            mistaken for a publishable sweep.
        out_dir: Override the output directory. Defaults to the
            ``public/<name>/`` the WebAssembly notebook reads.

    Returns:
        The directory written to.

    Raises:
        ValueError: If the adapter returns no points, or a view file exceeds
            :data:`MAX_VIEW_BYTES`.
    """
    import pandas as pd

    example_dir = example_dir.resolve()
    name = example_dir.name
    manifest = load_manifest(example_dir)
    sweep_cfg = manifest.get("sweep", {})
    adapter = load_adapter(example_dir)

    axis = sweep_cfg.get("axis", "value")
    columns = sweep_cfg.get("columns")
    views = sweep_cfg.get("views") or [{"name": "series", "kind": "full"}]
    out = out_dir or (example_dir / "public" / name)

    points = list(adapter.points(smoke=smoke))
    if not points:
        raise ValueError(f"{name}: points() returned nothing to sweep")

    label = "reduced sweep" if smoke else "sweep"
    print(f"{name}: {label} over {len(points)} point(s) of {axis!r}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    for view in views:
        (out / view["name"]).mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    ctx = adapter.setup(smoke=smoke)
    rows: list[dict] = []

    for i, point in enumerate(points):
        sweep_id = f"s{i:02d}"
        t0 = perf_counter()
        frame, summary = adapter.solve_point(ctx, point)
        elapsed = perf_counter() - t0

        for view in views:
            path = out / view["name"] / f"{sweep_id}.csv"
            reduce_view(frame, view, columns).to_csv(
                path, float_format=FLOAT_FORMAT, date_format=DATE_FORMAT
            )
            if path.stat().st_size > MAX_VIEW_BYTES:
                raise ValueError(
                    f"{_display(path)} is "
                    f"{path.stat().st_size / 1e6:.1f} MB, over the "
                    f"{MAX_VIEW_BYTES / 1e6:.0f} MB view limit. Reduce it with a "
                    f"narrower [[sweep.views]] or a shorter sweep.columns."
                )

        rows.append(
            {
                "sweep_id": sweep_id,
                "label": point["label"],
                axis: point.get(axis),
                **summary,
                "wall_seconds": round(elapsed, 2),
            }
        )
        print(
            f"  {sweep_id}  {point['label']:<34}  {elapsed:6.1f}s", flush=True
        )

    wall = perf_counter() - started
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, float_format="%.6g")
    pd.Series(build_provenance(rows, smoke=smoke, wall=wall)).rename_axis(
        "key"
    ).rename("value").to_csv(out / "provenance.csv")

    total = sum(p.stat().st_size for p in out.rglob("*.csv"))
    print(
        f"{name}: wrote {len(rows)} point(s) to "
        f"{_display(out)} ({total / 1024:.0f} KB) in {wall:.1f}s",
        flush=True,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run an example's sweep and write its site data files."
    )
    parser.add_argument("example", type=Path, help="path to an examples/<name> directory")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="solve a reduced instance (CI-sized); output is marked as reduced",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write here instead of the example's public/<name>/",
    )
    args = parser.parse_args(argv)
    run(args.example, smoke=args.smoke, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
