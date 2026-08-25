"""The sweep data contract, checked without a solver.

These are the fast gate: they need pandas and nothing else, so they run in
seconds on a bare runner and catch most of what goes wrong when someone adds an
example -- a manifest key left out, a sweep regenerated but not committed, an
orphaned CSV left behind by a shorter sweep, a blurb that grew into a paragraph.

The checks themselves live in ``tools/site/build.py`` so that the site build and
the test suite can never disagree about what a valid example is. These tests
drive those validators one example at a time, which is what turns a single
"the build failed" into a named failing test per example.
"""

import pytest

from conftest import example_dirs
from tools.site import build


@pytest.fixture(scope="module")
def discovered(examples_root):
    """Discover every example, and fail if discovery itself complained."""
    errors: list[str] = []
    examples = build.discover(examples_root, errors)
    assert not errors, "discovery failed:\n" + "\n".join(errors)
    return {example.name: example for example in examples}


def test_at_least_one_example(discovered):
    """A site with no examples is a bug, not an empty state."""
    assert discovered, "no examples/*/example.toml found"


@pytest.mark.parametrize(
    "example_dir", example_dirs(), ids=lambda p: p.name
)
def test_manifest_is_valid(example_dir, discovered):
    """Every manifest carries what the landing page needs."""
    errors: list[str] = []
    build.validate_manifest(discovered[example_dir.name], errors)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "example_dir", example_dirs(), ids=lambda p: p.name
)
def test_committed_sweep_matches_the_contract(example_dir, discovered):
    """The committed Parquet is present, readable and internally consistent."""
    errors: list[str] = []
    build.validate_data(discovered[example_dir.name], errors)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "example_dir", example_dirs(), ids=lambda p: p.name
)
def test_sweep_summary_is_ordered_and_finite(example_dir, discovered):
    """Sweep ids run s00, s01, ... with no gaps, and no cost is NaN.

    A gap means a point failed to solve and the sweep was committed anyway; the
    web notebook would then offer a case with no rows behind it.
    """
    import duckdb

    example = discovered[example_dir.name]
    path = example.data_dir / "summary.parquet"
    if not path.exists():
        pytest.skip(
            f"no committed sweep; run `python tools/sweep.py examples/{example.name}`"
        )
    summary = duckdb.sql(
        f"SELECT * FROM read_parquet('{path}') ORDER BY sweep_id"
    ).df()

    expected = [f"s{i:02d}" for i in range(len(summary))]
    assert list(summary["sweep_id"]) == expected, "sweep ids are not contiguous"

    for column in summary.select_dtypes("number").columns:
        assert summary[column].notna().all(), f"{column} has missing values"


@pytest.mark.parametrize("example_dir", example_dirs(), ids=lambda p: p.name)
def test_view_timestamps_are_typed(example_dir, discovered):
    """A view's time column keeps its type through the file.

    This is the thing Parquet buys that CSV could not promise: a `timestamp`
    column arrives as a timestamp rather than as text a reader has to remember to
    parse. If it ever regresses to a string the charts still draw, but the axis
    silently becomes categorical.
    """
    import duckdb

    example = discovered[example_dir.name]
    for view in example.manifest.get("sweep", {}).get("views", []):
        path = example.data_dir / f"{view['name']}.parquet"
        if not path.exists():
            pytest.skip("no committed sweep")
        types = dict(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).df()[["column_name", "column_type"]].values
        )
        assert "sweep_id" in types, f"{view['name']}.parquet has no sweep_id column"
        if "timestamp" in types:
            assert "TIMESTAMP" in types["timestamp"].upper(), (
                f"{view['name']}.parquet: timestamp is {types['timestamp']}, not a timestamp"
            )


@pytest.mark.parametrize(
    "example_dir", example_dirs(), ids=lambda p: p.name
)
def test_sweep_adapter_exposes_the_contract(example_dir):
    """The adapter defines the three names ``tools/sweep.py`` calls.

    Imported rather than inspected statically, because importing is what the
    runner does and an adapter that cannot import is as broken as one missing a
    function.

    Importing an adapter pulls in flexops via its sibling ``model``, so unlike
    the rest of this file it is not solver-free. It skips itself rather than
    being deselected by node id in the workflow -- a node id in YAML silently
    stops matching the day someone renames the test.
    """
    pytest.importorskip("flexops", reason="needs the flex-pse environment")

    from tools import sweep

    adapter = sweep.load_adapter(example_dir)
    for name in ("points", "setup", "solve_point"):
        assert callable(getattr(adapter, name)), f"{name} is not callable"

    points = adapter.points(smoke=False)
    assert points, "points() returned nothing"
    assert all("label" in p for p in points), "every point needs a 'label'"
    labels = [p["label"] for p in points]
    assert len(set(labels)) == len(labels), f"duplicate point labels: {labels}"
