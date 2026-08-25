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
    """The committed CSVs are present, parseable and internally consistent."""
    errors: list[str] = []
    build.validate_data(discovered[example_dir.name], errors)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "example_dir", example_dirs(), ids=lambda p: p.name
)
def test_sweep_summary_is_ordered_and_finite(example_dir, discovered):
    """Sweep ids run s00, s01, ... with no gaps, and no cost is NaN.

    A gap means a point failed to solve and the sweep was committed anyway; the
    web notebook would then offer a case whose series file does not exist.
    """
    import pandas as pd

    example = discovered[example_dir.name]
    path = example.data_dir / "summary.csv"
    if not path.exists():
        pytest.skip(
            f"no committed sweep; run `python tools/sweep.py examples/{example.name}`"
        )
    summary = pd.read_csv(path)

    expected = [f"s{i:02d}" for i in range(len(summary))]
    assert list(summary["sweep_id"]) == expected, "sweep ids are not contiguous"

    for column in summary.select_dtypes("number").columns:
        assert summary[column].notna().all(), f"{column} has missing values"


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
