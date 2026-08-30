"""The pump scheduling example still solves, and still says what it claims.

This is the one example that runs end to end on a free CI runner: 24 hourly
steps, four to twelve small LPs and MILPs, HiGHS, a couple of seconds. It is
therefore also the example that catches upstream drift in flexPSE, which
``pyproject.toml`` tracks at ``@main`` and so does not pin at all.

The assertions are deliberately about the *claim* the example makes, not just
about it returning without raising. A model that solves to a schedule where
flexibility is worthless has not failed loudly, but it has stopped being this
example.
"""

import pytest

pytest.importorskip("flexops", reason="needs the flex-pse environment")

from conftest import EXAMPLES_ROOT, load_example_module

EXAMPLE = EXAMPLES_ROOT / "pump_scheduling"


@pytest.fixture(scope="module")
def model():
    """The example's ``model`` module, under a name unique to this example."""
    return load_example_module(EXAMPLE)


@pytest.fixture(scope="module")
def cfg(model):
    """The problem instance."""
    return model.load_config()


@pytest.fixture(scope="module")
def scenarios(model, cfg):
    """The config's 2x2, built and solved once for the whole module."""
    return model.run_scenarios(cfg)


def test_solves_every_case(scenarios):
    """All four cases solve; ``solve_model`` asserts optimality itself."""
    assert len(scenarios) == 4
    assert all(s["frame"] is not None for s in scenarios)


def test_flexibility_beats_storage_at_the_same_battery(scenarios):
    """The example's entire claim, as an assertion.

    For each battery size, running the feed pump flexibly must cost less than
    holding it at constant duty. If this ever fails the example is telling a
    different story than its README.
    """
    by_case = {(s["strategy"], s["sizing"]): s["operating_cost"] for s in scenarios}
    for sizing in ("large", "small"):
        assert by_case[("flexible", sizing)] < by_case[("inflexible", sizing)], (
            f"flexibility did not pay at the {sizing} battery"
        )


def test_a_small_battery_plus_flexibility_beats_a_large_one_without(scenarios):
    """The headline trade-off: flexibility substitutes for storage.

    Flexible plus the *small* battery must come in under inflexible plus the
    *large* one -- less storage, lower bill.
    """
    by_case = {(s["strategy"], s["sizing"]): s["operating_cost"] for s in scenarios}
    assert by_case[("flexible", "small")] < by_case[("inflexible", "large")]


def test_every_case_moves_the_same_water(scenarios, cfg):
    """Product delivery is the plant's fixed obligation, not a decision."""
    demand = sum(cfg["product_demand_m3_per_hr"])
    for scenario in scenarios:
        delivered = scenario["frame"]["product_flow_m3_per_hr"].sum()
        assert delivered == pytest.approx(demand, rel=1e-6), scenario["label"]


def test_tank_stays_inside_its_bounds(scenarios, cfg):
    """The tank never over- or under-fills.

    A solver that reports optimal while violating a bound is the failure this
    catches -- rare, but silent, and it would flow straight into a published
    sweep.
    """
    tank = cfg["tank"]
    low = tank["level_min"] * tank["max_volume_m3"]
    high = tank["level_max"] * tank["max_volume_m3"]
    for scenario in scenarios:
        volume = scenario["frame"]["tank_volume_m3"]
        assert volume.min() >= low - 1e-6, scenario["label"]
        assert volume.max() <= high + 1e-6, scenario["label"]


def test_flexible_pump_respects_its_turndown_band(scenarios, cfg):
    """A flexible pump is off, or inside [min_flow_fraction, 1] x rated flow.

    This is the semicontinuous constraint ``flexops.logic.add_status`` adds; a
    flow strictly between zero and the turndown floor means the binary went
    fractional, which is what an accidentally relaxed model looks like.
    """
    rated = cfg["feed_pump"]["rated_flow_m3_per_hr"]
    floor = cfg["feed_pump"]["min_flow_fraction"] * rated
    for scenario in scenarios:
        if not scenario["flexible"]:
            continue
        for flow in scenario["frame"]["feed_flow_m3_per_hr"]:
            assert flow <= 1e-6 or flow >= floor - 1e-6, (
                f"{scenario['label']}: {flow:.3f} m3/hr is inside the turndown band"
            )


def test_smoke_sweep_runs_and_writes_the_contract(tmp_path):
    """``tools/sweep.py`` produces a valid sweep for this example.

    Runs the reduced sweep into a temporary directory rather than over the
    committed one, so CI can exercise the whole generation path without ever
    proposing a change to published data.
    """
    import duckdb

    from tools import sweep

    out = sweep.run(EXAMPLE, smoke=True, out_dir=tmp_path)

    summary_path = out / "summary.parquet"
    assert summary_path.exists() and summary_path.stat().st_size > 0
    assert (out / "provenance.parquet").exists()

    frame = duckdb.sql(f"SELECT * FROM read_parquet('{summary_path}')").df()
    assert len(frame) == 4, "the reduced sweep is the config's 2x2"
    assert set(frame["strategy"]) == {"flexible", "inflexible"}

    # Every point in the summary has rows in the view, and nothing else does.
    series_path = out / "series.parquet"
    assert series_path.exists()
    in_series = {
        row[0]
        for row in duckdb.sql(
            f"SELECT DISTINCT sweep_id FROM read_parquet('{series_path}')"
        ).fetchall()
    }
    assert in_series == set(frame["sweep_id"])
