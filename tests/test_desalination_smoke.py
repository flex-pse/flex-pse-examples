"""The desalination example still builds and solves -- at a size CI can afford.

The published example is a full calendar month at 15-minute resolution: ~27,000
binaries, and a *non-convex MIQCP*, because `construct_plant` unfixes RO recovery
and that makes ``permeate[t] == recovery * feed[t]`` bilinear. It needs Gurobi's
spatial branch and bound; HiGHS cannot take the problem at all. There is no
Gurobi licence on a public runner, so none of that can run in CI.

What runs here instead is a deliberately reduced instance:

* **a two-day horizon** rather than a month -- 192 steps instead of 2,976, still
  spanning the tariff's 16:00 and 21:00 boundaries and leaving room for a full
  three-step recuperation window;
* **recovery re-fixed** at its nominal value after the build, which linearizes
  the split and makes the problem a plain MILP that HiGHS can solve.

Both are applied from outside `model.py`, by monkeypatching the module constants
and re-fixing the variable, rather than by threading options through a
1,400-line module whose documentation is written around the month.

**What this does not cover**, and what therefore has no automated check at all:

* the non-convex path -- free recovery, Gurobi, ``NonConvex=2``, spatial B&B;
* ``SOLVER_OPTIONS["gurobi"]``, whose option names are only validated by a real
  Gurobi run;
* ``solve_relax_and_fix``'s fixing ladder at month scale, and its infeasibility
  branches;
* whether the 0.2 % MIP gap really holds the recuperation window to three steps
  *over a whole month* -- the exact failure the ``SOLVER_OPTIONS`` docstring says
  motivated 0.2 % over 1 %. Two days cannot reproduce it;
* the behaviour near the demand ceiling, where the month gets genuinely hard;
* **whether the committed sweep under `public/` still matches this model.**
  Regenerating it needs Gurobi, so that stays a human obligation.

A green run here means the flowsheet still builds and the scheduling logic still
holds together. It does not mean the published month is still right.
"""

import pytest

pytest.importorskip("flexops", reason="needs the flex-pse environment")

from conftest import EXAMPLES_ROOT, load_example_module

EXAMPLE = EXAMPLES_ROOT / "desalination_scheduling"

#: Two days at 15 minutes. Long enough for two full peak windows and a
#: recuperation window; short enough for a free runner.
SMOKE_START = "2026-07-01"
SMOKE_END = "2026-07-03"

#: Wall-clock ceiling for the solve. A smoke test that can run for half an hour
#: is not a smoke test, and a CI job that times out tells you nothing about why.
SOLVE_TIME_LIMIT_S = 300


@pytest.fixture(scope="module")
def model():
    """The example's ``model`` module, under a name unique to this example."""
    if not (EXAMPLE / "config.json").exists():
        pytest.skip("examples/desalination_scheduling/config.json is missing")
    return load_example_module(EXAMPLE)


@pytest.fixture(scope="module")
def solved(model):
    """A two-day, recovery-fixed month, solved with HiGHS."""
    from datetime import datetime

    import pyomo.environ as pyo

    hours = (
        datetime.fromisoformat(SMOKE_END) - datetime.fromisoformat(SMOKE_START)
    ).total_seconds() / 3600.0

    original = (model.START_DATE, model.END_DATE, model.HORIZON_HOURS)
    model.START_DATE, model.END_DATE, model.HORIZON_HOURS = (
        SMOKE_START,
        SMOKE_END,
        hours,
    )
    try:
        # Comfortably inside the ceiling: near it the problem gets hard, which is
        # the example's point but not this test's.
        m = model.main(demand_af=0.6 * model.max_product_af())

        # Re-fix what `construct_plant` unfixed. This is the whole reason HiGHS
        # can take the problem: with recovery fixed, `permeate == recovery * feed`
        # is linear and the month is a MILP rather than a non-convex MIQCP.
        for i in m.plant.trains:
            m.plant.ro[i].recovery.fix(model.RECOVERY)

        # Options matter here. Left to its defaults HiGHS chases a zero gap on a
        # unit-commitment MILP and will happily run for tens of minutes on two
        # days of schedule -- which is not a smoke test. 1% is loose enough to
        # finish quickly and tight enough that the recuperation window still
        # comes back at its constrained length.
        solver = pyo.SolverFactory("appsi_highs")
        solver.options["mip_rel_gap"] = 0.01
        solver.options["time_limit"] = SOLVE_TIME_LIMIT_S
        results = solver.solve(m)
        pyo.assert_optimal_termination(results)
        yield m, model.results_frame(m)
    finally:
        model.START_DATE, model.END_DATE, model.HORIZON_HOURS = original


def test_the_flowsheet_still_builds_and_solves(solved):
    """The model constructs against the installed flexPSE and reaches optimal.

    This is the assertion that catches upstream drift: `environment.yml` tracks
    flexPSE at `@main`, so a renamed unit-model option breaks this example with
    no commit landing in this repository.
    """
    _, frame = solved
    assert not frame.empty


def test_results_frame_has_the_columns_the_sweep_publishes(solved, model):
    """`example.toml`'s `sweep.columns` must all exist in the results frame.

    The sweep prunes to those columns silently, so a renamed column would show
    up as a chart that quietly lost a series rather than as an error.
    """
    import tomllib

    _, frame = solved
    manifest = tomllib.loads((EXAMPLE / "example.toml").read_text())
    missing = [c for c in manifest["sweep"]["columns"] if c not in frame.columns]
    assert not missing, f"sweep.columns names columns the model no longer returns: {missing}"


def test_delivers_the_water_it_owes(solved, model):
    """Product delivery meets the obligation the model was solved against.

    The obligation is a floor, not a target -- the plant may deliver more if
    that happens to be cheaper than throttling -- so this is a `>=`, with a
    tolerance for the solver's own noise on a sum of ~200 terms.
    """
    m, frame = solved
    step_hours = frame.index.to_series().diff().median().total_seconds() / 3600
    delivered_af = frame["product_m3_per_hr"].sum() * step_hours / model.M3_PER_AF
    assert delivered_af >= m.demand_af * (1 - 1e-6), (
        f"delivered {delivered_af:.3f} AF against an obligation of {m.demand_af:.3f} AF"
    )


def test_trains_run_as_a_lead_pair(solved):
    """Never exactly one train online.

    The skids are locked as a lead pair plus a swing, so the plant runs zero,
    two or three trains. A lone train means `register_parallel_group` stopped
    doing its job.
    """
    _, frame = solved
    assert 1 not in set(frame["trains_online"].round().astype(int))


def test_a_restart_costs_a_full_recuperation_window(solved, model):
    """Post-treatment is out for exactly `RECUP_STEPS` steps after a startup.

    This is the constraint the whole example exists to show, and the one the
    docstring for `SOLVER_OPTIONS` says a loose MIP gap quietly violates -- so
    assert the window's length rather than merely that it exists.
    """
    _, frame = solved
    starts = frame.index[frame["ro0_startup"].round() == 1]
    if len(starts) == 0:
        pytest.skip("this instance never restarts the RO system")

    post = frame["post_treatment_status"].round()
    for start in starts:
        position = frame.index.get_loc(start)
        window = post.iloc[position : position + model.RECUP_STEPS]
        assert (window == 0).all(), (
            f"post-treatment was online during the recuperation window at {start}"
        )


def test_no_permeate_reaches_the_header_while_post_treatment_is_out(solved):
    """With post-treatment out, *every* train's permeate goes to the outfall.

    This is the recuperation penalty, and the direction the model actually
    constrains: `permeate_to_header` is bounded by post-treatment's status, so a
    step with post-treatment off and any on-spec permeate would mean the
    recuperation window had come loose from the permeate split.

    The converse does **not** hold, and asserting it would give a flaky test.
    `offspec = permeate - permeate_to_header`, and nothing forbids diverting
    permeate while post-treatment runs. In practice that leaves a degenerate
    optimum: solving this instance at a 1%, a 0.1% and a 0% gap gives the same
    objective (7284.1) and the same total off-spec volume (1002.9 m3), but
    shuffles a single ~61 m3/hr diversion between different timestamps. The
    total is pinned by the recuperation windows; its attribution across steps is
    not.
    """
    _, frame = solved
    out = frame["post_treatment_status"].round() == 0
    if not out.any():
        pytest.skip("this instance never takes post-treatment out")
    assert frame.loc[out, "onspec_permeate_m3_per_hr"].max() < 1e-6, (
        "permeate reached the header while post-treatment was out"
    )
