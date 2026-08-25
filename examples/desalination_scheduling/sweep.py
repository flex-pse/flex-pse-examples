"""Sweep adapter for desalination_scheduling. Driven by `tools/sweep.py`.

The plant owes a *volume* of product water over the month rather than an hourly
profile, and has no storage, so the only lever it has is when to make water. How
much room that leaves is entirely a function of where the obligation sits against
`model.max_product_af()` -- the ~305 acre-feet three skids at rated feed and top
recovery would make if they never stopped. Near the ceiling the plant runs flat
out and there is nothing to schedule; well below it, the optimizer can afford to
sit out the whole peak window.

So the sweep is that ratio, and the page it feeds is a picture of the plant
running out of slack. Each point is a full calendar month at 15-minute
resolution -- ~27,000 binaries -- solved exactly through
`model.solve_relax_and_fix`, which takes about ten seconds a point with Gurobi.
"""

import model

#: Where each point's obligation sits against `model.max_product_af()`, which is
#: every skid at rated feed *and top recovery* for every step, never stopping.
#:
#: The range stops at 0.87 -- the example's own default of 265 AF -- rather than
#: reaching for the ceiling, and that bound is about solve time, not taste.
#: `max_product_af` is quoted at `RECOVERY_MAX`, so past roughly 0.9 the month is
#: only feasible with recovery pinned near the top of its window everywhere at
#: once, and the search gets very hard very quickly: 0.84 ran past ten minutes
#: without closing, against twelve seconds for every point at or below 0.78. Those
#: points are a corner of the feasible set rather than a schedule anyone would
#: run, and pricing them costs hours per regeneration.
#:
#: :data:`TIME_LIMIT_S` is the backstop if a point still runs long. A point that
#: hits it is reported with its achieved gap in `summary.csv` rather than
#: silently accepted.
FRACTIONS = (0.50, 0.58, 0.66, 0.72, 0.78, 0.83, 0.87)

#: The reduced sweep for CI: the two ends and the knee between them.
SMOKE_FRACTIONS = (0.58, 0.78)

#: Per-point wall-clock ceiling. The module default is 1800 s, which is a
#: reasonable backstop for one deliberate month but turns a seven-point sweep
#: into an afternoon.
TIME_LIMIT_S = 420


def points(*, smoke: bool = False) -> list[dict]:
    """Return the sweep points: monthly obligations, as acre-feet."""
    ceiling = model.max_product_af()
    fractions = SMOKE_FRACTIONS if smoke else FRACTIONS
    return [
        {
            "label": f"{fraction:.0%} of capacity ({fraction * ceiling:.0f} AF)",
            "demand_af": round(fraction * ceiling, 1),
            "capacity_fraction": fraction,
        }
        for fraction in fractions
    ]


def setup(*, smoke: bool = False):
    """Build the month once, and bound how long any one point may take.

    The obligation is a mutable `Param`, so every point after the first is a
    re-solve rather than a rebuild -- which is most of why this sweep is minutes
    rather than an afternoon. This is the same trick the solver notebook's demand
    slider relies on.
    """
    # A sweep is many months, not one, so the module's 1800 s backstop is far too
    # generous. Narrow it here rather than in `model.py`, where it is the right
    # value for someone solving a single month on purpose.
    for options in model.SOLVER_OPTIONS.values():
        for key in ("TimeLimit", "time_limit"):
            if key in options:
                options[key] = TIME_LIMIT_S

    return model.main(demand_af=points(smoke=smoke)[0]["demand_af"])


def solve_point(m, point: dict):
    """Retarget the obligation and re-solve the month.

    Args:
        m: The model from :func:`setup`, carried across points.
        point: One dict from :func:`points`.

    Returns:
        ``(frame, summary)`` -- the schedule indexed by timestamp, and the
        scalars that become this point's row in ``summary.csv``.
    """
    model.set_demand(m, point["demand_af"])
    results = model.solve_relax_and_fix(m)

    report = model.report_cost(m)
    frame = model.results_frame(m)
    step_hours = frame.index.to_series().diff().median().total_seconds() / 3600

    # Every RO startup is a plant restart, and every restart costs a 45-minute
    # window in which post-treatment is out and all three trains' permeate goes
    # to the outfall at full power. Counting both is the point of the example.
    restarts = round(sum(frame[f"ro{i}_startup"].sum() for i in range(3)))
    offspec_af = float(
        frame["offspec_permeate_m3_per_hr"].sum() * step_hours / model.M3_PER_AF
    )
    delivered_af = float(
        frame["product_m3_per_hr"].sum() * step_hours / model.M3_PER_AF
    )

    return frame, {
        "capacity_fraction": point["capacity_fraction"],
        "delivered_af": delivered_af,
        "objective": float(m.objective()),
        "operating_cost": float(report.operating.total),
        "electricity_cost": float(report.operating.electricity),
        "peak_kw": float(frame["grid_kw"].max()),
        "restarts": restarts,
        "offspec_af": offspec_af,
        "peak_window_kwh": float(
            frame.loc[
                frame["energy_price"] > frame["energy_price"].min(), "grid_kw"
            ].sum()
            * step_hours
        ),
        "solver": model.SOLVER,
        # Report what the solver actually said. Points near the top of the range
        # hit TIME_LIMIT_S and come back `maxTimeLimit` with a feasible schedule
        # and a non-zero gap -- which is a fine thing to publish, and a terrible
        # thing to publish as "optimal".
        "termination": str(results.solver.termination_condition),
        "mip_gap": float(getattr(m, "mip_gap", 0.0) or 0.0),
        # The relaxation is the valid lower bound; `mip_gap` is the gap of the
        # fixed subproblem, which is optimistic once statuses have been pinned.
        "relaxation_gap": float(getattr(m, "relaxation_gap", 0.0) or 0.0),
    }
