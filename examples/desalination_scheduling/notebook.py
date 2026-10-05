"""Marimo notebook for the from-code desalination model (`model.py`)."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pyomo.environ as pyo
    from pyomo.environ import units as pyunits

    try:
        here = Path(__file__).parent
    except NameError:  # pragma: no cover - interactive fallback
        here = Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import model

    return mo, model, pyo, pyunits


@app.cell
def _(mo, model):
    mo.md(rf"""
    # Desalination scheduling

    A seawater plant with three RO trains has to deliver a set volume of water
    each month. When it makes that water is up to the plant, and there's no
    storage.

    ```
                       ┌─► pretreatment[0] ─► RO[0] ─┬─► brine ─► ocean
    seawater ─► intake ─┼─► pretreatment[1] ─► RO[1] ─┤        (permeate)
                 pump   └─► pretreatment[2] ─► RO[2] ─┘            │
                                                                   ▼
      product water ◄─ product pump ◄─ post-treatment ◄─ permeate header
    ```

    A few things shape the schedule:

    - Each train is either off or at rated feed. The plant runs 3 trains, 2, or
      none. `ro[0]` and `ro[1]` always run together, and `ro[2]` is the swing train.
    - Going from 3 trains to 2 is free. Restarting from 0 is not: for 45 minutes
      post-treatment is down and all the water goes back to the ocean.
    - Energy costs 4.5× more from 4pm to 9pm, plus a \$21.50/kW demand charge
      on peak-window draw.

    The plant can make at most **{model.max_product_af():,.0f} AF** a month
    running flat out. The further below that the target sits, the more of the
    peak the plant can skip.

    The flowsheet is in [`model.py`](model.py). Only the tariff comes from
    `config.json`.
    """)
    return


@app.cell
def _(mo, model):
    demand_slider = mo.ui.slider(
        start=100,
        stop=280,
        step=5,
        value=model.DEMAND_AF_PER_MONTH,
        label="Water demand over the horizon (acre-feet)",
        show_value=True,
        full_width=True,
    )
    # 0 through 0.4: at 0 nothing is fixed and the solve is the plain exact
    # MILP, and past ~0.4 fixing starts cutting off feasible schedules outright.
    fix_slider = mo.ui.slider(
        steps=[0.0, 0.05, 0.1, 0.2, 0.4],
        value=model.FIX_TOL,
        label="Relax-and-fix tolerance (how close to 0/1 counts as decided)",
        show_value=True,
        full_width=True,
    )
    relax_switch = mo.ui.switch(value=False, label="LP relaxation only")
    run_button = mo.ui.run_button(label="Build and solve")

    mo.vstack([
        demand_slider,
        fix_slider,
        relax_switch,
        run_button,
        mo.md(rf"""
        - **Demand**: acre-feet delivered over the month. The slider stops at
          280, just under the {model.max_product_af():,.0f} AF ceiling. Changing it
          only re-solves; the model isn't rebuilt.
        - **Relax-and-fix tolerance**: solve the LP relaxation first, fix every
          on/off decision it already got within this distance of 0 or 1, then
          solve the MIP over what's left. At 0 nothing is fixed and it's the plain
          MILP. Higher is faster but can cost more; the relaxation gap in the
          results tells you how much. If fixing makes the month infeasible, the
          tolerance is stepped down automatically.
        - **LP relaxation only**: just the relaxation, as a lower bound. Its
          schedule isn't one the plant could actually run (trains at fractional
          feed), so leave this off unless you want the comparison.

        Nothing solves until you press the button.
                """),
    ])
    return demand_slider, fix_slider, relax_switch, run_button


@app.cell
def _(model):
    _cache = {}

    def built_model(relax_integrality):
        """The month, built once per setting of the relaxation switch.

        The demand is a mutable `Param` rather than a structural constant, so a
        new demand changes no constraint — `set_demand` retargets it and the
        same model goes back to the solver. Building the month is a few seconds
        against the relaxation's thirty-odd, but they are seconds spent
        rebuilding something that did not change.
        """
        key = bool(relax_integrality)
        if key not in _cache:
            _cache[key] = model.main(relax_integrality=key)
        return _cache[key]

    return (built_model,)


@app.cell
def _(
    built_model,
    demand_slider,
    fix_slider,
    mo,
    model,
    relax_switch,
    run_button,
):
    mo.stop(
        not run_button.value,
        mo.md("*Press **Build and solve** to run.*"),
    )

    # The build is cached and the demand is a Param, so a second press with a
    # new demand or a new tolerance goes straight to the solve. The tolerance is
    # not on the model at all — it is an argument to the routine, and
    # solve_relax_and_fix releases whatever the last press fixed before it
    # reads the relaxation again.
    with mo.status.spinner(title="Preparing the model…") as _spinner:
        m = built_model(relax_switch.value)
        model.set_demand(m, demand_slider.value)
        if m.is_relaxed:
            _spinner.update(title="Solving the relaxation…")
            results = model.solve_model(m)
        else:
            _spinner.update(
                title="Solving the relaxation, then the MIP over what "
                "it left undecided…"
            )
            results = model.solve_relax_and_fix(m, tol=fix_slider.value)
    return m, results


@app.cell
def _(m, mo, model, pyo, pyunits, results):
    tb = m.time_block
    plant = m.plant
    dt = pyo.value(pyunits.convert(tb.dt, pyunits.hr))

    demand_m3 = pyo.value(plant.potable.delivery_min)
    capacity_af = model.max_product_af()
    product_m3 = dt * sum(
        pyo.value(plant.potable.delivery[t]) for t in tb.time_index
    )
    power_kw = {
        t: pyo.value(m.costing.aggregate_power[t, "electrical"])
        for t in tb.time_index
    }
    energy_kwh = dt * sum(power_kw.values())
    # The fixed-duty intake pump and its pretreatment draw whether or not a skid
    # is running, so the plant's smallest draw is a floor no schedule can dodge.
    floor_kw = min(power_kw.values())
    # The demand charge bills against the largest draw inside 16:00-21:00.
    peak_kw = max(
        kw
        for t, kw in zip(tb.time_index, power_kw.values())
        if 16 <= tb.datetime_index[t].hour < 21
    )
    # ro[0] only, matching model.visualize. It is the whole RO system under the
    # symmetry breaking, and the only skid whose restart costs anything. Summing
    # all three would count ro[1] a second time -- the lead pair means it always
    # starts on the same step -- and count ro[2]'s free train-count steps as if
    # they were penalised restarts.
    restarts = sum(pyo.value(plant.ro[0].startup[t]) for t in plant.ro[0].startup)
    offspec_m3 = dt * sum(
        pyo.value(plant.permeate_split[i].flow_out_offspec[t])
        for i in plant.trains
        for t in tb.time_index
    )
    # What the obligation leaves on the table, in skid-hours: the water the
    # plant could have made and does not owe, divided by what one skid makes in
    # an hour at rated feed. This is the schedule's entire budget -- the hours it
    # gets to place wherever the tariff is cheapest.
    # RECOVERY_MAX, matching capacity_af: recovery is a degree of freedom, and
    # both ends of this subtraction have to be quoted at the same recovery or the
    # slack is measured in skid-hours the ceiling never counted.
    slack_skid_hours = (capacity_af - m.demand_af) * model.M3_PER_AF / (
        model.RATED_FEED_M3_PER_HR * model.RECOVERY_MAX
    )
    skid_hours = model.N_TRAINS * model.HORIZON_HOURS

    _mode = (
        "**LP relaxation** (a lower bound, not a real schedule)"
        if m.is_relaxed
        else "**relax and fix**"
    )
    # The two solves report different things, so the rows that describe the
    # heuristic only exist on the relax-and-fix path. On the ladder: fix_tol
    # below what was asked for means fixing at the asked-for tolerance came back
    # infeasible and the routine stepped down.
    if m.is_relaxed:
        _heuristic_rows = ""
    else:
        _stepped = (
            ""
            if len(m.fix_attempts) == 1
            else f" (stepped down from {m.fix_attempts[0][0]:g}, which was "
            f"infeasible; {len(m.fix_attempts)} tries)"
        )
        _heuristic_rows = f"""
    | Statuses fixed | {m.statuses_fixed:,} of {m.statuses_fixed + m.statuses_free:,} decided by the relaxation and fixed; {m.statuses_free:,} left to the MIP |
    | Fixing tolerance | {m.fix_tol:g}{_stepped} |
    | **Relaxation gap** | **{m.relaxation_gap:.3%}** vs. the \\${m.relaxed_objective:,.2f} lower bound, so the schedule is at most this far from optimal |"""
    # The bill is an EECO evaluation of the solved dispatch, never the
    # objective: the in-model cost is the relaxed, scalarized proxy the solver
    # optimizes over, and the demand charge in it is a proration.
    _cost = model.report_cost(m)
    mo.md(f"""
    ## Solved: {_mode}

    | | |
    |---|---|
    | Termination | `{results.solver.termination_condition}` |
    | MIP gap | {m.mip_gap:.3%} (only covers what fixing left free) |{_heuristic_rows}
    | **Electricity bill** | **\\${_cost.operating.electricity:,.2f}** for the month (`model.report_cost`) |
    | Objective | \\${pyo.value(m.objective):,.2f} (what the solver minimizes; not the bill) |
    | Demand | {m.demand_af:,.0f} AF = {demand_m3:,.0f} m³, {m.demand_af / capacity_af:.0%} of capacity |
    | Product water | {product_m3:,.0f} m³ |
    | Energy | {energy_kwh:,.0f} kWh = {energy_kwh / product_m3:.2f} kWh/m³ of product |
    | Peak-window draw | {peak_kw:,.0f} kW (billed by the demand charge) |
    | Floor draw | {floor_kw:,.0f} kW (intake + pretreatment, always on) |
    | RO restarts | {restarts:,.2f} |
    | Off-spec water dumped | {offspec_m3:,.0f} m³ |

    The plant makes exactly what it owes. Scheduling can't make a m³ cheaper to
    produce, only move it to a cheaper hour. At {m.demand_af:,.0f} AF the plant
    can leave **{slack_skid_hours:,.0f} of {skid_hours:,.0f} train-hours** idle,
    and it puts them in the peak window first.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The schedule

    Three days from the middle of the month. Shaded bands are the 4–9pm peak.
    Pass `start=` and `days=` to `model.visualize` to look at a different window.
    """)
    return


@app.cell
def _(m, model):
    frame = model.results_frame(m)
    model.visualize(frame)
    return (frame,)


@app.cell
def _(frame, mo, model):
    mo.vstack([
        mo.md(r"""
        ### Same window, as a table

        The full month is in `frame`.
        """),
        mo.ui.table(
            model.detail_window(frame).round(2).reset_index(),
            selection=None,
            page_size=24,
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
