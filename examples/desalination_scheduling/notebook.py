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
    # Desalination scheduling, built from code

    A seawater plant with three parallel treatment trains, scheduled against a
    time-of-use tariff. The flowsheet is written directly in
    [`model.py`](model.py) rather than assembled from `config.json` — only the
    tariff still comes from the config file, since it is ~80 EECO rate rows and
    there is no code form for it yet.

    ```
                       ┌─► pretreatment[0] ─► RO[0] ─┬─► brine ─► ocean
    seawater ─► intake ─┼─► pretreatment[1] ─► RO[1] ─┤        (permeate)
                 pump   └─► pretreatment[2] ─► RO[2] ─┘            │
                                                                   ▼
      product water ◄─ product pump ◄─ post-treatment ◄─ permeate header
    ```

    The plant owes a **volume** of product water over the month, and nothing says
    *when*. There is no product storage in the flowsheet, so the only way to
    dodge an expensive hour is to make less water in it and more water elsewhere.
    Each RO skid is off, or at its rated feed with recovery free inside the
    membrane window — there is no partial-load band, so the schedule is made of
    whole skids. And there are only **three plant states**, not four: three
    trains, two trains, or down. One skid alone is not a state the plant has,
    since the shared post-treatment step and product pump cannot be turned down
    that far, so `ro[0]` and `ro[1]` run as a locked lead pair and `ro[2]` is the
    one that swings.

    That leaves exactly two moves. Stepping the train count from three down to
    two is **free**. Shutting the *system* is not: for **45 minutes** after it
    comes back, post-treatment is out, and every running train's permeate — not
    just one skid's — leaves off-spec to the outfall while the plant pays full
    power to make it.

    The tariff is the base one in `config.json`: energy at 4.5× the off-peak
    price between **16:00 and 21:00**, plus a **\$21.50/kW** monthly demand
    charge on the peak-window draw.

    That obligation is the one thing you set here. It is a mutable `Param`, so
    the month is built once and every new demand costs only a re-solve. Where it
    sits against the plant's ceiling — **{model.max_product_af():,.0f} acre-feet**,
    what three skids at rated feed make if they never stop — is what decides
    whether there is a schedule to find at all: at the ceiling the plant runs
    flat out and the tariff is simply a bill, and the further below it the demand
    sits, the more of the peak window the optimizer can afford to sit out.
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
        > **On the slider.** The obligation, in acre-feet delivered downstream of
        > the product pump over the whole month — a volume, not a profile. The
        > default is {model.DEMAND_AF_PER_MONTH:,.0f} AF, and the slider stops at
        > 280 to keep you just under the plant's hard ceiling of
        > {model.max_product_af():,.0f} AF — three skids at rated feed for all
        > 744 hours. Go past that with `model.main(demand_af=...)` and the solver
        > reports an infeasible model, which is the honest answer rather than
        > something the model should paper over.
        >
        > Moving it does not rebuild the flowsheet. The demand enters the model
        > as a mutable `Param`, so a new setting changes no constraint — just
        > the right-hand side of the delivery balance — and `model.set_demand`
        > retargets it in place before the same model goes back to the solver.
        >
        > **On the tolerance.** With the switch off, the month is solved by
        > `model.solve_relax_and_fix`, which is two solves and a decision in
        > between. It relaxes every logic binary and solves that; then every
        > status the relaxation already decided — inside the tolerance of 0 or
        > 1 — is **fixed** there and stops being search space; then integrality
        > comes back and the MIP runs over what is left, which is the steps
        > where the LP was trading a fraction of a train against the tariff and
        > the plant has to pick a side. Of the {model.N_TRAINS + 1:,} × 2976
        > statuses, the default {model.FIX_TOL} typically hands about four
        > fifths of them to the solver already decided.
        >
        > It is a **heuristic**, and the table below reports it as one. The
        > relaxed objective is a valid lower bound on the exact cost, so
        > `m.relaxation_gap` — the returned schedule measured against that
        > bound — is what says how far off optimal it could be. `m.mip_gap` is
        > the gap of the *fixed* subproblem and says nothing about the statuses
        > that were fixed before it started.
        >
        > At **0** nothing is fixed and this is the plain exact MILP, with a
        > lower bound computed first. Turn it up and more of the month is
        > decided by an LP that is allowed to run a skid at 0.87 of rated feed,
        > which the plant cannot: the schedule stays feasible, but it can be
        > dearer, and the gap is where that shows. Far enough up and fixing cuts
        > off every feasible schedule — pin post-treatment on at a step and
        > `ro[0]` is forced on there *and* forbidden from restarting in the
        > preceding 45 minutes — so the routine steps the tolerance down and
        > fixes again rather than reporting an infeasible month.
        >
        > **On the switch.** Leave it off. It replaces the whole routine with a
        > single solve of the relaxation and stops there — the model is built
        > with `relax_integrality=True` and never gets its integrality back, so
        > it is the comparison case, not a schedule.
        >
        > The month is 2976 time points × 3 skids × (status, startup, shutdown)
        > ≈ 27,000 binaries, and `flexops.logic.relax` drops all of them to
        > `UnitInterval`. That is worth looking at once, because it does not
        > merely cost accuracy — it shows a **different plant**. A fractional
        > `status` scales a skid's feed with it, so the relaxed schedule never
        > shuts a train down: it dials all three to a fraction of rated feed and
        > the per-train permeate panel shows every skid trickling through what
        > should be a shutdown. A fractional `startup` then pays only a fraction
        > of the recuperation penalty. Read its cost as a **lower bound**, and
        > read its schedule as no schedule at all.
        >
        > Nothing solves until the button is pressed — moving the slider or
        > flipping the switch on its own is safe.
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
        mo.md("*Set the demand above and press **Build and solve** to run.*"),
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
                title="Solving — the relaxation first, then the MIP over what "
                "it left undecided…"
            )
            results = model.solve_relax_and_fix(m, tol=fix_slider.value)
    return m, results


@app.cell
def _(m, mo, model, pyo, pyunits, results):
    tb = m.time_block
    plant = m.plant
    dt = pyo.value(pyunits.convert(tb.dt, pyunits.hr))

    demand_m3 = pyo.value(m.demand_volume)
    capacity_af = model.max_product_af()
    product_m3 = dt * sum(
        pyo.value(plant.product_pump.outlet_state.flow_vol_phase[t, "Liq"])
        for t in tb.time_index
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
        pyo.value(plant.ro[i].permeate[t] - plant.permeate_to_header[i, t])
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
        "**LP relaxation** — an optimistic lower bound, not a runnable schedule"
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
            else f" — stepped down from {m.fix_attempts[0][0]:g}, which came "
            f"back infeasible ({len(m.fix_attempts)} rungs tried)"
        )
        _heuristic_rows = f"""
    | Statuses fixed | {m.statuses_fixed:,} of {m.statuses_fixed + m.statuses_free:,} decided by the relaxation and fixed; {m.statuses_free:,} left to the MIP |
    | Fixing tolerance | {m.fix_tol:g}{_stepped} |
    | **Relaxation gap** | **{m.relaxation_gap:.3%}** — the objective against the relaxation's \\${m.relaxed_objective:,.2f} lower bound. This is the number that bounds this schedule; the MIP gap above covers only the problem left after fixing |"""
    # The bill is an EECO evaluation of the solved dispatch, never the
    # objective: the in-model cost is the relaxed, scalarized proxy the solver
    # optimizes over, and the demand charge in it is a proration.
    _cost = model.report_cost(m)
    mo.md(f"""
    ## Solved: {_mode}

    | | |
    |---|---|
    | Termination | `{results.solver.termination_condition}` |
    | MIP gap | {m.mip_gap:.3%} — of the last solve, whose binaries are only the ones fixing left free |{_heuristic_rows}
    | **Electricity bill** | **\\${_cost.operating.electricity:,.2f}** over the month — EECO on the solved dispatch (`model.report_cost`), which is the cost |
    | Objective | \\${pyo.value(m.objective):,.2f} — the solve target, an in-model cost proxy. Not a bill |
    | Demand | {m.demand_af:,.0f} AF = {demand_m3:,.0f} m³, or {m.demand_af / capacity_af:.1%} of the {capacity_af:,.0f} AF the plant can make flat out |
    | Product water | {product_m3:,.0f} m³ against that obligation ({product_m3 / demand_m3:.2%}) |
    | Energy | {energy_kwh:,.0f} kWh = {energy_kwh / product_m3:.2f} kWh/m³ of product |
    | Peak-window draw | {peak_kw:,.0f} kW — what the \\$21.50/kW demand charge bills |
    | Floor draw | {floor_kw:,.0f} kW — the fixed-duty intake and its pretreatment, drawn whether or not a skid is running |
    | RO restarts | {restarts:,.2f} |
    | Off-spec permeate to brine | {offspec_m3:,.0f} m³ |

    The delivery constraint binds: the plant makes the water it owes and no more.
    What the schedule buys is a *cheaper hour*, not a cheaper m³ — the membranes
    cost the same energy per m³ wherever they run, and the fixed-duty intake
    costs its {floor_kw:,.0f} kW whether they run or not. That floor is why
    specific energy is not flat across the slider: it is the same bill spread
    over less water as the demand comes down.

    The demand is what says how many hours the schedule has to place. At
    {m.demand_af:,.0f} AF the plant can leave **{slack_skid_hours:,.0f} of its
    {skid_hours:,.0f} skid-hours** idle, and the peak window is the first place
    it puts them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The schedule itself

    The month is 2976 time points, far too dense to read a 45-minute
    recuperation window off, so `model.visualize` zooms into **three
    days** of it at the model's full 15-minute resolution. Shaded bands mark the
    daily peak window; reference lines are labelled out in the right margin —
    including the flat rate that would meet the demand you set, which is drawn
    off the frame rather than off the module default, so it moves with the
    slider.

    Pass `start=` and `days=` to move the window — the default opens on the
    14th day of the horizon.
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
        ### The solved schedule, row by row

        The table view of the same window — every decision variable the panels
        summarise. The full month is on `frame`.
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
