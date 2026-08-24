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

    The same plant as `notebook.py`, but the flowsheet is written directly in
    [`model.py`](model.py) rather than assembled from
    `config.json`. Only the tariff still comes from the config file — it is
    ~80 EECO rate rows, and there is no code form for it yet.

    ```
                       ┌─► pretreatment[0] ─► RO[0] ─┬─► brine ─► ocean
    seawater ─► intake ─┼─► pretreatment[1] ─► RO[1] ─┤        (permeate)
                 pump   └─► pretreatment[2] ─► RO[2] ─┘            │
                                                                   ▼
      product water ◄─ product pump ◄─ post-treatment ◄─ permeate header
    ```

    The plant owes **{model.DEMAND_AF_PER_MONTH:.0f} acre-feet** of
    product water over the month, and nothing says *when*. There is no product
    storage in the flowsheet, so the only way to dodge an expensive hour is to
    make less water in it and more water elsewhere. Each RO skid is off, or
    running inside its turndown band; restarting one costs a **45-minute
    recuperation window** in which it runs at full power while its permeate is
    off-spec and goes to brine.

    The tariff is the base one in `config.json`: energy at 4.5× the off-peak
    price between **16:00 and 21:00**, plus a **\$21.50/kW** monthly demand
    charge on the peak-window draw.

    This notebook does one thing — build the model and solve it.
    """)
    return


@app.cell
def _(mo):
    relax_switch = mo.ui.switch(value=True, label="LP relaxation")
    run_button = mo.ui.run_button(label="Build and solve")

    mo.vstack([
        relax_switch,
        run_button,
        mo.md(r"""
        > **On the switch.** The month is 2976 time points × 3 skids ×
        > (status, startup, shutdown) ≈ 27,000 binaries, and the exact MILP
        > does not solve in a sitting — it will run to the 1800 s time limit
        > in `config.json` and report whatever gap it reached.
        > `flexops.logic.relax` drops those binaries to `UnitInterval`, which
        > solves in well under a minute. The relaxation is **optimistic** in
        > two ways: a fractional `status` lets a skid run below its turndown
        > floor, and a fractional `startup` pays only a fraction of the
        > recuperation penalty. Read its cost as a **lower bound**.
        >
        > Nothing solves until the button is pressed — flipping the switch on
        > its own is safe.
        """),
    ])
    return relax_switch, run_button


@app.cell
def _(mo, model, relax_switch, run_button):
    mo.stop(
        not run_button.value,
        mo.md("*Press **Build and solve** above to run.*"),
    )

    with mo.status.spinner(title="Building the model…") as _spinner:
        m = model.main(relax_integrality=relax_switch.value)
        _spinner.update(
            title="Solving…"
            + ("" if m.is_relaxed else " (exact MILP — this takes a while)")
        )
        results = model.solve_model(m)
    return m, results


@app.cell
def _(m, mo, model, pyo, pyunits, results):
    tb = m.time_block
    plant = m.plant
    dt = pyo.value(pyunits.convert(tb.dt, pyunits.hr))

    demand_m3 = pyo.value(
        pyunits.convert(
            model.DEMAND_AF_PER_MONTH * pyunits.acre * pyunits.foot,
            pyunits.m**3,
        )
    )
    product_m3 = dt * sum(
        pyo.value(plant.product_pump.outlet_state.flow_vol_phase[t, "Liq"])
        for t in tb.time_index
    )
    power_kw = {
        t: pyo.value(m.costing.aggregate_power[t, "electrical"])
        for t in tb.time_index
    }
    energy_kwh = dt * sum(power_kw.values())
    # The demand charge bills against the largest draw inside 16:00-21:00.
    peak_kw = max(
        kw
        for t, kw in zip(tb.time_index, power_kw.values())
        if 16 <= tb.datetime_index[t].hour < 21
    )
    restarts = sum(
        pyo.value(plant.ro[i].startup[t])
        for i in plant.trains
        for t in plant.ro[i].startup
    )
    offspec_m3 = dt * sum(
        pyo.value(plant.ro[i].permeate[t] - plant.permeate_to_header[i, t])
        for i in plant.trains
        for t in tb.time_index
    )

    _mode = (
        "**LP relaxation** — an optimistic lower bound, not a runnable schedule"
        if m.is_relaxed
        else "**exact MILP**"
    )
    mo.md(f"""
    ## Solved: {_mode}

    | | |
    |---|---|
    | Termination | `{results.solver.termination_condition}` |
    | Relative gap | {m.mip_gap:.3%} |
    | **Operating cost** | **\\${pyo.value(m.objective):,.2f}** over the month |
    | Product water | {product_m3:,.0f} m³ against an obligation of {demand_m3:,.0f} m³ ({product_m3 / demand_m3:.2%}) |
    | Energy | {energy_kwh:,.0f} kWh = {energy_kwh / product_m3:.2f} kWh/m³ of product |
    | Peak-window draw | {peak_kw:,.0f} kW — what the \\$21.50/kW demand charge bills |
    | RO restarts | {restarts:,.2f} |
    | Off-spec permeate to brine | {offspec_m3:,.0f} m³ |

    The delivery constraint binds: the plant is sized close to its obligation,
    so it makes the water it owes and no more. Every m³ costs the same specific
    energy by construction — what the schedule buys is a *cheaper hour*, not a
    cheaper m³.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The schedule itself

    The month is 2976 time points, far too dense to read a 45-minute
    recuperation window off, so `model.visualize` zooms into **three
    days** of it at the model's full 15-minute resolution. Shaded bands mark the
    daily peak window; reference lines are labelled out in the right margin.

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
