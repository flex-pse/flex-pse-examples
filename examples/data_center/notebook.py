"""Data center under a grid cap: builds and solves the model locally."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from functools import lru_cache
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    try:
        here = Path(__file__).parent
    except NameError:
        here = Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import model

    cfg = model.load_config()

    @lru_cache(maxsize=32)
    def solve_cases(curtailed, event_hours, depth, firm_mw, battery_mw, full_mw):
        """Solve the three cases once per distinct set of inputs."""
        c = {
            **cfg,
            "data_center": {**cfg["data_center"], "full_load_mw": full_mw},
            "grid": {
                **cfg["grid"],
                "curtailed_fraction": curtailed,
                "event_hours": event_hours,
                "curtailment_depth": depth,
                "firm_mw": firm_mw,
            },
        }
        return c, model.compare(c, battery_mw)

    return cfg, mo, model, plt, solve_cases


@app.cell
def _(mo):
    mo.md(
        "# Is a battery worth it under a grid cap?\n\n"
        "A data center asks its utility for a large block of power. The utility "
        "guarantees part of it all the time; the rest can be curtailed. Change "
        "the variables below to see how the answer moves over a year. Everything "
        "held fixed is listed under Assumptions."
    )
    return


@app.cell
def _(cfg, mo):
    g, dc = cfg["grid"], cfg["data_center"]
    full_mw = mo.ui.number(
        1, 1000, step=1, value=dc["full_load_mw"], label="Full power request (MW)"
    )
    firm_mw = mo.ui.number(
        0, 1000, step=1, value=g["firm_mw"], label="Firm service (MW)"
    )
    how_often = mo.ui.dropdown(
        options={
            "Rarely (0.5% of the time)": 0.005,
            "Seldom (1%)": 0.01,
            "Sometimes (2%)": 0.02,
            "Often (5%)": 0.05,
            "Very often (10%)": 0.10,
            "Daily, about 4 hours (17%)": 0.17,
        },
        value="Seldom (1%)",
        label="Share of time curtailed",
    )
    event_hours = mo.ui.number(
        1, 24, step=1, value=g["event_hours"], label="Curtailment event length (hours)"
    )
    depth = mo.ui.number(
        0,
        100,
        step=5,
        value=g["curtailment_depth"] * 100,
        label="Curtailment depth (% of the curtailable block cut)",
    )
    battery = mo.ui.dropdown(
        options={f"{mw} MW": mw for mw in (5, 10, 15, 20, 30)},
        value="10 MW",
        label="Battery to add",
    )
    mo.vstack(
        [
            mo.md("## Variables"),
            mo.hstack([full_mw, firm_mw], justify="start", gap=2),
            mo.hstack([how_often, event_hours], justify="start", gap=2),
            mo.hstack([depth, battery], justify="start", gap=2),
            mo.md(
                "_Defaults lean pessimistic compared with the Duke study of flexible "
                "data center load._ Duke assumes curtailment 0.25-1% of the time, with "
                "events averaging about 2.5 hours at 1%, and finds that in most "
                "curtailment hours at least half of the new load keeps running. We use "
                "1% (the top of its range), 3-hour events, and a 75% cut. The power "
                "request and firm service are made up."
            ),
        ]
    )
    return battery, depth, event_hours, firm_mw, full_mw, how_often


@app.cell
def _(cfg, mo):
    _dc, _g, _b = cfg["data_center"], cfg["grid"], cfg["battery"]
    _rows = [
        # (group, assumption, value, status, source / note)
        ("Revenue", "Revenue per kWh of power drawn (R)", f"${_dc['revenue_per_unit']:.2f}", "Sourced",
         "Two independent estimates agree: CoreWeave Q3 2026 revenue per MW of active power ($0.92-$1.04) "
         "and IREN's $9.7B five-year Microsoft contract for 200 MW ($0.97). Low case $0.85, high case $2.00 "
         "(IREN's newer deals). From news coverage of company filings, not the filings themselves"),
        ("Revenue", "Compute is not the limit; power is", "yes", "Assumed",
         "Every kWh the site can draw earns R. Demand is unlimited"),
        ("Site", "Cost of owning compute capacity",
         f"${_dc['compute_cost_per_mw_year'] / 1e6:.2f}M per MW per year", "Sourced",
         "A fixed yearly charge for each MW of power the site is built to draw, paid whether or not the "
         "machines run (like rent). It covers servers, networking, building and non-energy running costs, "
         "with the upfront cost spread over each item's life. A 50 MW site pays 50 times this. Electricity "
         "is billed separately. From Epoch AI's 1 GW data center cost model: $7.9M per MW of IT power, "
         "divided by their cooling factor of 1.14 to get cost per MW at the meter"),
        ("Site", "Compute is built for the power the site can use", "full request, or firm MW only", "Assumed",
         "If the site accepts curtailment it builds for the full request; if not, only for the firm MW"),
        ("Site", "Power draw", "full contracted power at all times", "Assumed",
         "Whatever capacity the site takes, its GPUs draw it all the time. So kWh drawn = MW x hours, and R per kWh drawn "
         "equals R per hour of capacity. Real draw is probably around 70% of capacity, so this slightly overstates the "
         "electricity bill, which is about 10% of revenue"),
        ("Site", "Load range", "0 to full request", "Assumed",
         "No minimum level; the site can throttle to any load"),
        ("Grid", "Curtailment events", "evenly spaced", "Assumed",
         "Spread evenly across the year"),
        ("Grid", "Energy price", f"${_g['energy_price_per_kwh']:.2f}/kWh", "Made up", "Flat, no demand charges"),
        ("Battery", "Upfront cost", f"${_b['capex_per_kwh']:.0f}/kWh", "Sourced",
         "NREL 2025 update, 4-hour utility-scale Li-ion, 2024 dollars"),
        ("Battery", "Upkeep per year", f"{_b['fom_fraction_of_capex_per_year']:.0%} of upfront", "Sourced",
         "NREL, includes replacing capacity as it degrades"),
        ("Battery", "Lifetime", f"{_b['lifetime_years']} years", "Sourced", "NREL"),
        ("Battery", "Discount rate", f"{_b['discount_rate']:.0%}", "Assumed",
         "Yearly return the money spent on the battery has to earn. Used to spread the upfront price over its life, "
         "so a higher rate means a higher yearly battery cost. Our own assumption, not from NREL"),
        ("Battery", "Storage duration", f"{_b['storage_duration_hours']:g} h", "Sourced", "NREL's 4-hour system"),
        ("Battery", "Round-trip efficiency", f"{_b['eta_charge'] * _b['eta_discharge']:.0%}",
         "Sourced",
         "NREL 2025: 85% for utility-scale Li-ion, split evenly into charge and discharge. "
         "Field data runs lower (EIA's 2019 US fleet averaged 82%). "
         "No separate figure found for data centers or factories"),
    ]
    _table = "\n".join(f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in _rows)
    mo.md(
        "## Assumptions\n\n"
        "| Group | Assumption | Value | Status | Source or note |\n"
        "|---|---|---|---|---|\n" + _table
    )
    return


@app.cell
def _(battery, depth, event_hours, firm_mw, full_mw, how_often, solve_cases):
    cfg2, cases = solve_cases(
        how_often.value,
        event_hours.value,
        depth.value / 100,
        firm_mw.value,
        battery.value,
        full_mw.value,
    )
    return cases, cfg2


@app.cell
def _(cases, mo):
    base, accept, with_batt = cases
    rows = "\n".join(
        f"| {c['case']} | ${c['net_profit']:,.0f} | "
        f"${c['net_profit'] - base['net_profit']:+,.0f} | {c['curtailed_fraction']:.1%} |"
        for c in cases
    )
    gain_accept = accept["net_profit"] - base["net_profit"]
    gain_batt = with_batt["net_profit"] - accept["net_profit"]
    verdict = (
        f"Accepting curtailment {'adds' if gain_accept >= 0 else 'costs'} "
        f"**${abs(gain_accept):,.0f}** a year compared with firm service only. "
        f"The battery then {'adds' if gain_batt >= 0 else 'costs'} "
        f"**${abs(gain_batt):,.0f}** "
        f"({'worth it' if gain_batt > 0 else 'not worth it'} at this battery cost)."
    )
    mo.md(
        "| Case (one year) | Net profit | vs. firm only | Energy curtailed |\n"
        "|---|---|---|---|\n" + rows + "\n\n" + verdict
    )
    return


@app.cell
def _(cases, cfg2, plt):
    frame = cases[2]["frame"].iloc[: 24 * 7]
    _dc, _g = cfg2["data_center"], cfg2["grid"]
    cap_mw = _dc["full_load_mw"] - _g["curtailment_depth"] * (
        _dc["full_load_mw"] - _g["firm_mw"]
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(frame.index, frame["load_kw"] / 1000, label="Site load")
    ax.plot(frame.index, frame["grid_kw"] / 1000, label="Grid draw", linestyle="--")
    ax.axhline(cap_mw, color="red", linewidth=0.8, label="Grid cap during an event")
    ax.set_ylabel("MW")
    ax.set_title("First week of the year, with the battery")
    ax.legend()
    fig
    return


@app.cell
def _(cfg2, model, plt):
    sweep = model.sweep(cfg2)
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(
        [r["battery_mw"] for r in sweep],
        [r["net_profit"] / 1e6 for r in sweep],
        marker="o",
    )
    ax2.set_xlabel("Battery power (MW)")
    ax2.set_ylabel("Net profit ($M, one year)")
    ax2.set_title("Net profit vs battery size")
    fig2
    return


if __name__ == "__main__":
    app.run()
