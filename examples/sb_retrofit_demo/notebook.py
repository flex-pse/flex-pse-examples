"""Marimo notebook for the Santa Barbara retrofit demo.

Opens on the committed, pre-solved week under ``public/sb_retrofit_demo/`` so it
renders instantly. **Solve both cases** re-runs the model live with HiGHS and
streams the solver log into the cell; the charts and KPIs then switch to the
live results. A week takes a few minutes; two or three days take seconds.

    marimo run examples/sb_retrofit_demo/notebook.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    try:
        here = Path(__file__).parent
    except NameError:  # pragma: no cover - interactive fallback
        here = Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import diagram
    import plots

    import importlib

    import pandas as pd

    model = importlib.import_module("flex_pse_examples.examples.sb_retrofit_demo.model")
    stored_frames, stored_summary = plots.load_results()
    return diagram, mo, model, pd, plots, stored_frames, stored_summary


@app.cell
def _(mo):
    mo.md(r"""
    # Santa Barbara desalination: before and after a flexibility retrofit

    The same plant delivers the same water over the same week. **Before**, operators
    shed one RO skid in weekday peak hours. **After** the retrofit (+20% membrane
    area, and a battery sized to the load the plant can never shed), the
    optimizer drops most peak windows to near-zero grid draw and lowers the bill.
    """)
    return


@app.cell
def _(mo, model):
    _c = model.CASES
    _kw, _kwh = model.battery_rating()

    def _row(label, a, b):
        return f"| {label} | {a} | {b} |"

    _rows = [
        _row("Membranes per skid", f"{_c['A_before']['vessels_per_skid']} vessels",
             f"{_c['B_after']['vessels_per_skid']} vessels (+20% area)"),
        _row("Feed per skid (m³/h)", f"off or {_c['A_before']['rated_feed']:.0f}",
             f"off or {_c['B_after']['min_feed']:.0f}–{_c['B_after']['rated_feed']:.0f}"),
        _row("RO energy (kWh/m³ permeate)", f"{_c['A_before']['ro_intensity']:.2f}",
             f"{_c['B_after']['ro_intensity']:.2f} (−4%)"),
        _row("Recovery", f"fixed {_c['A_before']['recovery_fixed']:.1%}", "free, 40–50%"),
        _row("Battery", "none", f"{_kw:.0f} kW / {_kwh:,.0f} kWh"),
        _row("Schedule", "operators shed skid 3, weekdays 16:00–21:00",
             "optimized (relax-and-fix MILP, HiGHS)"),
        _row("Water target", "whatever its schedule makes", "exactly case A's water"),
    ]
    mo.accordion({
        "Assumptions": mo.md(
            "| | Before | After |\n|---|---|---|\n" + "\n".join(_rows)
            + "\n\nShared: 4–11 July 2026 at 15-min steps; fixed 1,063.5 m³/h intake; "
            "45-min post-treatment restart; tariff with a 16:00–21:00 peak "
            "($0.45 vs $0.10/kWh, $21.50/kW demand charge)."
        )
    })
    return


@app.cell
def _(mo):
    period = mo.ui.radio(
        options={"Off-peak": "offpeak", "Peak (16:00–21:00)": "peak"},
        value="Peak (16:00–21:00)",
        label="Plant state",
        inline=True,
    )
    period
    return (period,)


@app.cell
def _(diagram, mo, period):
    mo.ui.tabs({
        "Before retrofit": mo.Html(diagram.render_named("A_before", period.value)),
        "After retrofit": mo.Html(diagram.render_named("B_after", period.value)),
    })
    return


@app.cell
def _(mo):
    mo.md("""## Solve it""")
    return


@app.cell
def _(mo, stored_frames):
    _first = stored_frames["A_before"].index[0]
    solve_start = mo.ui.date(value=str(_first.date()), label="Start")
    solve_days = mo.ui.slider(2, 7, value=7, step=1, label="Days", show_value=True)
    solve_gap = mo.ui.dropdown(
        options={"1%": 0.01, "3%": 0.03, "5%": 0.05}, value="3%", label="MIP gap"
    )
    solve_button = mo.ui.run_button(label="Solve both cases")
    mo.hstack([solve_start, solve_days, solve_gap, solve_button], justify="start", gap=2)
    return solve_button, solve_days, solve_gap, solve_start


@app.cell
def _(
    mo,
    model,
    pd,
    solve_button,
    solve_days,
    solve_gap,
    solve_start,
    stored_frames,
    stored_summary,
):
    if not solve_button.value:
        frames, summary = stored_frames, stored_summary
        mo.output.replace(
            mo.callout(
                mo.md("Showing the **precomputed** week. Press **Solve both cases** to "
                      "re-run it live; the HiGHS log streams below."),
                kind="info",
            )
        )
    else:
        _start = pd.Timestamp(solve_start.value)
        _end = _start + pd.Timedelta(days=solve_days.value)
        model.SOLVER_OPTIONS["highs"]["mip_rel_gap"] = solve_gap.value
        with mo.status.spinner(title="Solving before and after with HiGHS…"):
            _results = model.run_both(start=str(_start.date()), end=str(_end.date()))
        frames = {k: v[1] for k, v in _results.items()}
        summary = pd.DataFrame({k: v[2] for k, v in _results.items()}).T
        _b = summary.loc["B_after"]
        mo.output.replace(
            mo.callout(
                mo.md(f"Solved live: {solve_days.value} days from {_start:%b %d}, "
                      f"case B in {_b.solve_seconds:,.0f} s, MIP gap {_b.mip_gap:.1%}."),
                kind="success",
            )
        )
    return frames, summary


@app.cell
def _(mo, summary):
    _a, _b = summary.loc["A_before"], summary.loc["B_after"]
    mo.hstack(
        [
            mo.stat(f"${_b.bill_usd:,.0f}", label="Bill after",
                    caption=f"from ${_a.bill_usd:,.0f} before", direction="decrease"),
            mo.stat(f"{_b.savings_pct:.0f}%", label="Bill savings"),
            mo.stat(f"{_b.peak_grid_kw_mean:,.0f} kW", label="Mean peak-window draw",
                    caption=f"from {_a.peak_grid_kw_mean:,.0f} kW", direction="decrease"),
            mo.stat(f"{_b.load_shed_pct:.0f}%", label="Peak load shed",
                    caption="vs. the plant's average load before"),
            mo.stat(f"{_b.water_af:,.1f} AF", label="Water delivered",
                    caption=f"same as before ({_a.water_af:,.1f} AF)"),
        ],
        justify="space-between",
    )
    return


@app.cell
def _(frames, mo, plots):
    start = mo.ui.date(
        value=str(frames["A_before"].index[0].date()),
        start=str(frames["A_before"].index[0].date()),
        stop=str(frames["A_before"].index[-1].date()),
        label="Window start",
    )
    start
    return (start,)


@app.cell
def _(frames, plots, start, summary):
    plots.plot_before_after(frames, summary, start=str(start.value))
    return


@app.cell
def _(mo, plots):
    import duckdb

    _prov = duckdb.sql(
        f"SELECT * FROM read_parquet('{plots.DATA / 'provenance.parquet'}')"
    ).df()
    mo.accordion({"Provenance": mo.ui.table(_prov, selection=None)})
    return


if __name__ == "__main__":
    app.run()
