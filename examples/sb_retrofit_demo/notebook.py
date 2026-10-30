"""Marimo notebook for the Santa Barbara retrofit demo.

Replays the committed, pre-solved results under ``public/sb_retrofit_demo/`` --
it never solves, so it opens instantly (made for recording). To re-solve both
cases, run ``python tools/sweep.py examples/sb_retrofit_demo`` (~20 min, HiGHS).

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

    frames, summary = plots.load_results()
    return diagram, frames, mo, plots, summary


@app.cell
def _(mo):
    mo.md(r"""
    # Santa Barbara desalination: before and after a flexibility retrofit

    The same plant delivers the same water for the month. **Before**, operators
    shed one RO skid in weekday peak hours. **After** the retrofit (+20% membrane
    area, and a battery sized to the load the plant can never shed), the
    optimizer drops most peak windows to near-zero grid draw and lowers the bill.
    """)
    return


@app.cell
def _(mo):
    import importlib

    _model_doc = importlib.import_module(
        "flex_pse_examples.examples.sb_retrofit_demo.model"
    )
    _c = _model_doc.CASES
    _kw, _kwh = _model_doc.battery_rating()

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
            + "\n\nShared: July 2026 at 15-min steps; fixed 1,063.5 m³/h intake; "
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
def _(mo, summary):
    _a, _b = summary.loc["A_before"], summary.loc["B_after"]
    mo.hstack(
        [
            mo.stat(f"${_b.bill_usd:,.0f}", label="Monthly bill after",
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
        value="2026-07-04", start=str(frames["A_before"].index[0].date()),
        stop=str((frames["A_before"].index[-1] - plots.pd.Timedelta(days=7)).date()),
        label="Window start (7 days)",
    )
    start
    return (start,)


@app.cell
def _(frames, plots, start, summary):
    plots.plot_before_after(frames, summary, start=str(start.value))
    return


@app.cell
def _(frames, plots, start):
    plots.plot_power_before_after(frames, start=str(start.value))
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
