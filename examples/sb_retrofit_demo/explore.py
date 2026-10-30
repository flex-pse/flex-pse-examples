# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "duckdb"]
# ///
"""Santa Barbara retrofit demo: the WebAssembly page.

Runs in the browser, so it imports only what Pyodide has: no pyomo, no flexops,
no sibling modules. It shows the committed figures and flowsheets and reads the
KPIs from ``summary.parquet`` with DuckDB. See CONTRIBUTING.md.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import marimo as mo

    EXAMPLE = "sb_retrofit_demo"
    DATA = f"{mo.notebook_location()}/public/{EXAMPLE}"
    return DATA, duckdb, mo


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
        # Santa Barbara retrofit: before and after

        One week of a seawater desalination plant, twice, delivering the same
        water. **Before**: the operators shed one of three RO skids on weekday
        peaks. **After**: +20% membrane area and a battery for the load the plant
        can never shed, with the schedule optimized.
        """),
        mo.callout(
            mo.md(
                "These results are **precomputed**: both weeks were solved ahead "
                "of time with HiGHS, and this page replays them. For the real "
                "model, see `notebook.py` and `model.py`."
            ),
            kind="info",
        ),
    ])
    return


@app.cell
def _(DATA, duckdb, mo):
    _s = duckdb.sql(
        f"SELECT \"case\", bill_usd, savings_pct, peak_grid_kw_mean, load_shed_pct, water_af "
        f"FROM read_parquet('{DATA}/summary.parquet') ORDER BY sweep_id"
    ).df().set_index("case")
    _a, _b = _s.loc["A_before"], _s.loc["B_after"]
    mo.hstack([
        mo.stat(f"${_b.bill_usd:,.0f}", label="Bill for the week, after",
                caption=f"from ${_a.bill_usd:,.0f}"),
        mo.stat(f"{_b.savings_pct:.0f}%", label="Bill savings"),
        mo.stat(f"{_b.load_shed_pct:.0f}%", label="Peak load shed"),
        mo.stat(f"{_b.water_af:,.1f} AF", label="Water delivered", caption="same in both"),
    ], justify="space-between")
    return


@app.cell
def _(DATA, mo):
    mo.vstack([
        mo.md("## Skids online and net water production, Jul 4–11"),
        mo.image(src=f"{DATA}/hero.png", alt="Skids online and net water production, before vs. after"),
        mo.md("## Grid power: the dashed line is the plant, the gap is the battery"),
        mo.image(src=f"{DATA}/power.png", alt="Grid power before vs. after"),
    ])
    return


@app.cell
def _(DATA, mo):
    mo.ui.tabs({
        "Before, peak": mo.image(src=f"{DATA}/pfd_A_peak.svg", alt="Flowsheet before, peak"),
        "After, peak": mo.image(src=f"{DATA}/pfd_B_peak.svg", alt="Flowsheet after, peak"),
        "Before, off-peak": mo.image(src=f"{DATA}/pfd_A_offpeak.svg", alt="Flowsheet before, off-peak"),
        "After, off-peak": mo.image(src=f"{DATA}/pfd_B_offpeak.svg", alt="Flowsheet after, off-peak"),
    })
    return


if __name__ == "__main__":
    app.run()
