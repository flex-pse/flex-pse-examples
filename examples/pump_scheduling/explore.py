# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib"]
# ///
"""Pump scheduling: the WebAssembly page.

This notebook ships to the browser, so it may import nothing that Pyodide cannot
install -- no pyomo, no flexops, no sibling `model`. It reads the sweep that
`tools/sweep.py` solved offline and committed under `public/`, and everything it
draws is pandas and matplotlib over those CSVs.

`tools/site/build.py` enforces that with an import allowlist; see CONTRIBUTING.md.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    #: Where `tools/sweep.py` wrote this example's data. `mo.notebook_location()`
    #: is the directory holding the notebook when run locally and the directory
    #: holding the exported HTML when run in the browser, so one f-string covers
    #: both. The name is doubled because every example's `public/` merges into a
    #: single directory in the export -- the inner folder is what keeps them apart.
    EXAMPLE = "pump_scheduling"
    DATA = f"{mo.notebook_location()}/public/{EXAMPLE}"
    return DATA, EXAMPLE, mdates, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Pump scheduling: does flexibility substitute for storage?

    A two-pump water plant on a time-of-use tariff, and one question:

    > Would you rather have a **bigger battery for an inflexible facility**, or a
    > **smaller battery for a flexible one**?

    A feed pump lifts water across a fixed pressure rise into a 1500 m³ tank; a
    product pump draws from the tank to meet a fixed 24-hour demand profile. Every
    case below moves the same water and buys the same total pump energy — the only
    thing that changes is *when* the energy is bought, and what is available to
    shift it: a battery, operational flexibility, or both.
    """
    )
    return


@app.cell
def _(EXAMPLE, mo):
    _repo = "https://github.com/flex-pse/flex-pse-examples"
    mo.callout(
        mo.md(
            f"""
    **This page replays precomputed results.** The optimization behind it needs
    Pyomo, IDAES and HiGHS, none of which run in a browser — so what you are
    selecting between are solutions solved offline and committed to the
    repository.

    For the model, the solver-level notebook and the code, see
    [`examples/{EXAMPLE}/`]({_repo}/tree/main/examples/{EXAMPLE}) — in particular
    [`model.py`]({_repo}/blob/main/examples/{EXAMPLE}/model.py) for the flowsheet
    and [`notebook.py`]({_repo}/blob/main/examples/{EXAMPLE}/notebook.py) for the
    walkthrough that builds and solves it.
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(DATA, mo, pd):
    # A failed fetch here is the most likely way this page breaks in a browser
    # (a stale export, a moved file, an offline visitor). Say so in the page
    # rather than dropping a traceback on a reader who cannot act on it.
    try:
        summary = pd.read_csv(f"{DATA}/summary.csv")
        provenance = pd.read_csv(f"{DATA}/provenance.csv", index_col="key")["value"]
        load_error = None
    except Exception as exc:
        summary, provenance, load_error = None, None, exc

    mo.stop(
        load_error is not None,
        mo.callout(
            mo.md(
                f"""
    **Could not load the sweep data.** Expected it under `{DATA}`.

    ```
    {load_error}
    ```
    """
            ),
            kind="danger",
        ),
    )
    return provenance, summary


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The trade-off curve

    Each point is one solved schedule. The battery is swept from nothing to a
    rating equal to the plant's whole peak electrical load, at both strategies:
    the **inflexible** feed pump holds a constant duty all day, while the
    **flexible** one may switch off or run anywhere in its turndown band.
    """
    )
    return


@app.cell
def _(np, plt, summary):
    def _plot_tradeoff(frame):
        fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
        colors = {"inflexible": "#b3543f", "flexible": "#2f6f8f"}

        for strategy, group in frame.groupby("strategy", sort=False):
            group = group.sort_values("battery_fraction")
            ax.plot(
                group["battery_fraction"],
                group["operating_cost"],
                marker="o",
                markersize=5,
                linewidth=2,
                color=colors[strategy],
                label=strategy,
                zorder=3,
            )

        # The headline: read the flexible curve's zero-battery cost across to the
        # inflexible curve and see how much storage it is worth.
        flex = frame[frame.strategy == "flexible"].sort_values("battery_fraction")
        infl = frame[frame.strategy == "inflexible"].sort_values("battery_fraction")
        flex_no_battery = float(flex["operating_cost"].iloc[0])
        # np.interp needs an increasing x, and cost falls with battery size.
        equivalent = float(
            np.interp(
                flex_no_battery,
                infl["operating_cost"].to_numpy()[::-1],
                infl["battery_fraction"].to_numpy()[::-1],
            )
        )
        ax.plot(
            [0, equivalent],
            [flex_no_battery, flex_no_battery],
            linestyle=":",
            color="#666666",
            linewidth=1.4,
            zorder=2,
        )
        ax.plot([equivalent], [flex_no_battery], marker="x", color="#666666", zorder=4)
        ax.annotate(
            f"flexibility alone ≈ a battery rated at\n{equivalent:.0%} of peak plant load",
            xy=(equivalent, flex_no_battery),
            xytext=(equivalent + 0.06, flex_no_battery + 14),
            fontsize=9,
            color="#444444",
            arrowprops={"arrowstyle": "-", "color": "#999999", "linewidth": 1},
        )

        ax.set_xlabel("Battery rating (fraction of peak plant load)")
        ax.set_ylabel("Operating cost for the day ($)")
        ax.set_title("What a day costs, by strategy and battery size")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.legend(frameon=False, loc="upper right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return fig, equivalent

    tradeoff_fig, equivalent_fraction = _plot_tradeoff(summary)
    tradeoff_fig
    return (equivalent_fraction,)


@app.cell
def _(equivalent_fraction, mo, summary):
    _flex = summary[summary.strategy == "flexible"].sort_values("battery_fraction")
    _infl = summary[summary.strategy == "inflexible"].sort_values("battery_fraction")
    _flex0 = float(_flex["operating_cost"].iloc[0])
    _infl0 = float(_infl["operating_cost"].iloc[0])
    _flex_kw = float(
        _flex["battery_kw"].iloc[-1] / max(_flex["battery_fraction"].iloc[-1], 1e-9)
    )

    mo.md(
        f"""
    **The answer is flexibility.** With no battery at all, running the feed pump
    flexibly costs **${_flex0:.2f}** for the day against **${_infl0:.2f}** for the
    constant-duty plant — a {1 - _flex0 / _infl0:.0%} saving bought with no capital
    at all. To reach that same bill by storage alone, the inflexible plant needs a
    battery rated at about **{equivalent_fraction:.0%}** of its entire peak
    electrical load ({equivalent_fraction * _flex_kw:.0f} kW /
    {equivalent_fraction * _flex_kw * 5:.0f} kWh).

    The flexible curve also flattens: past roughly 60 % it has already moved every
    kilowatt-hour it can out of the peak window, and further storage buys nothing.
    The inflexible curve is still falling at 100 %, because a plant that cannot
    stop pumping can only ever shift its load with a battery.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## One schedule at a time

    Pick a case to see the day it produces. Watch the tank: it is the only thing
    that lets the flexible plant stop pumping without missing a delivery.
    """
    )
    return


@app.cell
def _(mo, summary):
    case = mo.ui.dropdown(
        options=dict(zip(summary["label"], summary["sweep_id"])),
        value="flexible @ 40%",
        label="Case",
    )
    case
    return (case,)


@app.cell
def _(DATA, case, mo, pd):
    mo.stop(case.value is None, mo.md("*Pick a case above.*"))
    # One small file per point, fetched on selection rather than up front: in the
    # browser this is a synchronous request on the worker thread, so loading all
    # twelve at boot would stall the page for no benefit.
    schedule = pd.read_csv(
        f"{DATA}/series/{case.value}.csv", parse_dates=["timestamp"], index_col="timestamp"
    )
    return (schedule,)


@app.cell
def _(case, mdates, plt, schedule, summary):
    def _plot_schedule(frame, label, meta):
        fig, axes = plt.subplots(
            3, 1, figsize=(9, 7.6), sharex=True, layout="constrained"
        )
        hours = frame.index
        peak = frame["energy_price"] > frame["energy_price"].min()

        def shade(ax):
            """Band the on-peak energy hours behind every panel."""
            ax.fill_between(
                hours,
                0,
                1,
                where=peak,
                transform=ax.get_xaxis_transform(),
                color="#d9c9a3",
                alpha=0.35,
                linewidth=0,
                zorder=0,
                step="post",
            )

        ax = axes[0]
        shade(ax)
        ax.step(hours, frame["feed_flow_m3_per_hr"], where="post",
                color="#2f6f8f", linewidth=2, label="feed pump")
        ax.step(hours, frame["product_flow_m3_per_hr"], where="post",
                color="#b3543f", linewidth=2, linestyle="--", label="product delivery")
        ax.set_ylabel("Flow (m³/hr)")
        ax.set_title(f"{label} — ${meta['operating_cost']:.2f} for the day")
        ax.legend(frameon=False, ncols=2, fontsize=9)

        ax = axes[1]
        shade(ax)
        ax.fill_between(hours, frame["tank_volume_m3"], step="post",
                        color="#7fa8bf", alpha=0.45, linewidth=0)
        ax.step(hours, frame["tank_volume_m3"], where="post",
                color="#2f6f8f", linewidth=1.8)
        ax.set_ylabel("Tank volume (m³)")
        ax.set_ylim(bottom=0)

        ax = axes[2]
        shade(ax)
        ax.step(hours, frame["grid_kw"], where="post",
                color="#3f6f4f", linewidth=2, label="grid draw")
        if "battery_soc" in frame:
            twin = ax.twinx()
            twin.step(hours, frame["battery_soc"], where="post",
                      color="#8a6fb0", linewidth=1.6, linestyle=":", label="battery SOC")
            twin.set_ylabel("Battery SOC")
            twin.set_ylim(0, 1)
            twin.spines["top"].set_visible(False)
        ax.set_ylabel("Grid power (kW)")
        ax.set_ylim(bottom=0)
        ax.set_xlabel(
            f"Hour of {hours[0]:%-d %B %Y} — shaded band is the on-peak tariff window"
        )
        # The horizon is one day, so date-stamping every tick is noise.
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

        for a in axes:
            a.grid(axis="y", alpha=0.2, zorder=0)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        return fig

    schedule_fig = _plot_schedule(
        schedule,
        case.selected_key,
        summary.set_index("sweep_id").loc[case.value],
    )
    schedule_fig
    return


@app.cell
def _(mo):
    mo.md("## Every case")
    return


@app.cell
def _(mo, summary):
    _table = summary[
        [
            "label",
            "strategy",
            "battery_kw",
            "battery_kwh",
            "operating_cost",
            "peak_kw",
            "peak_window_grid_kwh",
        ]
    ].rename(
        columns={
            "label": "Case",
            "strategy": "Strategy",
            "battery_kw": "Battery (kW)",
            "battery_kwh": "Battery (kWh)",
            "operating_cost": "Operating cost ($)",
            "peak_kw": "Peak-window draw (kW)",
            "peak_window_grid_kwh": "Peak-window energy (kWh)",
        }
    )
    mo.ui.table(_table.round(2), selection=None, page_size=12)
    return


@app.cell
def _(mo, provenance):
    _keys = [
        ("generated_utc", "Solved"),
        ("flexpse_version", "flex-pse"),
        ("flexpse_commit", "flex-pse commit"),
        ("solver", "Solver"),
        ("n_points", "Sweep points"),
        ("total_wall_seconds", "Total solve time (s)"),
    ]
    _rows = "\n".join(
        f"| {label} | `{provenance.get(key, '—')}` |"
        for key, label in _keys
        if key in provenance.index
    )
    mo.accordion(
        {
            "How these numbers were made": mo.md(
                f"""
| | |
| --- | --- |
{_rows}

Regenerate with `python tools/sweep.py examples/pump_scheduling` in the conda
environment from `environment.yml`.
"""
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
