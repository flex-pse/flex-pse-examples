# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib", "duckdb"]
# ///
"""Pump scheduling: the WebAssembly page.

Runs in the browser, so it can only import what Pyodide has -- no pyomo, no
flexops, no sibling `model`. Everything here is DuckDB + matplotlib over the
Parquet files `tools/sweep.py` committed under `public/`. See CONTRIBUTING.md.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # `notebook_location()` is a path locally and a URL in the browser. The name
    # is doubled because every example's `public/` merges into one directory.
    EXAMPLE = "pump_scheduling"
    DATA = f"{mo.notebook_location()}/public/{EXAMPLE}"
    return DATA, EXAMPLE, duckdb, mdates, mo, np, pd, plt


@app.cell
def _():
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    SURFACE, PEAK = "#fcfcfb", "#f0efec"
    STRATEGY_COLOR = {"flexible": BLUE, "inflexible": ORANGE}

    def style(ax, ylabel=""):
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        return ax

    return AQUA, BLUE, INK, MUTED, ORANGE, PEAK, STRATEGY_COLOR, SURFACE, style


@app.cell
def _(mo):
    mo.md(r"""
    # Pump scheduling

    **Is it better to buy a bigger battery, or to let the pumps run flexibly?**

    A feed pump fills a tank, and a product pump draws from it to meet a fixed
    daily demand. Power costs 4.5× more from 4pm to 9pm. We solved one day
    for two pump strategies and a range of battery sizes:

    - **Inflexible**: the feed pump runs at the same rate all day.
    - **Flexible**: it can switch off, or run anywhere between 60 and 100%.
    """)
    return


@app.cell
def _(AQUA, BLUE, INK, MUTED, SURFACE, plt):
    def _diagram():
        fig, ax = plt.subplots(figsize=(8, 1.6))
        fig.patch.set_facecolor(SURFACE)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        unit = {"boxstyle": "round,pad=0.6", "fc": "#eef4fc", "ec": BLUE, "lw": 1}
        arrow = {"arrowstyle": "-|>", "color": MUTED, "lw": 1.2, "shrinkA": 9, "shrinkB": 9}
        wire = {"arrowstyle": "-", "color": AQUA, "lw": 1.2, "ls": "--",
                "shrinkA": 9, "shrinkB": 9}

        steps = ["water in", "feed pump", "tank", "product pump", "demand"]
        xs = [0.07, 0.29, 0.5, 0.71, 0.93]
        nodes = [
            ax.text(x, 0.75, s, ha="center", va="center", fontsize=9, color=INK,
                    bbox=unit if 0 < i < len(steps) - 1 else None)
            for i, (x, s) in enumerate(zip(xs, steps))
        ]
        battery = ax.text(0.5, 0.15, "battery", ha="center", va="center", fontsize=9,
                          color=INK, bbox={**unit, "fc": "#e6f6ef", "ec": AQUA})
        for a, b in zip(nodes, nodes[1:]):
            ax.annotate("", xy=(0, 0.5), xycoords=b, xytext=(1, 0.5), textcoords=a,
                        arrowprops=arrow)
        for pump, side in ((nodes[1], 0), (nodes[3], 1)):
            ax.annotate("", xy=(0.5, 0), xycoords=pump, xytext=(side, 0.5),
                        textcoords=battery, arrowprops=wire)
        ax.text(0.5, 0.37, "power", ha="center", va="center", fontsize=8, color=MUTED)
        return fig

    _diagram()
    return


@app.cell
def _(EXAMPLE, mo):
    _repo = f"https://github.com/flex-pse/flex-pse-examples/tree/main/examples/{EXAMPLE}"
    mo.callout(
        mo.md(
            f"""
    These are **precomputed** results. The model needs Pyomo and a MILP solver,
    which don't run in a browser, so every case was solved ahead of time. The
    model and a notebook that solves it live in [the repo]({_repo}).
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(DATA, duckdb, mo):
    try:
        summary = duckdb.sql(
            f"SELECT * FROM read_parquet('{DATA}/summary.parquet') ORDER BY sweep_id"
        ).df()
        provenance = (
            duckdb.sql(f"SELECT key, value FROM read_parquet('{DATA}/provenance.parquet')")
            .df()
            .set_index("key")["value"]
        )
        load_error = None
    except Exception as exc:
        summary, provenance, load_error = None, None, exc

    mo.stop(
        load_error is not None,
        mo.callout(
            mo.md(f"**Couldn't load the results** from `{DATA}`.\n\n```\n{load_error}\n```"),
            kind="danger",
        ),
    )
    return provenance, summary


@app.cell
def _(np, summary):
    # Read the flexible plant's no-battery cost across to the inflexible curve:
    # how much battery buys the same saving as flexibility does.
    _flex = summary[summary.strategy == "flexible"].sort_values("battery_fraction")
    _infl = summary[summary.strategy == "inflexible"].sort_values("battery_fraction")
    flex_no_battery = float(_flex["operating_cost"].iloc[0])
    infl_no_battery = float(_infl["operating_cost"].iloc[0])
    # np.interp wants x increasing, and cost falls as the battery grows.
    battery_equivalent = float(
        np.interp(
            flex_no_battery,
            _infl["operating_cost"].to_numpy()[::-1],
            _infl["battery_fraction"].to_numpy()[::-1],
        )
    )
    return battery_equivalent, flex_no_battery, infl_no_battery


@app.cell
def _(mo):
    mo.md(r"""
    ## The short answer
    """)
    return


@app.cell
def _(
    INK,
    MUTED,
    STRATEGY_COLOR,
    SURFACE,
    battery_equivalent,
    flex_no_battery,
    plt,
    style,
    summary,
):
    def _plot(frame):
        fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
        fig.patch.set_facecolor(SURFACE)
        style(ax, "Cost for the day ($)")

        for strategy, group in frame.groupby("strategy"):
            group = group.sort_values("battery_fraction")
            color = STRATEGY_COLOR[strategy]
            ax.plot(group["battery_fraction"], group["operating_cost"], color=color,
                    lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label=strategy)
            last = group.iloc[-1]
            ax.annotate(strategy, (last["battery_fraction"], last["operating_cost"]),
                        xytext=(8, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=9)

        ax.hlines(flex_no_battery, 0, battery_equivalent, color=MUTED, lw=1, ls=":")
        ax.vlines(battery_equivalent, 0, flex_no_battery, color=MUTED, lw=1, ls=":")
        ax.annotate(
            f"Flexible with no battery costs the same\n"
            f"as inflexible with a {battery_equivalent:.0%} battery",
            (battery_equivalent, 0.18 * flex_no_battery), xytext=(8, 0),
            textcoords="offset points", va="center", color=INK, fontsize=9,
        )

        ax.set_xlabel("Battery size (% of the plant's peak power)", color=MUTED, fontsize=9)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlim(-0.03, 1.15)
        ax.set_ylim(0, frame["operating_cost"].max() * 1.12)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower left")
        return fig

    _plot(summary)
    return


@app.cell
def _(battery_equivalent, flex_no_battery, infl_no_battery, mo):
    mo.md(f"""
    Flexibility on its own cuts the daily bill from \\${infl_no_battery:.0f} to
    \\${flex_no_battery:.0f}, and it costs nothing to build. Getting the same
    savings with storage alone takes a battery sized at about
    **{battery_equivalent:.0%}** of the plant's peak power.

    The flexible line flattens out past 60%: by then the pumps are already
    off for the whole peak window, so a bigger battery doesn't buy anything.
    """)
    return


@app.cell
def _(mo):
    strategy = mo.ui.radio(
        options=["flexible", "inflexible"], value="flexible", label="Feed pump", inline=True
    )
    battery = mo.ui.slider(
        steps=[0, 20, 40, 60, 80, 100], value=40, label="Battery size (%)", show_value=True
    )
    mo.vstack([mo.md("## Look at one day"), mo.hstack([strategy, battery], justify="start", gap=3)])
    return battery, strategy


@app.cell
def _(DATA, battery, duckdb, strategy, summary):
    picked = summary[
        (summary.strategy == strategy.value)
        & ((summary.battery_fraction * 100).round() == battery.value)
    ].iloc[0]
    day = (
        duckdb.sql(
            f"""
            SELECT * EXCLUDE (sweep_id)
            FROM read_parquet('{DATA}/series.parquet')
            WHERE sweep_id = '{picked.sweep_id}'
            ORDER BY timestamp
            """
        )
        .df()
        .set_index("timestamp")
    )
    return day, picked


@app.cell
def _(infl_no_battery, mo, picked):
    mo.hstack(
        [
            mo.stat(f"${picked.operating_cost:.0f}", label="Cost for the day"),
            mo.stat(
                f"{1 - picked.operating_cost / infl_no_battery:.0%}",
                label="Cheaper than the worst case",
                caption="inflexible, no battery",
            ),
            mo.stat(
                f"{picked.peak_window_grid_kwh:.0f} kWh",
                label="Bought from 4 to 9pm",
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(
    AQUA,
    BLUE,
    INK,
    MUTED,
    ORANGE,
    PEAK,
    SURFACE,
    day,
    mdates,
    pd,
    plt,
    style,
):
    def _plot_day(frame):
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
        fig.patch.set_facecolor(SURFACE)
        # Each row covers the hour that follows it. Repeat the last row an hour
        # later so the step lines draw that final hour too.
        step = frame.index[1] - frame.index[0]
        frame = pd.concat([frame, frame.iloc[[-1]].set_axis([frame.index[-1] + step])])
        t = frame.index
        peak = frame.index[:-1][frame["energy_price"].iloc[:-1] > frame["energy_price"].min()]

        for ax in axes:
            ax.axvspan(peak[0], peak[-1] + step, color=PEAK, lw=0, zorder=0)
        axes[2].annotate("4–9pm peak", (peak[0], 1), xytext=(4, -4),
                         textcoords="offset points", xycoords=("data", "axes fraction"),
                         color=MUTED, fontsize=8, va="top")

        ax = style(axes[0], "m³/hr")
        ax.step(t, frame["feed_flow_m3_per_hr"], where="post", color=BLUE, lw=2,
                label="feed pump")
        ax.step(t, frame["product_flow_m3_per_hr"], where="post", color=ORANGE, lw=2,
                label="demand")
        ax.set_ylim(0, None)
        ax.set_title("Pumping", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2, loc="lower right",
                  bbox_to_anchor=(1, 1), borderaxespad=0)

        ax = style(axes[1], "% full")
        level = frame["tank_level"].iloc[:-1] * 100
        ax.fill_between(level.index, level, color=BLUE, alpha=0.15, lw=0)
        ax.plot(level.index, level, color=BLUE, lw=2)
        ax.set_ylim(0, 100)
        ax.set_title("Tank: fills up before the peak, drains through it", loc="left",
                     color=INK, fontsize=10)

        # Stack the battery's output on top of what comes from the grid, so the
        # full height is what the plant uses and the aqua is what it didn't buy.
        ax = style(axes[2], "kW")
        grid = frame["grid_kw"]
        from_battery = frame["battery_discharge_kw"].fillna(0)
        ax.fill_between(t, grid, step="post", color=INK, alpha=0.12, lw=0)
        ax.step(t, grid, where="post", color=INK, lw=2, label="from the grid")
        if from_battery.max() > 0:
            ax.fill_between(t, grid, grid + from_battery, step="post", color=AQUA,
                            alpha=0.6, lw=0, label="from the battery")
        ax.set_ylim(0, None)
        ax.set_title("Power", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2, loc="lower right",
                  bbox_to_anchor=(1, 1), borderaxespad=0)

        axes[-1].set_xticks(pd.date_range(t[0], t[-1], freq="3h"))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
        axes[-1].set_xlim(t[0], t[-1])
        return fig

    _plot_day(day)
    return


@app.cell
def _(mo, provenance, summary):
    _table = summary[
        ["strategy", "battery_fraction", "battery_kw", "battery_kwh",
         "operating_cost", "peak_window_grid_kwh"]
    ].rename(
        columns={
            "strategy": "Feed pump",
            "battery_fraction": "Battery (% of peak)",
            "battery_kw": "Battery (kW)",
            "battery_kwh": "Battery (kWh)",
            "operating_cost": "Cost for the day ($)",
            "peak_window_grid_kwh": "Bought 4–9pm (kWh)",
        }
    )
    _table["Battery (% of peak)"] = (_table["Battery (% of peak)"] * 100).round().astype(int)

    _rows = "\n".join(
        f"| {label} | `{provenance.get(key, '—')}` |"
        for key, label in [
            ("generated_utc", "Solved"),
            ("flexpse_commit", "flex-pse commit"),
            ("solver", "Solver"),
            ("total_wall_seconds", "Total solve time (s)"),
        ]
        if key in provenance.index
    )
    mo.accordion(
        {
            "Every case": mo.ui.table(_table.round(1), selection=None, page_size=12),
            "How these were made": mo.md(
                f"""
| | |
| --- | --- |
{_rows}

To regenerate: `python tools/sweep.py examples/pump_scheduling`.
"""
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
