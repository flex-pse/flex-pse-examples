# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib", "duckdb"]
# ///
"""Desalination scheduling: the WebAssembly page.

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
    EXAMPLE = "desalination_scheduling"
    DATA = f"{mo.notebook_location()}/public/{EXAMPLE}"
    return DATA, EXAMPLE, duckdb, mdates, mo, np, pd, plt


@app.cell
def _():
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    SURFACE, PEAK = "#fcfcfb", "#f0efec"

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

    return BLUE, INK, MUTED, ORANGE, PEAK, SURFACE, style


@app.cell
def _(mo):
    mo.md(r"""
    # Desalination scheduling

    **How does a monthly water target change what the water costs?**

    A seawater plant with three reverse osmosis trains has to deliver a set
    volume of water each month. When it makes that water is up to the plant,
    and there's no storage. Power costs 4.5× more from 4pm to 9pm. Restarting
    has a cost too: for the first 45 minutes after a restart, everything the
    plant makes is off-spec and goes back to the ocean.

    We solved a full month at 15-minute steps for targets from half of what
    the plant can make up to nearly all of it.
    """)
    return


@app.cell
def _(BLUE, INK, MUTED, SURFACE, np, plt):
    def _diagram():
        fig, ax = plt.subplots(figsize=(8, 1.6))
        fig.patch.set_facecolor(SURFACE)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        unit = {"boxstyle": "round,pad=0.6", "fc": "#eef4fc", "ec": BLUE, "lw": 1}
        arrow = {"arrowstyle": "-|>", "color": MUTED, "lw": 1.2, "shrinkA": 9, "shrinkB": 9}

        steps = ["seawater", "intake", "3 RO trains", "post-treatment", "product water"]
        xs = np.linspace(0.07, 0.93, len(steps))
        nodes = [
            ax.text(x, 0.75, s, ha="center", va="center", fontsize=9, color=INK,
                    bbox=unit if 0 < i < len(steps) - 1 else None)
            for i, (x, s) in enumerate(zip(xs, steps))
        ]
        brine = ax.text(xs[2], 0.15, "brine to the ocean", ha="center", va="center",
                        fontsize=9, color=INK)
        for a, b in zip(nodes, nodes[1:]):
            ax.annotate("", xy=(0, 0.5), xycoords=b, xytext=(1, 0.5), textcoords=a,
                        arrowprops=arrow)
        ax.annotate("", xy=(0.5, 1), xycoords=brine, xytext=(0.5, 0), textcoords=nodes[2],
                    arrowprops=arrow)
        return fig

    _diagram()
    return


@app.cell
def _(EXAMPLE, mo):
    _repo = f"https://github.com/flex-pse/flex-pse-examples/tree/main/examples/{EXAMPLE}"
    mo.callout(
        mo.md(
            f"""
    These are **precomputed** results. Each month is a big mixed-integer
    problem (about 27,000 on/off decisions) solved with Gurobi, which doesn't
    run in a browser. The model and a notebook that solves it live in
    [the repo]({_repo}).
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(DATA, duckdb, mo):
    try:
        summary = duckdb.sql(
            f"SELECT * FROM read_parquet('{DATA}/summary.parquet') ORDER BY capacity_fraction"
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

    summary["usd_per_af"] = summary["operating_cost"] / summary["delivered_af"]
    summary["short_label"] = summary["capacity_fraction"].map(lambda f: f"{f:.0%}")
    return provenance, summary


@app.cell
def _(summary):
    # With the plant off from 4 to 9pm, all it buys in that window is the intake
    # pump's standby draw. The last target where on-peak energy is still at that
    # floor is the last one where the plant skips the peak entirely.
    _floor = summary["peak_window_kwh"].min()
    _skips = summary[summary["peak_window_kwh"] <= _floor * 1.01]
    last_skip = summary.loc[_skips.index[-1]]
    cheapest = summary.loc[summary["usd_per_af"].idxmin()]
    priciest = summary.loc[summary["usd_per_af"].idxmax()]
    return cheapest, last_skip, priciest


@app.cell
def _(mo):
    mo.md(r"""
    ## The short answer
    """)
    return


@app.cell
def _(
    BLUE,
    INK,
    MUTED,
    PEAK,
    SURFACE,
    cheapest,
    last_skip,
    plt,
    style,
    summary,
):
    def _plot(frame):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [3, 2]})
        fig.patch.set_facecolor(SURFACE)
        x = frame["capacity_fraction"]
        # Shade the targets where the plant can no longer skip the peak.
        nxt = frame[frame["capacity_fraction"] > last_skip["capacity_fraction"]]
        split = (last_skip["capacity_fraction"] + nxt["capacity_fraction"].iloc[0]) / 2

        for ax in axes:
            ax.axvspan(split, x.max() + 0.03, color=PEAK, lw=0, zorder=0)

        ax = style(axes[0], "$ per acre-foot")
        ax.plot(x, frame["usd_per_af"], color=BLUE, lw=2, marker="o", ms=6,
                mec=SURFACE, mew=2)
        ax.annotate("cheapest variable cost", (cheapest["capacity_fraction"], cheapest["usd_per_af"]),
                    xytext=(0, -16), textcoords="offset points", ha="center",
                    color=INK, fontsize=9)
        ax.annotate("off from 4–9pm\nevery day", (split, 1), xycoords=("data", "axes fraction"),
                    xytext=(-6, -6), textcoords="offset points",
                    ha="right", va="top", color=MUTED, fontsize=8)
        ax.annotate("has to run\nduring the peak", (split, 1), xycoords=("data", "axes fraction"),
                    xytext=(6, -6), textcoords="offset points", ha="left", va="top",
                    color=MUTED, fontsize=8)
        ax.set_ylim(0, frame["usd_per_af"].max() * 1.15)
        ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
        ax.set_title("What each acre-foot costs", loc="left", color=INK, fontsize=10)

        ax = style(axes[1], "MWh")
        ax.plot(x, frame["peak_window_kwh"] / 1000, color=BLUE, lw=2, marker="o", ms=6,
                mec=SURFACE, mew=2)
        ax.set_ylim(0, frame["peak_window_kwh"].max() / 1000 * 1.15)
        ax.set_title("Energy bought from 4 to 9pm, over the month", loc="left",
                     color=INK, fontsize=10)

        axes[1].set_xlabel("Monthly target (% of what the plant could make running flat out)",
                           color=MUTED, fontsize=9)
        axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        axes[1].set_xlim(x.min() - 0.03, x.max() + 0.03)
        return fig

    _plot(summary)
    return


@app.cell
def _(cheapest, last_skip, mo, priciest):
    mo.md(f"""
    Up to **{last_skip.capacity_fraction:.0%}** of capacity, the plant can make
    all its water outside the peak. It shuts down at 4pm every day and comes
    back after 9. Each acre-foot actually gets a bit cheaper up to that point,
    because the intake pump runs either way and its cost is spread over more
    water.

    Above that, there aren't enough off-peak hours left, so the plant has to
    run during the peak. At {priciest.capacity_fraction:.0%}, an acre-foot costs
    **\\${priciest.usd_per_af:,.0f}**, compared with \\${cheapest.usd_per_af:,.0f}
    at the cheapest point.
    """)
    return


@app.cell
def _(cheapest, mo, summary):
    month = mo.ui.radio(
        options=dict(zip(summary["short_label"], summary["sweep_id"])),
        value=cheapest["short_label"],
        label="Monthly target (% of capacity)",
        inline=True,
    )
    mo.vstack([mo.md("## Look at one month"), month])
    return (month,)


@app.cell
def _(DATA, duckdb, month, summary):
    picked = summary.set_index("sweep_id").loc[month.value]

    def _view(name, order):
        return (
            duckdb.sql(
                f"""
                SELECT * EXCLUDE (sweep_id)
                FROM read_parquet('{DATA}/{name}.parquet')
                WHERE sweep_id = '{month.value}'
                ORDER BY {order}
                """
            )
            .df()
            .set_index(order)
        )

    window = _view("series", "timestamp")
    profile = _view("profile", "time_of_day")
    return picked, profile, window


@app.cell
def _(mo, picked):
    mo.hstack(
        [
            mo.stat(f"${picked.operating_cost / 1000:,.0f}k", label="Power bill for the month"),
            mo.stat(f"${picked.usd_per_af:,.0f}", label="Per acre-foot"),
            mo.stat(f"{picked.restarts:.0f}", label="Restarts"),
            mo.stat(f"{picked.offspec_af:.1f} AF", label="Dumped while restarting"),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(BLUE, INK, MUTED, ORANGE, PEAK, SURFACE, mdates, pd, plt, style, window):
    def _plot_window(frame):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, layout="constrained")
        fig.patch.set_facecolor(SURFACE)
        # Each row covers the 15 minutes that follow it. Repeat the last row one
        # step later so the step lines draw that final interval too.
        step = frame.index[1] - frame.index[0]
        frame = pd.concat([frame, frame.iloc[[-1]].set_axis([frame.index[-1] + step])])
        t = frame.index

        peak = frame["energy_price"] > frame["energy_price"].min()
        starts = t[peak & ~peak.shift(fill_value=False)]
        ends = t[~peak & peak.shift(fill_value=False)]
        for ax in axes:
            for a, b in zip(starts, ends):
                ax.axvspan(a, b, color=PEAK, lw=0, zorder=0)
        axes[0].annotate("4–9pm peak", (starts[0], 1), xytext=(4, -4),
                         textcoords="offset points", xycoords=("data", "axes fraction"),
                         color=MUTED, fontsize=8, va="top")

        # A train is "restarting" when it's on but post-treatment isn't yet, so
        # its water goes to the outfall instead of the customer.
        trains = frame["trains_online"]
        dumping = frame["offspec_permeate_m3_per_hr"] > 0
        ax = style(axes[0], "")
        ax.fill_between(t, trains, step="post", color=BLUE, alpha=0.25, lw=0)
        ax.step(t, trains, where="post", color=BLUE, lw=2, label="making water")
        ax.fill_between(t, trains, where=dumping, step="post", color=ORANGE, lw=0,
                        label="restarting (water dumped)")
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(0, 3.6)
        ax.set_title("Trains running", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2, loc="lower right",
                  bbox_to_anchor=(1, 1), borderaxespad=0)

        ax = style(axes[1], "MW")
        ax.fill_between(t, frame["grid_kw"] / 1000, step="post", color=INK, alpha=0.12, lw=0)
        ax.step(t, frame["grid_kw"] / 1000, where="post", color=INK, lw=1.6)
        ax.set_ylim(0, None)
        ax.set_title("Power from the grid", loc="left", color=INK, fontsize=10)

        axes[-1].set_xticks(pd.date_range(t[0].normalize(), t[-1], freq="12h"))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %-d\n%-I%p"))
        axes[-1].set_xlim(t[0], t[-1])
        return fig

    _plot_window(window)
    return


@app.cell
def _(mo):
    mo.md(r"""
    That's three days from the middle of the month. Here's the whole month
    averaged into one day:
    """)
    return


@app.cell
def _(BLUE, INK, MUTED, PEAK, SURFACE, np, plt, profile, style):
    def _plot_profile(frame):
        fig, ax = plt.subplots(figsize=(8, 2.8), layout="constrained")
        fig.patch.set_facecolor(SURFACE)
        style(ax, "")
        hours = np.arange(len(frame) + 1) * 24 / len(frame)
        trains = np.append(frame["trains_online"].to_numpy(), frame["trains_online"].iloc[-1])
        peak = np.append(
            (frame["energy_price"] > frame["energy_price"].min()).to_numpy(), False
        )
        ax.axvspan(hours[peak.argmax()], hours[peak.argmax() + peak.sum()], color=PEAK,
                   lw=0, zorder=0)
        ax.annotate("4–9pm peak", (hours[peak.argmax()], 1), xytext=(4, -4),
                    textcoords="offset points", xycoords=("data", "axes fraction"),
                    color=MUTED, fontsize=8, va="top")
        ax.fill_between(hours, trains, step="post", color=BLUE, alpha=0.25, lw=0)
        ax.step(hours, trains, where="post", color=BLUE, lw=2)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels(["12am", "3am", "6am", "9am", "12pm", "3pm", "6pm", "9pm", "12am"])
        ax.set_xlim(0, 24)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(0, 3.6)
        ax.set_title("Trains running on an average day", loc="left", color=INK, fontsize=10)
        return fig

    _plot_profile(profile)
    return


@app.cell
def _(mo, provenance, summary):
    _table = summary[
        ["capacity_fraction", "demand_af", "operating_cost", "usd_per_af", "restarts",
         "offspec_af", "peak_window_kwh", "termination", "mip_gap"]
    ].rename(
        columns={
            "capacity_fraction": "Target (% of capacity)",
            "demand_af": "Target (AF)",
            "operating_cost": "Power bill ($)",
            "usd_per_af": "$ per AF",
            "restarts": "Restarts",
            "offspec_af": "Dumped (AF)",
            "peak_window_kwh": "Bought 4–9pm (kWh)",
            "termination": "Solver status",
            "mip_gap": "Gap",
        }
    )
    _table["Target (% of capacity)"] = (_table["Target (% of capacity)"] * 100).round().astype(int)

    _capped = summary[summary["termination"].astype(str).str.contains("maxTimeLimit", na=False)]
    _note = ""
    if len(_capped):
        _note = (
            f"\n\n{', '.join(_capped['short_label'])} hit the time limit before the solver "
            f"could prove it was optimal. The schedule is still valid, but the best one "
            f"could be up to {_capped['mip_gap'].max():.1%} cheaper."
        )

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
            "Every month": mo.vstack([
                mo.ui.table(_table.round(3), selection=None, page_size=12),
                mo.md(_note),
            ]),
            "How these were made": mo.md(
                f"""
    | | |
    | --- | --- |
    {_rows}

    To regenerate: `python tools/sweep.py examples/desalination_scheduling`. You'll
    need a Gurobi license, since the full month is a non-convex MIQCP.
    """
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
