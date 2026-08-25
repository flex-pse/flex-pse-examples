# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib"]
# ///
"""Desalination scheduling: the WebAssembly page.

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
    #: is the notebook's directory locally and the exported page's directory in
    #: the browser, so one f-string covers both. The name is doubled because
    #: every example's `public/` merges into one directory in the export.
    EXAMPLE = "desalination_scheduling"
    DATA = f"{mo.notebook_location()}/public/{EXAMPLE}"
    return DATA, EXAMPLE, mdates, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Desalination scheduling: what a monthly obligation costs

    A seawater plant with three parallel treatment trains, scheduled against a
    time-of-use tariff:

    ```
                       ┌─► pretreatment[0] ─► RO[0] ─┬─► brine ─► ocean
    seawater ─► intake ─┼─► pretreatment[1] ─► RO[1] ─┤        (permeate)
                 pump   └─► pretreatment[2] ─► RO[2] ─┘            │
                                                                   ▼
      product water ◄─ product pump ◄─ post-treatment ◄─ permeate header
    ```

    The plant owes a **volume** of product water over the month — not an hourly
    profile — and there is no storage in the flowsheet. So the only way to dodge
    an expensive hour is to make less water in it and more water elsewhere.

    Stepping the train count down from three to two is free. Restarting the RO
    *system* is not: for 45 minutes afterwards post-treatment is out, and every
    train's permeate — not just the restarting one's — leaves off-spec to the
    outfall while the plant pays full power to make it.

    Each point below is a full calendar month at 15-minute resolution, solved as
    a unit-commitment problem with roughly 27,000 binaries.
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
    Pyomo, IDAES and Gurobi — the exact month is a non-convex MIQCP, since
    unfixed RO recovery makes `permeate == recovery × feed` bilinear — and none
    of that runs in a browser. What you are selecting between are months solved
    offline and committed to the repository.

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
    ## Running out of slack

    The obligation is swept from half the plant's capacity up to almost all of
    it. Capacity here is every skid at rated feed and top recovery for every step
    of the month, never stopping — so the ratio on the x-axis is really *how much
    room the schedule has*.
    """
    )
    return


@app.cell
def _(plt, summary):
    def _plot_sweep(frame):
        frame = frame.sort_values("demand_af")
        fig, axes = plt.subplots(
            3, 1, figsize=(9, 8), sharex=True, layout="constrained"
        )
        x = frame["capacity_fraction"]

        ax = axes[0]
        ax.plot(x, frame["operating_cost"], marker="o", markersize=5,
                linewidth=2, color="#2f6f8f")
        ax.set_ylabel("Operating cost ($)")
        ax.set_title("What the month costs, and what it costs to get there")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

        ax = axes[1]
        ax.plot(x, frame["restarts"], marker="o", markersize=5,
                linewidth=2, color="#b3543f")
        ax.set_ylabel("RO startups")

        ax = axes[2]
        ax.plot(x, frame["offspec_af"], marker="o", markersize=5,
                linewidth=2, color="#8a6fb0")
        ax.set_ylabel("Off-spec permeate (AF)")
        ax.set_xlabel(
            "Monthly obligation, as a fraction of what the plant could make flat out"
        )
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

        for a in axes:
            a.grid(axis="y", alpha=0.25, zorder=0)
            a.set_ylim(bottom=0)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        return fig

    _plot_sweep(summary)
    return


@app.cell
def _(mo, summary):
    _sorted = summary.sort_values("capacity_fraction")
    _low, _high = _sorted.iloc[0], _sorted.iloc[-1]
    _cost_ratio = _high["operating_cost"] / _low["operating_cost"]
    _demand_ratio = _high["demand_af"] / _low["demand_af"]

    mo.md(
        f"""
    Between **{_low['capacity_fraction']:.0%}** and
    **{_high['capacity_fraction']:.0%}** of capacity the plant is asked for
    {_demand_ratio:.1f}× the water and the bill goes up {_cost_ratio:.1f}×. The
    interesting part is the middle two panels: the binding constraint is not the
    tariff, it is **headroom**. Low down, the optimizer can afford to shut the RO
    system through the whole peak window and make the water back later. As the
    obligation climbs there is less and less room to move water around, and each
    restart it does buy costs a 45-minute recuperation window in which all three
    trains make permeate the plant cannot sell.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## One month at a time

    Pick an obligation to see three days out of the middle of its schedule.
    Shaded bands are the on-peak tariff hours.
    """
    )
    return


@app.cell
def _(mo, summary):
    _labels = summary.sort_values("capacity_fraction")
    case = mo.ui.dropdown(
        options=dict(zip(_labels["label"], _labels["sweep_id"])),
        value=_labels["label"].iloc[len(_labels) // 2],
        label="Monthly obligation",
    )
    case
    return (case,)


@app.cell
def _(DATA, case, mo, pd):
    mo.stop(case.value is None, mo.md("*Pick an obligation above.*"))
    # One small file per point, fetched on selection: in the browser this is a
    # synchronous request on the worker thread, so loading all eight at boot
    # would stall the page for nothing.
    window = pd.read_csv(
        f"{DATA}/series/{case.value}.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    profile = pd.read_csv(f"{DATA}/profile/{case.value}.csv", index_col="time_of_day")
    return profile, window


@app.cell
def _(case, mdates, plt, summary, window):
    def _plot_window(frame, label, meta):
        fig, axes = plt.subplots(
            3, 1, figsize=(9, 7.8), sharex=True, layout="constrained"
        )
        t = frame.index
        peak = frame["energy_price"] > frame["energy_price"].min()

        def shade(ax):
            ax.fill_between(
                t, 0, 1, where=peak, transform=ax.get_xaxis_transform(),
                color="#d9c9a3", alpha=0.35, linewidth=0, zorder=0, step="post",
            )

        ax = axes[0]
        shade(ax)
        ax.step(t, frame["product_m3_per_hr"], where="post",
                color="#2f6f8f", linewidth=1.8, label="product water")
        if frame["offspec_permeate_m3_per_hr"].max() > 0:
            ax.fill_between(t, frame["offspec_permeate_m3_per_hr"], step="post",
                            color="#b3543f", alpha=0.75, linewidth=0,
                            label="off-spec to outfall")
        ax.set_ylabel("Flow (m³/hr)")
        ax.set_title(
            f"{label} — ${meta['operating_cost']:,.0f} and "
            f"{int(meta['restarts'])} RO startups over the month"
        )
        ax.legend(frameon=False, ncols=2, fontsize=9)

        ax = axes[1]
        shade(ax)
        ax.step(t, frame["trains_online"], where="post",
                color="#3f6f4f", linewidth=1.8, label="trains online")
        ax.step(t, frame["post_treatment_status"], where="post",
                color="#8a6fb0", linewidth=1.4, linestyle=":",
                label="post-treatment on")
        ax.set_ylabel("Trains / status")
        ax.set_yticks([0, 1, 2, 3])
        ax.legend(frameon=False, ncols=2, fontsize=9)

        ax = axes[2]
        shade(ax)
        ax.fill_between(t, frame["grid_kw"], step="post",
                        color="#7fa8bf", alpha=0.4, linewidth=0)
        ax.step(t, frame["grid_kw"], where="post", color="#2f6f8f", linewidth=1.6)
        ax.set_ylabel("Plant power (kW)")
        ax.set_xlabel("Shaded bands are the on-peak tariff window")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        for a in axes:
            a.grid(axis="y", alpha=0.2, zorder=0)
            a.set_ylim(bottom=0)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        return fig

    _plot_window(
        window, case.selected_key, summary.set_index("sweep_id").loc[case.value]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The average day

    The same month, folded onto a single 24 hours. This is where the tariff
    response shows up as a shape rather than as a decision: how much of the
    plant is running at each time of day, averaged over the whole month.
    """
    )
    return


@app.cell
def _(case, plt, profile):
    def _plot_profile(frame, label):
        fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
        x = range(len(frame))
        ax.fill_between(x, frame["trains_online"], step="mid",
                        color="#7fa8bf", alpha=0.45, linewidth=0)
        ax.step(x, frame["trains_online"], where="mid",
                color="#2f6f8f", linewidth=2)
        ax.set_ylabel("Trains online (month average)")
        ax.set_ylim(0, 3.05)
        ax.set_title(f"{label} — average day")

        twin = ax.twinx()
        twin.step(x, frame["energy_price"], where="mid",
                  color="#b3543f", linewidth=1.4, linestyle="--")
        twin.set_ylabel("Energy price ($/kWh)", color="#b3543f")
        twin.tick_params(axis="y", colors="#b3543f")
        twin.spines["top"].set_visible(False)

        ticks = [i for i, name in enumerate(frame.index) if name.endswith(":00")][::3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([frame.index[i] for i in ticks])
        ax.set_xlabel("Time of day")
        ax.grid(axis="y", alpha=0.2, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return fig

    _plot_profile(profile, case.selected_key)
    return


@app.cell
def _(mo, summary):
    _capped = summary[summary["termination"].astype(str).str.contains(
        "maxTimeLimit|aborted", case=False, na=False
    )]
    _note = ""
    if len(_capped):
        _labels = ", ".join(sorted(_capped["label"]))
        _worst = _capped["mip_gap"].max()
        _note = f"""

    {len(_capped)} of these hit the sweep's per-point time limit rather than
    proving optimality — {_labels}. Their schedules are feasible and their bills
    are real, but each is an upper bound: the true optimum is up to
    {_worst:.1%} cheaper. That is the difficulty cliff the middle panel above is
    really showing. Near the plant's ceiling there is so little room left that
    the search itself becomes the expensive part."""

    mo.md(f"## Every month solved{_note}")
    return


@app.cell
def _(mo, summary):
    _table = summary.sort_values("capacity_fraction")[
        [
            "demand_af",
            "capacity_fraction",
            "delivered_af",
            "operating_cost",
            "restarts",
            "offspec_af",
            "peak_kw",
            "termination",
            "mip_gap",
        ]
    ].rename(
        columns={
            "demand_af": "Obligation (AF)",
            "capacity_fraction": "Of capacity",
            "delivered_af": "Delivered (AF)",
            "operating_cost": "Operating cost ($)",
            "restarts": "RO startups",
            "offspec_af": "Off-spec (AF)",
            "peak_kw": "Peak draw (kW)",
            "termination": "Termination",
            "mip_gap": "Gap",
        }
    )
    _table["Of capacity"] = (_table["Of capacity"] * 100).round(0).astype(int).astype(str) + "%"
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

Regenerate with `python tools/sweep.py examples/desalination_scheduling` in an
environment with a Gurobi licence — the exact month is a non-convex MIQCP and
HiGHS cannot take the problem.
"""
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
