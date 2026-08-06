"""Interactive marimo notebook entrypoint for the pump scheduling example."""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        here = Path(__file__).parent
    except NameError:  # pragma: no cover - interactive fallback
        here = Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import model

    return mo, model, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Pump scheduling: does flexibility substitute for storage?

    A two-pump water plant on a time-of-use tariff, and one question:

    > Would you rather have a **bigger battery for an inflexible facility**, or a
    > **smaller battery for a flexible one**?

    ```
    feed pump  ──►  storage tank  ──►  product pump  ──►  user / product flow
    (the strategy    (buffer)          (follows demand)
     variable)
                         battery ── large, or small
    ```

    The two strategies for the **feed pump**:

    * **Inflexible** — a 100% duty cycle. The pump never shuts off and never
      modulates: it is pinned for the whole horizon at the single constant flow
      that meets the day's demand.
    * **Flexible** — the pump is off, or running anywhere between
      **60% and 100%** of rated flow, and the optimizer picks the hourly
      schedule against the tariff. "Off, or somewhere in a band" is a
      semicontinuous variable — `flexops.logic.add_status` attaches the on/off
      binary and the two links that pin flow to it, which makes this a MILP.

    Crossed with two battery sizes, that is a **2×2**. Because the inflexible
    duty is set by the water balance rather than by rated flow, every one of the
    four cases moves the *same* water and buys the *same* total pump energy — so
    the whole cost difference is a matter of **when** that energy is bought.

    The **product pump** draws from the tank to meet the user demand profile
    across a fixed pressure differential. Its flow is fixed by demand, so it is
    an inflexible load under either strategy.

    Everything below is driven by `config.json`; the model wiring lives in
    `model.py`.
    """)
    return


@app.cell
def _(mo, model):
    cfg = model.load_config()
    peak_kw = model.reference_load_kw(cfg)
    peak_hours = model.peak_window_hours(cfg)
    flat_duty = model.inflexible_flow_m3_per_hr(cfg)
    _sizes = cfg["battery"]["sizing_options"]
    _duration = cfg["battery"]["storage_duration_hours"]

    mo.md(
        f"""
        ## The plant, from `config.json`

        | | |
        |---|---|
        | Horizon | `{cfg["time"]["start_date"]}` → `{cfg["time"]["end_date"]}` at {cfg["time"]["time_step_hours"]} h steps |
        | Feed pump | {cfg["feed_pump"]["rated_flow_m3_per_hr"]:.0f} m³/hr rated, flexible over {cfg["feed_pump"]["min_flow_fraction"]:.0%}–100%, Δp = {cfg["feed_pump"]["delta_pressure_bar"]:.1f} bar, η = {cfg["feed_pump"]["efficiency"]:.2f} |
        | Inflexible duty | **{flat_duty:.1f} m³/hr** held every hour — {flat_duty / cfg["feed_pump"]["rated_flow_m3_per_hr"]:.0%} of rated, the flow that closes the day's water balance |
        | Tank | {cfg["tank"]["max_volume_m3"]:.0f} m³ max, starting at {cfg["tank"]["initial_volume_m3"]:.0f} m³, level held in [{cfg["tank"]["level_min"]:.2f}, {cfg["tank"]["level_max"]:.2f}] |
        | Product pump | follows demand, Δp = {cfg["product_pump"]["delta_pressure_bar"]:.1f} bar, η = {cfg["product_pump"]["efficiency"]:.2f} |
        | Product demand | {min(cfg["product_demand_m3_per_hr"]):.0f}–{max(cfg["product_demand_m3_per_hr"]):.0f} m³/hr, {sum(cfg["product_demand_m3_per_hr"]):.0f} m³ over the horizon |
        | Peak window | {peak_hours[0]:02d}:00–{peak_hours[-1] + 1:02d}:00, energy at 4.5× the off-peak price plus a demand charge |
        | **Plant peak load** | **{peak_kw:.1f} kW** — the basis battery sizing is quoted against |

        The **large** battery is {_sizes["large"]:.0%} of that peak load —
        **{_sizes["large"] * peak_kw:.1f} kW / {_sizes["large"] * peak_kw * _duration:.0f} kWh** —
        and the **small** one {_sizes["small"]:.0%}, or
        **{_sizes["small"] * peak_kw:.1f} kW / {_sizes["small"] * peak_kw * _duration:.0f} kWh**.
        """
    )
    return cfg, flat_duty, peak_hours


@app.cell
def _(cfg, model):
    # Solve all four cases in the 2x2.
    scenarios = model.run_scenarios(cfg)
    by_label = {s["label"]: s for s in scenarios}
    base = by_label[cfg["scenarios"]["selected"]]
    frame = base["frame"]

    inflexible = [s for s in scenarios if not s["flexible"]]
    flexible = [s for s in scenarios if s["flexible"]]
    return base, by_label, flexible, frame, inflexible, scenarios


@app.cell
def _():
    # Palette: slots 1-3 of the validated categorical order, plus chart ink.
    # Aqua sits below 3:1 on the light surface, so every series that uses it
    # also carries a direct label (and the table view at the bottom is the
    # documented relief).
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    SURFACE, PEAK_WASH = "#fcfcfb", "#f0efec"

    #: Battery size -> hue, held fixed everywhere the 2x2 is drawn so a colour
    #: always means the same asset rather than a rank within one chart.
    SIZE_COLOR = {"large": BLUE, "small": ORANGE}

    def style(ax, ylabel):
        """Recessive grid and axes; no top/right spines."""
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
        return ax

    return (
        AQUA,
        BLUE,
        INK,
        MUTED,
        ORANGE,
        PEAK_WASH,
        SIZE_COLOR,
        SURFACE,
        style,
    )


@app.cell
def _(mo):
    mo.md("""
    ## The headline: the 2×2

    Each column is a feed pump strategy, each line a battery size. The top row
    is what the pump is doing; the bottom row is what the meter sees.

    The inflexible pump is a flat line by construction, so its battery is the
    *only* thing standing between the plant and the peak window — and a small
    one runs out. The flexible pump pre-fills the tank and switches off for the
    whole window, leaving only the product pump for the battery to carry.

    (Hour 0 is the tank's initial-state snapshot: `Tank`'s holdup equation is a
    backward difference, so flow there never enters the water balance. The feed
    pump is pinned off at hour 0 under *both* strategies, so neither is billed
    for water that the model would otherwise let vanish.)
    """)
    return


@app.cell
def _(
    INK,
    MUTED,
    PEAK_WASH,
    SIZE_COLOR,
    SURFACE,
    cfg,
    flexible,
    inflexible,
    peak_hours,
    plt,
    style,
):
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 5.6), sharex=True,
        gridspec_kw={"hspace": 0.42, "wspace": 0.18},
    )
    fig.patch.set_facecolor(SURFACE)

    _columns = [("Inflexible — 100% duty cycle", inflexible),
                ("Flexible — scheduled against the tariff", flexible)]
    _rated = cfg["feed_pump"]["rated_flow_m3_per_hr"]
    _flow_top = _rated * 1.24
    # Headroom above the tallest trace so the stacked direct labels below have
    # clear space to sit in rather than crossing a line.
    _grid_top = max(
        s["frame"]["grid_kw"].max() for col in _columns for s in col[1]
    ) * 1.45
    # Dashing the small battery is redundant with hue, which is the point: the
    # two feed schedules in the top row are identical, and without a second
    # channel one series would simply be painted over and vanish.
    _dashes = {"large": (None, None), "small": (5, 2)}

    for _col, (_heading, _group) in enumerate(_columns):
        for _row in (0, 1):
            _ax = style(axes[_row][_col], "")
            for _h in peak_hours:
                _ax.axvspan(_h - 0.5, _h + 0.5, color=PEAK_WASH, lw=0, zorder=0)

        _flow_ax, _grid_ax = axes[0][_col], axes[1][_col]
        _flow_ax.set_ylim(0, _flow_top)
        _grid_ax.set_ylim(0, _grid_top)

        for _i, _s in enumerate(_group):
            _hours = _s["frame"].index.hour
            _color = SIZE_COLOR[_s["sizing"]]
            _dash = _dashes[_s["sizing"]]
            for _target, _column_name in ((_flow_ax, "feed_flow_m3_per_hr"),
                                          (_grid_ax, "grid_kw")):
                _target.step(_hours, _s["frame"][_column_name], where="mid",
                             color=_color, lw=2, dashes=_dash,
                             label=f"{_s['sizing']} battery")
            # Direct-label the peak-window draw -- it is what the demand charge
            # bills against. Stacked against the right edge rather than beside
            # each line, which would collide when the two draws are close and
            # run off the axis when they are high.
            _grid_ax.annotate(
                f"{_s['sizing']} battery — {_s['peak_window_grid_kw']:.1f} kW in peak",
                (23.4, _grid_top * (0.95 - 0.105 * _i)),
                color=_color, fontsize=8, va="center", ha="right",
            )

        _flow_ax.set_title(_heading, color=INK, fontsize=10, loc="left", pad=8)
        if _col == 0:
            _flow_ax.set_ylabel("feed pump (m³/hr)", color=MUTED, fontsize=9)
            _grid_ax.set_ylabel("net grid draw (kW)", color=MUTED, fontsize=9)
        _grid_ax.set_xlabel("hour of day", color=MUTED, fontsize=9)
        _grid_ax.set_xticks(range(0, 24, 4))
        _grid_ax.set_xlim(-0.6, 23.6)

    axes[0][0].annotate(
        "peak window", (sum(peak_hours) / len(peak_hours), 0.96),
        xycoords=("data", "axes fraction"), color=MUTED, fontsize=8,
        ha="center", va="top",
    )
    # Hour 0 is the tank's initial-state snapshot, so the feed pump is pinned
    # off there in every case; without this the inflexible line looks like it
    # breaks its own "never off" definition.
    for _ax in axes[0]:
        _ax.annotate(
            "hour 0:\ninitial state", (0.1, _flow_top * 0.30), color=MUTED,
            fontsize=7.5, va="center", ha="left",
        )
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2,
                      loc="upper left", bbox_to_anchor=(0, 0.94))
    fig.align_ylabels(axes[:, 0])
    fig
    return


@app.cell
def _(BLUE, INK, MUTED, ORANGE, SIZE_COLOR, SURFACE, plt, scenarios, style):
    fig2, ax2 = plt.subplots(figsize=(9, 2.6))
    fig2.patch.set_facecolor(SURFACE)
    style(ax2, "")

    # Grouped by strategy so the diagonal comparison -- inflexible+large versus
    # flexible+small -- sits one bar apart and can be read directly.
    _order = ["inflexible", "flexible"]
    _sizes = ["large", "small"]
    _height = 0.34
    for _i, _size in enumerate(_sizes):
        _costs, _ys = [], []
        for _j, _strategy in enumerate(_order):
            _s = next(
                s for s in scenarios
                if s["strategy"] == _strategy and s["sizing"] == _size
            )
            _costs.append(_s["operating_cost"])
            _ys.append(_j + (_i - 0.5) * (_height + 0.03))
        _bars = ax2.barh(_ys, _costs, height=_height, color=SIZE_COLOR[_size],
                         label=f"{_size} battery", zorder=2,
                         edgecolor=SURFACE, lw=1.5)
        ax2.bar_label(_bars, fmt="$%.2f", padding=6, color=INK, fontsize=9)

    ax2.set_yticks(range(len(_order)))
    ax2.set_yticklabels(_order)
    ax2.grid(False, axis="y")
    ax2.grid(True, axis="x", color="#e1e0d9", lw=0.8)
    ax2.set_xlim(0, max(s["operating_cost"] for s in scenarios) * 1.2)
    ax2.invert_yaxis()
    ax2.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2,
               loc="lower right")
    ax2.set_xlabel("operating cost over the horizon ($)", color=MUTED, fontsize=9)
    ax2.set_title("Horizon operating cost — flexibility beats battery size",
                  color=INK, fontsize=10, loc="left", pad=8)
    fig2
    return


@app.cell
def _(mo, scenarios):
    _worst = max(s["operating_cost"] for s in scenarios)
    rows = []
    for _scenario in scenarios:
        rows.append(
            {
                "strategy": _scenario["strategy"],
                "battery": _scenario["sizing"],
                "battery (kW)": f"{_scenario['battery_kw']:.1f}",
                "battery (kWh)": f"{_scenario['battery_kwh']:.0f}",
                "operating cost ($)": f"{_scenario['operating_cost']:.2f}",
                "vs worst case": f"-{(_worst - _scenario['operating_cost']) / _worst:.1%}",
                "peak-window draw (kW)": f"{_scenario['peak_window_grid_kw']:.2f}",
                "peak-window energy (kWh)": f"{_scenario['peak_window_grid_kwh']:.1f}",
            }
        )
    mo.ui.table(rows, selection=None)
    return


@app.cell
def _(base, mo):
    mo.md(f"""
    ## One schedule in full — {base["label"]}

    Shaded bands mark the peak window. Set `scenarios.selected` in
    `config.json` to render a different corner of the 2×2 here.
    """)
    return


@app.cell
def _(
    AQUA,
    BLUE,
    INK,
    MUTED,
    ORANGE,
    PEAK_WASH,
    SURFACE,
    base,
    cfg,
    flat_duty,
    frame,
    peak_hours,
    plt,
    style,
):
    hours = frame.index.hour.to_numpy()
    fig3, axes3 = plt.subplots(
        4, 1, figsize=(9, 9.4), sharex=True, gridspec_kw={"hspace": 0.55},
    )
    fig3.patch.set_facecolor(SURFACE)

    def shade(ax, label=False):
        """Wash the peak window; label it once, on the top panel."""
        for h in peak_hours:
            ax.axvspan(h - 0.5, h + 0.5, color=PEAK_WASH, lw=0, zorder=0)
        if label:
            ax.annotate(
                "peak window", (sum(peak_hours) / len(peak_hours), 0.97),
                xycoords=("data", "axes fraction"), color=MUTED, fontsize=8,
                ha="center", va="top",
            )

    # -- flows (m3/hr) -------------------------------------------------------
    ax = style(axes3[0], "m³/hr")
    shade(ax, label=True)
    rated = cfg["feed_pump"]["rated_flow_m3_per_hr"]
    ax.set_ylim(0, rated * 1.42)
    # Reference lines are labelled in the right margin, clear of the data. The
    # inflexible duty and the min-flow line sit close together, so they push
    # their labels apart vertically rather than both centring on the line.
    for _level, _text, _va in (
        (rated, "100% rated", "center"),
        (flat_duty, "inflexible duty", "bottom"),
        (rated * cfg["feed_pump"]["min_flow_fraction"], "60% — min when on", "top"),
    ):
        # hlines, not axhline: the line must stop at the data edge so it does
        # not run under its own margin label.
        ax.hlines(_level, -0.5, 23.5, color=BLUE, lw=1, ls=":", alpha=0.6)
        ax.annotate(_text, (24.1, _level), color=BLUE, fontsize=7.5,
                    va=_va, ha="left")
    ax.step(hours, frame["feed_flow_m3_per_hr"], where="mid", color=BLUE, lw=2,
            label="feed pump")
    ax.step(hours, frame["product_flow_m3_per_hr"], where="mid", color=ORANGE,
            lw=2, label="product demand")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2,
              loc="upper left", bbox_to_anchor=(0, 0.99))
    ax.set_title("Flows — the feed pump is off, or inside its 60–100% band",
                 color=INK, fontsize=10, loc="left", pad=10)

    # -- tank volume (m3) ----------------------------------------------------
    ax = style(axes3[1], "m³")
    shade(ax)
    max_volume = cfg["tank"]["max_volume_m3"]
    ax.set_ylim(0, max_volume)
    ax.axhspan(cfg["tank"]["level_min"] * max_volume,
               cfg["tank"]["level_max"] * max_volume,
               color=BLUE, alpha=0.07, lw=0, zorder=0)
    ax.plot(hours, frame["tank_volume_m3"], color=BLUE, lw=2)
    ax.annotate("allowed\nlevel band", (24.1, cfg["tank"]["level_max"] * max_volume),
                color=MUTED, fontsize=7.5, va="top", ha="left")
    ax.set_title("Tank volume — filled before the peak, drained through it",
                 color=INK, fontsize=10, loc="left", pad=10)

    # -- power (kW) ----------------------------------------------------------
    # Everything on the meter's load side is stacked, battery charging
    # included, so the net grid line is exactly the stack less any discharge.
    ax = style(axes3[2], "kW")
    shade(ax)
    _stack = [frame["feed_pump_kw"], frame["product_pump_kw"],
              frame["battery_charge_kw"]]
    ax.stackplot(hours, *_stack, colors=[BLUE, ORANGE, AQUA],
                 labels=["feed pump", "product pump", "battery charging"],
                 edgecolor=SURFACE, lw=1.5, zorder=2)
    ax.step(hours, frame["grid_kw"], where="mid", color=INK, lw=2, zorder=3,
            label="net grid draw")
    ax.set_ylim(0, max(frame["grid_kw"].max(), sum(_stack).max()) * 1.38)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=4,
              loc="upper left", bbox_to_anchor=(0, 0.99))
    ax.set_title(
        f"Plant power — {base['label']} "
        "(net grid draw = stack less battery discharge)",
        color=INK, fontsize=10, loc="left", pad=10,
    )

    # -- battery -------------------------------------------------------------
    ax = style(axes3[3], "kW")
    shade(ax)
    _net = frame["battery_charge_kw"] - frame["battery_discharge_kw"]
    _span = max(abs(_net.min()), _net.max()) * 1.75
    ax.set_ylim(-_span, _span)
    ax.fill_between(hours, _net, step="mid", color=AQUA, alpha=0.18, lw=0)
    ax.step(hours, _net, where="mid", color=AQUA, lw=2)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.annotate(f"charging  ▲ {_net.max():.0f} kW", (0.3, _span * 0.72),
                color=AQUA, fontsize=8, va="center")
    ax.annotate(f"discharging  ▼ {-_net.min():.0f} kW", (0.3, -_span * 0.72),
                color=AQUA, fontsize=8, va="center")
    ax.set_title("Battery — charges off-peak, carries the plant through the peak",
                 color=INK, fontsize=10, loc="left", pad=10)

    axes3[-1].set_xlabel("hour of day", color=MUTED, fontsize=9)
    axes3[-1].set_xticks(range(0, 24, 2))
    # Right margin holds the reference-line labels above.
    axes3[-1].set_xlim(-0.6, 27.5)
    fig3.align_ylabels(axes3)
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    ## The full solved schedule

    The table view — every decision variable the charts summarise.
    """)
    return


@app.cell
def _(frame, mo):
    mo.ui.table(frame.round(2).reset_index(), selection=None, page_size=24)
    return


@app.cell
def _(by_label, mo):
    _il = by_label["inflexible + large battery"]
    _is_ = by_label["inflexible + small battery"]
    _fl = by_label["flexible + large battery"]
    _fs = by_label["flexible + small battery"]

    mo.md(f"""
    ## What the model shows

    1. **The smaller battery on the flexible facility wins outright.** It runs at
       **\\${_fs["operating_cost"]:.2f}** against **\\${_il["operating_cost"]:.2f}**
       for the large battery on the inflexible facility — cheaper to operate
       *and* {(1 - _fs["battery_kw"] / _il["battery_kw"]):.0%} less battery to
       buy. That is the answer to the question this example asks.

    2. **Flexibility and storage are substitutes, not complements.** Upgrading
       the *inflexible* plant from the small battery to the large one is worth
       **\\${_is_["operating_cost"] - _il["operating_cost"]:.2f}**. Making the
       same upgrade on the *flexible* plant is worth only
       **\\${_fs["operating_cost"] - _fl["operating_cost"]:.2f}**, because the
       schedule has already removed the load the battery would have covered.

    3. **What each strategy leaves for the battery to carry.** The inflexible
       pump draws its constant duty straight through the peak window, so the
       battery has to cover both pumps; the small one runs out of energy and
       still leaves **{_is_["peak_window_grid_kw"]:.1f} kW** of billable demand.
       The flexible pump pre-fills the tank and shuts off for the whole window,
       leaving only the product pump — which even the small battery carries to
       **{_fs["peak_window_grid_kw"]:.2f} kW**.

    4. **None of this is an energy-efficiency story.** All four cases pump the
       same **{_fs["frame"]["feed_flow_m3_per_hr"].sum():.0f} m³** and buy the
       same **{_fs["frame"]["feed_pump_kw"].sum():.0f} kWh** of feed pump energy.
       The entire spread is *when* that energy is bought, priced by the peak
       energy rate and the demand charge.

    Note these are **operating** costs only: `FlexCosting`'s capital block is a
    v0 placeholder, so the large battery carries no capital penalty here. That
    understates the result rather than flattering it — the smaller battery
    already wins on operating cost alone, before its lower capital cost is
    counted at all.
    """)
    return


if __name__ == "__main__":
    app.run()
