"""Charts for the Santa Barbara retrofit demo, in a light theme.

Pure pandas + matplotlib, reading the committed Parquet under
``public/sb_retrofit_demo/`` -- nothing here imports Pyomo, so the notebook can
draw without a solver.

* :func:`plot_before_after` -- the hero figure: skids online on top, net water
  production underneath, before on the left and after on the right.
* :func:`plot_power_before_after` -- grid power, the secondary figure that shows
  the battery and the load shed.
* :func:`animate_before_after` -- the hero figure drawn left to right, for video.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path(__file__).parent / "public" / "sb_retrofit_demo"

CASES = ("A_before", "B_after")
TITLES = {"A_before": "BEFORE RETROFIT", "B_after": "AFTER RETROFIT"}

# Light theme. Text and axes are a soft charcoal; the "after" line is a deeper
# cyan than the deck's #32AFCE so it holds contrast on a light background.
BG = "#FBFBF8"
CYAN = "#1F8FB0"
GREY = "#7A8288"
LIGHT = "#5A6166"
GRIDLINE = "#E4E6E7"
INK = "#3C3C3C"
LINE = {"A_before": GREY, "B_after": CYAN}

WINDOW_START = "2026-07-04"
WINDOW_DAYS = 7
PEAK_HOURS = (16, 21)
Y_WATER_MAX = 600.0
Y_POWER_MAX = 2.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
})
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def load_results(data_dir: Path = DATA) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Read the committed sweep: ``({case: frame}, summary)``.

    Frames are indexed by timestamp; ``summary`` has one row per case.
    """
    import duckdb

    series = duckdb.sql(f"SELECT * FROM read_parquet('{data_dir / 'series.parquet'}')").df()
    summary = duckdb.sql(f"SELECT * FROM read_parquet('{data_dir / 'summary.parquet'}')").df()
    frames = {}
    for _, row in summary.iterrows():
        sub = series[series["sweep_id"] == row["sweep_id"]].copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        frames[row["case"]] = sub.set_index("timestamp").sort_index()
    return frames, summary.set_index("case")


def _window(frame: pd.DataFrame, start=WINDOW_START, days=WINDOW_DAYS) -> pd.DataFrame:
    start = pd.Timestamp(start)
    return frame.loc[start : start + pd.Timedelta(days=days) - pd.Timedelta(minutes=1)]


def _style(ax, ylabel: str | None):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT)
    ax.tick_params(colors=LIGHT, labelsize=11)
    ax.grid(True, axis="y", color=GRIDLINE, lw=0.75)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=LIGHT, fontsize=12)


def _shade_peaks(ax, index: pd.DatetimeIndex, label: bool = False):
    days = pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D")
    for k, day in enumerate(days):
        lo = day + pd.Timedelta(hours=PEAK_HOURS[0])
        hi = day + pd.Timedelta(hours=PEAK_HOURS[1])
        ax.axvspan(lo, hi, color=INK, alpha=0.06, lw=0, zorder=0)
        if label and k == 0:
            ax.annotate("peak", (lo + (hi - lo) / 2, 0.97), xycoords=("data", "axes fraction"),
                        ha="center", va="top", color=GREY, fontsize=9)


def _date_ticks(ax, index):
    days = pd.date_range(index[0].normalize(), index[-1].normalize() + pd.Timedelta(days=1), freq="D")
    ax.set_xticks(days)
    ax.set_xticklabels([d.strftime("%b %d") for d in days], rotation=45, ha="right")
    ax.set_xlim(index[0], index[-1] + pd.Timedelta(minutes=15))


def _figure(rows: int, height: float, height_ratios=None):
    fig, axes = plt.subplots(
        rows, 2, figsize=(13.33, height), sharex="col", sharey="row", squeeze=False,
        gridspec_kw={"height_ratios": height_ratios or [1] * rows, "hspace": 0.12, "wspace": 0.16},
    )
    fig.patch.set_facecolor(BG)
    for col, case in enumerate(CASES):
        axes[0, col].set_title(TITLES[case], loc="left", color=CYAN, fontsize=14,
                               fontweight="bold", pad=12, fontfamily="monospace")
    fig.text(0.5, 0.935, "→", color=CYAN, fontsize=34, ha="center", va="center", fontweight="bold")
    return fig, axes


def _draw_hero(axes, frames, average_rate, upto=None):
    """Draw (or redraw, up to ``upto``) the four hero panels."""
    for col, case in enumerate(CASES):
        w = frames[case]
        if upto is not None:
            w = w.loc[:upto]
        top, bottom = axes[0, col], axes[1, col]
        color = LINE[case]
        if len(w):
            top.step(w.index, w["trains_online"], where="post", color=color, lw=2)
            top.fill_between(w.index, 0, w["trains_online"], step="post", color=color, alpha=0.25, lw=0)
            bottom.step(w.index, w["product_m3_per_hr"], where="post", color=color, lw=2,
                        label="net water production")
            if case == "B_after" and w["offspec_permeate_m3_per_hr"].sum() > 0:
                bottom.fill_between(
                    w.index, w["product_m3_per_hr"],
                    w["product_m3_per_hr"] + w["offspec_permeate_m3_per_hr"], step="post",
                    facecolor="none", hatch="////", edgecolor=LIGHT, lw=0,
                    label="off-spec → outfall",
                )


def _frame_hero_axes(axes, index, average_rate):
    for col, case in enumerate(CASES):
        top, bottom = axes[0, col], axes[1, col]
        _style(top, "Skids online" if col == 0 else None)
        _style(bottom, "Net water production (m³/h)" if col == 0 else None)
        top.set_ylim(-0.2, 3.4)
        top.set_yticks([0, 1, 2, 3])
        bottom.set_ylim(0, Y_WATER_MAX)
        for ax in (top, bottom):
            _shade_peaks(ax, index, label=(ax is top and col == 0))
        bottom.axhline(average_rate, color=LIGHT, lw=1, ls=":", alpha=0.8)
        bottom.annotate("average", (index[-1], average_rate), xytext=(-4, 4),
                        textcoords="offset points", ha="right", va="bottom", color=LIGHT, fontsize=9)
        _date_ticks(bottom, index)


def plot_before_after(frames: dict[str, pd.DataFrame], summary: pd.DataFrame | None = None,
                      *, start=WINDOW_START, days=WINDOW_DAYS):
    """The hero figure: skids online (top) and net water production (bottom), A | B."""
    windows = {c: _window(frames[c], start, days) for c in CASES}
    average_rate = frames["A_before"]["product_m3_per_hr"].mean()
    fig, axes = _figure(2, 5.6, height_ratios=[1, 1.6])
    index = windows["A_before"].index
    _frame_hero_axes(axes, index, average_rate)
    _draw_hero(axes, windows, average_rate)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(handles, labels, frameon=False, labelcolor=LIGHT, fontsize=9,
                          loc="upper left", ncols=2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.16)
    return fig


def plot_power_before_after(frames: dict[str, pd.DataFrame], *, start=WINDOW_START, days=WINDOW_DAYS):
    """Grid power in MW, A | B. On B the dashed line is the plant; the gap is the battery."""
    fig, axes = _figure(1, 3.6)
    for col, case in enumerate(CASES):
        w = _window(frames[case], start, days)
        ax = axes[0, col]
        _style(ax, "Power (MW)" if col == 0 else None)
        _shade_peaks(ax, w.index, label=(col == 0))
        ax.step(w.index, w["grid_kw"] / 1000, where="post", color=LINE[case], lw=2, label="grid meter")
        if case == "B_after":
            ax.step(w.index, w["plant_kw"] / 1000, where="post", color=INK, lw=1, ls="--",
                    alpha=0.7, label="plant load")
            ax.legend(frameon=False, labelcolor=LIGHT, fontsize=9, ncols=2,
                      loc="lower right", bbox_to_anchor=(1.0, 1.0))
        ax.set_ylim(0, Y_POWER_MAX)
        _date_ticks(ax, w.index)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.24)
    return fig


def animate_before_after(frames: dict[str, pd.DataFrame], out: Path, *, fps: int = 30,
                         seconds: float = 8.0, hold: float = 1.0,
                         start=WINDOW_START, days=WINDOW_DAYS) -> Path:
    """Write the hero figure, drawn left to right with a playhead, as an MP4."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    windows = {c: _window(frames[c], start, days) for c in CASES}
    index = windows["A_before"].index
    average_rate = frames["A_before"]["product_m3_per_hr"].mean()
    fig, axes = _figure(2, 5.6, height_ratios=[1, 1.6])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.16)

    n_draw = int(fps * seconds)
    n_total = n_draw + int(fps * hold)

    def frame(k):
        for ax in axes.flat:
            ax.cla()
        _frame_hero_axes(axes, index, average_rate)
        pos = min(k, n_draw - 1) / (n_draw - 1)
        upto = index[0] + (index[-1] - index[0]) * pos
        _draw_hero(axes, windows, average_rate, upto=upto)
        if k < n_draw:
            for col in range(2):
                for row in range(2):
                    axes[row, col].axvline(upto, color=INK, alpha=0.35, lw=1.5)
        return []

    anim = FuncAnimation(fig, frame, frames=n_total, blit=False)
    anim.save(str(out), writer=FFMpegWriter(fps=fps, bitrate=4000))
    plt.close(fig)
    return Path(out)


def write_figures(data_dir: Path = DATA, *, video: bool = True) -> list[Path]:
    """Render ``hero.png``, ``power.png`` and (optionally) ``hero.mp4`` from the committed data."""
    frames, summary = load_results(data_dir)
    out = []
    fig = plot_before_after(frames, summary)
    fig.savefig(data_dir / "hero.png", dpi=200, facecolor=BG)
    plt.close(fig)
    out.append(data_dir / "hero.png")
    fig = plot_power_before_after(frames)
    fig.savefig(data_dir / "power.png", dpi=200, facecolor=BG)
    plt.close(fig)
    out.append(data_dir / "power.png")
    if video:
        out.append(animate_before_after(frames, data_dir / "hero.mp4"))
    return out


if __name__ == "__main__":
    for path in write_figures():
        print(path)
