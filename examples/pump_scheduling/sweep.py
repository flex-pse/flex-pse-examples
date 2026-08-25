"""Sweep adapter for pump_scheduling. Driven by `tools/sweep.py`.

The example asks whether operational flexibility substitutes for storage. Its
notebook answers with a 2x2 -- flexible or inflexible, crossed with a large or a
small battery -- but the question is really continuous, so the sweep walks the
battery from nothing to full peak-load rating at *both* strategies and lets the
site plot the two cost curves against each other. The notebook's four cases fall
out as points on those curves.

Each solve is a 24-step LP or MILP on HiGHS and takes a second or two, so the
whole sweep is cheap enough to regenerate on a whim.
"""

import model

#: Battery power ratings to sweep, as fractions of the plant's peak electrical
#: load (the same basis ``config.json`` uses for its ``sizing_options``). 0.0 is
#: no battery at all, which is the honest left-hand end of the trade-off; 1.0 is
#: the config's "large". The config's two named options, 0.4 and 1.0, are both on
#: the grid on purpose, so the notebook's 2x2 is reproduced exactly rather than
#: interpolated.
FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: The reduced sweep CI runs. Both strategies at the config's two named battery
#: sizes -- exactly the notebook's 2x2, which is what the assertions are written
#: against -- and nothing else.
SMOKE_FRACTIONS = (0.4, 1.0)


def points(*, smoke: bool = False) -> list[dict]:
    """Return the sweep points: every battery size at both strategies."""
    fractions = SMOKE_FRACTIONS if smoke else FRACTIONS
    return [
        {
            "label": f"{'flexible' if flexible else 'inflexible'} @ {fraction:.0%}",
            "battery_fraction": fraction,
            "flexible": flexible,
        }
        for flexible in (False, True)
        for fraction in fractions
    ]


def setup(*, smoke: bool = False) -> dict:
    """Load the config once. Every point rebuilds its own model from it."""
    return model.load_config()


def solve_point(cfg: dict, point: dict):
    """Build, solve and summarize one (strategy, battery size) pair.

    Args:
        cfg: The config from :func:`setup`.
        point: One dict from :func:`points`.

    Returns:
        ``(frame, summary)`` -- the results frame indexed by timestamp, and the
        scalars that become this point's row in ``summary.csv``.
    """
    fraction = point["battery_fraction"]
    peak_kw = model.reference_load_kw(cfg)
    duration = cfg["battery"]["storage_duration_hours"]

    # `build_model` takes a *name* from config's sizing_options, not a number, so
    # sweeping off-grid fractions means writing the fraction into a copy of the
    # config under a private key. A fraction of 0 gets no battery block at all,
    # which is not the same as a battery rated at zero: the latter still carries
    # SOC bounds and round-trip losses the plant cannot use.
    sizing = None
    if fraction > 0:
        cfg = {**cfg, "battery": {**cfg["battery"], "sizing_options": {
            **cfg["battery"]["sizing_options"], "_sweep": fraction}}}
        sizing = "_sweep"

    m = model.build_model(cfg, flexible=point["flexible"], battery_sizing=sizing)
    model.solve_model(m)

    report = m.costing.report_cost(m)
    frame = model.results_frame(m, cfg)
    in_peak = frame.index.hour.isin(model.peak_window_hours(cfg))

    return frame, {
        "strategy": "flexible" if point["flexible"] else "inflexible",
        "battery_kw": fraction * peak_kw,
        "battery_kwh": fraction * peak_kw * duration,
        "objective": float(report.operating.total),
        "operating_cost": float(report.operating.total),
        "electricity_cost": float(report.operating.electricity),
        "peak_kw": float(frame.loc[in_peak, "grid_kw"].max()),
        "peak_window_grid_kwh": float(frame.loc[in_peak, "grid_kw"].sum()),
        "solver": "highs",
        "termination": "optimal",  # solve_model asserts it, so anything else raised
        "mip_gap": 0.0,
    }
