"""Data center under a grid cap: how much shutdown does a battery avoid?

The site asks its utility for 50 MW. The utility can only guarantee 30 MW during
its constrained hours, so the site agrees to curtail then, in return for the full
50 MW the rest of the time. A battery charged while uncapped can carry the site
through the cap. This example sweeps the battery size and reports the revenue
that curtailment costs at each size.

The data center is a ``ConstantEnergyIntensityModel`` that can run anywhere from
0 to full load. It delivers into a ``Product`` priced negatively (revenue). Grid
power is billed on a flat tariff, and a constraint caps the grid draw in the
curtailed hours. With no on/off logic the model is an LP.

All numbers are made up and the cap is synthetic.
"""

import json
import math
from pathlib import Path

import pyomo.environ as pyo
from dateutil.relativedelta import relativedelta
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexops.core.time_block import TimeBlock
from flexops.costing import FlexCosting
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import (
    BatteryModel,
    ConstantEnergyIntensityModel,
    Feed,
    Product,
)

CONFIG_PATH = Path(__file__).parent / "config.json"


def weighted_revenue(mix: dict) -> float:
    """Return revenue per kWh at the meter from a mix of GPU types.

    Each GPU type earns ``price_per_gpu_hour`` and draws ``it_kw_per_gpu``. Types
    are weighted by units shipped, so the result is total revenue per hour divided
    by total IT power, then divided by ``pue`` to convert IT kWh to meter kWh.
    """
    gpus = mix["gpus"]
    revenue_per_hour = sum(g["units_millions"] * g["price_per_gpu_hour"] for g in gpus)
    it_kw = sum(g["units_millions"] * g["it_kw_per_gpu"] for g in gpus)
    return revenue_per_hour / it_kw / mix["pue"]


def load_config(path=None) -> dict:
    """Load the example's config JSON, computing R from the GPU mix if given."""
    cfg = json.loads(Path(path or CONFIG_PATH).read_text())
    dc = cfg["data_center"]
    if dc.get("revenue_per_unit") is None:
        dc["revenue_per_unit"] = round(weighted_revenue(dc["revenue_mix"]), 2)
    return cfg


def full_load_kw(cfg: dict) -> float:
    """Return the data center's full load in kW."""
    return cfg["data_center"]["full_load_mw"] * 1000.0


def capped_steps(cfg: dict, tb) -> list[int]:
    """Return the time indices during which the grid is capped at firm service.

    ``grid.curtailed_fraction`` of the horizon is capped, in events of
    ``grid.event_hours`` each, spread evenly across the horizon. The event count
    is rounded, so the realized fraction (see :func:`realized_curtailed_fraction`)
    can differ slightly from the requested one.
    """
    g = cfg["grid"]
    n = tb.n_points
    ev = max(1, math.ceil(g["event_hours"] / cfg["time"]["time_step_hours"] - 1e-9))
    n_events = round(g["curtailed_fraction"] * n / ev)
    steps: list[int] = []
    for k in range(n_events):
        start = int((k + 0.5) * n / n_events - ev / 2)
        steps.extend(range(max(start, 0), min(start + ev, n)))
    return sorted(set(steps))


def realized_curtailed_fraction(cfg: dict, tb) -> float:
    """Return the share of the horizon actually capped."""
    return len(capped_steps(cfg, tb)) / tb.n_points


def build_model(cfg: dict, *, battery_mw: float = 0.0, firm_only: bool = False):
    """Build the data center model.

    Args:
        cfg: The loaded config.
        battery_mw: Battery power rating in MW (charge and discharge). Energy
            capacity is that times ``storage_duration_hours``. ``0`` omits the
            battery block entirely.
        firm_only: ``True`` for the baseline: the site is held to ``firm_mw``
            at every step, with no curtailment events.

    Returns:
        A built ``pyo.ConcreteModel`` with ``objective`` (negative profit) and,
        when a battery is included, a ``battery`` block.
    """
    time = cfg["time"]
    dc = cfg["data_center"]
    grid = cfg["grid"]

    m = pyo.ConcreteModel(name=cfg["name"])
    m.time_block = TimeBlock(
        start_date=time["start_date"],
        end_date=time["end_date"],
        time_step=time["time_step_hours"] * pyunits.hr,
        # flexops limits a single time block to one calendar month by default.
        # This example compares whole years, so it widens the limit.
        max_length=relativedelta(years=1),
    )
    tb = m.time_block

    m.properties = SimpleAqueousFlow()
    m.costing = FlexCosting(time_block=tb, tariff=cfg["tariff"])

    rate = pyunits.m**3 / pyunits.hr
    m.source = Feed(property_package=m.properties)
    m.site = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=dc["energy_intensity_kwh_per_unit"] * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.sales = Product(
        property_package=m.properties,
        price=-dc["revenue_per_unit"],
        costing_package=m.costing,
    )
    m.source_to_site = Arc(source=m.source.outlet_a, destination=m.site.inlet)
    m.site_to_sales = Arc(source=m.site.outlet, destination=m.sales.inlet_a)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # The site can run anywhere from 0 to full load. One unit per hour draws one
    # kW, so full load in kW is also full output in units/hr.
    full = full_load_kw(cfg) * rate

    @m.Constraint(tb.time_index, doc="The site cannot exceed its full load.")
    def max_load(b, t):
        return b.site.flow_out[t] <= full

    n = tb.n_points
    if battery_mw > 0:
        b = cfg["battery"]
        m.battery = BatteryModel(
            capacity=battery_mw * 1000.0 * b["storage_duration_hours"] * pyunits.kWh,
            power_charge_max=battery_mw * 1000.0 * pyunits.kW,
            power_discharge_max=battery_mw * 1000.0 * pyunits.kW,
            eta_charge=b["eta_charge"],
            eta_discharge=b["eta_discharge"],
            soc_min=b["soc_min"],
            soc_max=b["soc_max"],
            initial_soc=b["initial_soc"],
            costing_package=m.costing,
        )
        m.battery_terminal = pyo.Constraint(
            expr=m.battery.charge[n - 1] >= m.battery.charge_init,
            doc="End the horizon at least as charged as it started.",
        )

    m.costing.cost_process()

    capped = list(tb.time_index) if firm_only else capped_steps(cfg, tb)
    # During an event the grid cap falls by `curtailment_depth` of the curtailable
    # block (full request minus firm service). Depth 1 cuts the site to firm_mw.
    full_mw = dc["full_load_mw"]
    cap_mw = full_mw - grid.get("curtailment_depth", 1.0) * (full_mw - grid["firm_mw"])
    cap_kw = (grid["firm_mw"] if firm_only else cap_mw) * 1000.0

    @m.Constraint(capped, doc="The grid supplies at most the cap in capped hours.")
    def grid_cap(b, t):
        return b.costing.aggregate_electrical_power[t] <= cap_kw

    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )
    m.battery_mw = battery_mw
    # Compute and facility are built for the power the site can actually use.
    m.capacity_mw = grid["firm_mw"] if firm_only else dc["full_load_mw"]
    m.curtailed_fraction = 1.0 if firm_only else realized_curtailed_fraction(cfg, tb)
    return m


def solve_model(m, *, prefer: str = "highs"):
    """Solve ``m`` and assert an optimal termination."""
    from flexcore.solvers import get_solver
    from pyomo.opt import assert_optimal_termination

    results = get_solver(model=m, prefer=prefer).solve(m)
    assert_optimal_termination(results)
    return results


def results_frame(m, cfg: dict):
    """Return the solved schedule as a DataFrame indexed by timestamp."""
    import pandas as pd

    tb = m.time_block
    ti = list(tb.time_index)
    data = {
        "load_kw": [pyo.value(m.site.power_electrical[t]) for t in ti],
        "grid_kw": [pyo.value(m.costing.aggregate_electrical_power[t]) for t in ti],
    }
    if hasattr(m, "battery"):
        data["battery_charge_kw"] = [pyo.value(m.battery.power_charge[t]) for t in ti]
        data["battery_discharge_kw"] = [
            pyo.value(m.battery.power_discharge[t]) for t in ti
        ]
        data["battery_soc"] = [pyo.value(m.battery.soc[t]) for t in ti]
    frame = pd.DataFrame(data, index=tb.datetime_index)
    frame.index.name = "timestamp"
    frame = frame.mask(frame.abs() < 1e-9, 0.0)
    frame["load_fraction"] = frame["load_kw"] / full_load_kw(cfg)
    return frame


def horizon_hours(cfg: dict) -> float:
    """Return the length of the horizon in hours."""
    import datetime

    t = cfg["time"]
    span = datetime.datetime.fromisoformat(t["end_date"]) - datetime.datetime.fromisoformat(t["start_date"])
    return span.total_seconds() / 3600.0


def compute_cost(cfg: dict, capacity_mw: float) -> float:
    """Return the cost of building and running compute for ``capacity_mw``, in dollars.

    Covers servers, networking, facility and non-energy operating costs, spread
    over their lives and prorated to the horizon. Energy is billed separately.
    """
    return cfg["data_center"]["compute_cost_per_mw_year"] * capacity_mw * horizon_hours(cfg) / 8760.0


def battery_cost(cfg: dict, battery_mw: float) -> float:
    """Return the battery's cost over the horizon, in dollars.

    Upfront capex is annualized with a capital recovery factor over the
    battery's life, fixed O&M is added as a share of capex per year, and the
    annual total is prorated to the length of the horizon.
    """
    b = cfg["battery"]
    capex = b["capex_per_kwh"] * battery_mw * 1000.0 * b["storage_duration_hours"]
    r, n = b["discount_rate"], b["lifetime_years"]
    crf = r / (1 - (1 + r) ** -n) if r > 0 else 1.0 / n
    annual = capex * (crf + b["fom_fraction_of_capex_per_year"])
    return annual * horizon_hours(cfg) / 8760.0


def summarize(m, cfg: dict) -> dict:
    """Return headline numbers for a solved model."""
    frame = results_frame(m, cfg)
    dt = cfg["time"]["time_step_hours"]
    full = full_load_kw(cfg)
    ideal_units = full * dt * len(frame)
    served_units = float(frame["load_kw"].sum() * dt)
    revenue = served_units * cfg["data_center"]["revenue_per_unit"]
    grid_cost = float(frame["grid_kw"].sum() * dt * cfg["grid"]["energy_price_per_kwh"])
    return {
        "battery_mw": m.battery_mw,
        "curtailed_time_fraction": m.curtailed_fraction,
        "revenue": revenue,
        "grid_cost": grid_cost,
        "profit": revenue - grid_cost,
        "battery_cost": battery_cost(cfg, m.battery_mw),
        "compute_cost": compute_cost(cfg, m.capacity_mw),
        "net_profit": (
            revenue
            - grid_cost
            - battery_cost(cfg, m.battery_mw)
            - compute_cost(cfg, m.capacity_mw)
        ),
        "curtailed_fraction": 1.0 - served_units / ideal_units,
        "hours_off": float((frame["load_kw"] == 0).sum() * dt),
    }


def with_curtailment(cfg: dict, fraction: float, event_hours: float | None = None) -> dict:
    """Return a copy of ``cfg`` with a different curtailed share of time."""
    grid = {**cfg["grid"], "curtailed_fraction": fraction}
    if event_hours is not None:
        grid["event_hours"] = event_hours
    return {**cfg, "grid": grid}


def compare(cfg: dict, battery_mw: float) -> list[dict]:
    """Solve the three cases on the same week and label them.

    1. 30 MW firm only (the baseline), 2. full request with curtailment and no
    battery, 3. full request with curtailment and a battery of ``battery_mw``.
    """
    cases = [
        ("Firm service only", dict(firm_only=True)),
        ("Full request, curtailable", dict()),
        (f"Full request + {battery_mw:g} MW battery", dict(battery_mw=battery_mw)),
    ]
    out = []
    for label, kw in cases:
        m = build_model(cfg, **kw)
        solve_model(m)
        out.append({"case": label, **summarize(m, cfg), "frame": results_frame(m, cfg)})
    return out


def sweep(cfg: dict, battery_mws=(0, 5, 10, 15, 20, 30)) -> list[dict]:
    """Solve the model at each battery size and summarize."""
    out = []
    for mw in battery_mws:
        m = build_model(cfg, battery_mw=mw)
        solve_model(m)
        out.append(summarize(m, cfg))
    return out


if __name__ == "__main__":
    import sys

    config = load_config()
    if len(sys.argv) > 1:  # e.g. `python model.py 0.10` for 10% of time curtailed
        config = with_curtailment(config, float(sys.argv[1]))
    print(f"curtailed share of time: {config['grid']['curtailed_fraction']:.1%}")
    print(f"{'battery MW':>10} {'profit':>12} {'battery cost':>13} {'net':>12} {'curtailed':>10}")
    for row in sweep(config):
        print(
            f"{row['battery_mw']:>10.0f} {row['profit']:>12,.0f} "
            f"{row['battery_cost']:>13,.0f} {row['net_profit']:>12,.0f} "
            f"{row['curtailed_fraction']:>10.1%}"
        )
