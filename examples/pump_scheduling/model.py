"""Pump scheduling: build the flex-pse model from ``config.json``.

The problem. A **feed pump** lifts water across a fixed pressure differential
into a **storage tank**; a **product pump** draws from that tank and delivers a
fixed user/product demand profile. Electricity is billed on a time-of-use
tariff with an expensive afternoon peak window, so the schedule wants to pump
off-peak and coast through the peak on stored water. A **battery** sits
alongside the plant, sized at either 100% or 40% of the plant's peak electrical
load.

The question this example is built to answer is whether operational flexibility
substitutes for battery capacity, so it runs a 2x2: two feed pump strategies
against two battery sizes.

* **Flexible.** The feed pump is either off, or running anywhere in
  ``[min_flow_fraction, 1.0]`` of its rated flow, and the optimizer picks the
  hourly schedule. That "off, or somewhere in a band" behavior is a
  semicontinuous variable, built by :func:`flexops.logic.add_status`, which
  attaches an on/off binary and the two links pinning flow to it — so the
  flexible model is a MILP.
* **Inflexible.** The feed pump runs at a 100% duty cycle: never off, never
  modulating, pinned for all 24 hours at the single constant flow that meets
  the day's demand (:func:`inflexible_flow_m3_per_hr`). No status binary is
  attached, so this model is an LP — with no battery it would have zero degrees
  of freedom and be a pure simulation.

Pinning the inflexible pump to a *constant* flow rather than to rated flow is
what makes the comparison mean anything: both strategies then move exactly the
same volume of water and buy exactly the same pump energy over the horizon, so
the cost difference is purely a matter of *when* that energy is bought. A pump
pinned at rated flow for 24 hours would instead over-pump the tank by a third
of a day's demand, and the comparison would be dominated by spilled water.

``flexops`` has no config-to-model bridge yet
(:meth:`OpsBlockData.build_from_config` raises ``NotImplementedError``), so
:func:`build_model` does that wiring by hand and is the only place config keys
are read.

Two modeling choices are worth calling out, because both were forced by how the
library composes and neither is obvious:

* **The pressure differential is fixed by pinning ``delta_pressure``**, not by
  fixing the inlet and outlet pressure states. ``Pump``'s hydraulic power law is
  ``power == delta_pressure[t] * flow[t] / efficiency``. ``delta_pressure`` is a
  free ``Var`` tied to the port pressures by the ``pressure_change`` constraint,
  so leaving it free makes ``power_eq`` a product of two free Vars — the model
  classifies as MINLP and HiGHS refuses it. Fixing ``delta_pressure`` directly
  is both the literal physical spec and what keeps the power law linear in flow.
  Suction pressure is fixed once at the plant inlet; every downstream pressure
  then follows from ``pressure_change`` and the arcs.

* **The property package uses ``fixed_density=False``.** With
  ``fixed_density=True`` every state block's ``dens_mass`` is fixed, and while
  :meth:`OpsBlockData.add_pass_through_constraints` skips a fully-fixed state
  variable, Pyomo's ``expand_arcs`` does not — each arc still gets a
  ``dens_mass`` equality. Over two arcs and 24 time points that is 48 equality
  constraints over zero free variables, which shows up as a structural
  singularity (an over-constrained set) and drags the reported degrees of
  freedom to -24. Fixing density once at the plant suction and letting the
  pass-throughs and arcs propagate it leaves a clean model: DoF == the 24 hourly
  feed-flow decisions, and no over-constrained set.
"""

import json
from pathlib import Path

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexops.core.time_block import TimeBlock
from flexops.costing import FlexCosting
from flexops.logic import add_status
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import BatteryModel, Pump, Tank

CONFIG_PATH = Path(__file__).parent / "config.json"

_BAR = 1e5 * pyunits.Pa


def load_config(path=None) -> dict:
    """Load and validate the example's config.

    Args:
        path: Path to a config JSON; defaults to this example's
            :data:`CONFIG_PATH`.

    Returns:
        The parsed config dict.

    Raises:
        ValueError: If the demand profile does not have one value per time
            point, if a scenario names an unknown battery sizing, or if the
            inflexible strategy's constant duty falls outside the feed pump's
            operating band.
    """
    cfg = json.loads(Path(path or CONFIG_PATH).read_text())

    time = cfg["time"]
    span_hours = (
        pyo.value(
            pyunits.convert(
                (
                    _as_datetime(time["end_date"]) - _as_datetime(time["start_date"])
                ).total_seconds()
                * pyunits.s,
                pyunits.hr,
            )
        )
    )
    n_points = int(round(span_hours / time["time_step_hours"]))
    demand = cfg["product_demand_m3_per_hr"]
    if len(demand) != n_points:
        raise ValueError(
            f"product_demand_m3_per_hr has {len(demand)} values but the horizon "
            f"{time['start_date']} -> {time['end_date']} at "
            f"{time['time_step_hours']} h resolution has {n_points} time points."
        )

    options = cfg["battery"]["sizing_options"]
    labels = {case["label"] for case in cfg["scenarios"]["cases"]}
    for case in cfg["scenarios"]["cases"]:
        if case["battery"] not in options:
            raise ValueError(
                f"scenario {case['label']!r} names battery sizing "
                f"{case['battery']!r}, which is not one of {sorted(options)}."
            )
    selected = cfg["scenarios"]["selected"]
    if selected not in labels:
        raise ValueError(
            f"scenarios.selected {selected!r} is not one of {sorted(labels)}."
        )

    # The inflexible strategy pins the feed pump at one flow for the whole
    # horizon, so that flow has to be a point the pump can actually hold.
    feed = cfg["feed_pump"]
    flat = inflexible_flow_m3_per_hr(cfg)
    lower = feed["min_flow_fraction"] * feed["rated_flow_m3_per_hr"]
    if not lower <= flat <= feed["rated_flow_m3_per_hr"]:
        raise ValueError(
            f"the inflexible strategy's constant duty is {flat:.2f} m3/hr, "
            f"outside the feed pump's band of {lower:.2f}-"
            f"{feed['rated_flow_m3_per_hr']:.2f} m3/hr. Adjust the demand "
            f"profile, the rated flow, or min_flow_fraction."
        )
    return cfg


def _as_datetime(value):
    """Parse an ISO-8601 date/datetime string into a ``datetime``."""
    import datetime

    return datetime.datetime.fromisoformat(value)


def _hydraulic_kw(delta_pressure_bar: float, flow_m3_per_hr: float, efficiency: float):
    """Return the hydraulic shaft power in kW, via Pyomo's unit conversion.

    Mirrors ``Pump``'s hydraulic law (``delta_pressure * flow / efficiency``)
    outside the model, so battery sizing can be expressed against the plant's
    peak load without hardcoding a Pa*m^3/hr -> kW factor.

    Args:
        delta_pressure_bar: Pressure differential across the pump, in bar.
        flow_m3_per_hr: Volumetric flow, in m^3/hr.
        efficiency: Pump hydraulic efficiency, a fraction in (0, 1].

    Returns:
        The power draw in kW, as a float.
    """
    expr = (
        (delta_pressure_bar * _BAR)
        * (flow_m3_per_hr * pyunits.m**3 / pyunits.hr)
        / efficiency
    )
    return pyo.value(pyunits.convert(expr, pyunits.kW))


def reference_load_kw(cfg: dict) -> float:
    """Return the plant's peak electrical load in kW: both pumps at max duty.

    This is the basis battery sizing is quoted against — a "100% capacity"
    battery is rated to carry this whole load, an "80%" one four fifths of it.

    Args:
        cfg: The loaded config.

    Returns:
        The peak combined pump load, in kW.
    """
    feed = cfg["feed_pump"]
    product = cfg["product_pump"]
    return _hydraulic_kw(
        feed["delta_pressure_bar"], feed["rated_flow_m3_per_hr"], feed["efficiency"]
    ) + _hydraulic_kw(
        product["delta_pressure_bar"],
        max(cfg["product_demand_m3_per_hr"]),
        product["efficiency"],
    )


def inflexible_flow_m3_per_hr(cfg: dict) -> float:
    """Return the constant feed flow the inflexible strategy holds all horizon.

    The inflexible facility runs its feed pump at a 100% duty cycle, so its one
    degree of freedom — the flow it settles at — is fixed by the horizon's water
    balance: it must deliver the demand profile plus whatever net change the
    terminal tank target asks for.

    ``Tank``'s holdup equation is a backward difference defined on ``t = 1..n-1``
    (``volume[0]`` is pinned to ``initial_volume`` as the rolling-horizon initial
    state), so the balance closes over ``n - 1`` intervals and the demand at
    ``t = 0`` never enters it. The flat duty has to be derived against that same
    convention or the pinned schedule is infeasible by construction.

    Args:
        cfg: The loaded config.

    Returns:
        The constant feed flow, in m^3/hr.
    """
    tank = cfg["tank"]
    demand = cfg["product_demand_m3_per_hr"]
    net_fill = tank["terminal_volume_m3"] - tank["initial_volume_m3"]
    intervals = (len(demand) - 1) * cfg["time"]["time_step_hours"]
    return (sum(demand[1:]) + net_fill) / intervals


def build_model(cfg: dict, *, flexible: bool = True, battery_sizing: str | None = None):
    """Build the pump scheduling model.

    Args:
        cfg: The loaded config (see :func:`load_config`).
        flexible: ``True`` to let the optimizer schedule the feed pump over its
            semicontinuous band (a MILP), ``False`` to pin it at the constant
            duty from :func:`inflexible_flow_m3_per_hr` (an LP).
        battery_sizing: A key of ``battery.sizing_options`` (e.g. ``"large"``)
            to include a battery of that size, or ``None`` to omit the battery.

    Returns:
        A built ``pyo.ConcreteModel`` carrying ``objective`` (the horizon
        operating cost) and, when a battery is included, a ``battery`` block.

    Raises:
        ValueError: If ``battery_sizing`` is not a known sizing option.
    """
    time = cfg["time"]
    fluid = cfg["fluid"]
    feed = cfg["feed_pump"]
    tank_cfg = cfg["tank"]
    product = cfg["product_pump"]
    demand = cfg["product_demand_m3_per_hr"]

    m = pyo.ConcreteModel(name=cfg["name"])
    m.time_block = TimeBlock(
        start_date=time["start_date"],
        end_date=time["end_date"],
        time_step=time["time_step_hours"] * pyunits.hr,
    )
    tb = m.time_block

    # has_pressure=True is required by the hydraulic power law; fixed_density is
    # deliberately False (see the module docstring).
    m.properties = SimpleAqueousFlow(fixed_density=False, has_pressure=True)
    m.costing = FlexCosting(time_block=tb, tariff=cfg["tariff"])

    m.feed_pump = Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=feed["efficiency"],
        costing_package=m.costing,
    )
    m.tank = Tank(
        property_package=m.properties,
        max_volume=tank_cfg["max_volume_m3"] * pyunits.m**3,
        initial_volume=tank_cfg["initial_volume_m3"] * pyunits.m**3,
        level_min=tank_cfg["level_min"],
        level_max=tank_cfg["level_max"],
    )
    m.product_pump = Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=product["efficiency"],
        costing_package=m.costing,
    )

    m.feed_to_tank = Arc(source=m.feed_pump.outlet, destination=m.tank.inlet)
    m.tank_to_product = Arc(source=m.tank.outlet, destination=m.product_pump.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    for t in tb.time_index:
        # Fixed pressure differential across each pump. Pin delta_pressure
        # itself, not the port pressures -- see the module docstring.
        m.feed_pump.delta_pressure[t].fix(
            pyo.value(feed["delta_pressure_bar"] * _BAR)
        )
        m.product_pump.delta_pressure[t].fix(
            pyo.value(product["delta_pressure_bar"] * _BAR)
        )
        # Plant boundary conditions: suction pressure and density enter once, at
        # the feed pump inlet, and propagate downstream through the arcs.
        m.feed_pump.inlet_state.pressure[t].fix(
            pyo.value(fluid["suction_pressure_bar"] * _BAR)
        )
        m.feed_pump.inlet_state.dens_mass[t].fix(fluid["density_kg_per_m3"])
        # The user/product delivery is the plant's fixed obligation.
        m.product_pump.outlet_state.flow_vol_phase[t, "Liq"].fix(demand[t])

    # add_status indexes its output variable by time alone, so expose the
    # [t, phase] flow state as a time-only Reference first (the same idiom
    # Tank.flow_in uses). The inflexible branch reuses the same Reference.
    m.feed_pump.flow_in = pyo.Reference(
        m.feed_pump.inlet_state.flow_vol_phase[:, "Liq"]
    )
    if flexible:
        # Off, or anywhere in [min_flow_fraction, 1.0] x rated flow.
        rated = feed["rated_flow_m3_per_hr"] * pyunits.m**3 / pyunits.hr
        add_status(
            m.feed_pump, m.feed_pump.flow_in, feed["min_flow_fraction"] * rated, rated
        )
    else:
        # 100% duty cycle: one constant flow, held over every interval. No
        # status binary, so the feed pump contributes no degrees of freedom.
        flat = inflexible_flow_m3_per_hr(cfg)
        for t in list(tb.time_index)[1:]:
            m.feed_pump.flow_in[t].fix(flat)

    # t=0 is the rolling-horizon initial-state snapshot: Tank pins volume[0] to
    # initial_volume, so the holdup equation never sees the feed flow there and
    # any water pumped at t=0 simply vanishes. Left free it is a phantom the
    # flexible strategy zeroes out and the inflexible one pays for, which would
    # charge one strategy for a modeling artifact rather than for its schedule.
    m.feed_pump.flow_in[0].fix(0.0)

    n = tb.n_points
    if battery_sizing is not None:
        battery_cfg = cfg["battery"]
        options = battery_cfg["sizing_options"]
        if battery_sizing not in options:
            raise ValueError(
                f"Unknown battery sizing {battery_sizing!r}; expected one of "
                f"{sorted(options)} or None."
            )
        fraction = options[battery_sizing]
        peak_kw = reference_load_kw(cfg)
        rate_kw = fraction * peak_kw
        m.battery = BatteryModel(
            capacity=rate_kw * battery_cfg["storage_duration_hours"] * pyunits.kWh,
            power_charge_max=rate_kw * pyunits.kW,
            power_discharge_max=rate_kw * pyunits.kW,
            eta_charge=battery_cfg["eta_charge"],
            eta_discharge=battery_cfg["eta_discharge"],
            soc_min=battery_cfg["soc_min"],
            soc_max=battery_cfg["soc_max"],
            initial_soc=battery_cfg["initial_soc"],
            costing_package=m.costing,
        )
        # Without this the schedule ends the horizon empty, selling off stored
        # energy it never has to buy back.
        m.battery_terminal = pyo.Constraint(
            expr=m.battery.charge[n - 1] >= m.battery.charge_init,
            doc="End the horizon at least as charged as it started.",
        )

    m.costing.cost_process()
    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )
    # Likewise for water: without a terminal volume the schedule just drains the
    # tank and under-pumps.
    m.tank_terminal = pyo.Constraint(
        expr=m.tank.volume[n - 1] >= tank_cfg["terminal_volume_m3"],
        doc="End the horizon at least as full as the terminal target.",
    )
    # Plain attributes, for reporting.
    m.battery_sizing = battery_sizing
    m.is_flexible = flexible
    return m


def solve_model(m, *, prefer: str = "highs"):
    """Solve ``m`` and assert an optimal termination.

    Args:
        m: A model from :func:`build_model`.
        prefer: Preferred solver name passed to ``flexcore.solvers.get_solver``.

    Returns:
        The Pyomo results object.
    """
    from pyomo.opt import assert_optimal_termination

    from flexcore.solvers import get_solver

    results = get_solver(model=m, prefer=prefer).solve(m)
    assert_optimal_termination(results)
    return results


def results_frame(m, cfg: dict):
    """Return the solved schedule as a pandas DataFrame indexed by timestamp.

    Args:
        m: A solved model from :func:`build_model`.
        cfg: The config ``m`` was built from, used to re-load the tariff for the
            energy-price column.

    Returns:
        A DataFrame with the feed/product flows, the pump on/off status, tank
        volume and level, per-unit and aggregate power, the tariff energy price,
        and (when a battery is present) its charge/discharge power and SOC.
    """
    import pandas as pd

    from flexops.costing import load_tariff, price_series

    tb = m.time_block
    ti = list(tb.time_index)
    # The inflexible strategy has no status binary -- it is on by definition.
    status = (
        [round(pyo.value(m.feed_pump.status[t])) for t in ti]
        if hasattr(m.feed_pump, "status")
        else [1] * len(ti)
    )
    data = {
        "feed_flow_m3_per_hr": [pyo.value(m.feed_pump.flow_in[t]) for t in ti],
        "feed_status": status,
        "product_flow_m3_per_hr": [
            pyo.value(m.product_pump.outlet_state.flow_vol_phase[t, "Liq"]) for t in ti
        ],
        "tank_volume_m3": [pyo.value(m.tank.volume[t]) for t in ti],
        "tank_level": [pyo.value(m.tank.level[t]) for t in ti],
        "feed_pump_kw": [pyo.value(m.feed_pump.power_electrical[t]) for t in ti],
        "product_pump_kw": [pyo.value(m.product_pump.power_electrical[t]) for t in ti],
        "grid_kw": [
            pyo.value(m.costing.aggregate_electrical_power[t]) for t in ti
        ],
    }
    if hasattr(m, "battery"):
        data["battery_charge_kw"] = [pyo.value(m.battery.power_charge[t]) for t in ti]
        data["battery_discharge_kw"] = [
            pyo.value(m.battery.power_discharge[t]) for t in ti
        ]
        data["battery_soc"] = [pyo.value(m.battery.soc[t]) for t in ti]

    frame = pd.DataFrame(data, index=tb.datetime_index)
    frame.index.name = "timestamp"
    # Snap solver noise (and negative zeros) so an idle asset reads as a clean 0.
    frame = frame.mask(frame.abs() < 1e-9, 0.0)
    frame["energy_price"] = price_series(
        load_tariff(cfg["tariff"]), tb.datetime_index
    ).to_numpy()
    return frame


def run_scenarios(cfg: dict, cases=None) -> list[dict]:
    """Build, solve, and summarize one scenario per strategy/battery pairing.

    Args:
        cfg: The loaded config.
        cases: Scenario dicts with ``label``, ``flexible``, and ``battery``
            keys; defaults to the config's ``scenarios.cases`` 2x2.

    Returns:
        One dict per scenario with its label, strategy, battery ratings,
        reported cost breakdown, peak-window grid draw, and the solved model.
    """
    battery_cfg = cfg["battery"]
    if cases is None:
        cases = cfg["scenarios"]["cases"]

    peak_kw = reference_load_kw(cfg)
    peak_hours = peak_window_hours(cfg)
    scenarios = []
    for case in cases:
        sizing = case["battery"]
        m = build_model(cfg, flexible=case["flexible"], battery_sizing=sizing)
        solve_model(m)
        report = m.costing.report_cost(m)
        frame = results_frame(m, cfg)
        in_peak = frame.index.hour.isin(peak_hours)
        fraction = None if sizing is None else battery_cfg["sizing_options"][sizing]
        scenarios.append(
            {
                "label": case["label"],
                "flexible": case["flexible"],
                "strategy": "flexible" if case["flexible"] else "inflexible",
                "sizing": sizing,
                "battery_kw": None if fraction is None else fraction * peak_kw,
                "battery_kwh": (
                    None
                    if fraction is None
                    else fraction * peak_kw * battery_cfg["storage_duration_hours"]
                ),
                "operating_cost": report.operating.total,
                "electricity_cost": report.operating.electricity,
                "peak_window_grid_kw": float(frame.loc[in_peak, "grid_kw"].max()),
                "peak_window_grid_kwh": float(frame.loc[in_peak, "grid_kw"].sum()),
                "model": m,
                "frame": frame,
            }
        )
    return scenarios


def peak_window_hours(cfg: dict) -> list[int]:
    """Return the hours covered by the tariff's demand-charge rows."""
    hours: set[int] = set()
    for row in cfg["tariff"]["tariff_data"]:
        if row.get("type") == "demand":
            hours.update(range(int(row["hour_start"]), int(row["hour_end"])))
    return sorted(hours)


if __name__ == "__main__":
    config = load_config()
    print(f"plant peak electrical load: {reference_load_kw(config):.2f} kW")
    print(f"inflexible constant duty:   {inflexible_flow_m3_per_hr(config):.2f} m3/hr\n")
    for scenario in run_scenarios(config):
        rating = (
            "--"
            if scenario["battery_kw"] is None
            else f"{scenario['battery_kw']:.1f} kW / {scenario['battery_kwh']:.0f} kWh"
        )
        print(
            f"{scenario['label']:>27}  battery={rating:>22}  "
            f"operating=${scenario['operating_cost']:8.2f}  "
            f"peak-window draw={scenario['peak_window_grid_kw']:6.2f} kW"
        )
