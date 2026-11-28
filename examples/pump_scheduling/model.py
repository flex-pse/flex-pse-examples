"""Pump scheduling: build the flex-pse model from code and write it to ``config.json``.

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
  semicontinuous variable, built by the ``semicontinuous_flow`` extension
  around :func:`flexops.logic.add_status`, which attaches an on/off binary and
  the two links pinning flow to it — so the flexible model is a MILP.
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

The plant's boundary is two flex-pse blocks. Raw water comes in through a
**Feed** (``water_source``), which carries the suction pressure as its boundary
condition and meters the withdrawal. Delivered water leaves through a
**Product** (``customer``), whose per-period ``min_demand`` is the hourly demand
profile -- a length-24 series, so the obligation is a limit the plant can be
held to rather than a fix written onto some pump's state.

The tank and the battery are both **cyclic** (``cyclic=True``): each has to end
the day at least as full as it started, so neither strategy can sell off stored
water or energy it never has to buy back.

Operating cost is more than the energy bill. Two **degradation penalties**
(:func:`flexops.logic.add_degradation`) price wear: every on/off transition of
the flexible feed pump, and every kWh the battery discharges. Both are billed
through ``FlexCosting`` as scalar operating costs, so the optimizer weighs a
start, or a cycle, against the energy it saves. The battery's **capital cost**
is annualized with :func:`flexops.costing.capital_recovery_factor` and reported
per day beside the operating cost (:func:`battery_capital_cost_per_day`); it is
reporting only, since the battery is not sized here.

The problem instance -- sizing, demand profile, battery options, wear and
capital costs, the 2x2 -- is in ``inputs.json``, and the tariff is in
``tariff.json``. :func:`build_model` wires a model from them in code and is the
only place input keys are read. ``config.json`` is not hand-written: it is the
flex-pse spec :func:`write_config` emits from the selected scenario's model with
``flexops.emit_model``, and ``flexops.build_model`` turns it back into that same
model (:func:`build_from_config`). The one piece without a spec element of its
own -- the feed pump's semicontinuous band -- is an ``@flexops.extension`` in
``extensions.py``, so the spec names it instead of dropping it.

Two modeling choices are worth calling out, because both were forced by how the
library composes and neither is obvious:

* **The pressure differential is fixed by pinning ``delta_pressure``**, not by
  fixing the inlet and outlet pressure states. ``Pump``'s hydraulic power law is
  ``power == delta_pressure[t] * flow[t] / efficiency``. ``delta_pressure`` is a
  free ``Var`` tied to the port pressures by the ``pressure_change`` constraint,
  so leaving it free makes ``power_eq`` a product of two free Vars — the model
  classifies as MINLP and HiGHS refuses it. Fixing ``delta_pressure`` directly
  is both the literal physical spec and what keeps the power law linear in flow.
  Suction pressure is fixed once, on the raw-water ``Feed``; every downstream
  pressure then follows from ``pressure_change`` and the arcs.

* **Density is a property-package constant, not a boundary condition.**
  ``SimpleAqueousFlow`` takes ``density=`` and builds ``dens_mass_phase`` as an
  always-fixed ``Var`` that it deliberately keeps out of
  :meth:`define_state_vars`. Because it is not a state variable, ``expand_arcs``
  writes no density equality across the arcs, and the model's degrees of freedom
  come out as the 24 hourly feed-flow decisions with no over-constrained set.

"""

import json
import warnings
from pathlib import Path

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops
from flexcore.config.schema import DegradationSpec, DegradationTermSpec
from flexops import (
    BatteryModel,
    Feed,
    FlexCosting,
    Product,
    Pump,
    SimpleAqueousFlow,
    Tank,
    TimeBlock,
)
from flexops.costing import capital_recovery_factor
from flexops.logic import add_degradation

# By package path, not as a sibling module: the extension element in
# config.json names this function by where it was imported from.
from flex_pse_examples.examples.pump_scheduling.extensions import semicontinuous_flow

HERE = Path(__file__).parent
INPUTS_PATH = HERE / "inputs.json"
TARIFF_PATH = HERE / "tariff.json"
#: The emitted flex-pse spec of the selected scenario. Written by
#: :func:`write_config`, never by hand.
CONFIG_PATH = HERE / "config.json"

_BAR = 1e5 * pyunits.Pa
_M3_HR = pyunits.m**3 / pyunits.hr


def load_inputs(path=None) -> dict:
    """Load and validate the example's problem instance.

    Args:
        path: Path to an inputs JSON; defaults to this example's
            :data:`INPUTS_PATH`.

    Returns:
        The parsed inputs dict.

    Raises:
        ValueError: If the demand profile does not have one value per time
            point, if a scenario names an unknown battery sizing, or if the
            inflexible strategy's constant duty falls outside the feed pump's
            operating band.
    """
    cfg = json.loads(Path(path or INPUTS_PATH).read_text())

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
        cfg: The loaded inputs.

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
    balance. The tank is cyclic (it must end at least as full as it started), so
    the pump has to replace every m^3 the customer draws: the mean demand.

    ``Tank``'s holdup balance covers every step, ``t = 0`` included (the
    initial volume is the level *before* step 0's flows), so the balance closes
    over all ``n`` intervals and the whole demand profile counts.

    Args:
        cfg: The loaded inputs.

    Returns:
        The constant feed flow, in m^3/hr.
    """
    demand = cfg["product_demand_m3_per_hr"]
    return sum(demand) / len(demand)


def battery_ratings(cfg: dict, fraction: float) -> tuple[float, float]:
    """Return ``(power_kw, energy_kwh)`` for a battery sized at ``fraction``.

    Both are quoted against :func:`reference_load_kw`: the power rating is
    ``fraction`` of the plant's peak load, and the energy is that rating held
    for ``storage_duration_hours``.
    """
    rate_kw = fraction * reference_load_kw(cfg)
    return rate_kw, rate_kw * cfg["battery"]["storage_duration_hours"]


def battery_capital_cost_per_day(cfg: dict, energy_kwh: float | None) -> float:
    """Return a battery's installed cost annualized and spread over one day.

    ``capital_cost_per_kwh * energy_kwh``, times the capital recovery factor
    ``flexops.costing.capital_recovery_factor(discount_rate, lifetime_years)``,
    divided by 365. This is what makes "a bigger battery" and "flexible pumps"
    comparable on one bill: flexibility costs nothing to build, a battery does.

    Args:
        cfg: The loaded inputs.
        energy_kwh: The battery's energy capacity, or ``None`` for no battery.

    Returns:
        The capital cost per day, in the tariff's currency.
    """
    if not energy_kwh:
        return 0.0
    battery = cfg["battery"]
    crf = capital_recovery_factor(battery["discount_rate"], battery["lifetime_years"])
    return battery["capital_cost_per_kwh"] * energy_kwh * crf / 365.0


def build_model(cfg: dict, *, flexible: bool = True, battery_sizing: str | None = None):
    """Build the pump scheduling model.

    Args:
        cfg: The loaded inputs (see :func:`load_inputs`).
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

    # has_pressure=True is required by the hydraulic power law. Density is a
    # property-package constant rather than a boundary condition (see the module
    # docstring).
    m.properties = SimpleAqueousFlow(
        has_pressure=True,
        density=fluid["density_kg_per_m3"] * pyunits.kg / pyunits.m**3,
    )
    # A file, not the parsed rows: a spec can name a tariff file but not hold a
    # tariff object.
    m.costing = FlexCosting(time_block=tb, tariff_file=str(TARIFF_PATH))

    # The plant boundary. The Feed carries the raw water's boundary state --
    # suction pressure, fixed below -- and meters the withdrawal; the Product
    # meters the delivery and holds it to the hourly demand profile. A
    # period-basis min_demand given as a series sets one limit per step.
    m.water_source = Feed(
        property_package=m.properties,
        resource_name="raw_water",
    )
    m.feed_pump = Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=feed["efficiency"],
        costing_package=m.costing,
    )
    # cyclic: end the day at least as full as it started. Without it the
    # schedule just drains the tank and under-pumps.
    m.tank = Tank(
        property_package=m.properties,
        max_volume=tank_cfg["max_volume_m3"] * pyunits.m**3,
        initial_volume=tank_cfg["initial_volume_m3"] * pyunits.m**3,
        level_min=tank_cfg["level_min"],
        level_max=tank_cfg["level_max"],
        cyclic=True,
    )
    m.product_pump = Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=product["efficiency"],
        costing_package=m.costing,
    )
    m.customer = Product(
        property_package=m.properties,
        resource_name="product_water",
        min_demand=[value * _M3_HR for value in demand],
    )

    m.source_to_feed = Arc(
        source=m.water_source.outlet_a, destination=m.feed_pump.inlet
    )
    m.feed_to_tank = Arc(source=m.feed_pump.outlet, destination=m.tank.inlet)
    m.tank_to_product = Arc(source=m.tank.outlet, destination=m.product_pump.inlet)
    m.product_to_customer = Arc(
        source=m.product_pump.outlet, destination=m.customer.inlet_a
    )
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
        # The plant's one boundary condition: suction pressure, on the Feed's
        # reference outlet. It propagates downstream through the arcs.
        m.water_source.outlet_a_state.pressure[t].fix(
            pyo.value(fluid["suction_pressure_bar"] * _BAR)
        )

    # Fixed on the [t, phase] state itself rather than through a Reference:
    # a spec's fix elements are applied before its extensions run, so they
    # have to name a variable the pump builds on its own.
    feed_flow = m.feed_pump.inlet_state.flow_vol_phase
    if flexible:
        # Off, or anywhere in [min_flow_fraction, 1.0] x rated flow, and every
        # start or stop is billed as wear (see extensions.py).
        rated = feed["rated_flow_m3_per_hr"]
        semicontinuous_flow(
            m,
            unit="feed_pump",
            min_flow=feed["min_flow_fraction"] * rated,
            max_flow=rated,
            start_stop_cost=feed["start_stop_cost"],
        )
    else:
        # 100% duty cycle: one constant flow, held over every step. No status
        # binary, so the feed pump contributes no degrees of freedom -- and
        # never switches, so it carries no start/stop wear.
        flat = inflexible_flow_m3_per_hr(cfg)
        for t in tb.time_index:
            feed_flow[t, "Liq"].fix(flat)

    if battery_sizing is not None:
        battery_cfg = cfg["battery"]
        options = battery_cfg["sizing_options"]
        if battery_sizing not in options:
            raise ValueError(
                f"Unknown battery sizing {battery_sizing!r}; expected one of "
                f"{sorted(options)} or None."
            )
        rate_kw, energy_kwh = battery_ratings(cfg, options[battery_sizing])
        # cyclic: end the day at least as charged as it started, so the
        # schedule cannot sell off stored energy it never has to buy back.
        m.battery = BatteryModel(
            energy_capacity=energy_kwh * pyunits.kWh,
            power_charge_max=rate_kw * pyunits.kW,
            power_discharge_max=rate_kw * pyunits.kW,
            eta_charge=battery_cfg["eta_charge"],
            eta_discharge=battery_cfg["eta_discharge"],
            soc_min=battery_cfg["soc_min"],
            soc_max=battery_cfg["soc_max"],
            initial_soc=battery_cfg["initial_soc"],
            cyclic=True,
            costing_package=m.costing,
        )
        # Wear: a throughput penalty on every kWh discharged, so cycling the
        # battery is no longer free.
        add_degradation(
            m.battery,
            DegradationSpec(
                name="cycling_wear",
                terms=[
                    DegradationTermSpec(
                        kind="throughput",
                        variable="power_discharge",
                        price=battery_cfg["wear_cost_per_kwh"],
                    )
                ],
            ),
            costing=m.costing,
        )

    m.costing.cost_process()
    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )
    # Plain attributes, for reporting. Not part of the spec: a model rebuilt
    # from config.json does not carry them.
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
    # A solver returns an idle grid connection as a few nanowatts either side
    # of zero, and EECO warns on any negative draw it is asked to bill.
    for entry in m.costing.aggregate_power.values():
        if entry.value is not None and abs(entry.value) < 1e-6:
            entry.set_value(0.0, skip_validation=True)
    return results


def write_config(cfg: dict, path=None):
    """Emit the selected scenario's model as a flex-pse spec.

    Builds the scenario named by ``scenarios.selected``, reads its spec back
    with ``flexops.emit_model`` and saves it. ``emit_model`` rebuilds what it
    wrote and warns with ``FlexEmitWarning`` about anything the spec cannot
    express; that warning is an error here, so a spec that would rebuild a
    different model is never written.

    Args:
        cfg: The loaded inputs.
        path: Where to write the spec; defaults to :data:`CONFIG_PATH`. The
            tariff is written relative to its directory.

    Returns:
        The emitted ``FlowsheetSpec``.
    """
    from flexcore.config.io import dump_spec
    from flexcore.exceptions import FlexEmitWarning

    path = Path(path or CONFIG_PATH)
    case = selected_case(cfg)
    m = build_model(cfg, flexible=case["flexible"], battery_sizing=case["battery"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexEmitWarning)
        spec = flexops.emit_model(m, relative_to=path.parent)
    dump_spec(spec, path)
    return spec


def build_from_config(path=None):
    """Build the model ``config.json`` describes, with ``flexops.build_model``.

    Args:
        path: A spec file; defaults to :data:`CONFIG_PATH`.

    Returns:
        The built ``pyo.ConcreteModel``, the same model :func:`build_model`
        makes for the selected scenario.
    """
    return flexops.build_model(Path(path or CONFIG_PATH))


def selected_case(cfg: dict) -> dict:
    """Return the scenario ``scenarios.selected`` names."""
    by_label = {case["label"]: case for case in cfg["scenarios"]["cases"]}
    return by_label[cfg["scenarios"]["selected"]]


def results_frame(m):
    """Return the solved schedule as a pandas DataFrame indexed by timestamp.

    Args:
        m: A solved model from :func:`build_model` or :func:`build_from_config`.

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
        "feed_flow_m3_per_hr": [
            pyo.value(m.feed_pump.inlet_state.flow_vol_phase[t, "Liq"]) for t in ti
        ],
        "feed_status": status,
        "product_flow_m3_per_hr": [
            pyo.value(m.customer.delivery[t]) for t in ti
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
        load_tariff(TARIFF_PATH), tb.datetime_index
    ).to_numpy()
    return frame


def run_scenarios(cfg: dict, cases=None) -> list[dict]:
    """Build, solve, and summarize one scenario per strategy/battery pairing.

    Args:
        cfg: The loaded inputs.
        cases: Scenario dicts with ``label``, ``flexible``, and ``battery``
            keys; defaults to the inputs' ``scenarios.cases`` 2x2.

    Returns:
        One dict per scenario with its label, strategy, battery ratings,
        reported cost breakdown, peak-window grid draw, and the solved model.
    """
    if cases is None:
        cases = cfg["scenarios"]["cases"]

    peak_hours = peak_window_hours()
    scenarios = []
    for case in cases:
        sizing = case["battery"]
        m = build_model(cfg, flexible=case["flexible"], battery_sizing=sizing)
        solve_model(m)
        frame = results_frame(m)
        in_peak = frame.index.hour.isin(peak_hours)
        battery_kw = battery_kwh = None
        if sizing is not None:
            battery_kw, battery_kwh = battery_ratings(
                cfg, cfg["battery"]["sizing_options"][sizing]
            )
        scenarios.append(
            {
                "label": case["label"],
                "flexible": case["flexible"],
                "strategy": "flexible" if case["flexible"] else "inflexible",
                "sizing": sizing,
                "battery_kw": battery_kw,
                "battery_kwh": battery_kwh,
                **cost_summary(cfg, m, battery_kwh),
                "peak_window_grid_kw": float(frame.loc[in_peak, "grid_kw"].max()),
                "peak_window_grid_kwh": float(frame.loc[in_peak, "grid_kw"].sum()),
                "model": m,
                "frame": frame,
            }
        )
    return scenarios


def cost_summary(cfg: dict, m, battery_kwh: float | None) -> dict:
    """Return one solved day's costs, split the way the example reports them.

    Args:
        cfg: The loaded inputs.
        m: A solved model.
        battery_kwh: The battery's energy capacity, or ``None``.

    Returns:
        ``operating_cost`` (the reported operating total: energy plus wear),
        ``electricity_cost`` (the EECO bill), ``wear_cost`` (the degradation
        penalties, billed as scalar operating costs),
        ``battery_capital_cost`` (the battery's annualized capital, per day)
        and ``total_cost`` (operating plus capital).
    """
    report = m.costing.report_cost(m)
    capital = battery_capital_cost_per_day(cfg, battery_kwh)
    return {
        "operating_cost": float(report.operating.total),
        "electricity_cost": float(report.operating.electricity),
        "wear_cost": float(report.operating.scalar),
        "battery_capital_cost": capital,
        "total_cost": float(report.operating.total) + capital,
    }


def peak_window_hours() -> list[int]:
    """Return the hours covered by the tariff's demand-charge rows."""
    hours: set[int] = set()
    for row in json.loads(TARIFF_PATH.read_text())["tariff_data"]:
        if row.get("type") == "demand":
            hours.update(range(int(row["hour_start"]), int(row["hour_end"])))
    return sorted(hours)


if __name__ == "__main__":
    config = load_inputs()
    write_config(config)
    print(f"wrote {CONFIG_PATH.name} ({selected_case(config)['label']})")
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
            f"(wear ${scenario['wear_cost']:5.2f})  "
            f"+capital=${scenario['total_cost']:8.2f}  "
            f"peak-window draw={scenario['peak_window_grid_kw']:6.2f} kW"
        )
