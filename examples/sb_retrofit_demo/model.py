"""Santa Barbara retrofit demo: the same month, before and after a flexibility retrofit.

Two cases of the Santa Barbara seawater desalination plant, both a full calendar
month at 15-minute resolution against the same tariff, both delivering the same
water:

* **A_before** -- the operators' baselining experiment. The existing plant (30
  membrane vessels per skid, recovery fixed at 46.5%) sheds one of its three RO
  skids every weekday from 16:00 to 21:00 and runs flat out otherwise. Nothing
  is optimized: every status is fixed, so the solve is only an evaluation.
* **B_after** -- the retrofit. Six more pressure vessels per skid (+20% membrane
  area, 36 vessels), so a skid can take up to 405 m^3/hr of feed at the same flux
  and runs ~4% more efficiently; recovery free in 40-50%; and a battery sized to
  carry the load the plant can never shed (intake pump + pretreatment) through
  the whole peak window. The schedule is optimized, and it has to deliver
  exactly the water A_before delivered.

The plant itself -- intake, three pretreatment/RO trains with bypass tees, the
permeate header, post-treatment with its 45-minute restart, the product pump --
is the one in ``desalination_scheduling``, rebuilt here with the case's numbers
passed in rather than read from module constants. See that module's docstring
for why every block is wired the way it is.

**Solver.** The upstream example unfixes RO recovery, which makes
``permeate == recovery * feed`` bilinear and the month a non-convex MIQCP that
needs Gurobi. Recovery appears in nothing else, so here it is replaced by the
exact linear window ``0.40 * feed <= permeate <= 0.50 * feed``
(:func:`extensions.linear_recovery_window`) and the month is a plain MILP that
HiGHS solves. Case B goes through the upstream ``solve_relax_and_fix``.

Sources: *Energy flexible design and operation reduce reverse osmosis system
costs* (Santa Barbara case study) for the upgrade sizing and efficiency, and the
operator's baselining experiment for case A.

Run ``python model.py`` to solve both cases and print the KPIs.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexops import (
    BatteryModel,
    ConstantEnergyIntensityModel,
    Feed,
    FlexCosting,
    Mixer,
    PlantBlock,
    Product,
    Pump,
    ReverseOsmosis,
    SimpleAqueousFlow,
    Splitter,
    TimeBlock,
)
from flexops.logic import (
    add_conditional,
    add_startup_delay,
    add_startup_shutdown,
    add_status,
    register_parallel_group,
)

from flex_pse_examples.examples.desalination_scheduling import model as desal
from flex_pse_examples.examples.sb_retrofit_demo import extensions as ext

HERE = Path(__file__).parent
TARIFF_PATH = HERE / "tariff.json"
CONFIG_PATH = HERE / "config.json"

# ---------------------------------------------------------------------------
# Horizon and the plant both cases share
# ---------------------------------------------------------------------------

START_DATE = "2026-07-01"
END_DATE = "2026-08-01"
TIME_STEP_HR = 0.25
HORIZON_HOURS = (
    datetime.fromisoformat(END_DATE) - datetime.fromisoformat(START_DATE)
).total_seconds() / 3600.0

N_TRAINS = 3
INTAKE_FLOW_M3_PER_HR = 1063.5      # fixed-duty intake, never sheds
INTAKE_KWH_M3 = 0.157
PRETREATMENT_KWH_M3 = 0.01
POST_TREATMENT_KWH_M3 = 0.11
PRODUCT_PUMP_KWH_M3 = 0.3
RECUP_STEPS = 3                     # 45 min post-treatment restart
RECOVERY_MIN, RECOVERY_MAX = 0.40, 0.50
PEAK_START_HOUR, PEAK_END_HOUR = 16, 21

M3_PER_AF = desal.M3_PER_AF

#: The existing skids: 30 pressure vessels, on/off at rated feed.
BASE_FEED_M3_PER_HR = 337.67
BASE_RO_KWH_M3 = 3.34

#: Every number that distinguishes the two cases, in one place.
CASES: dict[str, dict] = {
    "A_before": dict(
        label="Before retrofit",
        description="Existing plant; operators shed one skid on weekday peaks",
        vessels_per_skid=30,
        rated_feed=BASE_FEED_M3_PER_HR,
        min_feed=BASE_FEED_M3_PER_HR,
        ro_intensity=BASE_RO_KWH_M3,
        recovery_fixed=0.465,
        battery=None,
        schedule="operator_weekday_peak_shed",
    ),
    "B_after": dict(
        label="After retrofit",
        description="+20% membrane area and a battery for the non-sheddable load; optimized",
        vessels_per_skid=36,
        rated_feed=1.2 * BASE_FEED_M3_PER_HR,
        min_feed=BASE_FEED_M3_PER_HR,
        ro_intensity=BASE_RO_KWH_M3 * 0.96,
        recovery_fixed=None,
        battery=dict(
            hours=5.0,
            round_to_kw=50,
            eta_charge=0.95,
            eta_discharge=0.95,
            soc_min=0.1,
            soc_max=0.95,
            initial_soc=0.5,
        ),
        schedule="optimized",
    ),
}

#: HiGHS, with its log on. A 3% gap rather than upstream's 0.2%: HiGHS on a
#: month of unit commitment spends nearly all its time closing that last
#: fraction of a percent, and the bill moves far less than the gap.
SOLVER = "highs"
SOLVER_OPTIONS = {"highs": {"mip_rel_gap": 0.03, "time_limit": 1800, "output_flag": True}}

#: Relax-and-fix threshold: a relaxed status within this of 0 or 1 is taken as
#: decided and fixed before the MIP. Higher than upstream's 0.1 because the
#: semicontinuous skids and the battery leave more statuses fractional; the
#: routine steps down on its own if fixing makes the month infeasible.
FIX_TOL = 0.35

_M3_HR = pyunits.m**3 / pyunits.hr
_KWH_M3 = pyunits.kWh / pyunits.m**3


def non_sheddable_kw() -> float:
    """The draw the plant can never shed: the fixed intake and its pretreatment."""
    return INTAKE_FLOW_M3_PER_HR * (INTAKE_KWH_M3 + PRETREATMENT_KWH_M3)


def battery_rating(case: str = "B_after") -> tuple[float, float]:
    """``(kW, kWh)`` of the case's battery: the non-sheddable load, rounded up,
    carried through the whole peak window."""
    cfg = CASES[case]["battery"]
    kw = math.ceil(non_sheddable_kw() / cfg["round_to_kw"]) * cfg["round_to_kw"]
    return float(kw), float(kw * cfg["hours"])


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def construct_plant(m, *, rated_feed, min_feed, ro_intensity, recovery_fixed=None):
    """Wire the desalination flowsheet with one case's skid parameters.

    Same blocks, arcs and operating rules as
    ``desalination_scheduling.model.construct_plant``; only the skid sizing,
    RO intensity and recovery handling are arguments.
    """
    tb = m.time_block
    plant = m.plant
    plant.trains = pyo.Set(initialize=range(N_TRAINS), ordered=True)

    plant.intake_pump = Pump(
        property_package=m.properties,
        power_relation="constant_intensity",
        energy_intensity=INTAKE_KWH_M3 * _KWH_M3,
        costing_package=m.costing,
    )
    plant.pretreatment = ConstantEnergyIntensityModel(
        plant.trains,
        property_package=m.properties,
        energy_intensity=PRETREATMENT_KWH_M3 * _KWH_M3,
        costing_package=m.costing,
    )
    plant.ro = ReverseOsmosis(
        plant.trains,
        property_package=m.properties,
        recovery=recovery_fixed if recovery_fixed is not None else 0.465,
        recovery_min=RECOVERY_MIN,
        recovery_max=RECOVERY_MAX,
        energy_intensity=ro_intensity * _KWH_M3,
        costing_package=m.costing,
    )
    plant.post_treatment = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=POST_TREATMENT_KWH_M3 * _KWH_M3,
        costing_package=m.costing,
    )
    plant.product_pump = Pump(
        property_package=m.properties,
        power_relation="constant_intensity",
        energy_intensity=PRODUCT_PUMP_KWH_M3 * _KWH_M3,
        costing_package=m.costing,
    )
    plant.feed_header = Splitter(
        property_package=m.properties,
        outlet_names=tuple(f"t{i}" for i in plant.trains),
    )
    plant.train_split = Splitter(
        plant.trains, property_package=m.properties, outlet_names=("ro", "bypass")
    )
    plant.permeate_split = Splitter(
        plant.trains, property_package=m.properties, outlet_names=("header", "offspec")
    )
    plant.permeate_header = Mixer(
        property_package=m.properties,
        inlet_names=tuple(f"t{i}" for i in plant.trains),
    )
    plant.outfall = Product(
        property_package=m.properties,
        inlet_names=tuple(
            f"{s}_{i}" for i in plant.trains for s in ("brine", "bypass", "offspec")
        ),
        resource_name="outfall",
    )
    plant.potable = Product(
        property_package=m.properties,
        resource_name="potable_water",
        min_demand=0 * pyunits.m**3,
        demand_basis="horizon",
    )
    plant.seawater = Feed(property_package=m.properties, resource_name="seawater")

    plant.seawater_to_intake = Arc(
        source=plant.seawater.outlet_a, destination=plant.intake_pump.inlet
    )
    plant.intake_to_header = Arc(
        source=plant.intake_pump.outlet, destination=plant.feed_header.inlet
    )

    @plant.Arc(plant.trains)
    def header_to_pretreatment(b, i):
        return (plant.feed_header.find_component(f"outlet_t{i}"), plant.pretreatment[i].inlet)

    @plant.Arc(plant.trains)
    def pretreatment_to_split(b, i):
        return (plant.pretreatment[i].outlet, plant.train_split[i].inlet)

    @plant.Arc(plant.trains)
    def split_to_ro(b, i):
        return (plant.train_split[i].outlet_ro, plant.ro[i].inlet)

    @plant.Arc(plant.trains)
    def ro_to_permeate_split(b, i):
        return (plant.ro[i].outlet_a, plant.permeate_split[i].inlet)

    @plant.Arc(plant.trains)
    def permeate_split_to_header(b, i):
        return (
            plant.permeate_split[i].outlet_header,
            plant.permeate_header.find_component(f"inlet_t{i}"),
        )

    plant.header_to_post = Arc(
        source=plant.permeate_header.outlet, destination=plant.post_treatment.inlet
    )
    plant.post_to_product = Arc(
        source=plant.post_treatment.outlet, destination=plant.product_pump.inlet
    )
    plant.product_to_delivery = Arc(
        source=plant.product_pump.outlet, destination=plant.potable.inlet_a
    )

    @plant.Arc(plant.trains)
    def brine_to_outfall(b, i):
        return (plant.ro[i].outlet_b, plant.outfall.find_component(f"inlet_brine_{i}"))

    @plant.Arc(plant.trains)
    def bypass_to_outfall(b, i):
        return (
            plant.train_split[i].outlet_bypass,
            plant.outfall.find_component(f"inlet_bypass_{i}"),
        )

    @plant.Arc(plant.trains)
    def offspec_to_outfall(b, i):
        return (
            plant.permeate_split[i].outlet_offspec,
            plant.outfall.find_component(f"inlet_offspec_{i}"),
        )

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Operating rules, exactly as upstream.
    always_on = [(plant.intake_pump, INTAKE_FLOW_M3_PER_HR)]
    always_on += [(plant.pretreatment[i], INTAKE_FLOW_M3_PER_HR) for i in plant.trains]
    for unit, max_flow in always_on:
        add_status(unit, unit.flow_in, 0 * _M3_HR, max_flow * _M3_HR)
        for t in tb.time_index:
            unit.status[t].fix(1)
    for t in tb.time_index:
        plant.seawater.withdrawal[t].fix(INTAKE_FLOW_M3_PER_HR)
    for i in plant.trains:
        add_status(plant.ro[i], plant.ro[i].feed, min_feed * _M3_HR, rated_feed * _M3_HR)
        add_startup_shutdown(
            plant.ro[i], plant.ro[i].status, min_uptime=RECUP_STEPS, min_downtime=1
        )
    add_status(
        plant.post_treatment,
        plant.post_treatment.flow_in,
        0 * _M3_HR,
        N_TRAINS * rated_feed * RECOVERY_MAX * _M3_HR,
    )
    add_startup_delay(plant.post_treatment, plant.ro[0], k=RECUP_STEPS, initially_on=True)
    register_parallel_group([plant.ro[i] for i in plant.trains])
    add_conditional(plant.ro[0], plant.ro[1], then="on")

    # Recovery: fixed for case A; an exact linear window otherwise.
    for i in plant.trains:
        if recovery_fixed is not None:
            plant.ro[i].recovery.fix(recovery_fixed)
        else:
            ext.linear_recovery_window(
                m,
                unit=f"plant.ro[{i}]",
                recovery_min=RECOVERY_MIN,
                recovery_max=RECOVERY_MAX,
            )


def build_case(name: str, *, demand_m3: float = 0.0, start=START_DATE, end=END_DATE):
    """Build one case, ready to solve.

    Args:
        name: A key of :data:`CASES`.
        demand_m3: Product water the month must deliver (m^3). Case A is
            evaluated with 0 -- its schedule is fixed, so it makes what it makes;
            case B is handed what A delivered.
        start, end: The horizon. Shorter than a month only for tests.
    """
    cfg = CASES[name]
    m = pyo.ConcreteModel(name=f"sb_retrofit_{name}")
    m.case = name
    m.time_block = TimeBlock(
        start_date=start, end_date=end, time_step=TIME_STEP_HR * pyunits.hr
    )
    m.properties = SimpleAqueousFlow()
    m.costing = FlexCosting(time_block=m.time_block, tariff_file=str(TARIFF_PATH))
    m.plant = PlantBlock(time_block=m.time_block)
    construct_plant(
        m,
        rated_feed=cfg["rated_feed"],
        min_feed=cfg["min_feed"],
        ro_intensity=cfg["ro_intensity"],
        recovery_fixed=cfg["recovery_fixed"],
    )

    if cfg["battery"] is not None:
        b = cfg["battery"]
        kw, kwh = battery_rating(name)
        m.plant.battery = BatteryModel(
            capacity=kwh * pyunits.kWh,
            power_charge_max=kw * pyunits.kW,
            power_discharge_max=kw * pyunits.kW,
            eta_charge=b["eta_charge"],
            eta_discharge=b["eta_discharge"],
            soc_min=b["soc_min"],
            soc_max=b["soc_max"],
            initial_soc=b["initial_soc"],
            costing_package=m.costing,
        )
        ext.terminal_charge(m, battery="plant.battery")

    if cfg["schedule"] == "operator_weekday_peak_shed":
        ext.operator_weekday_peak_shed(
            m,
            swing_unit="plant.ro[2]",
            always_on=["plant.ro[0]", "plant.ro[1]", "plant.post_treatment"],
            peak_start_hour=PEAK_START_HOUR,
            peak_end_hour=PEAK_END_HOUR,
        )

    if cfg["schedule"] == "operator_weekday_peak_shed" and not demand_m3:
        # The operators deliver everything the fixed schedule makes.
        demand_m3 = operator_product_m3(m)
    m.plant.potable.delivery_min.set_value(demand_m3)
    m.demand_af = demand_m3 / M3_PER_AF
    ext.no_export(m)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost, sense=pyo.minimize)
    m.is_relaxed = False
    return m


def operator_product_m3(m) -> float:
    """Water a fixed-status case makes: every online skid at rated feed and its fixed recovery."""
    cfg = CASES[m.case]
    online = sum(
        pyo.value(m.plant.ro[i].status[t]) for i in m.plant.trains for t in m.time_block.time_index
    )
    return online * cfg["rated_feed"] * cfg["recovery_fixed"] * TIME_STEP_HR


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


def _use_highs_options():
    """Point the upstream solve helpers at HiGHS with this example's options."""
    desal.SOLVER = SOLVER
    desal.SOLVER_OPTIONS.update(SOLVER_OPTIONS)


def solve_case(m, *, time_limit: float | None = None):
    """Solve a built case. A is a plain evaluation; B goes through relax-and-fix."""
    from time import perf_counter

    _use_highs_options()
    if time_limit is not None:
        desal.SOLVER_OPTIONS["highs"]["time_limit"] = time_limit
    started = perf_counter()
    if CASES[m.case]["schedule"] == "optimized":
        results = desal.solve_relax_and_fix(m, tol=FIX_TOL, prefer=SOLVER)
    else:
        results = desal.solve_model(m, prefer=SOLVER)
        m.relaxation_gap = None
    m.solve_seconds = perf_counter() - started
    m.termination = str(results.solver.termination_condition)
    return results


def report_cost(m):
    """The month's bill: the solved dispatch evaluated by EECO (not the objective)."""
    return m.costing.report_cost(m)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def results_frame(m):
    """The solved month as a DataFrame indexed by timestamp."""
    import pandas as pd

    tb = m.time_block
    ti = list(tb.time_index)
    plant = m.plant
    trains = list(plant.trains)
    v = pyo.value

    def binary(var, t):
        return float(round(v(var[t])))

    data = {}
    for i in trains:
        data[f"ro{i}_feed_m3_per_hr"] = [v(plant.ro[i].feed[t]) for t in ti]
        data[f"ro{i}_permeate_m3_per_hr"] = [v(plant.ro[i].permeate[t]) for t in ti]
        data[f"ro{i}_status"] = [binary(plant.ro[i].status, t) for t in ti]
    data["ro0_startup"] = [
        float(round(v(plant.ro[0].startup[t]))) if t in plant.ro[0].startup else 0.0
        for t in ti
    ]
    data["trains_online"] = [sum(data[f"ro{i}_status"][k] for i in trains) for k in range(len(ti))]
    data["post_treatment_status"] = [binary(plant.post_treatment.status, t) for t in ti]
    data["permeate_m3_per_hr"] = [sum(v(plant.ro[i].permeate[t]) for i in trains) for t in ti]
    data["offspec_permeate_m3_per_hr"] = [
        sum(v(plant.permeate_split[i].flow_out_offspec[t]) for i in trains) for t in ti
    ]
    data["product_m3_per_hr"] = [v(plant.potable.delivery[t]) for t in ti]
    data["intake_pump_kw"] = [v(plant.intake_pump.power_electrical[t]) for t in ti]
    data["pretreatment_kw"] = [
        sum(v(plant.pretreatment[i].power_electrical[t]) for i in trains) for t in ti
    ]
    data["ro_kw"] = [sum(v(plant.ro[i].power_electrical[t]) for i in trains) for t in ti]
    data["post_treatment_kw"] = [v(plant.post_treatment.power_electrical[t]) for t in ti]
    data["product_pump_kw"] = [v(plant.product_pump.power_electrical[t]) for t in ti]
    data["plant_kw"] = [
        data["intake_pump_kw"][k] + data["pretreatment_kw"][k] + data["ro_kw"][k]
        + data["post_treatment_kw"][k] + data["product_pump_kw"][k]
        for k in range(len(ti))
    ]
    if plant.component("battery") is not None:
        bat = plant.battery
        data["battery_charge_kw"] = [v(bat.power_charge[t]) for t in ti]
        data["battery_discharge_kw"] = [v(bat.power_discharge[t]) for t in ti]
        data["battery_soc"] = [v(bat.soc[t]) for t in ti]
    else:
        zeros = [0.0] * len(ti)
        data["battery_charge_kw"] = zeros
        data["battery_discharge_kw"] = zeros
        data["battery_soc"] = zeros
    data["grid_kw"] = [v(m.costing.aggregate_power[t, "electrical"]) for t in ti]

    frame = pd.DataFrame(data, index=pd.DatetimeIndex(tb.datetime_index, name="timestamp"))
    frame = frame.mask(frame.abs() < 1e-6, 0.0)
    frame["in_peak"] = (
        (frame.index.hour >= PEAK_START_HOUR) & (frame.index.hour < PEAK_END_HOUR)
    ).astype(float)
    return frame


def kpis(m, frame, *, reference=None) -> dict:
    """The month's KPIs for one case.

    Args:
        m: The solved model.
        frame: Its :func:`results_frame`.
        reference: Case A's KPI dict, for the load-shed figure. ``None`` for A
            itself (then A is its own reference).
    """
    dt = TIME_STEP_HR
    peak = frame["in_peak"] > 0
    product_m3 = float(frame["product_m3_per_hr"].sum() * dt)
    grid_kwh = float(frame["grid_kw"].sum() * dt)
    plant_mean_all = float(frame["plant_kw"].mean())
    ref_plant_mean = reference["plant_kw_mean_all_hours"] if reference else plant_mean_all
    peak_mean = float(frame.loc[peak, "grid_kw"].mean())
    return {
        "bill_usd": float(report_cost(m).operating.electricity),
        "water_m3": product_m3,
        "water_af": product_m3 / M3_PER_AF,
        "peak_grid_kw_max": float(frame.loc[peak, "grid_kw"].max()),
        "peak_grid_kw_mean": peak_mean,
        "plant_kw_mean_all_hours": plant_mean_all,
        "load_shed_pct": 100.0 * (1.0 - peak_mean / ref_plant_mean),
        "kwh_per_m3": grid_kwh / product_m3 if product_m3 else float("nan"),
        "restarts": float(frame["ro0_startup"].sum()),
        "offspec_m3": float(frame["offspec_permeate_m3_per_hr"].sum() * dt),
        "solve_seconds": float(getattr(m, "solve_seconds", float("nan"))),
        "relaxation_gap": float(m.relaxation_gap) if m.relaxation_gap is not None else 0.0,
        "mip_gap": float(getattr(m, "mip_gap", 0.0) or 0.0),
    }


def run_both(*, start=START_DATE, end=END_DATE, time_limit=None):
    """Solve A, then B against A's delivered water. Returns ``{case: (m, frame, kpis)}``."""
    out = {}
    a = build_case("A_before", start=start, end=end)
    solve_case(a, time_limit=time_limit)
    fa = results_frame(a)
    ka = kpis(a, fa)
    out["A_before"] = (a, fa, ka)

    b = build_case("B_after", demand_m3=ka["water_m3"], start=start, end=end)
    solve_case(b, time_limit=time_limit)
    fb = results_frame(b)
    kb = kpis(b, fb, reference=ka)
    kb["savings_pct"] = 100.0 * (1.0 - kb["bill_usd"] / ka["bill_usd"])
    ka["savings_pct"] = 0.0
    out["B_after"] = (b, fb, kb)
    return out


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def write_config(path=None):
    """Emit case B's model as a flex-pse spec (``config.json``)."""
    import warnings

    import flexops
    from flexcore.config.io import dump_spec
    from flexcore.exceptions import FlexEmitWarning

    target = Path(path or CONFIG_PATH)
    m = build_case("B_after")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexEmitWarning)
        spec = flexops.emit_model(m, relative_to=target.parent)
    dump_spec(spec, target)
    return target


if __name__ == "__main__":
    results = run_both()
    for name, (_, _, k) in results.items():
        print(f"\n{CASES[name]['label']}")
        for key, value in k.items():
            print(f"  {key:>24}: {value:,.3f}")
