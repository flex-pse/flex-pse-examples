"""Desalination scheduling: build the flex-pse model from code and write to `config.json`

The plant. An **intake pump** lifts raw seawater into a feed header that splits
across **three parallel pretreatment units**; each pretreatment unit feeds its
own **reverse-osmosis skid**, or sends water around it through an **RO
bypass**. The three permeate streams recombine into a shared **post-treatment**
step and a **product water pump**, which is where the plant's water obligation
is measured. Brine, bypassed pretreated water and off-spec permeate all leave
to the ocean outfall and are not processed further::

                          ┌─► pretreatment[0] ─► RO[0] ─┬─► brine ─► ocean
    seawater ─► intake ───┼─► pretreatment[1] ─► RO[1] ─┤          (permeate)
                 pump     └─► pretreatment[2] ─► RO[2] ─┘             │
                                                                      ▼
      product water ◄── product pump ◄── post-treatment ◄── permeate header

Each train in detail — the RO bypass tees off the pretreatment outlet::

                         ┌─► RO bypass ────────────────────────┐
                         │                                     ▼
    feed ─► pretreat[i] ─┴─► RO[i] ─┬─► brine ───────────────► ocean outfall
                                    │                          ▲
                                    └─► permeate ─┬─ off-spec ─┘
                                                  └─► on-spec ─► post-treat

Every unit but the RO skids is a constant energy intensity — kWh per m^3 of
whatever passes through it. The RO skids are constant intensity, which may 
be replaced by a custom surrogate *and* a split: ``permeate == recovery * feed``,
brine takes the rest.

The **RO bypass** is what lets the intake pump and pretreatment keep flowing
while a skid is down: with the skid off its ``feed`` is pinned to zero by its
status binary, so every drop the train still pretreats has to leave through the
bypass. It is an open tee, not gated on status, so a skid running turned down
can bypass part of its feed too. Bypassed water earns nothing and still pays
intake + pretreatment energy, so at the flow floors shipped here (``0 m^3/hr``
for both always-on units) the optimizer leaves the bypass shut; it starts
carrying water as soon as either unit is given a real turndown floor.

Recuperation is a **plant-level** outage, not a per-skid one. ``post_treatment``
may only run while ``ro[0]`` is online and at least 45 minutes past its restart;
the symmetry breaking below makes ``ro[0]`` the last skid off and the first back
on, so "``ro[0]`` down" is exactly "the RO system down". While post-treatment is
out, the permeate header is pinned to zero and the *whole* permeate stream --
every train's, not just the restarting one's -- goes to the outfall off-spec.
Stepping the train count down from three to two therefore costs nothing; only a
full system restart does.

The plant owes **265 acre-feet a month** of product water, and nothing says 
*when*. That is the whole degree of freedom: with no product storage anywhere 
in the flowsheet, the only way to dodge an expensive tariff hour is to make less 
water in it and more water elsewhere.

"""

import json
from pathlib import Path

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexops.core.plant_block import PlantBlock
from flexops.core.time_block import TimeBlock
from flexops.costing import FlexCosting
from flexops.logic import (
    add_startup_shutdown,
    add_status,
    register_parallel_group,
    relax,
)
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import ConstantEnergyIntensityModel, Pump, ReverseOsmosis

CONFIG_PATH = Path(__file__).parent / "config.json"

#: Product water owed per month, measured downstream of the product pump. The
#: horizon below is exactly one calendar month, so this is the horizon's
#: obligation as it stands -- no proration.
DEMAND_AF_PER_MONTH = 265.0

#: The plant's sizing, module-level because the charts need it too: ``min_feed``
#: in particular is not recoverable from a solved model -- it goes in as a bound
#: inside ``add_status`` and is not a variable anywhere afterwards.
N_TRAINS = 3
RECOVERY = 0.465
RATED_FEED_M3_PER_HR = 337.67  # m3/hr of feed, per skid
MIN_FEED_M3_PER_HR = 337.67  # m3/hr; below this a skid must shut off entirely
RECUP_STEPS = 3  # 45 min of off-spec permeate after a startup, at 15-min steps

_M3_HR = pyunits.m**3 / pyunits.hr
_KWH_M3 = pyunits.kWh / pyunits.m**3

#: Flows and draws below this read as zero. A MILP solver returns an idle train
#: as a few nanolitres either side of nothing, and a negative flow plots as a
#: notch below the axis.
_FLOW_TOL_M3_HR = 1e-6

def main(relax_integrality: bool = False):
    """Build the desalination scheduling model, ready to solve.

    Args:
        relax_integrality: ``True`` to drop the RO skids' ``status``, ``startup``
            and ``shutdown`` -- and post-treatment's ``status`` -- from
            ``Binary`` to ``UnitInterval``. The month carries ~27k binaries and
            does not solve in a sitting; the relaxation solves in seconds, but
            it is **optimistic** -- a fractional status runs a skid below its
            turndown floor, and a fractional ``ro[0]`` startup lets
            post-treatment stay fractionally online through the recuperation
            window instead of paying for all of it. Read its cost as a lower
            bound.

    Returns:
        A ``pyo.ConcreteModel`` with the plant, the product-delivery
        obligation, and ``objective`` (the horizon operating cost).
    """
    m = pyo.ConcreteModel(name="desalination_example")

    m.time_block = TimeBlock(
        start_date= "2026-07-01",
        end_date= "2026-08-01",
        time_step= 0.25 * pyunits.hr,
    )
    m.properties = SimpleAqueousFlow()
    # The tariff is the one piece with no code form yet -- it is ~80 EECO rate
    # rows, so it stays in config.json.
    tariff = json.loads(CONFIG_PATH.read_text())["tariff"]
    m.costing = FlexCosting(time_block=m.time_block, tariff=tariff)

    m.plant = PlantBlock(time_block=m.time_block)
    construct_plant(m)

    # Applied after the plant is wired so it catches every binary the logic
    # pieces attached. Domain-only: constraints, bounds and fixed values are
    # untouched, so the always-on units stay pinned at 1.
    m.is_relaxed = bool(relax_integrality)
    if m.is_relaxed:
        for i in m.plant.trains:
            relax(m.plant.ro[i])
        # Post-treatment's status is a free binary too -- one per time step, so
        # leaving it out would keep 2976 of them and the relaxation would no
        # longer solve in seconds.
        relax(m.plant.post_treatment)

    add_demand_and_objective(m)

    return m


def demand_m3() -> float:
    """The horizon's product-water obligation in m^3.

    The horizon is exactly one calendar month, so this is
    :data:`DEMAND_AF_PER_MONTH` converted straight across -- no proration.
    """
    return pyo.value(
        pyunits.convert(
            DEMAND_AF_PER_MONTH * pyunits.acre * pyunits.foot, pyunits.m**3
        )
    )


def add_demand_and_objective(m):
    """Attach the monthly water obligation and the operating-cost objective."""
    tb = m.time_block
    plant = m.plant

    dt_hours = pyo.value(pyunits.convert(tb.dt, pyunits.hr))
    # TODO(#67) - replace this constraint with plant.add_product_demand once
    # flex-pse ships it.
    plant.product_delivery = pyo.Constraint(
        expr=sum(
            plant.product_pump.outlet_state.flow_vol_phase[t, "Liq"]
            for t in tb.time_index
        )
        * dt_hours
        * pyunits.hr
        >= demand_m3() * pyunits.m**3,
        doc="Deliver at least the month's obligation, measured downstream of "
        "the product pump. A volume, not a profile -- placing it in the "
        "cheap hours is the whole degree of freedom.",
    )

    m.costing.cost_process()
    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )
    return m


def solve_model(m, prefer: str | None = None):
    """Solve ``m`` with the solver and options from ``config.json``.

    Records the achieved relative MIP gap on ``m.mip_gap`` and snaps power
    noise to exactly zero -- a MILP solver returns an idle plant as a few
    nanowatts either side, and EECO refuses to bill a negative draw.

    Unlike ``model.solve_model``, a run that stops on the time limit is
    returned rather than raised on: the exact MILP here is expected to hit it,
    and the caller reports the termination condition and the gap.

    Args:
        m: A model from :func:`main`.
        prefer: Overrides ``solver.prefer`` from the config.

    Returns:
        The Pyomo results object.
    """
    from flexcore.solvers import get_solver

    solver_cfg = json.loads(CONFIG_PATH.read_text())["solver"]
    solver = get_solver(model=m, prefer=prefer or solver_cfg["prefer"])
    results = solver.solve(m, options=dict(solver_cfg["options"]))

    problem = results.problem
    lower, upper = problem.lower_bound, problem.upper_bound
    m.mip_gap = (
        0.0
        if lower is None or upper is None or not upper
        else abs(upper - lower) / abs(upper)
    )

    for entry in m.costing.aggregate_power.values():
        if entry.value is not None and abs(entry.value) < 1e-6:
            entry.set_value(0.0, skip_validation=True)
    return results

def construct_plant(m):

    tb = m.time_block    
    plant = m.plant
    
    plant.trains = pyo.Set(
        initialize=range(N_TRAINS),
        ordered=True,
        doc="Parallel treatment trains: one pretreatment unit + one RO skid each.",
    )
    
    recovery = RECOVERY
    rated_feed = RATED_FEED_M3_PER_HR
    min_feed = MIN_FEED_M3_PER_HR
    recup_steps = RECUP_STEPS

    plant.intake_pump = Pump(
        property_package=m.properties,
        power_relation="constant_intensity",
        energy_intensity=0.157 * _KWH_M3,
        costing_package=m.costing,
    )
    
    plant.pretreatment = ConstantEnergyIntensityModel(
        plant.trains,
        property_package=m.properties,
        energy_intensity=0.01 * _KWH_M3,
        costing_package=m.costing,
    )
    
    plant.ro = ReverseOsmosis(
        plant.trains,
        property_package=m.properties,
        recovery=recovery,
        recovery_min=0.4,
        recovery_max=0.5,
        energy_intensity=3.34 * _KWH_M3,
        costing_package=m.costing,
    )
    
    plant.post_treatment = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.11 * _KWH_M3,
        costing_package=m.costing,
    )
    
    plant.product_pump = Pump(
        property_package=m.properties,
        power_relation="constant_intensity",
        energy_intensity=0.3 * _KWH_M3,
        costing_package=m.costing,
    )
    
    # No pretreatment -> RO arc: an Arc is one-to-one, and each pretreatment
    # outlet now tees two ways -- skid feed, or the RO bypass. That tee is the
    # ro_feed_split balance below.
    plant.post_to_product = Arc(
        source=plant.post_treatment.outlet,
        destination=plant.product_pump.inlet,
        doc="Post-treated water to the product pump.",
    )
    
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    
    # TODO - replace this constraint with a mixer
    @plant.Constraint(
        tb.time_index,
        doc="Feed header: the intake pump's discharge splits across the trains.",
    )
    def feed_header(b, t):
        return plant.intake_pump.outlet_state.flow_vol_phase[t, "Liq"] == sum(
            plant.pretreatment[i].inlet_state.flow_vol_phase[t, "Liq"]
            for i in plant.trains
        )
    
    # TODO - replace this variable and constraint with a splitter
    plant.ro_bypass = pyo.Var(
        plant.trains,
        tb.time_index,
        domain=pyo.NonNegativeReals,
        bounds=(0, rated_feed),
        units=_M3_HR,
        doc="Pretreated water routed around this train's RO skid, straight to "
        "the ocean outfall.",
    )

    @plant.Constraint(
        plant.trains,
        tb.time_index,
        doc="RO bypass tee: pretreated water either feeds the skid or goes "
        "around it to the outfall.",
    )
    def ro_feed_split(b, i, t):
        return plant.pretreatment[i].flow_out[t] == (
            plant.ro[i].feed[t] + plant.ro_bypass[i, t]
        )

    # TODO - replace this variable and constraint with an automatic call to logic
    plant.permeate_to_header = pyo.Var(
        plant.trains,
        tb.time_index,
        domain=pyo.NonNegativeReals,
        units=_M3_HR,
        doc="On-spec permeate each train sends to the post-treatment header.",
    )

    @plant.Constraint(
        plant.trains,
        tb.time_index,
        doc="A train can send no more to the header than it makes.",
    )
    def permeate_to_header_cap(b, i, t):
        return plant.permeate_to_header[i, t] <= plant.ro[i].permeate[t]

    @plant.Constraint(
        tb.time_index,
        doc="Permeate header: the trains' on-spec permeate recombines into "
        "post-treatment.",
    )
    def permeate_header(b, t):
        return plant.post_treatment.inlet_state.flow_vol_phase[t, "Liq"] == sum(
            plant.permeate_to_header[i, t] for i in plant.trains
        )

    @plant.Expression(
        tb.time_index,
        doc="Ocean outfall: brine, bypassed pretreated water, and the permeate "
        "the plant dumps while post-treatment recuperates. Reporting only -- "
        "every term is already determined, so this adds no degrees of freedom.",
    )
    def outfall(b, t):
        return sum(
            plant.ro[i].brine[t]
            + plant.ro_bypass[i, t]
            + (plant.ro[i].permeate[t] - plant.permeate_to_header[i, t])
            for i in plant.trains
        )

    # The intake pump and the pretreatment units run whenever the plant does,
    # so their status is pinned at 1 and never becomes a decision. The floor is
    # 0, not a turndown limit: both sit upstream of the skids, so a train
    # shutting down has to be free to pass less water -- and the intake pump
    # none at all with every train off.
    always_on = [(plant.intake_pump, 3 * rated_feed)]
    always_on += [(plant.pretreatment[i], rated_feed) for i in plant.trains]
    for unit, max_flow in always_on:
        add_status(unit, unit.flow_in, 0 * _M3_HR, max_flow * _M3_HR)
        for t in tb.time_index:
            unit.status[t].fix(1)

    # Each skid is off, or running anywhere in [min_feed, rated_feed] -- the
    # plant's whole scheduling freedom, and what makes this a MILP. The
    # min-uptime is load-bearing on ro[0]: post_treatment_recuperation below
    # linearizes "45 minutes past the restart" as a rolling sum of startups, and
    # that is only valid while at most one startup falls inside any window.
    # On ro[1] and ro[2] it earns nothing now that their cycling is free -- it
    # just keeps the optimizer from switching a skid every 15 minutes.
    for i in plant.trains:
        add_status(
            plant.ro[i], plant.ro[i].feed, min_feed * _M3_HR, rated_feed * _M3_HR
        )
        add_startup_shutdown(
            plant.ro[i], plant.ro[i].status, min_uptime=recup_steps, min_downtime=1
        )

    # Post-treatment is the one downstream unit that is *not* always on: it is
    # what the recuperation window takes out. Floor of 0, so it can carry the
    # whole header or nothing.
    add_status(
        plant.post_treatment,
        plant.post_treatment.flow_in,
        0 * _M3_HR,
        N_TRAINS * rated_feed * recovery * _M3_HR,
    )

    @plant.Constraint(
        tb.time_index,
        doc="Recuperation: post-treatment may run only while the RO system is up "
        "and at least 45 minutes past its restart. While it is out, "
        "status_max_link pins its inlet -- the permeate header -- to zero, so "
        "every drop all three skids make goes to the outfall off-spec.",
    )
    def post_treatment_recuperation(b, t):
        # TODO(flex-pse#68) - replace with flexops.logic.add_startup_delay once
        # it enforces the whole [t-k, t] window instead of lagging the upstream
        # status by k. As shipped it samples only status[t-k], so a system that
        # restarts inside the window passes, and it never checks status[t] at
        # all -- post-treatment could run with every skid off.
        #
        # ro[0] stands in for the whole RO system: the symmetry breaking below
        # makes it the last skid off and the first back on. startup is indexed
        # from t=1 only, so a system already running at t=0 counts as warmed up
        # rather than as having just started.
        started = sum(
            plant.ro[0].startup[s] for s in range(max(t - recup_steps + 1, 1), t + 1)
        )
        return plant.post_treatment.status[t] <= plant.ro[0].status[t] - started

    # Interchangeable skids make the MILP degenerate: many equal-cost solutions
    # differ only in which skid is on. Pin the order -- skid 2 only runs if 1 is
    # running, skid 1 only if 0 is. register_parallel_group chains "units[i] on
    # implies units[i+1] on", so the skids go in last-first.
    #
    # The reversal is load-bearing, and flex-pse issue #64 is about to make it
    # wrong: register_parallel_group and break_parallel_symmetry take opposite
    # list orders today, and #64 resolves by reversing register_parallel_group
    # to match. Drop the reversed() when that lands, or this chain inverts
    # silently and ro[0] becomes the *first* skid off.
    register_parallel_group([plant.ro[i] for i in reversed(list(plant.trains))])


def peak_window_hours() -> list[int]:
    """Return the hours the tariff's demand-charge rows cover.

    Read off the tariff rather than hardcoded, so a tariff swap in
    ``config.json`` moves the shaded bands in :func:`visualize` with it.
    """
    tariff = json.loads(CONFIG_PATH.read_text())["tariff"]
    hours: set[int] = set()
    for row in tariff["tariff_data"]:
        if row.get("type") == "demand":
            hours.update(range(int(row["hour_start"]), int(row["hour_end"])))
    return sorted(hours)


def results_frame(m):
    """Return the solved schedule as a pandas DataFrame indexed by timestamp.

    Args:
        m: A solved model from :func:`main`.

    Returns:
        A DataFrame with per-train RO feed, status, startup and bypass; the
        plant's feed / permeate / brine / outfall / product flows; on-spec and
        off-spec permeate; the per-stage and aggregate power draw; the number of
        trains online and recuperating; post-treatment's on/off status; and the
        tariff energy price.
    """
    import pandas as pd

    from flexops.costing import load_tariff, price_series

    tb = m.time_block
    ti = list(tb.time_index)
    plant = m.plant
    trains = list(plant.trains)

    def _report_binary(var, t):
        # Under the LP relaxation the binary is a UnitInterval Var, so the value
        # is reported as it stands rather than rounded into a decision the model
        # never made.
        value = pyo.value(var[t])
        return value if m.is_relaxed else float(round(value))

    def status(i, t):
        return _report_binary(plant.ro[i].status, t)

    def post_treatment_status(t):
        return _report_binary(plant.post_treatment.status, t)

    def startup(i, t):
        """Startup indicator, or 0 at t=0 where the model declares none."""
        unit = plant.ro[i]
        if t not in unit.startup:
            return 0.0
        value = pyo.value(unit.startup[t])
        return value if m.is_relaxed else float(round(value))

    data = {}
    for i in trains:
        data[f"ro{i}_feed_m3_per_hr"] = [pyo.value(plant.ro[i].feed[t]) for t in ti]
        data[f"ro{i}_status"] = [status(i, t) for t in ti]
        data[f"ro{i}_startup"] = [startup(i, t) for t in ti]
        data[f"ro{i}_bypass_m3_per_hr"] = [
            pyo.value(plant.ro_bypass[i, t]) for t in ti
        ]

    data["trains_online"] = [sum(status(i, t) for i in trains) for t in ti]
    data["post_treatment_status"] = [post_treatment_status(t) for t in ti]
    # Recuperation is a plant state now, not a per-skid one: post-treatment is
    # out while the skids run, so every train online is recuperating at once.
    data["trains_recuperating"] = [
        sum(status(i, t) for i in trains) * (1.0 - post_treatment_status(t))
        for t in ti
    ]
    data["feed_m3_per_hr"] = [
        pyo.value(plant.intake_pump.inlet_state.flow_vol_phase[t, "Liq"]) for t in ti
    ]
    data["permeate_m3_per_hr"] = [
        sum(pyo.value(plant.ro[i].permeate[t]) for i in trains) for t in ti
    ]
    data["onspec_permeate_m3_per_hr"] = [
        sum(pyo.value(plant.permeate_to_header[i, t]) for i in trains) for t in ti
    ]
    # Off-spec permeate is diverted into the brine line, so the outfall carries
    # both. Reported apart as well, since it is the recuperation penalty made
    # visible: water the plant paid full power to make and then threw away.
    data["offspec_permeate_m3_per_hr"] = [
        sum(
            pyo.value(plant.ro[i].permeate[t] - plant.permeate_to_header[i, t])
            for i in trains
        )
        for t in ti
    ]
    data["bypass_m3_per_hr"] = [
        sum(pyo.value(plant.ro_bypass[i, t]) for i in trains) for t in ti
    ]
    data["brine_m3_per_hr"] = [
        sum(pyo.value(plant.ro[i].brine[t]) for i in trains) for t in ti
    ]
    # The Expression the flowsheet already carries: brine + bypass + off-spec.
    data["outfall_m3_per_hr"] = [pyo.value(plant.outfall[t]) for t in ti]
    data["product_m3_per_hr"] = [
        pyo.value(plant.product_pump.outlet_state.flow_vol_phase[t, "Liq"]) for t in ti
    ]
    data["intake_pump_kw"] = [
        pyo.value(plant.intake_pump.power_electrical[t]) for t in ti
    ]
    data["pretreatment_kw"] = [
        sum(pyo.value(plant.pretreatment[i].power_electrical[t]) for i in trains)
        for t in ti
    ]
    data["ro_kw"] = [
        sum(pyo.value(plant.ro[i].power_electrical[t]) for i in trains) for t in ti
    ]
    data["post_treatment_kw"] = [
        pyo.value(plant.post_treatment.power_electrical[t]) for t in ti
    ]
    data["product_pump_kw"] = [
        pyo.value(plant.product_pump.power_electrical[t]) for t in ti
    ]
    # The two-index form, which is what solve_model snapped.
    data["grid_kw"] = [pyo.value(m.costing.aggregate_power[t, "electrical"]) for t in ti]

    frame = pd.DataFrame(data, index=tb.datetime_index)
    frame.index.name = "timestamp"
    # Snap solver noise (and negative zeros) so an idle train reads as a clean 0.
    frame = frame.mask(frame.abs() < _FLOW_TOL_M3_HR, 0.0)
    frame["energy_price"] = price_series(
        load_tariff(json.loads(CONFIG_PATH.read_text())["tariff"]), tb.datetime_index
    ).to_numpy()
    return frame


def detail_window(frame, *, start=None, days: int = 3):
    """Return the slice of ``frame`` the detail charts render.

    Args:
        frame: A results frame from :func:`results_frame`.
        start: First timestamp of the window, as a string or Timestamp.
            Defaults to the 14th day of the horizon -- read off the frame, not
            from ``config.json``, whose ``reporting.detail_window`` sits in a
            different year than the horizon built here.
        days: Length of the window in days.

    Returns:
        The sub-frame covering ``days`` from ``start``.

    Raises:
        ValueError: If the window falls outside the horizon.
    """
    import pandas as pd

    if start is None:
        begin = frame.index[0] + pd.Timedelta(days=13)
    else:
        begin = pd.Timestamp(start)
    end = begin + pd.Timedelta(days=days)
    window = frame.loc[(frame.index >= begin) & (frame.index < end)]
    if window.empty:
        raise ValueError(
            f"No time points in [{begin}, {end}); the horizon runs "
            f"{frame.index[0]} to {frame.index[-1]}."
        )
    return window


# --- chart ink ---------------------------------------------------------------
# Palette: slots 1-3 of the validated categorical order, plus chart ink. Aqua
# sits below 3:1 on the light surface, so every series that uses it also carries
# a direct label.
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"
_SURFACE, _PEAK_WASH = "#fcfcfb", "#f0efec"

#: The trains are *ordered*, not merely distinct: symmetry breaking forces
#: train[i] on before train[i+1]. An ordinal one-hue ramp says that; three
#: categorical hues would imply they are interchangeable. Steps 250 / 450 / 650
#: of the blue ramp, the light end clear of the surface.
_TRAIN_RAMP = ["#86b6ef", "#2a78d6", "#104281"]

#: Plant sections for the power stack. Three groups, not five units:
#: pretreatment at 0.01 kWh/m3 is a sliver that would cost a hue to show, so it
#: rides with the intake pump on the feed side of the membranes.
_SECTION_COLOR = {
    "intake + pretreatment": _BLUE,
    "RO skids": _ORANGE,
    "post-treatment + product pump": _AQUA,
}


def _style(ax, ylabel):
    """Recessive grid and axes; no top/right spines."""
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    ax.grid(True, axis="y", color=_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=_MUTED, labelsize=8)
    return ax


def visualize(source, *, start=None, days: int = 3):
    """Render the solved schedule's timeseries as a three-panel figure.

    The month is 2976 points at 15-minute resolution, far too dense to read a
    45-minute recuperation window off, so the panels zoom into ``days`` of it at
    the model's full resolution. Shaded bands mark the daily peak window;
    reference lines are labelled out in the right margin, clear of the data.

    The panels, top to bottom:

    1. **Product water**, with the permeate the plant dumps while post-treatment
       recuperates stacked on top -- hatched grey rather than a fourth hue,
       because it is a loss, not a peer category. Product goes to zero for the
       whole window, since post-treatment carries the entire header.
    2. **RO feed by train**, hand-stacked. Most of the time the trains are
       *identical*, so three overlaid traces would paint over each other and
       only the last would be visible; stacked, each band's thickness is that
       train's own feed and the stack top is the plant's draw on the sea.
    3. **Plant power** by section. Nothing generates or stores on site, so the
       stack top *is* the meter -- no separate net-grid trace, which would only
       cover the thin post-treatment band underneath.

    Args:
        source: A solved model from :func:`main`, or a frame from
            :func:`results_frame`. Passing the frame avoids a second extraction
            when the caller already has one.
        start: First timestamp of the window, as a string or Timestamp.
            Defaults to the 14th day of the horizon -- read off the frame, not
            from ``config.json``, whose ``reporting.detail_window`` sits in a
            different year than the horizon built here.
        days: Length of the window in days.

    Returns:
        The matplotlib ``Figure``, unshown -- marimo renders it as a cell value,
        and a headless caller can ``savefig`` it.
    """
    import matplotlib.pyplot as plt

    frame = source if hasattr(source, "columns") else results_frame(source)
    detail = detail_window(frame, start=start, days=days)
    dt_hours = (frame.index[1] - frame.index[0]).total_seconds() / 3600.0

    # Hours elapsed since the window opened: a plain float axis, so the peak
    # bands, the day rules and the margin labels all place arithmetically at any
    # time resolution.
    hours = (detail.index - detail.index[0]).total_seconds().to_numpy() / 3600.0
    span = hours[-1] + dt_hours
    margin = span * 0.22
    label_x = span + span * 0.015
    peak_hours = peak_window_hours()
    # Midnights on the same float axis. The window need not open on one, so this
    # counts from the first midnight inside it and steps back a day as well, to
    # catch the peak window of a partial opening day.
    opening_hour = detail.index[0].hour + detail.index[0].minute / 60.0
    first_midnight = (24.0 - opening_hour) % 24.0
    day_starts = [
        first_midnight + 24.0 * k for k in range(-1, int(span // 24) + 2)
    ]
    # Centres of the peak bands that fall wholly inside the window -- where the
    # one "peak window" caption can sit without being clipped.
    band_centres = [
        _d + (peak_hours[0] + peak_hours[-1] + 1) / 2
        for _d in day_starts
        if _d + peak_hours[0] >= 0 and _d + peak_hours[-1] + 1 <= span
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 7.6), sharex=True, gridspec_kw={"hspace": 0.5},
    )
    fig.patch.set_facecolor(_SURFACE)

    full_tilt = N_TRAINS * RATED_FEED_M3_PER_HR * RECOVERY
    horizon_hours = len(frame) * dt_hours
    flat_rate = demand_m3() / horizon_hours
    peak_mask = frame.index.hour.isin(peak_hours)
    billed_peak_kw = frame.loc[peak_mask, "grid_kw"].max()
    # ro[0] is the whole RO system under the symmetry breaking, and it is the
    # only skid whose restart costs anything now -- summing all three would
    # count the free train-count steps as if they were penalised restarts.
    restarts = frame["ro0_startup"].sum()
    offspec_m3 = frame["offspec_permeate_m3_per_hr"].sum() * dt_hours

    def shade(ax, label=False):
        """Wash each day's peak window; label it once, on the top panel."""
        for _d in day_starts:
            # Clipped to the data region: the right margin is reserved for the
            # reference labels, and a band spilling into it would read as a
            # peak window the window does not actually cover.
            _lo = max(_d + peak_hours[0], 0.0)
            _hi = min(_d + peak_hours[-1] + 1, span)
            if _hi > _lo:
                ax.axvspan(_lo, _hi, color=_PEAK_WASH, lw=0, zorder=0)
        for _d in day_starts:
            if 0 < _d < span:
                ax.axvline(_d, color=_GRID, lw=1, zorder=1)
        # A window short enough to hold no whole peak band gets no caption.
        if label and band_centres:
            # The *last* band, not the first: the legend runs the width of the
            # panel from the left, and the caption would land inside it.
            _mid = max(band_centres)
            ax.annotate(
                "peak window", (_mid, 0.97),
                xycoords=("data", "axes fraction"), color=_MUTED, fontsize=8,
                ha="center", va="top",
            )

    def reference(ax, level, text, va="center"):
        """Dotted reference line, labelled out in the right margin.

        ``va`` pushes a label off its own line, for the case where two
        references sit close enough together to overlap.
        """
        # hlines, not axhline: the line must stop at the data edge so it does
        # not run under its own margin label.
        ax.hlines(level, 0, span, color=_MUTED, lw=1, ls=":", alpha=0.7)
        ax.annotate(text, (label_x, level), color=_MUTED, fontsize=7.5,
                    va=va, ha="left")

    # -- product water and the recuperation loss (m3/hr) ---------------------
    ax = _style(axes[0], "m³/hr")
    shade(ax, label=True)
    ax.set_ylim(0, full_tilt * 1.3)
    # The plant is sized close to its obligation, so these two land within a few
    # percent of each other -- their labels push apart rather than overlap.
    reference(ax, full_tilt, f"full tilt — {N_TRAINS} trains at rated feed",
              va="bottom")
    reference(ax, flat_rate, "flat rate that would meet demand", va="top")
    ax.fill_between(hours, 0, detail["product_m3_per_hr"], step="post",
                    color=_TRAIN_RAMP[1], alpha=0.85, lw=0, zorder=2,
                    label="product water")
    ax.fill_between(hours, detail["product_m3_per_hr"],
                    detail["product_m3_per_hr"] + detail["offspec_permeate_m3_per_hr"],
                    step="post", facecolor="none", hatch="////",
                    edgecolor=_MUTED, lw=0.0, zorder=3,
                    label="off-spec permeate → brine")
    ax.legend(frameon=False, fontsize=8, labelcolor=_MUTED, ncols=2,
              loc="upper left", bbox_to_anchor=(0, 0.99))
    ax.set_title(
        f"Product water — {offspec_m3:,.0f} m³ of the month's permeate went to "
        f"brine across {restarts:,.2f} RO-system restarts",
        color=_INK, fontsize=10, loc="left", pad=10,
    )

    # -- per-train RO feed (m3/hr) ------------------------------------------
    ax = _style(axes[1], "m³/hr")
    shade(ax)
    ax.set_ylim(0, N_TRAINS * RATED_FEED_M3_PER_HR * 1.3)
    _base = detail["ro0_feed_m3_per_hr"] * 0
    for _i in range(N_TRAINS):
        _top = _base + detail[f"ro{_i}_feed_m3_per_hr"]
        ax.fill_between(hours, _base, _top, step="post", color=_TRAIN_RAMP[_i],
                        lw=1.0, edgecolor=_SURFACE, zorder=2,
                        label=f"train {_i + 1}")
        _base = _top
    for _k in range(1, N_TRAINS + 1):
        reference(ax, _k * RATED_FEED_M3_PER_HR,
                  f"{_k} train{'s' if _k > 1 else ''} at rated feed")
    ax.legend(frameon=False, fontsize=8, labelcolor=_MUTED, ncols=N_TRAINS,
              loc="upper left", bbox_to_anchor=(0, 0.99))
    ax.set_title(
        f"RO feed by train — each skid is off, or inside its "
        f"{MIN_FEED_M3_PER_HR:,.0f}–{RATED_FEED_M3_PER_HR:,.0f} m³/hr band; "
        "symmetry breaking fills them in order",
        color=_INK, fontsize=10, loc="left", pad=10,
    )

    # -- power (kW) ----------------------------------------------------------
    # Hand-stacked with step="post" rather than stackplot, which interpolates
    # linearly between points and would round off every switching edge.
    ax = _style(axes[2], "kW")
    shade(ax)
    _sections = {
        "intake + pretreatment": detail["intake_pump_kw"] + detail["pretreatment_kw"],
        "RO skids": detail["ro_kw"],
        "post-treatment + product pump": detail["post_treatment_kw"]
        + detail["product_pump_kw"],
    }
    _base = detail["grid_kw"] * 0
    for _name, _series in _sections.items():
        _top = _base + _series
        ax.fill_between(hours, _base, _top, step="post",
                        color=_SECTION_COLOR[_name], lw=1.0, edgecolor=_SURFACE,
                        zorder=2, label=_name)
        _base = _top
    ax.set_ylim(0, frame["grid_kw"].max() * 1.42)
    reference(ax, billed_peak_kw,
              f"{billed_peak_kw:,.0f} kW — the month's billed peak")
    ax.legend(frameon=False, fontsize=8, labelcolor=_MUTED, ncols=2,
              loc="upper left", bbox_to_anchor=(0, 0.99))
    ax.set_title(
        f"Plant power — the stack top is the meter, and the membranes are "
        f"{frame['ro_kw'].sum() / frame['grid_kw'].sum():.0%} of the bill",
        color=_INK, fontsize=10, loc="left", pad=10,
    )

    # Ticks every 6 h, labelled with the wall-clock hour they land on.
    _ticks = [h for h in range(0, int(span) + 1, 6)]
    axes[-1].set_xlabel("hour of day", color=_MUTED, fontsize=9)
    axes[-1].set_xticks(_ticks)
    axes[-1].set_xticklabels([f"{int(opening_hour + h) % 24:02d}" for h in _ticks])
    # Right margin holds the reference-line labels above.
    axes[-1].set_xlim(0, span + margin)
    # Date each midnight, read straight off the (regularly spaced) index rather
    # than reconstructed with timedelta arithmetic.
    for _d in day_starts:
        _row = int(round(_d / dt_hours))
        if not 0 <= _row < len(detail.index):
            continue
        axes[-1].annotate(
            detail.index[_row].strftime("%b %-d"),
            (_d + 0.3, -0.30), xycoords=("data", "axes fraction"),
            color=_MUTED, fontsize=7.5, ha="left", va="top",
        )
    fig.align_ylabels(axes)
    return fig


if __name__ == "__main__":
    m = main()
    results = solve_model(m)
    print(results)