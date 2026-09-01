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

Every box and every junction in those diagrams is a flex-pse block wired with
``Arc``s -- the tees are ``Splitter``s, the permeate header is a ``Mixer``, and
the three streams that cross the facility boundary are a ``Feed`` (seawater) and
two ``Product``s (potable water, and the ocean outfall). None of the mass
balances are written here: a ``Splitter`` carries conservation and deliberately
no split fraction, so the routing stays the decision the objective makes. The
one constraint this module still writes by hand is the recuperation window, and
it is marked with the flex-pse issue that would remove it.

Every unit but the RO skids is a constant energy intensity — kWh per m^3 of
whatever passes through it. The RO skids are constant intensity, which may
be replaced by a custom surrogate *and* a split: ``permeate == recovery * feed``,
brine takes the rest. Note the RO intensity is quoted per m^3 of **permeate**,
which is the basis ``ReverseOsmosis`` applies it on.

The intake pump is **fixed duty**: 1063.5 m^3/hr of seawater every step, no
turndown, whatever the skids are doing. Three skids at rated feed can swallow
only 1013.01 of that, so the plant always lifts and pretreats at least 50.49
m^3/hr more than its membranes can take, and far more with a skid down. How the
feed header splits that flow across the trains is *not* pinned -- only mass
conservation is -- but an even third is what a solved model reports, since it is
the split that lets all three skids run at rated feed.

The **RO bypass** is where the surplus goes, and it is what lets the intake pump
and pretreatment keep flowing while a skid is down: with the skid off its
``feed`` is pinned to zero by its status binary, so every drop that train
pretreats leaves through the bypass. It is an open tee, not gated on status, so
a running skid bypasses its surplus the same way. Bypassed water earns nothing
and still pays intake + pretreatment energy; with the intake fixed that energy
is a constant the schedule cannot dodge, so it shifts the whole bill up rather
than changing where the cheap hours are.

The plant has **three operating states, not four**: all three trains, two
trains, or down. A single skid running is not a state it has -- the shared
post-treatment step and product pump cannot be turned down to one train's
permeate -- so ``ro[0]`` and ``ro[1]`` are pinned together as a lead pair and
``ro[2]`` is the one that swings. The only two moves the schedule can make are
therefore *shut one skid* and *shut the system*, which is what the solved
``trains_online`` steps between.

Recuperation is a **plant-level** outage, not a per-skid one. ``post_treatment``
may only run while ``ro[0]`` is online and at least 45 minutes past its restart;
the symmetry breaking below makes ``ro[0]`` the last skid off and the first back
on, so "``ro[0]`` down" is exactly "the RO system down". While post-treatment is
out, the permeate header is pinned to zero and the *whole* permeate stream --
every train's, not just the restarting one's -- goes to the outfall off-spec.
Stepping the train count down from three to two therefore costs nothing; only a
full system restart does.

The lead pair is what makes that penalty bite. Left free, a lone ``ro[0]`` is a
loophole: the cheapest way through a 45-minute window is to restart the one skid
the constraint watches, spill only its third of the permeate, and bring the
other two up on the step post-treatment returns. Pinning ``ro[1]`` to it means a
restart carries two trains through the window at full power for nothing, which
is what a restart actually costs.

The plant owes a **fixed volume of product water over the horizon** -- 265
acre-feet by default -- and nothing says *when*. That is the whole degree of
freedom: with no product storage anywhere in the flowsheet, the only way to dodge
an expensive tariff hour is to make less water in it and more water elsewhere.

That obligation is the model's one free knob. :func:`main` takes it as
``demand_af``, and it lands in the model as a **mutable** ``Param``, so
:func:`set_demand` can retarget it and the model can be re-solved without being
rebuilt -- which is how the notebook's demand slider works. How much slack the
schedule has is entirely a function of where it sits against
:func:`max_product_af`, the most the three skids can deliver if they run at rated
feed for the whole horizon: near that ceiling the plant has to run flat out and
there is nothing to schedule, and the further below it the demand sits, the more
of the peak window the optimizer can afford to sit out.

The objective is the in-model operating cost and nothing else. What makes the
month tractable is not a term in it but *how it is solved*:
:func:`solve_relax_and_fix` solves the integrality-relaxed model first, takes
every status the relaxation already decided -- within :data:`FIX_TOL` of 0 or 1
-- as a decision and fixes it there, restores integrality, and re-solves as a
MIP over the statuses that came back genuinely fractional. Those are the steps
where the plant is actually choosing; the rest were never a search. It is a
heuristic and says so: the relaxed objective is a valid lower bound, so
``m.relaxation_gap`` bounds how far the schedule it returns can be from optimal.

The objective is still not the bill, though it was never the bill: the in-model
cost is a relaxed, scalarized proxy the solver can optimize over, and the
month's actual cost is :func:`report_cost` -- the solved dispatch handed back to
EECO after the fact.

"""

import json
from datetime import datetime
from pathlib import Path

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

# The blocks and unit models come off the top-level namespace, which is the
# public API `docs/how_to/build_a_plant.md` is written against. The logic
# helpers are not exported there and keep their `flexops.logic` path.
from flexops import (
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
    add_startup_shutdown,
    add_status,
    register_parallel_group,
    relax,
    unrelax,
)

CONFIG_PATH = Path(__file__).parent / "config.json"

#: The scheduling horizon, module-level because the demand ceiling is a rate
#: times its length: exactly one calendar month at 15-minute resolution. The step
#: is 15 minutes rather than an hour because the 45-minute recuperation delay has
#: to land on a whole number of steps, and because 15 minutes divides the
#: tariff's 16:00 and 21:00 boundaries exactly.
START_DATE = "2026-07-01"
END_DATE = "2026-08-01"
TIME_STEP_HR = 0.25
HORIZON_HOURS = (
    datetime.fromisoformat(END_DATE) - datetime.fromisoformat(START_DATE)
).total_seconds() / 3600.0

#: Product water owed over the horizon, measured downstream of the product pump.
#: A default, not a constant: :func:`main` takes ``demand_af``, and
#: :func:`set_demand` retargets it on a model already built. The horizon is
#: exactly one calendar month, so this is the month's obligation as it stands --
#: no proration.
DEMAND_AF_PER_MONTH = 265.0

#: How close to an integer a relaxed status has to land before
#: :func:`solve_relax_and_fix` takes it as a decision rather than a search: a
#: status below ``FIX_TOL`` is fixed off, one above ``1 - FIX_TOL`` is fixed on,
#: and everything between is handed to the MIP as a free binary.
#:
#: It is the routine's one knob, and it trades tractability against optimality.
#: At 0 nothing is fixed and the routine *is* the exact MILP -- the relaxation
#: becomes a wasted solve that buys only the bound. Turn it up and more of the
#: month is decided by an LP that is allowed to run a skid at 0.87 of rated feed,
#: which the plant cannot; the schedule it hands down is feasible (the MIP still
#: has to satisfy every constraint around what was fixed) but it can be dearer
#: than the optimum, and ``m.relaxation_gap`` is what says by how much. Past
#: ~0.4 fixing starts to cut off every feasible schedule outright, which is what
#: the tolerance ladder in :func:`solve_relax_and_fix` is there to survive.
FIX_TOL = 0.1

#: The plant's sizing, module-level because the charts need it too: ``min_feed``
#: in particular is not recoverable from a solved model -- it goes in as a bound
#: inside ``add_status`` and is not a variable anywhere afterwards.
N_TRAINS = 3
#: Water recovery is a **degree of freedom**, not a plant constant: each skid's
#: ``recovery`` Var is unfixed in :func:`construct_plant` and the optimizer picks
#: it anywhere in ``[RECOVERY_MIN, RECOVERY_MAX]``. ``RECOVERY`` is only the
#: starting point the Var is initialized at -- the nominal seawater figure -- and
#: is what the *reporting* helpers that need a nominal number still quote.
#: Anything that is a **ceiling** (the header's status cap, the delivery ceiling,
#: the "full tilt" reference line) reads ``RECOVERY_MAX`` instead, or it would
#: bind before the optimizer ever got to the top of the window.
RECOVERY = 0.465
RECOVERY_MIN = 0.4
RECOVERY_MAX = 0.5
RATED_FEED_M3_PER_HR = 337.67  # m3/hr of feed, per skid
MIN_FEED_M3_PER_HR = 337.67  # m3/hr; below this a skid must shut off entirely
RECUP_STEPS = 3  # 45 min of off-spec permeate after a startup, at 15-min steps

#: The intake pump is a fixed-duty machine: one speed, no throttling, so the
#: plant lifts the same seawater every step whatever the skids are doing. It is
#: *fixed*, not bounded -- ``feed_m3_per_hr`` is a flat line in the results
#: frame, and the intake + pretreatment power it costs is a constant the
#: schedule cannot dodge. It sits above what the membranes can swallow (three
#: skids at rated feed take 1013.01), so 50.49 m^3/hr goes to the outfall
#: through the RO bypass even at full tilt.
INTAKE_FLOW_M3_PER_HR = 1063.5

_M3_HR = pyunits.m**3 / pyunits.hr
_KWH_M3 = pyunits.kWh / pyunits.m**3

#: One acre-foot in m^3. The demand is quoted in acre-feet because that is how a
#: water contract is written; every flow in the flowsheet is metric.
M3_PER_AF = pyo.value(
    pyunits.convert(1 * pyunits.acre * pyunits.foot, pyunits.m**3)
)

#: Flows and draws below this read as zero. A MILP solver returns an idle train
#: as a few nanolitres either side of nothing, and a negative flow plots as a
#: notch below the axis.
_FLOW_TOL_M3_HR = 1e-6

#: Which solver :func:`solve_model` asks for first. ``get_solver`` takes this as
#: ``prefer`` and uses it whenever that solver is installed and can take the
#: problem class; otherwise it warns and falls through its own priority list
#: (``gurobi``, ``scip``, ``highs``, ``cbc``, ``ipopt``). HiGHS is the default
#: because it ships with ``flex-pse[solvers]`` and needs no license.
SOLVER = "gurobi"

#: Solver options, keyed by the solver's own name for its own option spellings
#: -- they are not translated anywhere, and a solver rejects a name it does not
#: know (Gurobi raises outright on HiGHS's ``mip_rel_gap``). :func:`solve_model`
#: looks these up under the solver ``get_solver`` *returned*, not the one asked
#: for, so a fallback still gets options it understands. A solver with no entry
#: here runs on its defaults: no gap, no time limit.
#:
#: ``mip_rel_gap``/``MIPGap`` ends the branch and bound once the incumbent is
#: provably within that fraction of optimal, which is what keeps a month of unit
#: commitment tractable; the time limit is the backstop, and a run that hits it
#: is reported rather than silently accepted.
#:
#: 0.2%, not the usual 1%. The gap is not just a cost tolerance here -- it is a
#: tolerance on the *schedule*, and a whole recuperation window is small against
#: the month's bill. At 1% the exact month came back holding post-treatment out
#: for 2h15m after a restart rather than the 45 minutes the constraint requires:
#: feasible, within tolerance, and wrong on the one behaviour the example exists
#: to show. At 0.2% the window is exactly three steps and the month still solves
#: in a few minutes -- about 6 on a laptop, most of it the relax-and-fix MIP.
#:
#: ``NonConvex=2`` is what an unfixed recovery costs: the split
#: ``permeate[t] == recovery * feed[t]`` is bilinear, so the month is a
#: non-convex MIQCP and Gurobi has to run spatial branch and bound over it.
#: Gurobi's default (``NonConvex=-1``) already does this; it is set explicitly
#: so a run that gets slow points at its own cause. HiGHS cannot take the
#: problem at all -- see :func:`_register_gurobi_for_quadratics`.
SOLVER_OPTIONS = {
    "highs": {"mip_rel_gap": 0.002, "time_limit": 1800},
    "gurobi": {"MIPGap": 0.002, "TimeLimit": 1800, "NonConvex": 2},
}


def _register_gurobi_for_quadratics() -> None:
    """Teach flex-pse's solver registry that Gurobi can take this model.

    Unfixing recovery makes ``permeate[t] == recovery * feed[t]`` bilinear, and
    flex-pse classifies *any* nonlinear constraint as ``NLP``/``MINLP`` --
    quadratic included, since ``QP`` there means a quadratic *objective* only.
    So the exact model is ``MINLP`` and the relaxation, whose binaries are gone,
    is ``NLP``. The shipped ``CAPABILITIES`` lists Gurobi for ``LP``/``QP``/
    ``MILP``, so ``get_solver`` passes over an installed Gurobi on both: with no
    SCIP here the exact model raises outright, and the relaxation silently falls
    through to IPOPT.

    That IPOPT fallback is the worse of the two. IPOPT is a *local* solver, and
    a relaxation that is only locally optimal is not a lower bound on the exact
    cost -- which is the one thing :func:`main`'s ``relax_integrality`` promises.

    Gurobi solves non-convex QCP and MIQCP by spatial branch and bound (what
    ``NonConvex=2`` above asks for), so the registry entry is what is wrong here,
    not the routing; ``CAPABILITIES`` is documented as the extension point for
    exactly this. Idempotent, and a no-op if a future flex-pse ships the entries
    itself.
    """
    from flexcore.solvers.classify import ProblemClass
    from flexcore.solvers.registry import CAPABILITIES

    CAPABILITIES.setdefault("gurobi", set()).update(
        {ProblemClass.NLP, ProblemClass.MINLP}
    )

def main(
    relax_integrality: bool = False,
    demand_af: float = DEMAND_AF_PER_MONTH,
):
    """Build the desalination scheduling model, ready to solve.

    Args:
        relax_integrality: ``True`` to drop the RO skids' ``status``, ``startup``
            and ``shutdown`` -- and post-treatment's ``status`` -- from
            ``Binary`` to ``UnitInterval``, and leave them there. The month
            carries ~27k binaries; the relaxation solves in seconds, but it is
            **optimistic** -- a fractional status runs a skid below its turndown
            floor, and a fractional ``ro[0]`` startup lets post-treatment stay
            fractionally online through the recuperation window instead of
            paying for all of it. Read its cost as a lower bound and its
            schedule as no schedule at all. It is a comparison case, not the way
            to make the exact month tractable -- that is
            :func:`solve_relax_and_fix`, which relaxes and restores integrality
            inside a single solve and is what a model built with the default
            wants.
        demand_af: Product water owed over the horizon, in acre-feet. Has to sit
            under :func:`max_product_af` or the model is infeasible. It goes in
            as a mutable ``Param``, so a *different* demand needs only
            :func:`set_demand` and another solve, not another build.

    Returns:
        A ``pyo.ConcreteModel`` with the plant, the product-delivery
        obligation, and ``objective`` (the in-model operating cost -- a solve
        target, not a bill; :func:`report_cost` is the bill). The obligation is
        on ``m.plant.potable.delivery_min`` (a mutable ``Param``, m^3, which
        :func:`set_demand` rewrites) and ``m.demand_af``
        (the acre-feet it was set from).
    """
    m = pyo.ConcreteModel(name="desalination_example")

    m.time_block = TimeBlock(
        start_date=START_DATE,
        end_date=END_DATE,
        time_step=TIME_STEP_HR * pyunits.hr,
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
    # untouched, so the always-on units stay pinned at 1 -- which is why
    # logic_units can sweep the whole plant rather than naming the RO skids and
    # post-treatment, and is the same sweep solve_relax_and_fix makes.
    m.is_relaxed = bool(relax_integrality)
    if m.is_relaxed:
        for unit in logic_units(m.plant):
            relax(unit)

    add_demand_and_objective(m, demand_af)

    return m


def demand_m3(demand_af: float = DEMAND_AF_PER_MONTH) -> float:
    """Convert a product-water obligation from acre-feet to m^3.

    The horizon is exactly one calendar month, so an obligation quoted per month
    converts straight across -- no proration.
    """
    return float(demand_af) * M3_PER_AF


def max_product_af() -> float:
    """The most product water the plant could deliver over the horizon, in AF.

    Every skid at rated feed *and at the top of its recovery window* for every
    step of the horizon, and never a restart to spill permeate off-spec. Any
    ``demand_af`` has to sit under this or the model is infeasible, and how far
    under is exactly how much slack the schedule has: at the ceiling the plant
    runs flat out and there is nothing to schedule.

    ``RECOVERY_MAX``, not the nominal ``RECOVERY``: recovery is a degree of
    freedom, so a demand between the two is deliverable and quoting the nominal
    figure would call it infeasible.
    """
    return (
        N_TRAINS * RATED_FEED_M3_PER_HR * RECOVERY_MAX * HORIZON_HOURS / M3_PER_AF
    )


def set_demand(m, demand_af: float) -> float:
    """Retarget the horizon's product-water obligation on a built model.

    The obligation is the mutable ``Param`` the product ``Product`` block built
    for its horizon-basis ``min_demand`` -- ``plant.potable.delivery_min``, a
    scalar because the limit is on the month's total and not on any one step. So
    this is all that stands between one demand and the next: no constraint in
    the flowsheet changes, only that Param, and the model is ready to re-solve.
    That is what the notebook's demand slider moves -- the month is built once
    and each new demand pays for its solve alone.

    Args:
        m: A model from :func:`main`, solved or not.
        demand_af: The new obligation in acre-feet. Above
            :func:`max_product_af` the model becomes infeasible; nothing here
            checks, because the solver's report is the honest answer.

    Returns:
        The new obligation in m^3.
    """
    m.demand_af = float(demand_af)
    volume = demand_m3(m.demand_af)
    m.plant.potable.delivery_min.set_value(volume)
    return volume


def logic_units(plant):
    """Yield every unit block in the flowsheet carrying flex-pse logic binaries.

    Walks the plant rather than naming the units, so a unit added to
    :func:`construct_plant` later is relaxed and restored without a second edit
    here. ``status`` is the marker because ``add_status`` is the base piece
    every other one hangs off, and ``relax``/``unrelax`` work off the unit's own
    registry of tracked binaries -- so one call per unit covers its ``status``,
    ``startup`` and ``shutdown`` together.

    The always-on units come back too. Their statuses are *fixed* at 1, and
    ``relax`` is domain-only, so relaxing them is a no-op rather than a hole in
    the plant.
    """
    for unit in plant.component_data_objects(pyo.Block, descend_into=True):
        if unit.component("status") is not None:
            yield unit


def status_vars(plant):
    """Yield every status variable in the flowsheet that is still a decision.

    The set :func:`solve_relax_and_fix` reads off the relaxation and fixes.
    Statuses only -- ``startup`` and ``shutdown`` follow from them through the
    constraints ``add_startup_shutdown`` attached, so fixing a status decides
    its startup too.

    The always-on units' statuses are **fixed at 1** by :func:`construct_plant`
    and are skipped: they were never a decision. Anything the routine itself
    fixed is skipped for the same reason, which is why it keeps its own list of
    those and releases them before it reads this again.
    """
    for unit in plant.component_data_objects(pyo.Block, descend_into=True):
        status = unit.component("status")
        if status is None:
            continue
        for datum in status.values():
            if not datum.fixed:
                yield datum


def add_demand_and_objective(m, demand_af: float = DEMAND_AF_PER_MONTH):
    """Target the horizon's water obligation and add the objective the solver minimizes.

    The obligation's *constraint* is part of the flowsheet -- the product
    ``Product`` block carries it -- so this only sets the number and builds the
    objective. The objective is the in-model operating cost, which is a solve
    target and not a statement about money -- see :func:`report_cost`.
    """
    # The obligation itself is a constraint on the product Product block, built
    # with the flowsheet in construct_plant -- `min_demand` on the horizon basis,
    # which bounds the scalar `delivery_total` rather than any one step. All that
    # is left here is to point it at this demand.
    set_demand(m, demand_af)

    m.costing.cost_process()

    # The operating cost and nothing else. It is not a dollar figure anyone
    # should quote -- it is the relaxed, scalarized proxy the solver can
    # optimize over, and the bill is report_cost. What keeps the month
    # tractable is solve_relax_and_fix, not a term added here.
    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )
    return m


def solve_model(m, prefer: str | None = None):
    """Solve ``m`` with :data:`SOLVER` and the matching :data:`SOLVER_OPTIONS`.

    Records the achieved relative MIP gap on ``m.mip_gap`` and snaps power
    noise to exactly zero -- a MILP solver returns an idle plant as a few
    nanowatts either side, and EECO refuses to bill a negative draw.

    A run that stops on the time limit is returned rather than raised on: the
    exact MILP is expected to hit it, and the caller reports the termination
    condition and the gap.

    Args:
        m: A model from :func:`main`.
        prefer: Overrides :data:`SOLVER` for this solve. ``get_solver`` honours
            it only if that solver is installed and can take the problem class;
            otherwise it warns and falls through its own priority list.

    Returns:
        The Pyomo results object.
    """
    from flexcore.solvers import get_solver

    _register_gurobi_for_quadratics()
    solver = get_solver(model=m, prefer=prefer or SOLVER)
    # Keyed on what came back, not on what was asked for: the options are the
    # solver's own spellings, and a fallback would choke on another's.
    results = solver.solve(m, options=dict(SOLVER_OPTIONS.get(solver.name, {})))

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


#: Termination conditions the fixing ladder in :func:`solve_relax_and_fix`
#: reads as "these fixes cut off every schedule". ``infeasibleOrUnbounded`` is
#: in here because a presolve that folds the two cannot tell them apart, and
#: this model is bounded below by construction.
_INFEASIBLE = frozenset(
    {
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    }
)


def solve_relax_and_fix(m, *, tol: float = FIX_TOL, prefer: str | None = None):
    """Solve the month by relaxing, fixing what the relaxation decided, re-solving.

    The exact month is ~27k binaries and the schedule the plant actually has is
    made of long flat runs, so most of those binaries are not a decision anyone
    is making -- they are a search the solver has to close anyway. This routine
    lets the relaxation say which:

    1. Relax every logic binary to ``UnitInterval`` and solve. The relaxed
       objective is a **valid lower bound** on the exact one, and it is kept on
       ``m.relaxed_objective`` for exactly that reason.
    2. Fix every status the relaxation already decided -- below ``tol``, or
       above ``1 - tol`` -- at that integer. Statuses only: ``startup`` and
       ``shutdown`` follow from them through their own constraints.
    3. Restore integrality and re-solve as a MIP over what is left, which is the
       genuinely fractional statuses -- the steps where the LP is trading a
       fraction of a train against the tariff and the plant has to pick a side.

    It is a heuristic, and the honest number for it is ``m.relaxation_gap``, not
    ``m.mip_gap``: the final solve's gap is the gap of the *fixed* subproblem and
    says nothing at all about the statuses that were fixed before it started.
    The gap against the relaxation's bound covers both.

    **The fixing ladder.** Fixing can cut off every feasible schedule -- pin
    post-treatment on at a step and ``post_treatment_recuperation`` forces
    ``ro[0]`` on there *and* forbids a restart in the preceding 45 minutes, which
    may be the only way the month meets its demand. So an infeasible re-solve is
    not an answer here; it steps the tolerance down (``tol``, ``tol/2``,
    ``tol/4``, then 0) and fixes again from the same relaxed values, handing
    more of the decision back to the solver each time. The last rung fixes
    nothing and *is* the exact MILP, so the ladder always ends somewhere honest.
    Every rung tried is recorded on ``m.fix_attempts``.

    Args:
        m: A model from :func:`main`, built exact. A model built with
            ``relax_integrality=True`` is the LP comparison case and is
            rejected: this routine would hand it back with its integrality
            restored, which is not the model that was asked for.
        tol: Distance from an integer within which a relaxed status is taken as
            decided. See :data:`FIX_TOL`. ``0`` fixes nothing, which makes this
            the exact MILP with a lower bound computed first.
        prefer: Passed to :func:`solve_model` for both solves.

    Returns:
        The Pyomo results object of the solve that produced the returned
        schedule. Alongside it, on the model:

        * ``m.relaxed_objective`` -- the relaxation's cost, a lower bound.
        * ``m.relaxation_gap`` -- how far the returned schedule sits above it.
        * ``m.fix_tol`` -- the rung that produced the answer, which is ``tol``
          unless the ladder had to step down.
        * ``m.statuses_fixed`` / ``m.statuses_free`` -- how much of the month
          the relaxation decided and how much went to the MIP.
        * ``m.fix_attempts`` -- ``(tol, fixed, termination)`` per rung tried.

    Raises:
        ValueError: If ``m`` was built with ``relax_integrality=True``.
    """
    if m.is_relaxed:
        raise ValueError(
            "solve_relax_and_fix needs a model built exact -- it relaxes and "
            "restores integrality itself. This model came from "
            "main(relax_integrality=True), which is the LP comparison case; "
            "solve it with solve_model, or rebuild it with main()."
        )

    # Release what an earlier call fixed. Without this a second solve at a new
    # demand would inherit the first one's schedule as hard constraints --
    # status_vars skips fixed data, so they would not even show up as decisions
    # to reconsider.
    for datum in getattr(m, "fixed_statuses", ()):
        datum.unfix()

    units = list(logic_units(m.plant))
    for unit in units:
        relax(unit)
    m.is_relaxed = True
    relaxed_results = solve_model(m, prefer=prefer)
    m.relaxed_objective = pyo.value(m.objective)
    # Snapshot rather than read the Vars again later: the solves below overwrite
    # them, and every rung of the ladder fixes from the same relaxed answer.
    relaxed_statuses = [(datum, pyo.value(datum)) for datum in status_vars(m.plant)]

    for unit in units:
        unrelax(unit)
    m.is_relaxed = False

    if relaxed_results.solver.termination_condition in _INFEASIBLE:
        # No bound and nothing to fix. The exact model is infeasible too -- the
        # relaxation is a superset of it -- so this is the answer, and a demand
        # above max_product_af is the usual reason.
        m.relaxed_objective = None
        m.relaxation_gap = None
        m.fix_tol = None
        m.fixed_statuses = []
        m.statuses_fixed, m.statuses_free = 0, len(relaxed_statuses)
        m.fix_attempts = []
        return relaxed_results

    ladder = [tol, tol / 2, tol / 4, 0.0]
    m.fix_attempts = []
    results = relaxed_results  # rebound on the first rung; the ladder is never empty
    for rung, rung_tol in enumerate(ladder):
        for datum in getattr(m, "fixed_statuses", ()):
            datum.unfix()
        fixed = []
        # Strict, so the last rung fixes nothing rather than pinning every
        # status that landed exactly on an integer.
        for datum, value in relaxed_statuses:
            if value < rung_tol:
                datum.fix(0)
            elif value > 1 - rung_tol:
                datum.fix(1)
            else:
                continue
            fixed.append(datum)
        m.fixed_statuses = fixed

        results = solve_model(m, prefer=prefer)
        condition = results.solver.termination_condition
        m.fix_attempts.append((rung_tol, len(fixed), condition))
        if condition not in _INFEASIBLE or rung == len(ladder) - 1:
            break

    m.fix_tol, m.statuses_fixed, _ = m.fix_attempts[-1]
    m.statuses_free = len(relaxed_statuses) - m.statuses_fixed
    objective = pyo.value(m.objective)
    m.relaxation_gap = (
        abs(objective - m.relaxed_objective) / abs(objective) if objective else 0.0
    )
    return results


def report_cost(m):
    """Return the month's bill: an EECO evaluation of the realized dispatch.

    **This, and not the objective, is the cost.** ``m.objective`` is what the
    solver can optimize over: the tariff's pricing non-convexity has to be
    relaxed and scalarized to stay in a MILP, and the demand charge in
    particular is a proration, not a meter read.

    ``FlexCosting.report_cost`` recomputes the cost from scratch by handing the
    solved aggregate power profile back to EECO. Once the dispatch is fixed the
    non-convexity is harmless, so the evaluation is exact.

    Args:
        m: A solved model from :func:`main`.

    Returns:
        A ``flexops.costing.CostReport``. ``report.operating.electricity`` is
        the EECO electricity bill, ``report.operating.total`` the operating
        total, and ``report.currency`` names the currency they are magnitudes
        in.
    """
    return m.costing.report_cost(m)

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
    intake_flow = INTAKE_FLOW_M3_PER_HR

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
    
    # TODO - replace this with a custom surrogate, so intensity rises with
    # recovery instead of being flat across the window.
    #
    # 3.34 kWh per m^3 of *permeate*, which is how an SWRO figure is normally
    # quoted. It used to be written ``3.34 * 0.465`` -- the same figure converted
    # to a per-feed basis, because ReverseOsmosis applied energy_intensity to the
    # feed. flex-pse bc69357 moved the relation onto the permeate stream, so the
    # conversion has to come back out or the plant draws recovery times too
    # little power. (The class docstring still says feed while the config option
    # says product; that contradiction is flex-pse#85, not this example's bug --
    # the code at reverseosmosis.py:108 is what runs.)
    plant.ro = ReverseOsmosis(
        plant.trains,
        property_package=m.properties,
        recovery=recovery,
        recovery_min=RECOVERY_MIN,
        recovery_max=RECOVERY_MAX,
        energy_intensity=3.34 * _KWH_M3,
        costing_package=m.costing,
    )

    # Recovery is a degree of freedom, not a plant constant. flex-pse builds a
    # process parameter as a scalar Var *fixed* at its configured value, exactly
    # so it can be unfixed in place; unfixing it here hands the optimizer the
    # membrane window, with the configured RECOVERY left behind as the starting
    # point. One value per skid for the whole horizon -- the Var is scalar, not
    # time-indexed -- so this is a design choice the schedule is solved around,
    # not a knob that moves step to step.
    #
    # It costs the model its problem class: ``permeate[t] == recovery * feed[t]``
    # is bilinear the moment recovery stops being a constant, so the month is a
    # non-convex MIQCP rather than a MILP. See SOLVER_OPTIONS for what that
    # takes. The bilinearity is only in 3 scalars, so spatial branch and bound
    # has very little to branch on, but it is not free.
    #
    # And until the surrogate in the TODO above lands, the answer is known in
    # advance: recovery pins to RECOVERY_MAX at every demand. With intensity on a
    # per-permeate basis a m^3 of product costs the same 3.34 kWh whatever the
    # recovery, so recovery buys nothing on the *energy* side -- what it buys is
    # capacity. The intake is fixed-duty, so a higher recovery turns the same
    # seawater into more permeate per skid-hour, and the month's obligation is
    # met in fewer running hours that the schedule can then place in the cheap
    # ones. The window only starts to trade off once intensity rises with
    # recovery, which is what the surrogate above is for.
    for i in plant.trains:
        plant.ro[i].recovery.unfix()


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
    
    # Every junction below is a flex-pse block, not a balance written here. A
    # Splitter carries conservation and nothing else -- no split fraction -- so
    # the routing stays the decision the objective makes, which is exactly what
    # the hand-written tees it replaces were for. A Mixer is its mirror.
    #
    # The property package is flow-only (SimpleAqueousFlow defaults
    # has_pressure/has_temperature to False), so each junction adds its
    # volumetric balance and nothing else: no pressure or temperature
    # pass-through equations come along for the ride.
    plant.feed_header = Splitter(
        property_package=m.properties,
        outlet_names=tuple(f"t{i}" for i in plant.trains),
    )

    plant.train_split = Splitter(
        plant.trains,
        property_package=m.properties,
        outlet_names=("ro", "bypass"),
    )

    # The permeate tee. The constraint this replaces was an inequality --
    # "a train can send no more to the header than it makes" -- with the
    # remainder implicit. As a Splitter it is the equality
    # `permeate == header + offspec` with `offspec >= 0`, which says the same
    # thing and gives the dumped stream a name the outfall can be wired to.
    plant.permeate_split = Splitter(
        plant.trains,
        property_package=m.properties,
        outlet_names=("header", "offspec"),
    )

    plant.permeate_header = Mixer(
        property_package=m.properties,
        inlet_names=tuple(f"t{i}" for i in plant.trains),
    )

    # The ocean outfall, as a boundary sink rather than the reporting-only
    # Expression it used to be. A Product deliberately does *not* blend its
    # inlets -- it aggregates flow -- which is what an outfall wants: three
    # streams per train arriving on their own ports. Unpriced, so it adds no
    # operating cost; it meters the discharge into total_product["outfall"].
    plant.outfall = Product(
        property_package=m.properties,
        inlet_names=tuple(
            f"{stream}_{i}"
            for i in plant.trains
            for stream in ("brine", "bypass", "offspec")
        ),
        resource_name="outfall",
    )

    # The delivery obligation, as a horizon-basis limit on a boundary sink. The
    # bound lands on the scalar `delivery_total`, so it is a volume over the
    # month and not a profile -- placing it in the cheap hours is the whole
    # degree of freedom. `min_demand` becomes the mutable Param `delivery_min`,
    # which is what set_demand rewrites; the value here is only a starting
    # point. Unpriced, so the objective stays the energy bill alone.
    plant.potable = Product(
        property_package=m.properties,
        resource_name="potable_water",
        min_demand=demand_m3() * pyunits.m**3,
        demand_basis="horizon",
    )

    # The seawater boundary, which meters the intake into total_feed. No
    # min/max_withdrawal: those build a pair of inequalities, and the intake is a
    # fixed-duty machine, so the withdrawal is *fixed* below rather than boxed.
    # Equal limits would pin the same number but leave a column the fix removes.
    plant.seawater = Feed(
        property_package=m.properties,
        resource_name="seawater",
    )

    # Arcs. `naming_dict` renames a unit's flows and state blocks but never its
    # ports, so the RO skids are wired through `outlet_a` (permeate) and
    # `outlet_b` (brine) even though the components read `permeate`/`brine`.
    plant.seawater_to_intake = Arc(
        source=plant.seawater.outlet_a,
        destination=plant.intake_pump.inlet,
        doc="Raw seawater to the intake pump.",
    )
    plant.intake_to_header = Arc(
        source=plant.intake_pump.outlet,
        destination=plant.feed_header.inlet,
        doc="The intake pump's discharge into the feed header.",
    )

    @plant.Arc(plant.trains, doc="Feed header to each train's pretreatment.")
    def header_to_pretreatment(b, i):
        return (
            plant.feed_header.find_component(f"outlet_t{i}"),
            plant.pretreatment[i].inlet,
        )

    @plant.Arc(plant.trains, doc="Pretreated water to this train's RO tee.")
    def pretreatment_to_split(b, i):
        return (plant.pretreatment[i].outlet, plant.train_split[i].inlet)

    @plant.Arc(plant.trains, doc="The tee's skid leg into the RO membranes.")
    def split_to_ro(b, i):
        return (plant.train_split[i].outlet_ro, plant.ro[i].inlet)

    @plant.Arc(plant.trains, doc="Raw permeate into this train's permeate tee.")
    def ro_to_permeate_split(b, i):
        return (plant.ro[i].outlet_a, plant.permeate_split[i].inlet)

    @plant.Arc(plant.trains, doc="On-spec permeate into the permeate header.")
    def permeate_split_to_header(b, i):
        return (
            plant.permeate_split[i].outlet_header,
            plant.permeate_header.find_component(f"inlet_t{i}"),
        )

    plant.header_to_post = Arc(
        source=plant.permeate_header.outlet,
        destination=plant.post_treatment.inlet,
        doc="The permeate header into post-treatment.",
    )
    plant.post_to_product = Arc(
        source=plant.post_treatment.outlet,
        destination=plant.product_pump.inlet,
        doc="Post-treated water to the product pump.",
    )
    plant.product_to_delivery = Arc(
        source=plant.product_pump.outlet,
        destination=plant.potable.inlet_a,
        doc="Product water across the facility boundary.",
    )

    # The three streams that leave through the outfall: brine, the water routed
    # around a skid, and the permeate dumped while post-treatment recuperates.
    @plant.Arc(plant.trains, doc="Brine to the outfall.")
    def brine_to_outfall(b, i):
        return (plant.ro[i].outlet_b, plant.outfall.find_component(f"inlet_brine_{i}"))

    @plant.Arc(plant.trains, doc="Bypassed pretreated water to the outfall.")
    def bypass_to_outfall(b, i):
        return (
            plant.train_split[i].outlet_bypass,
            plant.outfall.find_component(f"inlet_bypass_{i}"),
        )

    @plant.Arc(plant.trains, doc="Off-spec permeate to the outfall.")
    def offspec_to_outfall(b, i):
        return (
            plant.permeate_split[i].outlet_offspec,
            plant.outfall.find_component(f"inlet_offspec_{i}"),
        )

    # Once, after every arc above is declared -- the transformation only expands
    # the arcs that exist when it runs.
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # The intake pump and the pretreatment units run whenever the plant does, so
    # their status is pinned at 1 and never becomes a decision. Every one of
    # them is sized for the whole fixed intake: the pump because it passes all
    # of it, a pretreatment unit because the header split is free and it may be
    # handed all of it. Sizing a pretreatment unit for an even third instead
    # would be a *constraint*, not a sizing -- 3 * (1063.5 / 3) is exactly the
    # intake, so the caps would bind and force the split.
    always_on = [(plant.intake_pump, intake_flow)]
    always_on += [(plant.pretreatment[i], intake_flow) for i in plant.trains]
    for unit, max_flow in always_on:
        add_status(unit, unit.flow_in, 0 * _M3_HR, max_flow * _M3_HR)
        for t in tb.time_index:
            unit.status[t].fix(1)

    # Fixed-duty intake: the pump has no turndown, so this is a fixed value and
    # not a bound. It is set on the seawater Feed's metered withdrawal, which the
    # arc carries into the pump. It buys nothing but a floor under the outfall:
    # the feed header conserves it into the trains, and everything the skids will
    # not take has to leave through the bypass. Nothing dictates *how* it splits:
    # an even third is what the solver reports because every skid can then run at
    # rated feed, not because the model requires it.
    for t in tb.time_index:
        plant.seawater.withdrawal[t].fix(intake_flow)

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
    #
    # Sized at RECOVERY_MAX, not the nominal RECOVERY: this cap is a status link
    # (flow <= max * status), so at the nominal 0.465 it would cap the header at
    # 471.15 m^3/hr while three skids at rated feed and the top of the window
    # make 506.5 -- the recovery window would be real on paper and clipped at
    # 0.465 in every solve where all three trains run.
    add_status(
        plant.post_treatment,
        plant.post_treatment.flow_in,
        0 * _M3_HR,
        N_TRAINS * rated_feed * RECOVERY_MAX * _M3_HR,
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
        # all -- post-treatment could run with every skid off. Re-checked against
        # flex-pse main at 50ff1d8: still a single lagged sample, so this stays
        # the one constraint in the flowsheet written out by hand.
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
    # running, skid 1 only if 0 is.
    #
    # The list goes lead-first. register_parallel_group takes its units in
    # priority order and chains them descending:
    #
    #     units[0].status[t] >= units[1].status[t] >= ... >= units[-1].status[t]
    #
    # so ro[0] is the last skid off and the first back on, which is what lets
    # post_treatment_recuperation use it as the proxy for the whole RO system.
    #
    # This call used to pass reversed(), because the ordering ran the other way
    # before flex-pse #64. That was fixed in #76 (commit bc69357) and the
    # reversal came out with it -- putting it back would silently make ro[0] the
    # *first* skid off and quietly decouple the recuperation window.
    register_parallel_group([plant.ro[i] for i in plant.trains])

    # The lead pair. Symmetry breaking alone leaves four plant states -- 3, 2, 1
    # or 0 skids online -- and a single skid running is not one the plant has:
    # the shared post-treatment and product pump cannot be turned down to one
    # train's permeate. Pinning ro[0] to ro[1] deletes exactly that state, and
    # the chain above does the rest:
    #
    #     register_parallel_group  ->  status[0] >= status[1] >= status[2]
    #     lead_pair                ->  status[0] <= status[1]
    #     together                 ->  status[0] == status[1] >= status[2]
    #
    # so the plant is at 3 trains, 2 trains, or down -- shut one skid, or shut
    # the system. That is not only a fidelity fix. Without it the optimizer
    # games the plant-level recuperation below: the penalty is "whatever is
    # running goes off-spec", so the cheapest restart is to bring ro[0] up
    # *alone*, spill a third of the permeate for 45 minutes, and snap the other
    # two on at the exact step post-treatment returns. Every restart in the
    # solved month did that. With the lead pair the window costs two trains'
    # permeate and the restart reads as a real one.
    #
    # "ro[0] on implies ro[1] on" is what add_conditional writes, as
    # ro[1].status[t] >= ro[0].status[t]. It lands on ro[0] as
    # ``ro[0].conditional``; register_parallel_group puts its own ``conditional``
    # on the *later* unit of each pair, so ro[1] and ro[2] carry those and there
    # is no collision.
    add_conditional(plant.ro[0], plant.ro[1], then="on")


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
        A DataFrame with per-train RO feed, permeate, status, startup and
        bypass; the
        plant's feed / permeate / brine / outfall / product flows; on-spec and
        off-spec permeate; the per-stage and aggregate power draw; the number of
        trains online and recuperating; post-treatment's on/off status; and the
        tariff energy price. The obligation the schedule was solved against
        rides along in ``frame.attrs`` as ``demand_af`` / ``demand_m3``, so a
        frame is self-describing and :func:`visualize` needs no second argument.
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
        data[f"ro{i}_permeate_m3_per_hr"] = [
            pyo.value(plant.ro[i].permeate[t]) for t in ti
        ]
        data[f"ro{i}_status"] = [status(i, t) for t in ti]
        data[f"ro{i}_startup"] = [startup(i, t) for t in ti]
        data[f"ro{i}_bypass_m3_per_hr"] = [
            pyo.value(plant.train_split[i].flow_out_bypass[t]) for t in ti
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
    # The permeate header's own outlet, rather than a sum over the trains: the
    # Mixer already carries it.
    data["onspec_permeate_m3_per_hr"] = [
        pyo.value(plant.permeate_header.flow_out[t]) for t in ti
    ]
    # Off-spec permeate is diverted into the brine line, so the outfall carries
    # both. Reported apart as well, since it is the recuperation penalty made
    # visible: water the plant paid full power to make and then threw away. It
    # used to be computed as permeate minus what reached the header; the permeate
    # tee now names it outright.
    data["offspec_permeate_m3_per_hr"] = [
        sum(pyo.value(plant.permeate_split[i].flow_out_offspec[t]) for i in trains)
        for t in ti
    ]
    data["bypass_m3_per_hr"] = [
        sum(pyo.value(plant.train_split[i].flow_out_bypass[t]) for i in trains)
        for t in ti
    ]
    data["brine_m3_per_hr"] = [
        sum(pyo.value(plant.ro[i].brine[t]) for i in trains) for t in ti
    ]
    # The outfall block's metered discharge: brine + bypass + off-spec, summed by
    # the Product rather than by an Expression written here.
    data["outfall_m3_per_hr"] = [pyo.value(plant.outfall.delivery[t]) for t in ti]
    # Read off the boundary block the obligation is written against, so the
    # column and the constraint cannot drift apart.
    data["product_m3_per_hr"] = [pyo.value(plant.potable.delivery[t]) for t in ti]
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
    # Set last: attrs do not survive every pandas operation, and the mask above
    # is one that returns a new frame.
    frame.attrs["demand_af"] = m.demand_af
    frame.attrs["demand_m3"] = pyo.value(m.plant.potable.delivery_min)
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
    2. **Permeate by train**, hand-stacked. Most of the time the trains are
       *identical*, so three overlaid traces would paint over each other and
       only the last would be visible; stacked, each band's thickness is that
       train's own permeate and the stack top is what the membranes made. Read
       against panel 1: through a recuperation window this stack keeps its
       height while product falls to zero, which is the whole of that permeate
       going to brine.
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

    # RECOVERY_MAX: recovery is a degree of freedom, and a "full tilt" line drawn
    # at the nominal recovery would sit *below* product the plant can legitimately
    # make, reading as an impossible schedule rather than a high-recovery one.
    full_tilt = N_TRAINS * RATED_FEED_M3_PER_HR * RECOVERY_MAX
    horizon_hours = len(frame) * dt_hours
    # Off the frame, not off the module constant: the demand is a knob, and a
    # reference line drawn at the default would quietly libel every other
    # setting of it. A frame from an older results_frame carries neither attr.
    demand_af = frame.attrs.get("demand_af", DEMAND_AF_PER_MONTH)
    flat_rate = frame.attrs.get("demand_m3", demand_m3(demand_af)) / horizon_hours
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
    # At the shipped demand the plant is sized close to its obligation, so these
    # two land within a few percent of each other -- their labels push apart
    # rather than overlap. The gap between them *is* the schedule's slack, and it
    # opens as the demand comes down.
    reference(ax, full_tilt,
              f"full tilt — {N_TRAINS} trains at rated feed, {RECOVERY_MAX:.0%} "
              "recovery",
              va="bottom")
    reference(ax, flat_rate, f"flat rate that would meet {demand_af:,.0f} AF",
              va="top")
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

    # -- per-train permeate (m3/hr) ------------------------------------------
    ax = _style(axes[1], "m³/hr")
    shade(ax)
    ax.set_ylim(0, full_tilt * 1.3)
    _base = detail["ro0_permeate_m3_per_hr"] * 0
    for _i in range(N_TRAINS):
        _top = _base + detail[f"ro{_i}_permeate_m3_per_hr"]
        ax.fill_between(hours, _base, _top, step="post", color=_TRAIN_RAMP[_i],
                        lw=1.0, edgecolor=_SURFACE, zorder=2,
                        label=f"train {_i + 1}")
        _base = _top
    # Drawn at RECOVERY_MAX for the same reason the panel above is: at max
    # recovery each band is as tall as that skid can make it, so a stack top
    # short of the line is a skid dialled down, not an infeasible one.
    for _k in range(1, N_TRAINS + 1):
        reference(ax, _k * RATED_FEED_M3_PER_HR * RECOVERY_MAX,
                  f"{_k} train{'s' if _k > 1 else ''} at {RECOVERY_MAX:.0%} "
                  "recovery")
    ax.legend(frameon=False, fontsize=8, labelcolor=_MUTED, ncols=N_TRAINS,
              loc="upper left", bbox_to_anchor=(0, 0.99))
    ax.set_title(
        f"Permeate by train — each skid is off, or at its "
        f"{RATED_FEED_M3_PER_HR:,.0f} m³/hr feed with recovery free in "
        f"{RECOVERY_MIN:.0%}–{RECOVERY_MAX:.0%}",
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
    # An energy share, so it is quoted as one: the bill also carries a demand
    # charge on the peak-window maximum, and only report_cost prices that.
    ax.set_title(
        f"Plant power — Top of the stack is the facility meter, and RO is "
        f"{frame['ro_kw'].sum() / frame['grid_kw'].sum():.0%} of the energy",
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
    results = solve_relax_and_fix(m)
    print(results)
    print(
        f"relax-and-fix: fixed {m.statuses_fixed:,} of "
        f"{m.statuses_fixed + m.statuses_free:,} statuses at tol={m.fix_tol:g}"
        f" ({len(m.fix_attempts)} rung(s) tried), "
        f"relaxation gap {m.relaxation_gap:.3%}, "
        f"MIP gap on the fixed problem {m.mip_gap:.3%}"
    )
    print(f"electricity bill: ${report_cost(m).operating.electricity:,.2f}")