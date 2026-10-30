"""Spec extensions for the Santa Barbara retrofit demo.

Everything :mod:`model` builds that has no element kind of its own lives here as
an ``@flexops.extension``, so ``flexops.emit_model`` can name it in
``config.json`` and ``flexops.build_model`` can re-run it on rebuild. They are
imported by package path (``flex_pse_examples.examples.sb_retrofit_demo.
extensions``) so the spec names a module that imports the same way everywhere.
"""

import pyomo.environ as pyo

import flexops


@flexops.extension
def linear_recovery_window(model, *, unit: str, recovery_min: float, recovery_max: float):
    """Replace an RO skid's bilinear split with an exact linear window.

    ``ReverseOsmosis`` carries ``permeate[t] == recovery * feed[t]``. With
    ``recovery`` a free Var that product is bilinear, and the month becomes a
    non-convex MIQCP only Gurobi can take. But ``recovery`` appears nowhere else
    -- the RO energy intensity is charged per m^3 of *permeate*, and brine is the
    mass-balance remainder -- so letting it vary per step in
    ``[recovery_min, recovery_max]`` is *exactly* the pair of linear bounds::

        recovery_min * feed[t] <= permeate[t] <= recovery_max * feed[t]

    The bilinear ``split_definition`` is deactivated and ``recovery`` is fixed
    at its initial value (it is no longer read by any active constraint, and a
    free Var with no constraint would only be noise for the solver). The model
    is then a plain MILP that HiGHS solves.

    The per-step recovery a solved model ran at is ``permeate[t] / feed[t]``
    wherever the skid is on.

    Args:
        model: The model.
        unit: Path of the RO skid, e.g. ``"plant.ro[0]"``.
        recovery_min: Lowest recovery the membranes may run at.
        recovery_max: Highest recovery the membranes may run at.
    """
    ro = model.find_component(unit)
    ro.split_definition.deactivate()
    ro.recovery.fix()
    times = model.time_block.time_index

    ro.recovery_window_low = pyo.Constraint(
        times,
        rule=lambda b, t: ro.permeate[t] >= recovery_min * ro.feed[t],
        doc="Permeate is at least recovery_min of the feed.",
    )
    ro.recovery_window_high = pyo.Constraint(
        times,
        rule=lambda b, t: ro.permeate[t] <= recovery_max * ro.feed[t],
        doc="Permeate is at most recovery_max of the feed.",
    )


@flexops.extension
def terminal_charge(model, *, battery: str):
    """End the horizon at least as charged as ``battery`` started.

    Without it the schedule ends the month empty, spending stored energy it
    never has to buy back. Same as ``pump_scheduling.extensions``.
    """
    block = model.find_component(battery)
    last = model.time_block.time_index.last()
    block.terminal_charge = pyo.Constraint(
        expr=block.charge[last] >= block.charge_init,
        doc="End the horizon at least as charged as it started.",
    )


@flexops.extension
def no_export(model, *, plant: str = "plant"):
    """Forbid the plant from exporting power to the grid.

    ``BatteryModel`` is behind the meter and may push the plant's net draw
    negative; its docstring puts the "net draw >= 0" rule at the plant level.
    The facility meter is the sum of every unit's ``power_electrical`` (the
    battery's is its net charge), so that sum is held non-negative. Written on
    the units rather than on ``costing.aggregate_power`` so the extension does
    not depend on the costing having been built first -- which it has not when
    ``flexops.build_model`` replays the spec.
    """
    block = model.find_component(plant)
    units = [
        unit
        for unit in block.component_data_objects(pyo.Block, descend_into=True)
        if unit.component("power_electrical") is not None
    ]
    times = model.time_block.time_index
    block.no_export = pyo.Constraint(
        times,
        rule=lambda b, t: sum(u.power_electrical[t] for u in units) >= 0,
        doc="The facility never exports electricity.",
    )


@flexops.extension
def operator_weekday_peak_shed(
    model,
    *,
    swing_unit: str,
    always_on: list[str],
    peak_start_hour: int,
    peak_end_hour: int,
):
    """Fix every RO status to the operator's baselining experiment.

    The schedule the Santa Barbara operators ran for a month: shut down one of
    the three skids (``swing_unit``) on weekdays from ``peak_start_hour`` to
    ``peak_end_hour``, run everything else all the time. Nothing about it is
    optimized, so every status is *fixed*; the solve is only an evaluation.

    Args:
        model: The model.
        swing_unit: Path of the skid the operators shed, e.g. ``"plant.ro[2]"``.
        always_on: Paths of units held on at every step (the lead RO pair and
            post-treatment).
        peak_start_hour: First hour of the shed (inclusive).
        peak_end_hour: Hour the shed ends (exclusive).
    """
    tb = model.time_block
    for path in always_on:
        unit = model.find_component(path)
        for t in tb.time_index:
            unit.status[t].fix(1)
    swing = model.find_component(swing_unit)
    for t in tb.time_index:
        stamp = tb.datetime_index[t]
        shed = stamp.weekday() < 5 and peak_start_hour <= stamp.hour < peak_end_hour
        swing.status[t].fix(0 if shed else 1)
