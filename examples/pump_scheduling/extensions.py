"""Spec extensions for the pump scheduling model.

Everything :mod:`model` builds that has no element kind of its own lives here,
as an ``@flexops.extension``. Calling one records it on the model, so
``flexops.emit_model`` writes it into ``config.json`` as an ``extension``
element naming ``flex_pse_examples.examples.pump_scheduling.extensions:<fn>``,
and ``flexops.build_model`` re-runs it on rebuild. That entry point is why these
live in their own module rather than in ``model.py``: ``model`` is imported
under a different name by every caller (the notebook, the sweep tool, the
tests), and a spec has to name a module that imports the same way everywhere.
"""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

import flexops
from flexops.logic import add_status

_M3_HR = pyunits.m**3 / pyunits.hr


@flexops.extension
def semicontinuous_flow(model, *, unit: str, min_flow: float, max_flow: float):
    """Make a pump's inlet flow off, or anywhere in ``[min_flow, max_flow]`` m^3/hr.

    An extension rather than a ``status`` element because the element needs a
    time-only variable on the unit, and ``Pump`` carries its flow only as the
    ``[t, phase]`` state ``inlet_state.flow_vol_phase``. So this exposes that as
    a time-only ``flow_in`` Reference first (the idiom ``Tank.flow_in`` uses),
    then attaches the on/off binary and the two links pinning flow to it.

    Args:
        model: The model.
        unit: Path of the pump, e.g. ``"feed_pump"``.
        min_flow: Lowest flow while on, in m^3/hr.
        max_flow: Rated flow, in m^3/hr.
    """
    pump = model.find_component(unit)
    pump.flow_in = pyo.Reference(pump.inlet_state.flow_vol_phase[:, "Liq"])
    add_status(pump, pump.flow_in, min_flow * _M3_HR, max_flow * _M3_HR)


@flexops.extension
def terminal_volume(model, *, tank: str, volume: float):
    """End the horizon with at least ``volume`` m^3 in ``tank``.

    Without it the schedule just drains the tank and under-pumps.
    """
    block = model.find_component(tank)
    last = model.time_block.time_index.last()
    block.terminal_volume = pyo.Constraint(
        expr=block.volume[last] >= volume,
        doc="End the horizon at least as full as the terminal target.",
    )


@flexops.extension
def terminal_charge(model, *, battery: str):
    """End the horizon at least as charged as ``battery`` started.

    Without it the schedule ends the horizon empty, selling off stored energy
    it never has to buy back.
    """
    block = model.find_component(battery)
    last = model.time_block.time_index.last()
    block.terminal_charge = pyo.Constraint(
        expr=block.charge[last] >= block.charge_init,
        doc="End the horizon at least as charged as it started.",
    )
