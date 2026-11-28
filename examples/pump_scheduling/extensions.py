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
from flexcore.config.schema import DegradationSpec, DegradationTermSpec
from flexops.logic import add_degradation, add_status

_M3_HR = pyunits.m**3 / pyunits.hr


@flexops.extension
def semicontinuous_flow(
    model,
    *,
    unit: str,
    min_flow: float,
    max_flow: float,
    start_stop_cost: float = 0.0,
):
    """Make a pump's inlet flow off, or anywhere in ``[min_flow, max_flow]`` m^3/hr.

    An extension rather than a ``status`` element because the element needs a
    time-only variable on the unit, and ``Pump`` carries its flow only as the
    ``[t, phase]`` state ``inlet_state.flow_vol_phase``. So this exposes that as
    a time-only ``flow_in`` Reference first (the idiom ``Tank.flow_in`` uses),
    then attaches the on/off binary and the two links pinning flow to it.

    With a ``start_stop_cost`` it also prices the switching, as a
    :func:`flexops.logic.add_degradation` penalty: a ``variation`` term on the
    status charges ``|status[t] - status[t-1]|``, so every start and every stop
    costs ``start_stop_cost``, billed through the model's ``costing`` block.
    It lives here, not as a ``degradation`` spec element, because the
    ``status`` it watches only exists once this extension has run -- and a
    spec builds degradation elements before extensions.

    Args:
        model: The model.
        unit: Path of the pump, e.g. ``"feed_pump"``.
        min_flow: Lowest flow while on, in m^3/hr.
        max_flow: Rated flow, in m^3/hr.
        start_stop_cost: Wear cost per on/off transition, in the tariff's
            currency. ``0`` adds no penalty.
    """
    pump = model.find_component(unit)
    pump.flow_in = pyo.Reference(pump.inlet_state.flow_vol_phase[:, "Liq"])
    add_status(pump, pump.flow_in, min_flow * _M3_HR, max_flow * _M3_HR)
    if start_stop_cost:
        add_degradation(
            pump,
            DegradationSpec(
                name="start_stop_wear",
                terms=[
                    DegradationTermSpec(
                        kind="variation", variable="status", price=start_stop_cost
                    )
                ],
            ),
            costing=model.costing,
        )
