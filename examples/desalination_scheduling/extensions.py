"""Spec extensions for the desalination scheduling model.

Everything :mod:`model` builds that has no element kind of its own lives here,
as an ``@flexops.extension``. Calling one records it on the model, so
``flexops.emit_model`` writes it into ``config.json`` as an ``extension``
element naming ``flex_pse_examples.examples.desalination_scheduling.extensions:<fn>``,
and ``flexops.build_model`` re-runs it on rebuild. They live in their own module
rather than in ``model.py`` because a spec has to name a module that imports the
same way everywhere, and ``model`` is imported under a different name by every
caller.
"""


import flexops


@flexops.extension
def recovery_window(model, *, unit: str):
    """Replace each RO skid's bilinear split with its exact linear hull.

    ``ReverseOsmosis`` writes ``permeate[t] == recovery[t] * feed[t]``. With
    recovery free at every step in ``[recovery_min, recovery_max]`` that is
    bilinear, so the month is a non-convex MIQCP that only a global solver
    (Gurobi, SCIP) can take. But because ``recovery[t]`` is chosen per step and
    ``feed[t] >= 0``, the set of reachable ``(feed, permeate)`` pairs is exactly

        recovery_min * feed[t] <= permeate[t] <= recovery_max * feed[t]

    -- two linear rows and no ``recovery`` at all. So this deactivates the
    split and adds those rows, which makes the month a MILP that HiGHS solves.
    ``recovery[t]`` stays fixed at its configured value and out of every
    constraint; the recovery a solved schedule actually ran at is
    ``permeate[t] / feed[t]`` (see ``model.results_frame``).

    The reformulation is exact only while nothing else depends on
    ``recovery[t]``. A surrogate that makes energy intensity a function of
    recovery would bring the product back, and this would have to go.

    Args:
        model: The model.
        unit: Path of the (possibly indexed) ``ReverseOsmosis`` block, e.g.
            ``"plant.ro"``.
    """
    component = model.find_component(unit)
    members = component.values() if component.is_indexed() else [component]
    tb = model.time_block
    for ro in members:
        low, high = ro.recovery[tb.time_index.first()].bounds
        ro.split_definition.deactivate()

        @ro.Constraint(tb.time_index, doc="Permeate at least recovery_min of feed.")
        def recovery_window_min(b, t, low=low):
            return b.permeate[t] >= low * b.feed[t]

        @ro.Constraint(tb.time_index, doc="Permeate at most recovery_max of feed.")
        def recovery_window_max(b, t, high=high):
            return b.permeate[t] <= high * b.feed[t]

