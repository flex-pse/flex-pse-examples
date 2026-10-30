"""Sweep adapter for sb_retrofit_demo. Driven by `tools/sweep.py`.

The "sweep" is the example's two cases, not a range of one knob: point 0 is the
operators' schedule on the existing plant, point 1 the optimized retrofit. Case
B has to deliver exactly the water case A delivers and its load-shed figure is
measured against case A's plant load, so :func:`setup` solves case A once (a
couple of seconds: every status is fixed) and hands both numbers to every point.
"""

from flex_pse_examples.examples.sb_retrofit_demo import model

#: CI horizon: a weekend and three weekdays, enough peak windows to schedule.
SMOKE_START, SMOKE_END = "2026-07-04", "2026-07-08"


def _horizon(smoke: bool) -> dict:
    return {"start": SMOKE_START, "end": SMOKE_END} if smoke else {}


def points(*, smoke: bool = False) -> list[dict]:
    """The two cases, in order."""
    return [
        {"label": model.CASES[name]["label"], "case": name, "case_index": i}
        for i, name in enumerate(model.CASES)
    ]


def setup(*, smoke: bool = False):
    """Solve case A once: its delivered water and plant load anchor case B."""
    horizon = _horizon(smoke)
    a = model.build_case("A_before", **horizon)
    model.solve_case(a)
    frame = model.results_frame(a)
    return {"horizon": horizon, "reference": model.kpis(a, frame), "smoke": smoke}


def solve_point(ctx, point: dict):
    """Build and solve one case. Returns ``(frame, summary)``."""
    reference = ctx["reference"]
    name = point["case"]
    if name == "A_before":
        m = model.build_case(name, **ctx["horizon"])
        model.solve_case(m)
        frame = model.results_frame(m)
        k = model.kpis(m, frame)
    else:
        m = model.build_case(name, demand_m3=reference["water_m3"], **ctx["horizon"])
        model.solve_case(m, time_limit=600 if ctx["smoke"] else None)
        frame = model.results_frame(m)
        k = model.kpis(m, frame, reference=reference)
    k["savings_pct"] = 100.0 * (1.0 - k["bill_usd"] / reference["bill_usd"])
    k["termination"] = m.termination
    k["case_index"] = point["case_index"]
    k["case"] = name
    return frame, k
