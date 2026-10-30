"""The retrofit demo's flowsheet, before and after, as icon-library diagrams.

The water side is the ``desalination_scheduling`` flowsheet. Case B adds the
electrical side: the grid and the battery on a bus feeding the plant. Rendered by
``flex_pse_examples.diagrams`` and written to ``public/sb_retrofit_demo/pfd_*.svg``
by :func:`write_all`.

:func:`render_state` draws one *state* of the plant -- which units are shed, and
whether the battery is discharging -- by laying translucent overlays on the
rendered SVG. ``render.py`` itself is not modified.
"""

from __future__ import annotations

from pathlib import Path

import importlib

_render = importlib.import_module("flex_pse_examples.diagrams.render")

TRAINS = range(3)
BG = "#3C3C3C"
CYAN = "#1F8FB0"

#: Brine runs down a lane between the RO column and the permeate header.
BRINE_LANE = [{"x": 3.6}]


def nodes(case: str) -> list[dict]:
    """The flowsheet's units for ``case`` (``"A_before"`` or ``"B_after"``)."""
    vessels = 30 if case == "A_before" else 36
    out = [
        {"id": "seawater", "kind": "terminal", "c": 0, "r": 1, "label": "Seawater"},
        {"id": "intake", "icon": "pump", "c": 1, "r": 1, "label": "Intake pump"},
        *({"id": f"pre{i}", "icon": "filter", "c": 2, "r": i, "label": f"Pretreatment {i + 1}"} for i in TRAINS),
        *({"id": f"ro{i}", "icon": "membrane", "c": 3, "r": i, "label": f"RO {i + 1} · {vessels} vessels"} for i in TRAINS),
        {"id": "permeate", "kind": "junction", "c": 4, "r": 1},
        {"id": "post", "icon": "filter", "c": 5, "r": 1, "label": "Post-treatment"},
        {"id": "product_pump", "icon": "pump", "c": 6, "r": 1, "label": "Product pump"},
        {"id": "product", "kind": "terminal", "c": 7, "r": 1, "label": "Product water"},
        {"id": "brine", "kind": "junction", "c": 4, "r": 3},
        {"id": "ocean", "kind": "terminal", "c": 5, "r": 3, "label": "Ocean outfall"},
    ]
    if case == "B_after":
        out += [
            {"id": "grid", "icon": "grid", "c": 0, "r": 4, "label": "Grid"},
            {"id": "bus", "kind": "junction", "c": 1, "r": 4},
            {"id": "battery", "icon": "battery", "c": 2, "r": 4, "label": "Battery · 200 kW / 1 MWh"},
        ]
    return out


def edges(case: str) -> list[tuple]:
    """The flowsheet's connections for ``case``."""
    out = [
        ("seawater.e", "intake.w", "fluid"),
        *(("intake.e", f"pre{i}.w", "fluid") for i in TRAINS),
        *((f"pre{i}.e", f"ro{i}.w", "fluid") for i in TRAINS),
        *((f"ro{i}.e", "permeate.w", "fluid") for i in TRAINS),
        *((f"ro{i}.s", "brine.w", "fluid", BRINE_LANE) for i in TRAINS),
        ("permeate.e", "post.w", "fluid"),
        ("post.e", "product_pump.w", "fluid"),
        ("product_pump.e", "product.w", "fluid"),
        ("brine.e", "ocean.w", "fluid"),
    ]
    if case == "B_after":
        out += [
            ("grid.e", "bus.w", "electricity"),
            ("battery.w", "bus.e", "electricity"),
            ("bus.n", "intake.s", "electricity"),
        ]
    return out


#: The after-retrofit flowsheet, for ``tools/render_diagrams.py`` and
#: ``tests/test_diagrams.py``: it writes ``public/sb_retrofit_demo/flowsheet.svg``.
NODES = nodes("B_after")
EDGES = edges("B_after")

#: Units off in each state. In the peak state of case A the operators shed the
#: swing train; in case B the whole RO system and post-treatment are down.
STATES = {
    ("A_before", "offpeak"): dict(off=set(), discharging=False),
    ("A_before", "peak"): dict(off={"ro2"}, discharging=False),
    ("B_after", "offpeak"): dict(off=set(), discharging=False),
    ("B_after", "peak"): dict(off={"ro0", "ro1", "ro2", "post", "product_pump"}, discharging=True),
}


def render_state(case: str, *, off=frozenset(), discharging: bool = False) -> str:
    """The flowsheet as SVG, with shed units dimmed and the battery lit if discharging."""
    ns, es = nodes(case), edges(case)
    svg = _render.render(ns, es)
    by_id = {n["id"]: n for n in ns}
    overlays = []
    for node_id in off:
        x, y = _render._origin(by_id[node_id])
        overlays.append(
            f'<rect x="{x - 4}" y="{y - 4}" width="{_render.ICON + 8}" height="{_render.ICON + 8}" '
            f'rx="6" fill="{BG}" fill-opacity="0.65"/>'
        )
        overlays.append(
            f'<text x="{x + _render.ICON / 2}" y="{y + _render.ICON / 2 + 4}" text-anchor="middle" '
            f'font-size="11" font-weight="700" fill="#ffffff">OFF</text>'
        )
    if discharging and "battery" in by_id:
        x, y = _render._origin(by_id["battery"])
        overlays.append(
            f'<rect x="{x - 6}" y="{y - 6}" width="{_render.ICON + 12}" height="{_render.ICON + 12}" '
            f'rx="8" fill="none" stroke="{CYAN}" stroke-width="3"/>'
        )
    return svg.replace("</svg>", "\n".join(overlays) + "\n</svg>")


def render_named(case: str, period: str) -> str:
    """One of the four named states: ``period`` is ``"offpeak"`` or ``"peak"``."""
    return render_state(case, **STATES[(case, period)])


def _png(paths) -> None:
    """Rasterize SVGs at 2x with headless Chromium (cairosvg drops nested icons)."""
    import re

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for path in paths:
            svg = path.read_text()
            w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
            page = browser.new_page(viewport={"width": int(w), "height": int(h)}, device_scale_factor=2)
            page.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
            page.screenshot(path=str(path.with_suffix(".png")))
        browser.close()


def write_all(out_dir: Path | None = None) -> list[Path]:
    """Write the four ``pfd_<case>_<period>.svg`` files, plus 2x PNGs when playwright is installed."""
    out_dir = out_dir or Path(__file__).parent / "public" / "sb_retrofit_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for case, period in STATES:
        short = "A" if case == "A_before" else "B"
        path = out_dir / f"pfd_{short}_{period}.svg"
        path.write_text(render_named(case, period))
        written.append(path)
    try:
        _png(written)
    except ImportError:  # playwright is optional: the SVGs are what the page uses
        pass
    return written


if __name__ == "__main__":
    for p in write_all():
        print(p)
