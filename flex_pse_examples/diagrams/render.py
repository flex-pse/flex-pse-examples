"""Assemble process-flow diagrams from the bundled icon library.

A diagram is two lists:

* nodes -- dicts ``{"id", "c", "r", ...}`` placed on a grid of CELL_W x CELL_H
  cells (column ``c``, row ``r``). ``kind`` is ``"icon"`` (the default; needs
  ``"icon"``, an id from icons.json), ``"terminal"`` (a labelled source or
  sink) or ``"junction"`` (a header or splitter dot). ``label`` is optional on
  icons and junctions.
* edges -- ``(from_ref, to_ref, flow)`` or ``(from_ref, to_ref, flow, via)``.
  A ref is ``"node_id.port"`` with port one of n/e/s/w, and ``flow`` is a key
  of FLOWS. ``via`` is a list of lanes to pass through, ``{"x": c}`` or
  ``{"y": r}`` in grid units, for routing around other units.

Lines are orthogonal: out of the source port by STUB px, along that port's
axis first, then into the target port. ``render`` returns one self-contained
SVG string -- show it with ``mo.Html``, or write it to a file.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

LIBRARY = json.loads(Path(__file__).with_name("icons.json").read_text())
ICONS = {icon["id"]: icon for icon in LIBRARY["icons"]}
FLOWS = LIBRARY["flows"]

CELL_W, CELL_H = 128, 120
ICON = 64
PAD_X, PAD_Y = (CELL_W - ICON) // 2, 14
STUB = 12
FONT = "Helvetica, Arial, sans-serif"
INK = "#1a1a1a"
DIRS = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}


def _n(v: float) -> str:
    """Coordinates to at most two decimals, without trailing zeros."""
    return f"{round(v, 2):g}"


def _origin(node: dict) -> tuple[float, float]:
    return node["c"] * CELL_W + PAD_X, node["r"] * CELL_H + PAD_Y


def _ports(node: dict) -> dict[str, tuple[float, float]]:
    kind = node.get("kind", "icon")
    if kind == "junction":
        return {p: (32, 32) for p in DIRS}
    if kind == "terminal":
        half = max(20, len(node.get("label", "")) * 3.4)
        return {"n": (32, 24), "e": (32 + half + 4, 32), "s": (32, 40), "w": (32 - half - 4, 32)}
    return {k: tuple(v) for k, v in ICONS[node["icon"]]["ports"].items()}


def _lane(spec: dict, at: tuple[float, float]) -> tuple[float, float]:
    x, y = at
    if "x" in spec:
        x = spec["x"] * CELL_W + PAD_X + ICON / 2
    if "y" in spec:
        y = spec["y"] * CELL_H + PAD_Y + ICON / 2
    return x, y


def route(a, da, b, db, via=(), stub_a=STUB, stub_b=STUB):
    """Orthogonal polyline from port ``a`` (facing ``da``) to port ``b``."""
    a2 = (a[0] + DIRS[da][0] * stub_a, a[1] + DIRS[da][1] * stub_a)
    b2 = (b[0] + DIRS[db][0] * stub_b, b[1] + DIRS[db][1] * stub_b)
    horizontal = da in ("e", "w")
    pts = [a, a2]
    for spec in [*via, None]:
        p = pts[-1]
        q = b2 if spec is None else _lane(spec, p)
        pts += [(q[0], p[1]) if horizontal else (p[0], q[1]), q]
    pts.append(b)
    out: list[tuple[float, float]] = []
    for p in pts:
        if out and p == out[-1]:
            continue
        if len(out) >= 2 and (
            out[-2][0] == out[-1][0] == p[0] or out[-2][1] == out[-1][1] == p[1]
        ):
            out[-1] = p
            continue
        out.append(p)
    return out


def _label_side(node_id: str, edges) -> str:
    """Below the unit, unless a line leaves that way: then above, right, left."""
    used = {
        ref.split(".")[1]
        for edge in edges
        for ref in edge[:2]
        if ref.split(".")[0] == node_id
    }
    for side in ("s", "n", "e", "w"):
        if side not in used:
            return side
    return "s"


def _label(node: dict, side: str) -> str:
    ox, oy = _origin(node)
    x, y, anchor = {
        "s": (ox + 32, oy + ICON + 16, "middle"),
        "n": (ox + 32, oy - 4, "middle"),
        "e": (ox + ICON + 6, oy + 36, "start"),
        "w": (ox - 6, oy + 36, "end"),
    }[side]
    return (
        f'<text x="{_n(x)}" y="{_n(y)}" text-anchor="{anchor}" font-size="12" '
        f'fill="{INK}" stroke="#ffffff" stroke-width="4" stroke-linejoin="round" '
        f'paint-order="stroke">{escape(node["label"])}</text>'
    )


def render(nodes: list[dict], edges: list[tuple]) -> str:
    """The diagram as one SVG string."""
    by_id = {node["id"]: node for node in nodes}
    width = (max(node["c"] for node in nodes) + 1) * CELL_W
    height = (max(node["r"] for node in nodes) + 1) * CELL_H
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="{FONT}" '
        f'style="max-width:100%;height:auto">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        "<defs>",
    ]
    for key, flow in FLOWS.items():
        out.append(
            f'<marker id="arrow-{key}" viewBox="0 0 8 8" refX="7" refY="4" '
            f'markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" '
            f'orient="auto"><path d="M0 0L8 4L0 8Z" fill="{flow["color"]}"/></marker>'
        )
    out.append("</defs>")

    def end(ref: str):
        node_id, port = ref.split(".")
        node = by_id[node_id]
        ox, oy = _origin(node)
        px, py = _ports(node)[port]
        junction = node.get("kind") == "junction"
        return (ox + px, oy + py), port, 0 if junction else STUB, junction

    for edge in edges:
        a_ref, b_ref, flow = edge[:3]
        via = edge[3] if len(edge) > 3 else ()
        a, da, sa, _ = end(a_ref)
        b, db, sb, b_junction = end(b_ref)
        pts = route(a, da, b, db, via, sa, sb)
        style = FLOWS[flow]
        heat = flow == "heat"
        points = " ".join(f"{_n(x)},{_n(y)}" for x, y in pts)
        attrs = [
            f'points="{points}"',
            'fill="none"',
            f'stroke="{style["color"]}"',
            f'stroke-width="{3 if heat else 2.5}"',
            f'stroke-linecap="{"round" if heat else "butt"}"',
            'stroke-linejoin="round"',
        ]
        if style["dasharray"]:
            attrs.append(f'stroke-dasharray="{style["dasharray"]}"')
        if not b_junction:
            attrs.append(f'marker-end="url(#arrow-{flow})"')
        out.append(f"<polyline {' '.join(attrs)}/>")

    for node in nodes:
        ox, oy = _origin(node)
        kind = node.get("kind", "icon")
        if kind == "terminal":
            out.append(
                f'<text x="{_n(ox + 32)}" y="{_n(oy + 36)}" text-anchor="middle" '
                f'font-size="12" font-weight="600" fill="{INK}">'
                f'{escape(node["label"])}</text>'
            )
            continue
        if kind == "junction":
            out.append(f'<circle cx="{_n(ox + 32)}" cy="{_n(oy + 32)}" r="3.5" fill="{INK}"/>')
        else:
            svg = ICONS[node["icon"]]["svg"]
            out.append(re.sub(r"^<svg ", f'<svg x="{_n(ox)}" y="{_n(oy)}" ', svg, count=1))
        if node.get("label"):
            out.append(_label(node, _label_side(node["id"], edges)))
    out.append("</svg>")
    return "\n".join(out)
