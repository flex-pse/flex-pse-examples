"""Every example's flowsheet diagram is wired to real icons and ports, and the
SVG the WebAssembly page shows is the one diagram.py renders today."""

import pytest

from flex_pse_examples.diagrams import FLOWS, ICONS, render
from flex_pse_examples.diagrams.render import DIRS
from tools import render_diagrams

CASES = list(render_diagrams.targets())
IDS = [example.name for example, _ in CASES]


@pytest.mark.parametrize(("example", "svg_path"), CASES, ids=IDS)
def test_diagram_is_wired(example, svg_path):
    diagram = render_diagrams.load(example / "diagram.py")
    ids = [node["id"] for node in diagram.NODES]
    assert len(ids) == len(set(ids)), "duplicate node ids"
    for node in diagram.NODES:
        if node.get("kind", "icon") == "icon":
            assert node["icon"] in ICONS, f"{node['id']}: no icon {node['icon']!r}"
    for edge in diagram.EDGES:
        for ref in edge[:2]:
            node_id, port = ref.split(".")
            assert node_id in ids, f"{ref}: no such node"
            assert port in DIRS, f"{ref}: port must be one of {sorted(DIRS)}"
        assert edge[2] in FLOWS, f"{edge}: flow must be one of {sorted(FLOWS)}"


@pytest.mark.parametrize(("example", "svg_path"), CASES, ids=IDS)
def test_committed_svg_is_current(example, svg_path):
    diagram = render_diagrams.load(example / "diagram.py")
    assert svg_path.exists(), f"run python tools/render_diagrams.py ({svg_path.name} missing)"
    assert svg_path.read_text() == render(diagram.NODES, diagram.EDGES), (
        "flowsheet.svg is stale: run python tools/render_diagrams.py"
    )
