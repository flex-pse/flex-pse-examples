"""The pump-scheduling flowsheet as a diagram: units on a grid, lines between ports.

Rendered by flex_pse_examples.diagrams -- live in notebook.py, and into
public/pump_scheduling/flowsheet.svg for the WebAssembly page by
tools/render_diagrams.py. Keep it in step with the blocks model.py wires.
"""

NODES = [
    {"id": "source", "kind": "terminal", "c": 0, "r": 0, "label": "Source"},
    {"id": "feed_pump", "icon": "pump", "c": 1, "r": 0, "label": "Feed pump"},
    {"id": "tank", "icon": "tank", "c": 2, "r": 0, "label": "Storage tank"},
    {"id": "product_pump", "icon": "pump", "c": 3, "r": 0, "label": "Product pump"},
    {"id": "user", "kind": "terminal", "c": 4, "r": 0, "label": "User demand"},
    {"id": "grid", "icon": "grid", "c": 2, "r": 1, "label": "Grid"},
    {"id": "battery", "icon": "battery", "c": 2, "r": 2, "label": "Battery"},
]

EDGES = [
    ("source.e", "feed_pump.w", "fluid"),
    ("feed_pump.e", "tank.w", "fluid"),
    ("tank.e", "product_pump.w", "fluid"),
    ("product_pump.e", "user.w", "fluid"),
    ("grid.w", "feed_pump.s", "electricity"),
    ("grid.e", "product_pump.s", "electricity"),
    ("grid.s", "battery.n", "electricity"),
]
