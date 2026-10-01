"""The desalination flowsheet as a diagram: units on a grid, lines between ports.

Rendered by flex_pse_examples.diagrams -- live in notebook.py, and into
public/desalination_scheduling/flowsheet.svg for the WebAssembly page by
tools/render_diagrams.py. Keep it in step with the blocks model.py wires.
"""

TRAINS = range(3)

NODES = [
    {"id": "seawater", "kind": "terminal", "c": 0, "r": 1, "label": "Seawater"},
    {"id": "intake", "icon": "pump", "c": 1, "r": 1, "label": "Intake pump"},
    *({"id": f"pre{i}", "icon": "filter", "c": 2, "r": i, "label": f"Pretreatment[{i}]"} for i in TRAINS),
    *({"id": f"ro{i}", "icon": "membrane", "c": 3, "r": i, "label": f"RO[{i}]"} for i in TRAINS),
    {"id": "permeate", "kind": "junction", "c": 4, "r": 1},
    {"id": "post", "icon": "filter", "c": 5, "r": 1, "label": "Post-treatment"},
    {"id": "product_pump", "icon": "pump", "c": 6, "r": 1, "label": "Product pump"},
    {"id": "product", "kind": "terminal", "c": 7, "r": 1, "label": "Product water"},
    {"id": "brine", "kind": "junction", "c": 4, "r": 3},
    {"id": "ocean", "kind": "terminal", "c": 5, "r": 3, "label": "Ocean outfall"},
]

#: Brine runs down a lane between the RO column and the permeate header.
BRINE_LANE = [{"x": 3.6}]

EDGES = [
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
