# flex-pse-examples

A repository of example test problems built with [flex-pse](https://github.com/flex-pse/flexPSE).

**→ Browse them at [flex-pse.github.io/flex-pse-examples](https://flex-pse.github.io/flex-pse-examples/)**,
where each example has an interactive page you can open without installing anything.

Each example lives in its own directory under [examples/](examples/) and follows the same
layout: a `config.json` describing the problem instance, a `model.py` holding the flowsheet, a
`notebook.py` [marimo](https://marimo.io) notebook that builds and solves it, and a
`explore.py` notebook that becomes that example's page on the website.

> **How the website works.** The optimization needs Pyomo, IDAES and a MILP solver, none of
> which have a WebAssembly build — so nothing solves in your browser. Instead each model is
> solved offline across a sweep of one parameter, and the results are committed as
> Parquet under `examples/<name>/public/`, which the pages query with DuckDB. The pages replay that sweep: the charts and controls are live, the
> schedules behind them were computed in advance. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Table of Contents

- [Examples](#examples)
  - [Desalination Scheduling](#desalination-scheduling)
  - [Pump Scheduling](#pump-scheduling)
- [Getting Started](#getting-started)
- [Running an Example](#running-an-example)
- [Building the Website](#building-the-website)
- [Running the Tests](#running-the-tests)
- [Repository Layout](#repository-layout)
- [Adding a New Example](#adding-a-new-example)
- [License](#license)

## Examples

| Example | Directory | Page | Solver |
| --- | --- | --- | --- |
| [Desalination Scheduling](#desalination-scheduling) | [examples/desalination_scheduling/](examples/desalination_scheduling/) | [open](https://flex-pse.github.io/flex-pse-examples/notebooks/desalination_scheduling.html) | Gurobi (non-convex MIQCP) |
| [Pump Scheduling](#pump-scheduling) | [examples/pump_scheduling/](examples/pump_scheduling/) | [open](https://flex-pse.github.io/flex-pse-examples/notebooks/pump_scheduling.html) | HiGHS |

## Getting Started

Dependencies are defined in [pyproject.toml](pyproject.toml), which installs `flex-pse[solvers]`
from the upstream `main` branch.

```bash
pip install -e .[notebooks]
pip install --group dev  # pytest, jinja2, ruff -- only needed to run the test suite
```

To pick up newer upstream changes later, reinstall `flex-pse` from `main`:

```bash
pip install --force-reinstall --no-deps "flex-pse[solvers] @ git+https://github.com/flex-pse/flexPSE.git@main"
```

## Using the Examples Programmatically

Once installed, both examples are importable without cloning the repo:

```python
from flex_pse_examples import list_examples, load_model

list_examples()  # ['desalination_scheduling', 'pump_scheduling']

m = load_model("pump_scheduling")
cfg = m.load_config()
model = m.build_model(cfg)
m.solve_model(model)
```

`load_model` returns the example's `model` module itself -- a model isn't built until you call
`build_model(cfg)` on it. See each example's `model.py` for what else it exposes.

## Running an Example

Examples are marimo notebooks, so they can be opened as an interactive app or executed as a
plain Python script:

```bash
# read-only app view
marimo run examples/desalination_scheduling/notebook.py

# straight through, no UI
python examples/desalination_scheduling/notebook.py
```

An example whose model lives in its own `model.py` can also be run headless, for a text
summary of every scenario without the charts:

```bash
python examples/desalination_scheduling/model.py
```

## Building the Website

```bash
python tools/site/build.py --out _site
python -m http.server --directory _site 8000
```

A WebAssembly export cannot be opened over `file://`, so the second command is not optional.
The build validates every example before exporting anything — a manifest with a missing field,
a sweep whose CSVs were never committed, or an `explore.py` that imports something a browser
cannot run all fail the build rather than the page.

`marimo export html-wasm` requires [`uv`](https://docs.astral.sh/uv/) on PATH.

Regenerate an example's published data with:

```bash
python tools/sweep.py examples/<name>
```

## Running the Tests

Pytest is configured repository-wide via [conftest.py](conftest.py):

```bash
pytest
```

The suite has two halves. `tests/test_sweep_data.py` and `tests/test_wasm_notebooks.py` need
only pandas: they check that every example's committed data matches the contract the website
reads, and that no page imports something a browser cannot run. The rest solve real models, and
so need `flex-pse` and its solvers installed (`pip install -e .` above).

`pyproject.toml` installs flex-pse from `git+…@main`, which pins nothing, so an example can
break without a commit landing here. The `examples` workflow re-runs the solves weekly against
whatever upstream is that day.

## Repository Layout

```
.
├── conftest.py                  # repository-wide pytest configuration
├── pyproject.toml               # dependencies (Python + flex-pse[solvers]) and packaging
├── flex_pse_examples/__init__.py  # list_examples()/load_model(): the installed package's API
├── CONTRIBUTING.md              # how to add an example
├── tools/
│   ├── sweep.py        # solves an example across its sweep, writes the site's data
│   ├── check_drift.py  # does a committed sweep still reproduce?
│   └── site/build.py   # validates every example, exports it, renders the landing page
├── tests/              # data contract, WASM safety, and the real solves
└── examples/<name>/
    ├── example.toml    # site + sweep metadata
    ├── config.json     # the problem instance
    ├── model.py        # the only place config keys are read
    ├── notebook.py     # marimo walkthrough (local; may import anything)
    ├── sweep.py        # the sweep adapter, driven by tools/sweep.py
    ├── explore.py      # the WebAssembly page (browser-safe imports only)
    └── public/<name>/  # generated by tools/sweep.py, and committed (Parquet)
```

## Adding a New Example

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full procedure and the CI checks that enforce
each step. In short: add the model and its `notebook.py`, write a `sweep.py` adapter, run
`python tools/sweep.py examples/<name>` and commit the CSVs it writes, add an `explore.py`
that reads them, and describe the example in `example.toml`.


## License

See [LICENSE](LICENSE).
