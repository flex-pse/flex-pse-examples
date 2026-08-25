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
> solved offline across a sweep of one parameter, and the results are committed under
> `examples/<name>/public/`. The pages replay that sweep: the charts and controls are live, the
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

### Desalination Scheduling

[examples/desalination_scheduling/](examples/desalination_scheduling/) — a seawater
desalination plant with three parallel treatment trains, scheduled against a time-of-use
tariff.

```
                   ┌─► pretreatment[0] ─► RO[0] ─┬─► brine ─► ocean
seawater ─► intake ─┼─► pretreatment[1] ─► RO[1] ─┤        (permeate)
             pump   └─► pretreatment[2] ─► RO[2] ─┘            │
                                                               ▼
  product water ◄─ product pump ◄─ post-treatment ◄─ permeate header
```

An intake pump feeds a header that splits across three constant-energy-intensity
pretreatment units; each feeds its own reverse-osmosis skid, whose permeate recombines
into a shared post-treatment step and a product water pump. Brine leaves each skid to the
ocean outfall.

The plant owes a **volume** of product water over the month — 265 acre-feet by default, not
an hourly profile — and there is no storage in the flowsheet, so the only way to dodge an
expensive hour is to make less water in it and more water elsewhere. The example solves a
full calendar month at 15-minute resolution as a unit-commitment problem: each RO skid is
off, or running inside its turndown band, via `flexops.logic.add_status`,
`add_startup_shutdown` and `register_parallel_group`.

Stepping the train count down from three to two is free; restarting the RO *system* is
not. For 45 minutes afterwards post-treatment is out, and every train's permeate — not
just the restarting one's — leaves off-spec to the outfall while the plant pays full power
to make it. That, plus a narrow turndown band and — at the default demand — a plant sized
close to its obligation, is what makes the answer interesting: the headroom, not the
tariff, is the binding constraint.

That obligation is the notebook's one knob. It is a slider on the notebook page, and it
enters the model as a mutable `Param`, so the month is built once and each new demand
costs only a re-solve. How much slack the schedule has is entirely a function of where the
demand sits against `model.max_product_af()` — the ~284 acre-feet three skids at rated
feed make if they never stop. Near the ceiling the plant runs flat out and there is
nothing to schedule; well below it, the optimizer can afford to sit out the whole peak
window.

The month carries ~27,000 binaries and does not solve to optimality in a sitting, so the
notebook defaults to the LP relaxation (`flexops.logic.relax`), which solves in well under
a minute. The relaxation is optimistic — a fractional status runs a skid below its
turndown floor and pays only a fraction of a restart — so read its cost as a **lower
bound**, and flip the switch off for the exact MILP when a runnable schedule is what is
wanted.

The flowsheet is written in code in [model.py](examples/desalination_scheduling/model.py);
[config.json](examples/desalination_scheduling/config.json) supplies the one piece with no
code form yet, the EECO tariff. The solver and its options are `SOLVER` and `SOLVER_OPTIONS`
in `model.py`. Run `model.py` directly for a text
summary, or open the notebook for the slider and the charts.

### Pump Scheduling

[examples/pump_scheduling/](examples/pump_scheduling/) — a two-pump water plant scheduling
against a time-of-use tariff, framed as a flexibility-versus-storage trade-off.

A feed pump lifts water across a fixed pressure rise into a 1500 m³ tank; a product pump draws
from the tank to meet a fixed 24-hour demand profile. Every case moves the same water and buys
the same total pump energy, so the whole cost spread is *when* the energy is bought.

The question is whether operational flexibility substitutes for storage. Sweeping the battery
from nothing to a rating equal to the plant's entire peak electrical load, at both a flexible
feed pump (semicontinuous — off, or 60–100 % of rated flow, a MILP) and an inflexible one
(pinned at constant duty, an LP), answers it directly: **flexibility alone, with no battery at
all, beats a constant-duty plant carrying a battery rated at ~65 % of its peak load.**

It is a 24-step LP/MILP on HiGHS and solves in well under a second, which makes it the one
example CI can exercise end to end.

## Getting Started

The environment is defined in [environment.yml](environment.yml) and pins Python 3.13 plus
`flex-pse[solvers, dev]` from the upstream `main` branch.

```bash
conda env create -f environment.yml
conda activate flex-pse-examples
```

To pick up newer upstream changes later:

```bash
conda env update -f environment.yml --prune
```

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
so need the conda environment.

`environment.yml` installs flex-pse from `git+…@main`, which pins nothing, so an example can
break without a commit landing here. The `examples` workflow re-runs the solves weekly against
whatever upstream is that day.

## Repository Layout

```
.
├── conftest.py         # repository-wide pytest configuration
├── environment.yml     # conda environment (Python + flex-pse[solvers, dev])
├── CONTRIBUTING.md     # how to add an example
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
    └── public/<name>/  # generated by tools/sweep.py, and committed
```

## Adding a New Example

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full procedure and the CI checks that enforce
each step. In short: add the model and its `notebook.py`, write a `sweep.py` adapter, run
`python tools/sweep.py examples/<name>` and commit the CSVs it writes, add an `explore.py`
that reads them, and describe the example in `example.toml`.


## License

See [LICENSE](LICENSE).
