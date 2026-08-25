# flex-pse-examples

A repository of example test problems built with [flex-pse](https://github.com/flex-pse/flexPSE).

Each example lives in its own directory under [examples/](examples/) and follows the same
layout: a `config.json` describing the problem instance, and a `notebook.py`
[marimo](https://marimo.io) notebook that loads the config, builds the model, solves it, and
walks through the results interactively.

> **Status:** the table below tracks each example's status. A *scaffold* has its `config.json`
> and `notebook.py` in place but no problem definition yet.

## Table of Contents

- [Examples](#examples)
  - [Desalination Scheduling](#desalination-scheduling)
  - [Pump Scheduling](#pump-scheduling)
- [Getting Started](#getting-started)
- [Running an Example](#running-an-example)
- [Running the Tests](#running-the-tests)
- [Repository Layout](#repository-layout)
- [Adding a New Example](#adding-a-new-example)
- [License](#license)

## Examples

| Example | Directory | Status |
| --- | --- | --- |
| [Desalination Scheduling](#desalination-scheduling) | [examples/desalination_scheduling/](examples/desalination_scheduling/) | Complete |
| [Pump Scheduling](#pump-scheduling) | [examples/pump_scheduling/](examples/pump_scheduling/) | Scaffold |

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

## Running the Tests

Pytest is configured repository-wide via [conftest.py](conftest.py):

```bash
pytest
```

## Repository Layout

```
.
├── conftest.py        # repository-wide pytest configuration
├── environment.yml    # conda environment (Python + flex-pse[solvers, dev])
└── examples/
    ├── desalination_scheduling/
    │   ├── config.json    # the problem instance
    │   ├── model.py       # the only place config keys are read
    │   └── notebook.py    # marimo walkthrough
    └── pump_scheduling/
```

## Adding a New Example

1. Create `examples/<example_name>/`.
2. Add a `config.json` with at least a `name` and `description` field, alongside the
   parameters that define the problem instance.
3. For anything beyond a couple of units, put the model wiring in a `model.py` beside it —
   the notebook then reads as a walkthrough rather than as a build script.
4. Add a `notebook.py` marimo notebook that reads the config, builds and solves the model,
   and presents the results.
5. Add the example to the [table of contents](#table-of-contents) and the
   [examples table](#examples) above.

## License

See [LICENSE](LICENSE).
