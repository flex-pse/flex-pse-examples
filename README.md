# flex-pse-examples

A repository of example test problems built with [flex-pse](https://github.com/flex-pse/flex-pse).

Each example lives in its own directory under [examples/](examples/) and follows the same
layout: a `config.json` describing the problem instance, and a `notebook.py`
[marimo](https://marimo.io) notebook that loads the config, builds the model, solves it, and
walks through the results interactively.

> **Status:** the example directories are currently scaffolds — the config and notebook files
> are in place, but the problem definitions have not been filled in yet. The table below
> tracks what each one is intended to cover.

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
| [Desalination Scheduling](#desalination-scheduling) | [examples/desalination_scheduling/](examples/desalination_scheduling/) | Scaffold |
| [Pump Scheduling](#pump-scheduling) | [examples/pump_scheduling/](examples/pump_scheduling/) | Scaffold |


## Getting Started

The environment is defined in [environment.yml](environment.yml) and pins Python 3.13 plus
`flex-pse[solvers]` from the upstream `main` branch.

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
marimo run examples/pump_scheduling/notebook.py

# straight through, no UI
python examples/pump_scheduling/notebook.py
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
├── environment.yml    # conda environment (Python + flex-pse[solvers])
└── examples/
    ├── desalination_scheduling/
    └── pump_scheduling/
```

## Adding a New Example

1. Create `examples/<example_name>/`.
2. Add a `config.json` with at least a `name` and `description` field, alongside the
   parameters that define the problem instance.
3. Add a `notebook.py` marimo notebook that reads the config, builds and solves the model,
   and presents the results.
4. Add the example to the [table of contents](#table-of-contents) and the
   [examples table](#examples) above.

## License

See [LICENSE](LICENSE).
