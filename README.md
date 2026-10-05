# flex-pse-examples

Example problems built with [flex-pse](https://github.com/flex-pse/flexPSE).

**Browse them at [flex-pse.github.io/flex-pse-examples](https://flex-pse.github.io/flex-pse-examples/).**
Every example has a page that runs in your browser, nothing to install.

| Example | Question | Page | Solver |
| --- | --- | --- | --- |
| [Pump scheduling](examples/pump_scheduling/) | Bigger battery, or flexible pumps? | [open](https://flex-pse.github.io/flex-pse-examples/notebooks/pump_scheduling.html) | HiGHS |
| [Desalination scheduling](examples/desalination_scheduling/) | How does a monthly water target change the cost per acre-foot? | [open](https://flex-pse.github.io/flex-pse-examples/notebooks/desalination_scheduling.html) | Gurobi |

The solvers don't run in a browser, so each model is solved ahead of time over a
range of one input, and the results are committed as Parquet under
`examples/<name>/public/`. The pages read those files.

## Setup

```bash
pip install -e ".[notebooks]"
pip install --group dev   # only for the tests
```

flex-pse is installed from its `main` branch. To pick up newer changes:

```bash
pip install --force-reinstall --no-deps "flex-pse[solvers] @ git+https://github.com/flex-pse/flexPSE.git@main"
```

## Running an example

```bash
marimo run examples/pump_scheduling/notebook.py   # as an app
python examples/pump_scheduling/notebook.py       # as a script
python examples/pump_scheduling/model.py          # text summary, no charts
```

Or from Python:

```python
from flex_pse_examples import list_examples, load_model

m = load_model("pump_scheduling")
model = m.build_model(m.load_config())
m.solve_model(model)
```

## Building the site

```bash
python tools/sweep.py examples/<name>     # re-solve and rewrite an example's data
python tools/site/build.py --out _site    # needs uv on PATH
python -m http.server --directory _site 8000
```

The pages won't load over `file://`, so you do need the local server. The build
checks every example first and fails if, for example, a page imports something
that can't run in a browser.

## Tests

```bash
pytest
```

`test_sweep_data.py` and `test_wasm_notebooks.py` only need pandas. They check the
committed data and that the pages are browser-safe. The other tests solve real
models and need flex-pse installed. Since flex-pse comes from `main` unpinned, CI
also re-runs the solves weekly.

## Layout

```
examples/<name>/
  example.toml    site + sweep metadata
  config.json     problem inputs
  model.py        the model
  notebook.py     marimo notebook that builds and solves it
  sweep.py        sweep adapter for tools/sweep.py
  explore.py      the browser page (browser-safe imports only)
  public/<name>/  solved results (generated, committed)
tools/
  sweep.py        solve an example across its sweep
  check_drift.py  check a committed sweep still reproduces
  site/build.py   validate and build the site
tests/
```

To add an example, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).
