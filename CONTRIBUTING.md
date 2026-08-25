# Adding an example

Every example in this repository is two things at once: a model you can solve
locally, and a page on <https://flex-pse.github.io/flex-pse-examples/> that
someone can open without installing anything. This describes how to add one.

## Why an example is split in two

The optimization needs Pyomo, IDAES and a MILP solver. None of them have a
WebAssembly build, `flex-pse` installs from git (which `micropip` cannot do), and
`highspy` publishes no `emscripten` wheel — so **nothing solves in a browser**,
and no amount of build cleverness changes that.

So each example has two notebooks and a script between them:

| File | Runs where | Does what |
| --- | --- | --- |
| `notebook.py` | your machine | The solver-level walkthrough. May import anything. |
| `sweep.py` | your machine, and CI | An adapter that solves the model at each point of one sweep. |
| `explore.py` | **a browser** | Replays the committed sweep. May import almost nothing. |

`tools/sweep.py` drives `sweep.py` and writes CSVs into `public/<name>/`, which
are **committed to the repository**. `explore.py` reads them. The website build
never solves anything.

## The layout

```
examples/<name>/
    example.toml          site + sweep metadata
    config.json           the problem instance
    model.py              the flex-pse flowsheet
    notebook.py           the solver-level walkthrough
    sweep.py              the sweep adapter
    explore.py            the WebAssembly page
    public/<name>/        generated, and committed
        provenance.csv
        summary.csv
        series/sNN.csv
```

The doubled `public/<name>/` is not a typo. Every example's `public/` folder is
merged into a single directory in the export, so the inner folder is what keeps
one example's data from overwriting another's.

## The checklist

Each step names the check that enforces it, so you find out at build time rather
than in a reader's browser.

**1. Create `examples/<name>/`**, `lower_snake_case`.
→ *`example.toml`'s `name` must equal the directory name.*

**2. Add `config.json`** with at least `name` and `description`, alongside the
parameters that define the instance.

**3. Add `model.py`** — the flowsheet, and the only place `config.json` keys are
read. Take the horizon, and anything that hardens the problem class, as keyword
arguments rather than module constants; that is what makes a CI-sized smoke test
possible later.
→ *`tests/test_<name>.py`, which you write in step 9.*

**4. Add `notebook.py`** — the solver-level walkthrough. Local only, so it may
import anything. This is what the website links to when it says "for the real
thing, see here".
→ *`tools/site/build.py` checks the file exists.*

**5. Add `sweep.py`**, the adapter. Three names, and nothing else:

```python
def points(*, smoke=False) -> list[dict]:
    """Each dict needs a "label"; usually also the manifest's sweep.axis key."""

def setup(*, smoke=False):
    """Anything worth building once. Returns an opaque context."""

def solve_point(ctx, point) -> tuple[DataFrame, dict]:
    """Solve one point. Returns the results frame and a flat dict of scalars."""
```

Pass `smoke=True` through to a reduced instance that a free CI runner can finish
in a couple of minutes — a shorter horizon, fewer points, a linearized
constraint. If the full model needs a commercial solver, this is the only thing
CI will ever run.

**6. Add the `[sweep]` block** to `example.toml` and generate the data:

```bash
python tools/sweep.py examples/<name>
```

`columns` prunes the results frame, and `[[sweep.views]]` reduces it — `window`
for a few days at native resolution, `diurnal` for a time-of-day average, `full`
for a horizon short enough to draw whole. A month at 15-minute resolution is
~2,976 rows per point; shipping it raw costs megabytes for detail nobody can
see, and the browser fetches these files *synchronously*, so it is a stall as
well as a download. The build refuses any view file over 2 MB.

**7. Commit `public/<name>/`.**
→ *The build fails without `summary.csv`, and fails if the committed sweep is
marked as a reduced `--smoke` run.*

**8. Add `explore.py`**, the page. It must:

- start with a PEP 723 header naming its dependencies;
- import only from the allowlist — `marimo`, `numpy`, `pandas`, `matplotlib`,
  `scipy` and safe stdlib. No `pyomo`, no `flexops`, no sibling `model`;
- read its data through `mo.notebook_location()`, never `open()` or
  `Path(__file__)` — in the browser that location is a *URL*;
- open with the precomputed-results callout (copy an existing one), so no reader
  is left thinking the page is solving anything.

→ *`tools/site/build.py`'s import allowlist and `tests/test_wasm_notebooks.py`.*
This is the check that matters most: a bad import here exports cleanly,
publishes cleanly, and then throws in a reader's browser.

**9. Add `tests/test_<name>.py`.** Load the model with
`load_example_module(EXAMPLE)` from `conftest.py`, **not** a bare `import model`.
Every example has a `model.py`, so the first bare import wins `sys.modules` and
every later example silently gets the wrong one — which surfaces as a pile of
`AttributeError`s the moment two example test files run in the same session. Assert the *claim* the example makes, not
just that it returns without raising. If the full model cannot run on a public
runner, set `solve.ci_runnable = false` in the manifest and write a reduced smoke
test anyway — and say in that file's docstring exactly what is not covered.
→ *The build fails if `ci_runnable = true` and no test file exists.*

**10. Fill in the site fields** in `example.toml`: `title`, `order`, `tags`, and
a `blurb` of **one or two sentences**. The blurb is a card on the landing page,
not a paragraph.
→ *The build counts the sentences.*

**11. Build and look at it.**

```bash
python tools/site/build.py --out _site
python -m http.server --directory _site 8000
```

A WebAssembly export **cannot** be opened over `file://`. Serve it over HTTP or
you will see a blank page and blame the wrong thing.

**12. Add the example to the table in `README.md`.**
→ *Not enforced. Sorry.*

**13. Open a pull request.** `pages.yml` builds the site and uploads it as a
`site-preview` artifact without deploying; `examples.yml` runs the contract tests
and the solves.

## What CI does and does not cover

`pages.yml` installs no flex-pse and runs no solver — the sweeps are committed,
so the site build is fast and cannot be broken by an upstream change.

`examples.yml` is what notices upstream changes. `environment.yml` installs
flex-pse from `git+…@main`, which pins nothing, so an example can break with no
commit landing here; the weekly run is the thing that finds out. Its conda cache
key includes the ISO week for exactly that reason — a permanent cache would test
a frozen upstream forever.

**Not covered, and worth knowing:**

- The desalination example's full month. It is a non-convex MIQCP needing Gurobi,
  and there is no Gurobi license on a public runner. CI solves a linearized short
  horizon instead.
- Whether a committed sweep still matches what the model produces today. The
  weekly `drift` job checks `pump_scheduling` only. For anything Gurobi-bound
  that remains a human obligation — a green badge over a two-year-old sweep is
  the most likely way this site goes quietly wrong.

Regenerate a sweep whenever the model behind it changes. Nothing will remind you.

## Gotchas

- **Do not create a top-level `site/` directory.** `.gitignore` carries mkdocs'
  `/site` rule, so it would be silently untracked. The site source lives in
  `tools/site/`. `build/` and `lib/` are ignored too.
- **Relative links only** in the landing page template. This is a project site
  served under `/flex-pse-examples/`, so a `/`-rooted href resolves against the
  user site and 404s. The build checks this.
- **`marimo export html-wasm` requires `uv`** on PATH, for every notebook,
  whether or not it has local imports — it shells out to `uv tool run ruff` to
  build the import graph.
- **Pyodide boots in ten to twenty seconds** and pulls ~30 MB of wheels from a
  CDN. The build passes `--execute` so marimo bakes the outputs in and the page
  paints immediately; `--no-execute` is the escape hatch if that ever breaks.
- **Never `import model` in a test.** See step 9 — `conftest.load_example_module`
  exists because every example's model module is called the same thing.
- **matplotlib is `agg`-only** in the browser. Return the `Figure` from a cell
  rather than calling `plt.show()`, and keep box-drawing characters out of chart
  labels — DejaVu Sans has no glyphs for them.
