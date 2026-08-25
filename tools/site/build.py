"""Build the flex-pse-examples site: validate every example, export, render.

Run it from the repository root::

    python tools/site/build.py --out _site
    python -m http.server --directory _site 8000

A WebAssembly export cannot be opened over ``file://`` -- it has to be served
over HTTP -- so the second command is not optional when checking the result.

What this does, in order:

1. **Discover** every ``examples/*/example.toml``. Discovery is by glob so that
   adding an example needs no central registry, and the inverse is checked too:
   an example directory with a notebook but no manifest fails the build rather
   than being silently skipped.
2. **Validate** all of them, collecting every error before exiting. Nothing is
   exported against input that failed, because a marimo export of a broken
   notebook succeeds and fails later, in a reader's browser.
3. **Export** each example's WebAssembly notebook into a shared ``notebooks/``
   directory. Shared, because marimo copies ~28 MB of runtime assets into every
   output directory it is given; one directory keeps the published site the same
   size whether there are two examples or twenty.
4. **Render** the landing page from ``templates/index.html.j2``.
5. **Verify** the output actually exists and that every internal link resolves.

Dependencies are ``marimo`` (which brings ``jinja2``) and nothing else.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _display(path: Path) -> str:
    """Return ``path`` relative to the repository when it is inside it.

    Error messages read better as ``examples/foo/public`` than as an absolute
    path, but ``--examples-root`` and ``--out`` may both point anywhere, and
    ``Path.relative_to`` raises rather than falling back.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


#: Where the exported notebooks land inside the output directory. Never the
#: output root: `export_assets` copies everything in marimo's `_static` except
#: its `index.html`, including marimo's own logo, manifest and CLAUDE.md, all of
#: which would sit next to -- and confuse -- the hand-written landing page.
NOTEBOOK_SUBDIR = "notebooks"

#: The repository the site links back to for source.
REPO_URL = "https://github.com/flex-pse/flex-pse-examples"

#: Manifest keys every example must set, as dotted paths.
REQUIRED_KEYS = (
    "schema_version",
    "name",
    "title",
    "order",
    "blurb",
    "tags",
    "notebooks.solver",
    "notebooks.wasm",
    "sweep.axis",
    "sweep.axis_label",
    "solve.solver",
    "solve.ci_runnable",
)

#: A card is a card. A blurb past this is a paragraph and breaks the grid.
MAX_BLURB_CHARS = 360
MAX_BLURB_SENTENCES = 2

#: Columns `tools/sweep.py` guarantees in every `summary.parquet`, and which the
#: landing page and the notebooks both read.
REQUIRED_SUMMARY_COLUMNS = ("sweep_id", "label", "wall_seconds")

#: Facts every `provenance.parquet` must carry.
REQUIRED_PROVENANCE_KEYS = ("generated_utc", "generator", "flexpse_version", "solver")

#: Top-level modules a WebAssembly notebook may import. Pyodide ships numpy,
#: pandas, scipy, matplotlib and duckdb; everything else here is stdlib that
#: works under Emscripten. Anything outside this list either cannot be installed in a browser
#: at all (pyomo's solvers, flexops, idaes) or is a sibling module that will not
#: be on the path in the export.
ALLOWED_IMPORTS = frozenset(
    {
        "marimo", "numpy", "pandas", "matplotlib", "mpl_toolkits", "scipy",
        "duckdb",
        "__future__", "abc", "base64", "collections", "dataclasses", "datetime",
        "decimal", "enum", "functools", "io", "itertools", "json", "math",
        "operator", "random", "re", "statistics", "string", "textwrap", "typing",
        "unicodedata", "warnings",
    }
)

#: Names that cannot work under Pyodide but are the natural thing to copy over
#: from a solver-level notebook. `open()` fails because `mo.notebook_location()`
#: is a URL in the browser, not a path; the other two are how `notebook.py`
#: reaches its sibling `model.py`, which is exactly what must not happen here.
FORBIDDEN_CALLS = {"open"}
FORBIDDEN_ATTRS = {"sys.path", "Path(__file__)"}

#: A view file past this is someone shipping a raw results frame. The browser
#: fetches these synchronously on the worker thread, so it is a stall, not just
#: a download. `tools/sweep.py` refuses to write one; this is the second gate.
#: Generous: a reduced view is ~10 KB of Parquet.
MAX_VIEW_BYTES = 2 * 1024 * 1024

#: An export below this is truncated. A partial export otherwise passes silently.
MIN_EXPORT_BYTES = 10 * 1024


@dataclass
class Example:
    """One example, as the site sees it."""

    dir: Path
    manifest: dict
    name: str
    title: str
    order: int
    blurb: str
    tags: list[str]
    wasm_notebook: str
    solver_notebook: str
    provenance: dict = field(default_factory=dict)
    summary_rows: int = 0

    @property
    def page(self) -> str:
        """The landing page's relative link to this example's notebook."""
        return f"{NOTEBOOK_SUBDIR}/{self.name}.html"

    @property
    def source_url(self) -> str:
        """The GitHub URL for this example's directory."""
        return f"{REPO_URL}/tree/main/examples/{self.name}"

    @property
    def solver_url(self) -> str:
        """The GitHub URL for this example's solver-level notebook."""
        return f"{REPO_URL}/blob/main/examples/{self.name}/{self.solver_notebook}"

    @property
    def data_dir(self) -> Path:
        """Where `tools/sweep.py` writes this example's committed CSVs."""
        return self.dir / "public" / self.name


def _dig(mapping: dict, dotted: str):
    """Return ``mapping`` at a dotted path, or None if any level is missing."""
    node = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# --------------------------------------------------------------------------
# Discovery


def discover(examples_root: Path, errors: list[str]) -> list[Example]:
    """Return every example with a manifest, and complain about those without.

    Args:
        examples_root: The ``examples/`` directory.
        errors: Accumulator appended to in place.

    Returns:
        The examples that parsed, sorted by ``(order, name)``.
    """
    found: list[Example] = []

    for directory in sorted(p for p in examples_root.iterdir() if p.is_dir()):
        manifest_path = directory / "example.toml"
        if not manifest_path.exists():
            # Glob discovery would skip this silently, and the example would
            # vanish from the site with nothing to show for it.
            if (directory / "notebook.py").exists():
                errors.append(
                    f"{_display(directory)}: has a notebook.py but no "
                    f"example.toml, so it would be silently left off the site. "
                    f"Add a manifest, or delete the directory. See CONTRIBUTING.md."
                )
            continue

        try:
            manifest = tomllib.loads(manifest_path.read_text())
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"{_display(manifest_path)}: invalid TOML -- {exc}")
            continue

        missing = [k for k in REQUIRED_KEYS if _dig(manifest, k) is None]
        if missing:
            errors.append(
                f"{_display(manifest_path)}: missing required "
                f"key(s) {', '.join(missing)}"
            )
            continue

        found.append(
            Example(
                dir=directory,
                manifest=manifest,
                name=manifest["name"],
                title=manifest["title"],
                order=int(manifest["order"]),
                blurb=" ".join(manifest["blurb"].split()),
                tags=list(manifest["tags"]),
                wasm_notebook=_dig(manifest, "notebooks.wasm"),
                solver_notebook=_dig(manifest, "notebooks.solver"),
            )
        )

    seen_order: dict[int, str] = {}
    for example in found:
        if example.name != example.dir.name:
            errors.append(
                f"{_display(example.dir)}: manifest name "
                f"{example.name!r} does not match the directory name"
            )
        if example.order in seen_order:
            errors.append(
                f"{example.name}: order {example.order} collides with "
                f"{seen_order[example.order]}"
            )
        seen_order[example.order] = example.name

    return sorted(found, key=lambda e: (e.order, e.name))


# --------------------------------------------------------------------------
# Validation


def validate_manifest(example: Example, errors: list[str]) -> None:
    """Check the presentation fields the landing page depends on."""
    sentences = len(re.findall(r"[.!?](?:\s|$)", example.blurb))
    if not 1 <= sentences <= MAX_BLURB_SENTENCES:
        errors.append(
            f"{example.name}: blurb is {sentences} sentence(s); a card takes "
            f"1-{MAX_BLURB_SENTENCES}"
        )
    if len(example.blurb) > MAX_BLURB_CHARS:
        errors.append(
            f"{example.name}: blurb is {len(example.blurb)} chars, over the "
            f"{MAX_BLURB_CHARS} a card fits"
        )
    if not example.tags:
        errors.append(f"{example.name}: tags is empty")

    for key in ("notebooks.solver", "notebooks.wasm", "notebooks.sweep"):
        filename = _dig(example.manifest, key)
        if filename and not (example.dir / filename).exists():
            errors.append(f"{example.name}: {key} points at missing {filename}")

    if _dig(example.manifest, "solve.ci_runnable"):
        test = REPO_ROOT / "tests" / f"test_{example.name}.py"
        if not test.exists():
            errors.append(
                f"{example.name}: solve.ci_runnable is true but "
                f"tests/test_{example.name}.py does not exist"
            )


def validate_data(example: Example, errors: list[str]) -> None:
    """Check the committed sweep against the contract the notebooks read."""
    import duckdb

    data = example.data_dir
    rel = _display(data)

    summary_path = data / "summary.parquet"
    provenance_path = data / "provenance.parquet"

    if not summary_path.exists():
        errors.append(
            f"{example.name}: {rel}/summary.parquet is missing. Generate it with "
            f"`python tools/sweep.py examples/{example.name}` and commit the result."
        )
        return
    if not provenance_path.exists():
        errors.append(f"{example.name}: {rel}/provenance.parquet is missing")
        return

    con = duckdb.connect()
    try:
        try:
            summary = con.sql(f"SELECT * FROM read_parquet('{summary_path}')").df()
            provenance = con.sql(
                f"SELECT key, value FROM read_parquet('{provenance_path}')"
            ).df().set_index("key")["value"]
        except Exception as exc:
            errors.append(f"{example.name}: could not read the sweep Parquet -- {exc}")
            return

        if summary.empty:
            errors.append(f"{example.name}: {rel}/summary.parquet has no rows")
            return

        axis = _dig(example.manifest, "sweep.axis")
        for column in (*REQUIRED_SUMMARY_COLUMNS, axis):
            if column not in summary.columns:
                errors.append(
                    f"{example.name}: summary.parquet has no {column!r} column"
                )

        ids = summary.get("sweep_id")
        if ids is not None:
            if ids.duplicated().any():
                errors.append(
                    f"{example.name}: summary.parquet has duplicate sweep_id values"
                )
            bad = [i for i in ids if not re.fullmatch(r"s\d{2,}", str(i))]
            if bad:
                errors.append(f"{example.name}: malformed sweep_id(s) {bad}")

        for key in REQUIRED_PROVENANCE_KEYS:
            if key not in provenance.index:
                errors.append(f"{example.name}: provenance.parquet has no {key!r} row")

        if str(provenance.get("smoke", "no")) == "yes":
            errors.append(
                f"{example.name}: the committed sweep is marked as a reduced (smoke) "
                f"run. Regenerate it without --smoke before publishing."
            )

        # Each view must cover exactly the points summary.parquet lists. A missing
        # sweep_id is a point whose data never got written; an extra one is a
        # leftover from a longer sweep, and would show the reader a case the
        # selector cannot reach.
        expected = set(map(str, ids)) if ids is not None else set()
        for view in example.manifest.get("sweep", {}).get("views", []):
            path = data / f"{view['name']}.parquet"
            if not path.exists():
                errors.append(f"{example.name}: {rel}/{view['name']}.parquet is missing")
                continue
            try:
                present = {
                    str(row[0])
                    for row in con.sql(
                        f"SELECT DISTINCT sweep_id FROM read_parquet('{path}')"
                    ).fetchall()
                }
            except Exception as exc:
                errors.append(
                    f"{example.name}: could not read {view['name']}.parquet -- {exc}"
                )
                continue

            for missing in sorted(expected - present):
                errors.append(
                    f"{example.name}: {view['name']}.parquet has no rows for "
                    f"{missing} -- that point's data was never written"
                )
            for orphan in sorted(present - expected):
                errors.append(
                    f"{example.name}: {view['name']}.parquet has rows for {orphan}, "
                    f"which has no row in summary.parquet -- a leftover from a "
                    f"longer sweep?"
                )

            size = path.stat().st_size
            if size > MAX_VIEW_BYTES:
                errors.append(
                    f"{example.name}: {view['name']}.parquet is {size / 1e6:.1f} MB, "
                    f"over the {MAX_VIEW_BYTES / 1e6:.0f} MB view limit"
                )
    finally:
        con.close()

    example.provenance = {k: str(v) for k, v in provenance.items()}
    example.summary_rows = len(summary)


def validate_wasm(example: Example, errors: list[str]) -> None:
    """Reject anything in the WebAssembly notebook a browser cannot run.

    This is the check that matters most. Every other failure in this script
    shows up here, at build time, in front of someone who can fix it. A bad
    import in a WebAssembly notebook exports cleanly, publishes cleanly, and
    fails in a reader's browser with a traceback they cannot act on.
    """
    path = example.dir / example.wasm_notebook
    where = f"{example.name}/{example.wasm_notebook}"
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{where}:{exc.lineno}: syntax error -- {exc.msg}")
        return

    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                errors.append(
                    f"{where}:{node.lineno}: relative import -- a WebAssembly "
                    f"notebook has no sibling modules in the export"
                )
                continue
            modules = [(node.module or "").split(".")[0]]
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                errors.append(
                    f"{where}:{node.lineno}: `{func.id}()` cannot work under "
                    f"Pyodide -- mo.notebook_location() is a URL in the browser, "
                    f"not a path. Query it with duckdb.sql(read_parquet(...)) instead."
                )
            if isinstance(func, ast.Attribute) and func.attr == "insert":
                target = ast.unparse(func.value)
                if target.endswith("sys.path") or target == "path":
                    errors.append(
                        f"{where}:{node.lineno}: `{target}.insert(...)` -- a "
                        f"WebAssembly notebook must not reach for sibling modules"
                    )
            if (
                isinstance(func, ast.Name)
                and func.id == "Path"
                and any(
                    isinstance(a, ast.Name) and a.id == "__file__" for a in node.args
                )
            ):
                errors.append(
                    f"{where}:{node.lineno}: `Path(__file__)` -- there is no "
                    f"filesystem in the browser; use mo.notebook_location()"
                )
            continue
        else:
            continue

        for module in modules:
            if module and module not in ALLOWED_IMPORTS:
                errors.append(
                    f"{where}:{node.lineno}: `{module}` is not available in "
                    f"Pyodide. A WebAssembly notebook may import only: "
                    f"{', '.join(sorted(ALLOWED_IMPORTS))}."
                )


# --------------------------------------------------------------------------
# Export


def export(example: Example, out_dir: Path, *, execute: bool) -> None:
    """Export one example's WebAssembly notebook into the shared notebook dir.

    Raises:
        subprocess.CalledProcessError: If marimo's export fails.
    """
    target = out_dir / NOTEBOOK_SUBDIR / f"{example.name}.html"
    target.parent.mkdir(parents=True, exist_ok=True)

    wasm_cfg = example.manifest.get("wasm", {})
    cmd = [
        sys.executable, "-m", "marimo", "export", "html-wasm",
        str(example.dir / example.wasm_notebook),
        "-o", str(target),
        "--mode", wasm_cfg.get("mode", "run"),
        "--show-code" if wasm_cfg.get("show_code", False) else "--no-show-code",
        "--force",
    ]
    if execute and wasm_cfg.get("execute", True):
        # Runs the notebook and embeds its outputs, so the page paints charts at
        # once instead of showing a spinner for the ten-odd seconds Pyodide needs
        # to boot and pull its wheels. Re-invokes marimo under `uv run
        # --isolated`, which is why the notebook carries a PEP 723 header.
        cmd.append("--execute")
    else:
        # Without this the PEP 723 header makes marimo *prompt* to re-run in a
        # sandbox, which hangs a local build on a terminal.
        cmd += ["--no-execute", "--no-sandbox"]

    print(f"  export {example.name} -> {target.relative_to(out_dir)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


# --------------------------------------------------------------------------
# Index


def render_index(examples: list[Example], out_dir: Path) -> None:
    """Render the landing page from the jinja2 template."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    css = (STATIC_DIR / "style.css").read_text()
    html = env.get_template("index.html.j2").render(
        examples=examples, repo_url=REPO_URL, inline_css=css
    )
    (out_dir / "index.html").write_text(html)


def copy_static(out_dir: Path) -> None:
    """Copy static assets and write the Jekyll opt-out.

    marimo touches a `.nojekyll` in the directory it exports into, but not in the
    site root. Without one there, GitHub Pages runs Jekyll over the site and
    drops every underscore-prefixed path.
    """
    (out_dir / ".nojekyll").touch()
    if STATIC_DIR.is_dir():
        for item in STATIC_DIR.iterdir():
            if item.name == "style.css":
                continue  # inlined into the page instead
            dest = out_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)


# --------------------------------------------------------------------------
# Post-conditions


class _LinkFinder(HTMLParser):
    """Collect every href/src in a document."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.links.append(value)


def verify(examples: list[Example], out_dir: Path, errors: list[str]) -> None:
    """Check the built site is actually a site."""
    index = out_dir / "index.html"
    if not index.exists():
        errors.append("index.html was not written")
        return
    if not (out_dir / ".nojekyll").exists():
        errors.append(".nojekyll is missing; GitHub Pages would run Jekyll")

    notebooks = out_dir / NOTEBOOK_SUBDIR
    if not (notebooks / "assets").is_dir():
        errors.append(f"{NOTEBOOK_SUBDIR}/assets/ is missing -- the export ran but "
                      f"copied no runtime")
    elif not list((notebooks / "assets").glob("index-*.js")):
        errors.append(f"{NOTEBOOK_SUBDIR}/assets/ has no index-*.js bundle")

    for example in examples:
        page = out_dir / example.page
        if not page.exists():
            errors.append(f"{example.name}: {example.page} was not exported")
        elif page.stat().st_size < MIN_EXPORT_BYTES:
            errors.append(
                f"{example.name}: {example.page} is only "
                f"{page.stat().st_size} bytes -- a truncated export"
            )
        data = notebooks / "public" / example.name / "summary.parquet"
        if not data.exists():
            errors.append(
                f"{example.name}: {data.relative_to(out_dir)} missing -- the "
                f"export did not copy public/, so the page will have no data"
            )

    # Relative links only: this is a project site served under a path prefix, so
    # a `/`-rooted href resolves against the user site and 404s. Such a path also
    # will not resolve under out_dir, which is what catches it here.
    finder = _LinkFinder()
    finder.feed(index.read_text())
    for link in finder.links:
        if link.startswith(("http://", "https://", "mailto:", "#", "data:")):
            continue
        if link.startswith("/"):
            errors.append(
                f"index.html: `{link}` is root-relative; this is a project site "
                f"served under /flex-pse-examples/, so it must be relative"
            )
            continue
        target = (out_dir / urlparse(link).path).resolve()
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            errors.append(f"index.html: broken internal link -> {link}")


def report_size(out_dir: Path) -> None:
    """Print what was built and how big it is."""
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    files = sum(1 for p in out_dir.rglob("*") if p.is_file())
    print(f"\nBuilt {_display(out_dir)}: {files} files, {total / 1e6:.1f} MB")


# --------------------------------------------------------------------------


def build(out_dir: Path, examples_root: Path, *, execute: bool) -> int:
    """Build the site. Returns a process exit code."""
    errors: list[str] = []

    examples = discover(examples_root, errors)
    for example in examples:
        validate_manifest(example, errors)
        validate_data(example, errors)
        validate_wasm(example, errors)

    if not examples and not errors:
        errors.append(f"no examples found under {_display(examples_root)}")

    # Never export against input that failed validation: marimo will happily
    # export a notebook that cannot run, and the failure resurfaces in a browser.
    if errors:
        return _fail(errors)

    print(f"Building {len(examples)} example(s) into {_display(out_dir)}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for example in examples:
        export(example, out_dir, execute=execute)

    render_index(examples, out_dir)
    copy_static(out_dir)
    verify(examples, out_dir, errors)

    if errors:
        return _fail(errors)

    report_size(out_dir)
    print(
        f"Serve it with: python -m http.server --directory "
        f"{_display(out_dir)} 8000"
    )
    return 0


def _fail(errors: list[str]) -> int:
    """Print every collected error and return a failing exit code."""
    print(f"\n{len(errors)} problem(s):\n", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print("\nSee CONTRIBUTING.md for what each example must provide.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "_site")
    parser.add_argument(
        "--examples-root", type=Path, default=REPO_ROOT / "examples"
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help=(
            "skip marimo's pre-execution of the notebooks. Pages then show a "
            "loading spinner until Pyodide boots, but the build stops depending "
            "on network resolution of a pyodide lockfile."
        ),
    )
    args = parser.parse_args(argv)
    return build(
        args.out.resolve(), args.examples_root.resolve(), execute=not args.no_execute
    )


if __name__ == "__main__":
    raise SystemExit(main())
