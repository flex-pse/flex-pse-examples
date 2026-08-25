"""Every WebAssembly notebook must be runnable in a browser.

This is the check with the worst failure mode if it is missing. A notebook that
imports pyomo exports cleanly, publishes cleanly, and then throws in a reader's
browser with a traceback they can do nothing about -- so it is worth asserting
in two places, here and in the site build.
"""

import ast

import pytest

from conftest import example_dirs
from tools.site import build


def _wasm_notebook(example_dir):
    """Return the path to an example's WebAssembly notebook."""
    import tomllib

    manifest = tomllib.loads((example_dir / "example.toml").read_text())
    return example_dir / manifest["notebooks"]["wasm"]


@pytest.mark.parametrize("example_dir", example_dirs(), ids=lambda p: p.name)
def test_imports_only_what_pyodide_has(example_dir, examples_root):
    """No pyomo, no flexops, no sibling modules, no open()."""
    errors: list[str] = []
    discovery: list[str] = []
    examples = {e.name: e for e in build.discover(examples_root, discovery)}
    build.validate_wasm(examples[example_dir.name], errors)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("example_dir", example_dirs(), ids=lambda p: p.name)
def test_declares_its_dependencies_inline(example_dir):
    """A PEP 723 header is what pins the browser's packages.

    marimo's ``--execute`` export resolves this header against Pyodide's package
    set; without it the pre-executed build has no pandas and the page falls back
    to a cold boot.
    """
    text = _wasm_notebook(example_dir).read_text()
    assert text.lstrip().startswith("# /// script"), (
        "the WebAssembly notebook needs a PEP 723 header; see CONTRIBUTING.md"
    )
    assert "marimo" in text.split("# ///")[1]


@pytest.mark.parametrize("example_dir", example_dirs(), ids=lambda p: p.name)
def test_reads_data_through_notebook_location(example_dir):
    """Data comes from ``mo.notebook_location()``, the one path that works in both.

    A notebook that hardcodes a relative path works locally and 404s in the
    browser, which is exactly the class of bug this whole file exists to stop.
    """
    text = _wasm_notebook(example_dir).read_text()
    assert "notebook_location()" in text


@pytest.mark.parametrize("example_dir", example_dirs(), ids=lambda p: p.name)
def test_says_the_results_are_precomputed(example_dir):
    """The page must not imply it is solving anything.

    Every visitor arrives assuming an interactive notebook is running the model.
    Saying otherwise, on the page, is a correctness requirement rather than a
    stylistic one.
    """
    text = _wasm_notebook(example_dir).read_text()
    assert "precomputed" in text.lower(), (
        "the WebAssembly notebook must carry the precomputed-results callout"
    )
    assert "kind=\"info\"" in text or "kind='info'" in text


@pytest.mark.parametrize("example_dir", example_dirs(), ids=lambda p: p.name)
def test_is_a_marimo_app(example_dir):
    """It parses, and it is a marimo notebook rather than a plain script."""
    path = _wasm_notebook(example_dir)
    tree = ast.parse(path.read_text(), filename=str(path))
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "app" in assigned, f"{path.name} does not define a marimo App"
