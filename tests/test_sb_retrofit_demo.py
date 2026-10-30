"""The Santa Barbara retrofit demo: committed results hold together, and the model still solves.

Two halves:

* **Committed data** (needs only pandas/duckdb): the checks the demo's claims
  rest on, read straight off ``public/sb_retrofit_demo/``. Same water in both
  cases, a lower bill after, >50% peak load shed, no export, battery inside its
  window, and case A matching the operators' experiment.
* **Smoke solve** (needs flex-pse + HiGHS): both cases over two weekdays, which
  a free runner finishes in about a minute. It checks the model builds and the
  same invariants hold; it does not regenerate the committed month.
"""

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

DATA = Path(__file__).resolve().parents[1] / "examples" / "sb_retrofit_demo" / "public" / "sb_retrofit_demo"
PEAK = (16, 21)


def _df(name):
    return duckdb.sql(f"SELECT * FROM read_parquet('{DATA / name}')").df()


@pytest.fixture(scope="module")
def committed():
    if not (DATA / "summary.parquet").exists():
        pytest.skip("no committed sweep; run python tools/sweep.py examples/sb_retrofit_demo")
    summary = _df("summary.parquet").set_index("case")
    series = _df("series.parquet")
    ids = summary["sweep_id"].to_dict()
    frames = {
        case: series[series["sweep_id"] == sid].set_index("timestamp").sort_index()
        for case, sid in ids.items()
    }
    return summary, frames


def _checks(summary, frames):
    a, b = summary.loc["A_before"], summary.loc["B_after"]
    fa, fb = frames["A_before"], frames["B_after"]

    # 1. Same water.
    assert abs(b.water_m3 - a.water_m3) / a.water_m3 < 1e-3
    # 2. Lower bill.
    assert b.bill_usd < a.bill_usd
    # 3. >50% load shed in the peak window, against case A's average plant load.
    assert b.peak_grid_kw_mean < 0.5 * a.plant_kw_mean_all_hours, (
        f"peak-window mean {b.peak_grid_kw_mean:.0f} kW is not under half of "
        f"{a.plant_kw_mean_all_hours:.0f} kW"
    )
    # 4. No export.
    assert (fb["grid_kw"] >= -1e-6).all()
    assert (fa["grid_kw"] >= -1e-6).all()
    # 5. Battery inside its window.
    assert fb["battery_soc"].between(0.1 - 1e-6, 0.95 + 1e-6).all()
    # 6. Case A is the operators' schedule: 2 skids on weekday peaks, 3 otherwise.
    idx = fa.index
    shed = (idx.weekday < 5) & (idx.hour >= PEAK[0]) & (idx.hour < PEAK[1])
    assert (fa.loc[shed, "trains_online"] == 2).all()
    assert (fa.loc[~shed, "trains_online"] == 3).all()
    # 7. Case A's draw matches slide 3's "before" chart levels.
    assert 1300 <= fa.loc[shed, "grid_kw"].median() <= 1500
    assert 1900 <= fa.loc[~shed, "grid_kw"].median() <= 2100


def test_committed_results_hold_together(committed):
    summary, frames = committed
    _checks(summary, frames)


def test_committed_relaxation_gap_is_small(committed):
    summary, _ = committed
    assert summary.loc["B_after", "relaxation_gap"] < 0.01


def test_smoke_solve():
    pytest.importorskip("flexops", reason="needs the flex-pse environment")
    pytest.importorskip("highspy", reason="needs HiGHS")
    import pandas as pd

    from flex_pse_examples.examples.sb_retrofit_demo import model

    results = model.run_both(start="2026-07-06", end="2026-07-08", time_limit=240)
    summary = pd.DataFrame({k: v[2] for k, v in results.items()}).T
    frames = {k: v[1] for k, v in results.items()}
    a, b = summary.loc["A_before"], summary.loc["B_after"]
    assert abs(b.water_m3 - a.water_m3) / a.water_m3 < 1e-3
    assert b.bill_usd < a.bill_usd
    assert (frames["B_after"]["grid_kw"] >= -1e-6).all()
