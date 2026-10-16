from pathlib import Path

import numpy as np
import pytest

from successors.frem_portfolio_v1.engine import Forecast
from successors.frem_s6_real.prepare import stamp, write
from successors.frem_s8.policy import continuous_target, holding_policy, objective
from successors.frem_s8.prepare import ASSETS, build_segment, market_open
from successors.frem_s8.router import SoftGate
from successors.frem_s8.run import validate_scope


def test_soft_gate_exact_delayed_update_and_common_loss_invariance():
    a, b = SoftGate(), SoftGate()
    for gate, loss in [(a, [0.0, -1.0]), (b, [100.0, 99.0])]:
        assert np.array_equal(gate.weights(1), [0.5, 0.5])
        gate.observe(1, 3, loss)
        assert np.array_equal(gate.weights(3), [0.5, 0.5])
        assert gate.weights(4)[1] > 0.5
    assert np.allclose(a.probabilities, b.probabilities)
    expected = (1 - 1 / 52) * np.array([1, np.e]) / (1 + np.e) + 1 / 104
    assert np.allclose(a.probabilities, expected)


def test_soft_gate_does_not_permanently_discard_loser():
    g = SoftGate()
    for t in range(1, 80):
        w = g.weights(t)
        assert (w >= 1 / 104).all() and np.isclose(w.sum(), 1)
        g.observe(t, t + 0.1, [0, 100] if t < 40 else [100, 0])
    assert g.weights(80)[1] > 0.9


def test_soft_gate_duplicate_and_backward_rejected():
    g = SoftGate()
    g.weights(1)
    g.observe(1, 2, [0, 0])
    with pytest.raises(ValueError):
        g.observe(1, 3, [0, 0])
    with pytest.raises(ValueError):
        g.weights(1)
    with pytest.raises(ValueError):
        SoftGate(0)


def test_concave_solution_beats_dense_feasible_grid():
    h = np.array([0.2, 0.3])
    mu, v = np.array([0.006, 0.003]), np.array([0.002, 0.001])
    w = continuous_target(h, mu, v, horizon=4, fee=0.001, budget=0.5, cap=0.4)
    best = objective(w, h, mu, v, 4, 0.001)
    for a in np.linspace(0, 0.4, 81):
        for b in np.linspace(0, min(0.4, 0.5 - a), 81):
            assert objective([a, b], h, mu, v, 4, 0.001) <= best + 1e-12
    assert w.sum() <= 0.5 + 1e-12 and (w <= 0.4).all()


def test_partial_entry_and_horizon_cost_tradeoff():
    one = continuous_target([0], [0.001], [0.002], horizon=1, fee=0.001)
    four = continuous_target([0], [0.001], [0.002], horizon=4, fee=0.001)
    assert one[0] == 0
    assert 0 < four[0] < 0.1
    assert np.isclose(four[0], 0.03125)
    assert continuous_target([0], [-0.01], [0.002], horizon=4)[0] == 0


def test_no_cost_analytic_and_permutation_equivariance():
    mu, v = np.array([0.001, 0.002]), np.array([0.001, 0.002])
    w = continuous_target([0, 0], mu, v, fee=0)
    assert np.allclose(w, mu / (8 * v))
    assert np.allclose(w[::-1], continuous_target([0, 0], mu[::-1], v[::-1], fee=0))


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 1.0])
def test_bad_cost_rejected(bad):
    with pytest.raises(ValueError):
        continuous_target([0], [0.01], [0.001], fee=bad)


def periods(n=9):
    return [
        {
            "time": t,
            "end": t + 1,
            "currency": "USD",
            "forecasts": [Forecast("A", 0.003, 0.001), Forecast("B", 0.002, 0.001)],
            "returns": {"A": 0.01, "B": -0.005},
        }
        for t in range(n)
    ]


def test_committed_horizon_and_self_financing_all_periods():
    p = periods()
    ledger = holding_policy(p, 10)
    for i, row in enumerate(ledger["periods"]):
        assert (row["decision"] is not None) == (i % 4 == 0)
        if i % 4:
            assert row["notional"] == row["fee"] == 0
        expected = (
            row["nav_start"] - row["fee"] + sum(row["contributions"].values()) - row.get("liquidation_fee", 0)
        )
        assert row["nav_end"] == pytest.approx(expected)
    assert ledger["periods"][-1]["decision"]["horizon"] == 1
    assert ledger["terminal_wealth"] == pytest.approx(
        100000 * np.prod([r["net_factor"] for r in ledger["periods"]])
    )


def test_outcome_cannot_change_own_or_committed_trade():
    p, changed = periods(), periods()
    changed[0]["returns"] = {"A": -0.4, "B": 0.5}
    a, b = holding_policy(p, 10), holding_policy(changed, 10)
    assert a["periods"][0]["decision"] == b["periods"][0]["decision"]
    assert [r["notional"] for r in a["periods"][1:4]] == [0, 0, 0]
    assert [r["notional"] for r in b["periods"][1:4]] == [0, 0, 0]


def test_ledger_calendar_gap_rejected():
    p = periods()
    p[1]["time"] += 0.1
    with pytest.raises(ValueError, match="continuous"):
        holding_policy(p, 10)


def test_scope_guard_runs_before_bad_outcomes():
    payload = {"partitions": {"evaluation": {"origin_times": [stamp("2019-03-01")], "y": ["SECRET"]}}}
    with pytest.raises(ValueError, match="2019"):
        validate_scope(payload)


def test_ny_open_dst_and_adapter_no_gap_bridging(tmp_path):
    from datetime import date, timedelta

    assert market_open("2020-03-09") - market_open("2020-03-02") == 7 * 86400 - 3600
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(60)]
    days = [d for d in days if d.weekday() < 5]
    for asset in ASSETS:
        write(
            tmp_path / f"{asset}.json",
            {
                "rows": [
                    {"timestamp": stamp(str(d)) + 14 * 3600, "open": 100 + i, "close": 100.5 + i}
                    for i, d in enumerate(days)
                ],
                "dividends": {},
            },
        )
    parts, exclusions = build_segment(tmp_path, fresh=True)
    p = parts["evaluation"]
    dates = np.unique(p["origin_times"])
    assert exclusions and len(dates) >= 3
    assert all(t >= stamp("2020-01-01") for row in p["window_times"] for t in row)
    assert p["maturity_times"][0] == dates[1]
    assert not parts["train"]["y"] and not parts["calibration"]["y"]


def test_source_timestamp_guard_rejects_2019_without_reading_quotes(tmp_path):
    for asset in ASSETS:
        write(
            tmp_path / f"{asset}.json",
            {
                "rows": [{"timestamp": stamp("2019-12-31"), "open": "SECRET", "close": "SECRET"}],
                "dividends": {},
            },
        )
    with pytest.raises(ValueError, match="2019"):
        build_segment(tmp_path, fresh=True)


def test_write_preserves_previous_attempt(tmp_path):
    path = tmp_path / "receipt.json"
    write(path, {"failed": True})
    with pytest.raises(FileExistsError):
        write(path, {"failed": False})
    assert "true" in Path(path).read_text()


def test_old_dividend_before_every_possible_holding_is_excluded_by_date(tmp_path):
    from datetime import date, timedelta

    days = [date(2016, 1, 1) + timedelta(days=i) for i in range(60)]
    days = [d for d in days if d.weekday() < 5]
    for asset in ASSETS:
        write(
            tmp_path / f"{asset}.json",
            {
                "rows": [
                    {"timestamp": stamp(str(d)) + 14 * 3600, "open": 100 + i, "close": 100.5 + i}
                    for i, d in enumerate(days)
                ],
                "dividends": {"prior": {"date": stamp("2004-12-31"), "amount": "DO NOT READ"}},
            },
        )
    parts, _ = build_segment(tmp_path, fresh=False)
    assert len(parts["train"]["y"]) > 0
