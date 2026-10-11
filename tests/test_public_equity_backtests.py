"""Information-boundary and accounting tests for the disclosed public study."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.public_equity_backtests import (
    build_research_windows,
    fold_indices,
    local_stock_ledgers,
    monthly_prices,
    paired_month_bootstrap,
    save,
)


def prices():
    index = pd.period_range("1998-01", "2024-12", freq="M")
    return pd.Series(np.exp(np.arange(len(index)) * 0.003) * 100, index=index)


def test_public_inputs_ignore_blind_month_and_all_future_prices():
    original = prices()
    data, months, _ = build_research_windows({"test": original}, 12)
    row = np.flatnonzero(months == "2008-01")[0]
    changed = original.copy()
    changed.loc["2007-11":] *= np.linspace(1.1, 3, len(changed.loc["2007-11":]))
    other, other_months, _ = build_research_windows({"test": changed}, 12)
    other_row = np.flatnonzero(other_months == "2008-01")[0]
    np.testing.assert_array_equal(data.x[row], other.x[other_row])
    assert data.y[row] != other.y[other_row]
    assert data.time[row] == pd.Timestamp("2007-12-01", tz="UTC").value
    assert data.label_time[row] == pd.Timestamp("2008-02-03", tz="UTC").value
    assert data.entry_time is None  # No fabricated execution timestamp.


def test_fold_purges_maturing_labels_and_reports_missing_stock():
    whole = prices()
    missing = whole.copy()
    missing.loc["2008-06"] = np.nan
    data, months, _ = build_research_windows({"complete": whole, "missing": missing}, 12)
    cfg = {
        "protected_start": "2025-01-01",
        "validation_months": 12,
        "minimum_train_samples_per_stock": 36,
        "minimum_validation_samples_per_stock": 6,
    }
    ix, receipt = fold_indices(
        data, months, {"name": "crisis", "return_start": "2007-12", "return_end": "2009-06"}, cfg
    )
    assert receipt["included_stocks"] == ["complete"]
    assert "missing" in receipt["excluded_stocks"]
    assert data.label_time[ix[0]].max() < receipt["validation_start_ns"]
    assert data.label_time[ix[1]].max() < data.time[ix[2]].min()
    assert len(ix[2]) == 19


def test_monthly_adapter_does_not_fill_last_missing_close_or_read_protected_price():
    stamps = ["2024-11-29T14:30Z", "2024-12-30T14:30Z", "2024-12-31T14:30Z", "2025-01-02T14:30Z"]
    chart = {
        "timestamp": [int(pd.Timestamp(t).timestamp()) for t in stamps],
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "indicators": {"adjclose": [{"adjclose": [100, 110, None, -1]}]},
    }
    cfg = {"source_start": "2024-01-01", "source_end_exclusive": "2025-01-01"}
    result = monthly_prices(chart, cfg)
    assert list(result.index.astype(str)) == ["2024-11", "2024-12"]
    assert np.isnan(result.iloc[-1])
    chart["indicators"]["adjclose"][0]["adjclose"][0] = 0
    with pytest.raises(ValueError, match="terminal losses"):
        monthly_prices(chart, cfg)


def test_local_round_trip_costs_and_initial_wealth_drawdown():
    frame = pd.DataFrame(
        {"asset": ["test", "test"], "time": [1, 2], "prediction": [0.1, -0.1], "actual": np.log([0.5, 1.2])}
    )
    ledger = local_stock_ledgers(frame, 10)["test"]
    expected = 0.5 * 0.999 / 1.001 - 1
    assert ledger["net_local_return"] == pytest.approx(expected)
    assert ledger["max_drawdown"] == pytest.approx(expected)
    assert ledger["long_months"] == 1


def test_pairing_does_not_drop_missing_outcomes():
    frame = pd.DataFrame(
        {"time": range(12), "asset": ["test"] * 12, "actual": np.zeros(12), "nll": np.zeros(12)}
    )
    control = frame.copy()
    control["nll"] = 0.5
    settings = {"block_months": 3, "repetitions": 100, "seed": 1, "confidence": 0.95}
    result = paired_month_bootstrap(frame, control, settings)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci"] == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="Incomplete prediction pairing"):
        paired_month_bootstrap(frame, control.iloc[:-1], settings)


def test_receipts_refuse_overwrite_and_accept_numpy_scalars(tmp_path):
    path = tmp_path / "receipt.json"
    save(path, {"value": np.int64(5)})
    assert json.loads(path.read_text()) == {"value": 5}
    with pytest.raises(FileExistsError):
        save(path, {"value": 6})
