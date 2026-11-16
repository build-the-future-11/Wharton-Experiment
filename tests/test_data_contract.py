"""Prospective data-integrity tests, now covered by runtime verification."""

import numpy as np
import pandas as pd
import pytest

from frem.data import (
    DataValidationError,
    TrainOnlyRobustScaler,
    WindowData,
    _local_wealth,
    add_causal_peer_features,
    align_asset_universe,
    build_inference_windows,
    build_windows,
    data_fingerprint,
    load_panel,
    point_in_time_macro,
    validate_panel,
)
from frem.splits import chronological_split, walk_forward_splits


def panel(n=16, asset="TEST"):
    times = pd.date_range("2020-01-01T21:00:00Z", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "available_at": times + np.timedelta64(1, "m"),
            "asset": asset,
            "close": 100.0 + np.arange(n),
            "total_return_index": 100.0 + np.arange(n),
            "currency": "USD",
            "asset_type": "etf",
            "price_kind": "total_return_index",
            "eligible": True,
            "executable": True,
            "eligibility_known_at": pd.Timestamp("2019-01-01T00:00:00Z"),
            "source": "constructed integrity-test fixture, never empirical data",
            "universe_source": "fixed fixture manifest",
            "corporate_action_source": "fixture identity total-return index",
        }
    )
    frame.attrs.update(
        {
            "universe_status": "fixed_ex_ante",
            "universe_frozen_at": "2019-01-01T00:00:00Z",
            "availability_policy": "fixture event plus one minute",
            "total_return_methodology": "fixture TRI equals positive price",
            "base_currency": "USD",
        }
    )
    return frame


def window_fixture(n_dates=100, assets=2, horizon=5):
    dates = pd.date_range("2020-01-01T21:01:00Z", periods=n_dates, freq="D").asi8
    times = np.repeat(dates, assets)
    ids = np.tile(np.arange(assets), n_dates)
    n = len(times)
    day = 86_400_000_000_000
    return WindowData(
        np.zeros((n, 3, 2)),
        np.zeros((n, horizon, 2)),
        np.zeros(n),
        times,
        times + (horizon + 1) * day,
        ids,
        ["return", "return__missing"],
        [f"ASSET_{i}" for i in range(assets)],
        times + day,
        times + (horizon + 1) * day,
    )


def test_execution_label_starts_after_decision_and_has_h_intervals():
    frame = panel()
    data = build_windows(frame, context=2, horizon=2)
    # Decision row 1 -> entry row 2 -> exit row 4; never row 1 -> row 3.
    assert data.y[0] == pytest.approx(np.log(104.0 / 102.0))
    assert data.entry_time[0] == frame.timestamp.iloc[2].value
    assert data.exit_time[0] == frame.timestamp.iloc[4].value
    assert data.label_time[0] == frame.available_at.iloc[4].value
    assert data.entry_time[0] > data.time[0]


@pytest.mark.parametrize("entry,exit", [(None, "existing"), ("existing", None)])
def test_window_data_requires_complete_execution_interval(entry, exit):
    data = window_fixture(n_dates=3, assets=1, horizon=1)
    with pytest.raises(DataValidationError, match="entry_time and exit_time"):
        WindowData(
            data.x,
            data.future_x,
            data.y,
            data.time,
            data.label_time,
            data.asset_id,
            data.feature_names,
            data.asset_names,
            data.entry_time if entry == "existing" else None,
            data.exit_time if exit == "existing" else None,
        )


def test_delayed_publication_changes_next_executable_entry():
    frame = panel()
    frame["available_at"] += np.timedelta64(2, "D")
    data = build_windows(frame, context=2, horizon=1)
    # Decision row1 arrives after row3's close; earliest later close is row4.
    assert data.entry_time[0] == frame.timestamp.iloc[4].value
    assert data.y[0] == pytest.approx(np.log(105.0 / 104.0))


def test_macro_vintages_never_backfill_or_replace_newer_period_with_old_revision():
    macro = pd.DataFrame(
        {
            "series": ["M", "M", "M"],
            "observation_time": ["2020-01-01T00:00:00Z", "2020-02-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "available_at": ["2020-02-01T00:00:00Z", "2020-03-01T00:00:00Z", "2020-04-01T00:00:00Z"],
            "value": [10.0, 20.0, 15.0],
        }
    )
    joined = point_in_time_macro(
        macro,
        ["2020-01-15T00:00:00Z", "2020-02-02T00:00:00Z", "2020-03-02T00:00:00Z", "2020-04-02T00:00:00Z"],
    )
    assert np.isnan(joined.macro_M.iloc[0])
    assert joined.macro_M.iloc[1:].tolist() == [10.0, 20.0, 20.0]


def test_macro_declared_schema_survives_development_filter():
    macro = pd.DataFrame(
        {
            "series": ["early", "late"],
            "observation_time": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "available_at": ["2020-02-01T00:00:00Z", "2021-02-01T00:00:00Z"],
            "value": [1.0, 2.0],
        }
    )
    macro.attrs["series_universe"] = ["early", "late"]
    development = macro.iloc[:1].copy()
    decisions = ["2020-03-01T00:00:00Z"]
    full = point_in_time_macro(macro, decisions)
    filtered = point_in_time_macro(development, decisions)
    assert list(full) == list(filtered)
    np.testing.assert_allclose(full.to_numpy(), filtered.to_numpy(), equal_nan=True)


def test_fx_total_return_is_base_currency_and_waits_for_publication():
    frame = panel()
    frame["currency"] = "EUR"
    frame["base_fx"] = 1.1 + np.arange(len(frame)) * 0.01
    frame["fx_available_at"] = frame.available_at + np.timedelta64(1, "m")
    frame["fx_source"] = "fixture"
    data = build_windows(frame, context=2, horizon=1)
    assert data.y[0] == pytest.approx(np.log((103.0 * 1.13) / (102.0 * 1.12)))
    assert data.label_time[0] == frame.fx_available_at.iloc[3].value


def test_raw_action_accounting_split_and_distribution():
    frame = pd.DataFrame(
        {
            "price_kind": ["raw_with_actions"] * 3,
            "close": [100.0, 50.0, 52.0],
            "split_ratio": [1.0, 2.0, 1.0],
            "cash_distribution": [0.0, 0.0, 1.0],
        }
    )
    np.testing.assert_allclose(_local_wealth(frame), [1.0, 1.0, 1.06])


def test_bond_yields_and_current_constituents_fail_closed():
    frame = panel()
    frame["asset_type"] = "bond"
    frame["price_kind"] = "yield"
    frame.attrs["universe_status"] = "current_constituents"
    with pytest.raises(DataValidationError):
        validate_panel(frame)


def test_date_only_price_timestamps_are_not_assumed_close(tmp_path):
    frame = panel()
    frame["timestamp"] = "2020-01-01"
    target = tmp_path / "naive.csv"
    frame.to_csv(target, index=False)
    with pytest.raises(DataValidationError, match="timezone"):
        load_panel(target)


def test_duplicate_asset_event_rows_rejected():
    frame = panel()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicated.attrs.update(frame.attrs)
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_panel(duplicated)


def test_missing_values_are_imputed_only_from_training_context():
    data = window_fixture(n_dates=12, assets=1)
    data.x[:, :, 0] = 2.0
    data.x[0, 0, 0] = np.nan
    data.x[0, 0, 1] = 1.0
    data.future_x[:, :, 0] = 1e9  # This must never affect fitted statistics.
    scaler = TrainOnlyRobustScaler().fit(data)
    assert scaler.to_dict()["median"][0] == 2.0
    result = scaler.transform_x(data.x)
    assert np.isfinite(result).all()
    assert result[0, 0, 0] == 0.0
    assert result[0, 0, 1] == 1.0
    with pytest.raises(ValueError, match="already fitted"):
        scaler.fit(data)


def test_scaler_roundtrip_preserves_transform():
    data = window_fixture(n_dates=10, assets=1)
    data.x[:, :, 0] = np.arange(10)[:, None]
    first = TrainOnlyRobustScaler().fit(data)
    second = TrainOnlyRobustScaler.from_dict(first.to_dict())
    np.testing.assert_array_equal(first.transform_x(data.x), second.transform_x(data.x))


def test_split_preserves_date_groups_and_purges_label_maturity():
    data = window_fixture()
    split = chronological_split(data, embargo_bars=2)
    assert data.label_time[split.train].max() < data.time[split.validation].min()
    assert data.label_time[split.validation].max() < data.time[split.test].min()
    for indices in (split.train, split.validation, split.test):
        for time in np.unique(data.time[indices]):
            assert np.sum(data.time[indices] == time) == 2
    assert not set(data.time[split.train]).intersection(data.time[split.test])


def test_protected_tail_excluded_before_fraction_boundaries():
    data = window_fixture()
    protected_start = int(np.unique(data.time)[80])
    split = chronological_split(data, embargo_bars=1, protected_start=protected_start)
    for indices in (split.train, split.validation, split.test):
        assert np.all(data.label_time[indices] < protected_start)
    assert np.all(data.time[split.protected] >= protected_start)
    assert split.metadata["excluded_protected_crossing_labels"] > 0


def test_walk_forward_test_blocks_never_reuse_dates():
    data = window_fixture(n_dates=140, horizon=2)
    folds = walk_forward_splits(data, min_train_dates=30, validation_dates=10, test_dates=10, embargo_bars=3)
    used = set()
    for fold in folds:
        current = set(data.time[fold.test])
        assert not used.intersection(current)
        used.update(current)
    with pytest.raises(ValueError, match="reuse"):
        walk_forward_splits(data, 30, 10, 10, step_dates=5)


def test_inference_has_no_actual_or_future_and_uses_latest_available():
    frame = panel(n=5)
    data = build_inference_windows(frame, context=4, horizon=20)
    assert len(data) == 1
    assert np.isnan(data.y).all() and np.isnan(data.future_x).all()
    assert data.time[0] == frame.available_at.iloc[-1].value
    assert data.label_time[0] == np.iinfo(np.int64).max
    assert data.entry_time is None and data.exit_time is None


def test_narrow_inference_reuses_asset_ids_and_rejects_unknown():
    data = build_inference_windows(panel(asset="B"), context=3, horizon=1)
    aligned = align_asset_universe(data, ["A", "B", "C"])
    assert aligned.asset_id.tolist() == [1]
    assert aligned.asset_names == ["A", "B", "C"]
    with pytest.raises(DataValidationError, match="Unseen"):
        align_asset_universe(data, ["A"])


def test_peer_features_ignore_future_prices():
    a, b = panel(asset="A"), panel(asset="B")
    frame = pd.concat([a, b], ignore_index=True)
    frame.attrs.update(a.attrs)
    frame["market"] = "fixture_market"
    frame["sector"] = "fixture_sector"
    frame["classification_known_at"] = pd.Timestamp("2019-01-01T00:00:00Z")
    first = add_causal_peer_features(frame, min_peers=1)
    changed = frame.copy()
    last = (changed.asset == "B") & (changed.timestamp == changed.timestamp.max())
    changed.loc[last, "total_return_index"] *= 1.5
    second = add_causal_peer_features(changed, min_peers=1)
    earlier = first.timestamp < first.timestamp.max()
    np.testing.assert_allclose(
        first.loc[earlier, "feature_market_peer_return"],
        second.loc[earlier, "feature_market_peer_return"],
        equal_nan=True,
    )


def test_fingerprint_changes_for_value_or_provenance_change():
    frame = panel()
    original = data_fingerprint(frame)
    frame.attrs["availability_policy"] = "changed source policy"
    assert data_fingerprint(frame) != original
