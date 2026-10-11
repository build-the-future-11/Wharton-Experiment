from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from reng.data import (
    annual_split_indices,
    build_examples,
    fit_feature_scaler,
    opening_log_return,
    select_fold_universe,
    synthetic_panel,
    validate_panel,
)


@pytest.mark.parametrize(
    "kind", ["calm_trend", "mean_reversion", "correlation_break", "missing_actions"]
)
def test_all_engineering_panels_validate_and_build(kind: str) -> None:
    frame, manifest = synthetic_panel(kind, sessions=220, assets=4)
    receipt = validate_panel(frame, manifest)
    assert receipt["synthetic"] is True
    universe = select_fold_universe(frame, frame["timestamp"].max())
    examples = build_examples(frame, universe, context_length=128, horizons=(5, 21))
    assert examples.context.shape[1:] == (4, 128, 18)
    assert examples.future[21].shape[1:] == (4, 21, 18)
    assert np.isfinite(examples.log_returns[21]).all()


def test_future_availability_is_rejected() -> None:
    frame, manifest = synthetic_panel("calm_trend")
    frame["available_at"] = [pd.Timestamp(value).isoformat() for value in frame["available_at"]]
    observed = pd.Timestamp(frame.at[0, "timestamp"]).to_pydatetime()
    frame.at[0, "available_at"] = (observed - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="before"):
        validate_panel(frame, manifest)


def test_split_preserves_economic_wealth() -> None:
    frame, _ = synthetic_panel("missing_actions", sessions=220, assets=2)
    history = frame[frame["permanent_id"] == "SYN00"].reset_index(drop=True)
    value, _ = opening_log_return(history, 118, 2)
    assert abs(value) < 0.1


def test_annual_split_offsets_are_exact() -> None:
    dates = pd.bdate_range("2014-01-01", "2020-12-31", tz="UTC").to_numpy()
    split = annual_split_indices(dates, 2020, fitting_origins=504)
    first_test = split["test"][0]
    assert split["fit"][-1] == first_test - 319
    assert split["stopping"][0] == first_test - 254
    assert split["stopping"][-1] == first_test - 192
    assert split["calibration"][0] == first_test - 127
    assert split["calibration"][-1] == first_test - 65
    assert len(split["fit"]) == 504


def test_mutating_future_rows_does_not_change_earlier_context() -> None:
    frame, _ = synthetic_panel("calm_trend", sessions=220, assets=4)
    universe = select_fold_universe(frame, frame["timestamp"].max())
    original = build_examples(frame, universe, context_length=128, horizons=(5, 21))
    first = original.decision_times[0]
    changed = frame.copy()
    threshold = pd.Timestamp(first).tz_localize("UTC")
    changed.loc[pd.to_datetime(changed["timestamp"], utc=True) > threshold, "close"] *= 2
    rebuilt = build_examples(
        changed, universe, context_length=128, horizons=(5, 21), decision_times=[first]
    )
    assert np.array_equal(original.context[:1], rebuilt.context)


def test_scaler_changes_only_continuous_channels_and_uses_masks() -> None:
    frame, _ = synthetic_panel("calm_trend", sessions=220, assets=4)
    universe = select_fold_universe(frame, frame["timestamp"].max())
    examples = build_examples(frame, universe, context_length=128, horizons=(5, 21))
    fitting = examples.context[:10]
    scaler = fit_feature_scaler(fitting)
    transformed = scaler.transform_feature_channels(fitting)
    assert np.array_equal(transformed[..., 6:], fitting[..., 6:])
    assert np.max(np.abs(transformed[..., :6])) <= 10
