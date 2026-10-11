import pandas as pd

from rejepa.data import (
    features_available_as_of,
    information_available_as_of,
    select_phase_rows,
    validate_point_in_time_features,
    validate_point_in_time_frame,
    verify_snapshot,
    write_deterministic_snapshot,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": ["A", "B"],
            "event_time": ["2026-01-01T21:00:00Z", "2026-01-01T21:00:00Z"],
            "availability_time": ["2026-01-01T21:05:00Z", "2026-01-02T01:00:00Z"],
            "outcome_available_time": ["2026-01-08T21:00:00Z", "2026-01-08T21:00:00Z"],
            "market": ["US", "JP"],
            "currency": ["USD", "JPY"],
            "eligible": [True, True],
            "point_in_time_membership": [True, True],
            "close": [100.0, 200.0],
            "total_return": [0.01, -0.02],
            "volume": [1_000.0, 2_000.0],
            "source_id": ["licensed-1", "licensed-1"],
            "fx_to_usd": [1.0, 0.007],
            "delisted": [False, False],
            "corporate_action_id": ["none", "none"],
        }
    )


def test_valid_point_in_time_frame_and_as_of_filter() -> None:
    report = validate_point_in_time_frame(_frame())
    assert report.valid
    visible = information_available_as_of(_frame(), "2026-01-01T22:00:00Z")
    assert visible["asset_id"].tolist() == ["A"]


def test_future_or_missing_fx_is_rejected() -> None:
    frame = _frame().drop(columns="fx_to_usd")
    report = validate_point_in_time_frame(frame)
    assert not report.valid
    assert any("fx_to_usd" in error for error in report.errors)


def test_boolean_strings_extreme_losses_and_invalid_fx_fail_closed() -> None:
    frame = _frame()
    frame["eligible"] = ["False", "True"]
    frame.loc[0, "total_return"] = -1.0
    frame.loc[1, "fx_to_usd"] = 0.0
    report = validate_point_in_time_frame(frame)
    assert not report.valid
    assert any("explicit booleans" in error for error in report.errors)
    assert any("-100 percent" in error for error in report.errors)
    assert any("finite and positive" in error for error in report.errors)


def test_deterministic_snapshot_is_hash_verified(tmp_path) -> None:
    path = tmp_path / "panel.csv"
    manifest = write_deterministic_snapshot(
        _frame(),
        path,
        snapshot_id="engineering-fixture",
        source_name="authorized-test-source",
        license_or_right="local engineering fixture only",
        retrieved_at="2026-10-11T00:00:00Z",
    )
    assert verify_snapshot(path, manifest)
    assert manifest.row_count == 2
    assert path.with_suffix(".csv.manifest.json").exists()


def test_duplicate_observation_is_rejected() -> None:
    frame = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    report = validate_point_in_time_frame(frame)
    assert not report.valid
    assert "duplicate asset_id/event_time rows" in report.errors


def test_protected_boundary_fails_closed() -> None:
    frame = _frame()
    development = select_phase_rows(frame, phase="development", protected_start="2026-01-01T21:30:00Z")
    assert development["asset_id"].tolist() == ["A", "B"]
    try:
        select_phase_rows(frame, phase="protected", protected_start="2026-01-01T20:00:00Z")
    except PermissionError as error:
        assert "authorization" in str(error)
    else:
        raise AssertionError("Protected rows were exposed without authorization")


def test_macro_and_fundamental_vintages_obey_publication_lags() -> None:
    features = pd.DataFrame(
        {
            "entity_id": ["US", "US"],
            "feature_name": ["inflation", "inflation"],
            "observation_time": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "availability_time": ["2026-01-15T13:30:00Z", "2026-02-15T13:30:00Z"],
            "value": [2.0, 2.1],
            "source_id": ["authorized-macro", "authorized-macro"],
            "vintage_id": ["initial", "revision-1"],
            "revision_of_vintage_id": [None, "initial"],
            "frequency": ["monthly", "monthly"],
        }
    )
    assert validate_point_in_time_features(features).valid
    visible = features_available_as_of(features, "2026-01-20T00:00:00Z")
    assert visible["vintage_id"].tolist() == ["initial"]


def test_invalid_feature_frequency_is_rejected() -> None:
    features = pd.DataFrame(
        {
            "entity_id": ["ACME"],
            "feature_name": ["earnings"],
            "observation_time": ["2026-01-01T00:00:00Z"],
            "availability_time": ["2026-01-02T00:00:00Z"],
            "value": [1.0],
            "source_id": ["authorized-fundamental"],
            "vintage_id": ["v1"],
            "frequency": ["sometimes"],
        }
    )
    report = validate_point_in_time_features(features)
    assert not report.valid
    assert any("unsupported feature frequencies" in error for error in report.errors)
