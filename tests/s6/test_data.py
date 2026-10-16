"""Constructed input-contract checks; no market or protected data is loaded."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from successors.frem_s6.data import PROTECTED_LIMIT, load_dataset, validate_dataset


def payload():
    def partition(dates, train=False):
        origin = np.repeat(dates, 2).astype(float)
        n = len(origin)
        times = origin[:, None] + np.array([-3, -2, -1])[None]
        y = np.linspace(-.02, .03, n)
        result = {"windows": np.zeros((n, 3, 1)).tolist(),
                  "window_times": times.tolist(), "window_available": (times + .5).tolist(),
                  "origin_times": origin.tolist(), "maturity_times": (origin + 5).tolist(),
                  "available_times": (origin + 6).tolist(), "y": y.tolist(),
                  "simple_returns": np.expm1(y).tolist(), "assets": ["A", "B"] * len(dates)}
        if train:
            future = origin[:, None] + np.array([1, 2, 3])[None]
            result.update({"future_windows": np.ones((n, 3, 1)).tolist(),
                           "future_times": future.tolist(), "future_available": (future + .5).tolist()})
        return result

    return {"schema_version": "s6-windows-v1", "return_units": "log_return", "currency": "USD",
            "horizon_seconds": 5, "fit_cutoff": 40, "protected_start": 1000,
            "asset_universe": ["A", "B"],
            "partitions": {"train": partition([10, 20], True),
                           "calibration": partition([50, 60]), "evaluation": partition([80, 85])}}


def save_dataset(tmp_path, data=None, **overrides):
    data = payload() if data is None else data
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(data))
    manifest = {"schema_version": "s6-manifest-v1", "kind": "admitted_development",
                "protected_start": 1000, "last_outcome_time": 90,
                "source": "constructed test input only", "rights": "generated test values",
                "point_in_time": "constructed timestamps, not market provenance",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest.update(overrides)
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest))
    return path


def test_valid_complete_synchronized_dataset_and_digest(tmp_path):
    result = load_dataset(save_dataset(tmp_path))
    assert result["metadata"]["last_outcome_time"] == 90
    assert result["metadata"]["calibration_cutoff"] == 66
    assert result["metadata"]["input_dim"] == 1
    assert result["metadata"]["asset_universe"] == ("A", "B")
    assert result["partitions"]["train"]["future_available"].shape == (4, 3)
    assert result["partitions"]["evaluation"]["windows"].shape == (4, 3, 1)
    assert result["metadata"]["sha256"] == result["metadata"]["manifest"]["sha256"]


@pytest.mark.parametrize("changes", [{"last_outcome_time": 1000},
                                      {"protected_start": PROTECTED_LIMIT + 1},
                                      {"kind": "raw_unreviewed"},
                                      {"source": ""}, {"rights": ""}, {"point_in_time": ""},
                                      {"sha256": "bad"}])
def test_failed_sidecar_gate_never_reads_payload(tmp_path, monkeypatch, changes):
    path = save_dataset(tmp_path, **changes)
    original = Path.read_bytes

    def guarded(target):
        if target == path:
            raise AssertionError("payload opened before passing sidecar gate")
        return original(target)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_digest_mismatch_rejected_before_payload_json_parse(tmp_path):
    path = save_dataset(tmp_path, sha256="0" * 64)
    path.write_bytes(b"not valid JSON or UTF-8 \xff")
    with pytest.raises(ValueError, match="digest"):
        load_dataset(path)


def test_all_timestamps_checked_before_any_outcome_or_feature_numeric_conversion():
    data = payload()
    data["partitions"]["train"]["windows"] = "NEVER_NUMERICALLY_PARSE"
    data["partitions"]["train"]["y"] = "NEVER_NUMERICALLY_PARSE"
    data["partitions"]["evaluation"]["available_times"][-1] = 1000
    with pytest.raises(ValueError, match="protected boundary"):
        validate_dataset(data)


@pytest.mark.parametrize("field", ["window_times", "window_available"])
def test_feature_equal_to_issue_is_not_available(field):
    data = payload()
    data["partitions"]["train"][field][0][-1] = 10
    with pytest.raises(ValueError, match="strictly before origin"):
        validate_dataset(data)


def test_future_representation_and_labels_must_be_available_before_fit():
    for field, value in [("future_available", 40), ("available_times", 40)]:
        data = payload()
        if field == "future_available":
            data["partitions"]["train"][field][0][-1] = value
        else:
            data["partitions"]["train"][field][0] = value
        with pytest.raises(ValueError, match="before fit_cutoff"):
            validate_dataset(data)


def test_future_representation_cannot_contain_origin_or_past():
    data = payload()
    data["partitions"]["train"]["future_times"][0][0] = 10
    with pytest.raises(ValueError, match="post-origin"):
        validate_dataset(data)


def test_duplicate_or_incomplete_asset_dates_fail():
    for change in ("duplicate", "missing", "undeclared"):
        data = payload()
        assets = data["partitions"]["evaluation"]["assets"]
        if change == "duplicate":
            assets[0] = "B"
        elif change == "missing":
            data["asset_universe"].append("C")
        else:
            assets[0] = "C"
        with pytest.raises(ValueError, match="duplicate|complete"):
            validate_dataset(data)


def test_labels_cannot_be_available_before_outcome_or_have_wrong_horizon():
    for field, value in [("available_times", 84), ("maturity_times", 86)]:
        data = payload()
        data["partitions"]["evaluation"][field][0] = value
        with pytest.raises(ValueError, match="availability|horizon"):
            validate_dataset(data)


def test_calibration_must_finish_before_evaluation():
    data = payload()
    data["partitions"]["calibration"]["available_times"][-1] = 80
    with pytest.raises(ValueError, match="complete calibration"):
        validate_dataset(data)


def test_overlapping_evaluation_intervals_rejected_with_same_valid_horizon():
    data = payload()
    part = data["partitions"]["evaluation"]
    for field in ("origin_times", "maturity_times", "available_times"):
        part[field][2:] = [value - 1 for value in part[field][2:]]
    for field in ("window_times", "window_available"):
        part[field][2:] = [[value - 1 for value in row] for row in part[field][2:]]
    with pytest.raises(ValueError, match="nonoverlapping"):
        validate_dataset(data)


@pytest.mark.parametrize("value", [-1, -.999, float("nan"), float("inf")])
def test_log_simple_return_reconciliation_rejects_invented_or_invalid_values(value):
    data = payload()
    data["partitions"]["evaluation"]["simple_returns"][0] = value
    with pytest.raises(ValueError, match="finite|expm1"):
        validate_dataset(data)


def test_finite_log_cannot_hide_total_loss_underflow():
    data = payload()
    data["partitions"]["evaluation"]["y"][0] = -1000
    data["partitions"]["evaluation"]["simple_returns"][0] = -1
    with pytest.raises(ValueError, match="positive gross"):
        validate_dataset(data)


def test_sidecar_outcome_declaration_must_match_actual_metadata(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        load_dataset(save_dataset(tmp_path, last_outcome_time=91))


def test_payload_is_not_mutated_by_validation():
    data = payload()
    original = copy.deepcopy(data)
    validate_dataset(data)
    assert data == original


def test_duplicate_json_fields_rejected(tmp_path):
    path = save_dataset(tmp_path)
    sidecar = Path(str(path) + ".manifest.json")
    sidecar.write_text('{"kind":"admitted_development","kind":"admitted_development"}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_dataset(path)
