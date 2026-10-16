"""Causal, numerical and serialization contracts for the additive S3 lineage."""

import copy
import itertools
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from frem.s3 import DAY, KeyMetric, Replay, ReplayConfig, arms, calibrate, run_replay, seed_memory
from frem.s3_forecaster import S3Base, partitions


def frame(n=12, start=0):
    t = np.arange(start, start + n) * DAY
    return pd.DataFrame(
        {
            "time": t,
            "label_time": t + 2 * DAY,
            "asset_id": 0,
            "asset": "a",
            "actual": 0.1,
            "prediction": 0.0,
            "scale": 0.2,
            "student_df": 5.0,
        }
    )


def test_maturity_is_strict_and_unknown_outcomes_do_not_retrieve():
    memory = Replay(ReplayConfig("x"), "v1")
    row = frame(1).iloc[0].to_dict()
    memory.schedule(row, [1.0], 0.0, 0.2, [])
    memory.advance(2 * DAY)
    assert memory.retrieve([1.0], 0, 2 * DAY)["count"] == 0
    memory.advance(3 * DAY)
    assert memory.retrieve([1.0], 0, 3 * DAY)["z"] == pytest.approx(0.5)


def test_duplicate_version_and_backward_rejected():
    memory = Replay(ReplayConfig("x"), "v1")
    row = frame(1).iloc[0].to_dict()
    with pytest.raises(ValueError, match="version"):
        memory.schedule({**row, "version": "v2"}, [1], 0, 0.2, [])
    memory.schedule(row, [1], 0, 0.2, [])
    with pytest.raises(ValueError, match="Duplicate"):
        memory.schedule(row, [1], 0, 0.2, [])
    memory.advance(3 * DAY)
    with pytest.raises(ValueError, match="backwards"):
        memory.advance(2 * DAY)


@pytest.mark.parametrize("retention", ["fifo", "balanced", "dual", "utility"])
def test_capacity_and_checkpoint_roundtrip(retention):
    cfg = ReplayConfig("x", retention=retention, capacity=4)
    initial = seed_memory(frame(), np.ones((12, 2)), cfg, "v1", 20 * DAY)
    assert len(initial["records"]) == 4
    assert initial["evicted"] == 8
    clone = Replay.from_dict(json.loads(json.dumps(initial)))
    assert clone.to_dict() == initial


def test_future_outcome_perturbation_cannot_change_earlier_forecasts():
    cfg = ReplayConfig("x", confidence=False)
    initial = seed_memory(frame(), np.ones((12, 1)), cfg, "v1", 20 * DAY)
    stream = frame(8, 21)
    altered = stream.copy()
    altered.loc[4:, "actual"] = 1000
    a = run_replay(initial, stream, np.ones((8, 1)), cfg, 1, 1)
    b = run_replay(initial, altered, np.ones((8, 1)), cfg, 1, 1)
    # Row four's label matures at day27, only decisions strictly after that may change.
    pd.testing.assert_frame_equal(a.loc[:6, ["prediction", "scale"]], b.loc[:6, ["prediction", "scale"]])


def test_frozen_and_off_do_not_use_evaluation_actuals():
    for cfg in [replace(ReplayConfig("x"), online=False), arms()[0]]:
        initial = seed_memory(frame(), np.ones((12, 1)), cfg, "v1", 20 * DAY)
        stream = frame(8, 21)
        a = run_replay(initial, stream, np.ones((8, 1)), cfg, 1, 1)
        stream.actual = 100
        b = run_replay(initial, stream, np.ones((8, 1)), cfg, 1, 1)
        pd.testing.assert_frame_equal(a[["prediction", "scale"]], b[["prediction", "scale"]])


def test_absolute_age_attenuates_confidence_without_changing_residual():
    cfg = ReplayConfig("x")
    state = seed_memory(frame(), np.ones((12, 1)), cfg, "v1", 20 * DAY)
    memory = Replay.from_dict(state)
    a = memory.retrieve([1], 0, 20 * DAY)
    b = memory.retrieve([1], 0, 200 * DAY)
    assert b["z"] == pytest.approx(a["z"])
    assert b["confidence"] == pytest.approx(a["confidence"] / 2)


def test_policies_receive_same_matured_observations_and_random_is_reproducible():
    for cfg in [
        replace(ReplayConfig("x"), retrieval="random"),
        replace(ReplayConfig("x"), retrieval="recent"),
    ]:
        state = seed_memory(frame(), np.ones((12, 1)), cfg, "v1", 20 * DAY)
        stream = frame(8, 21)
        a, final = run_replay(state, stream, np.ones((8, 1)), cfg, 0.5, 0.5, keep_state=True)
        b = run_replay(state, stream, np.ones((8, 1)), cfg, 0.5, 0.5)
        pd.testing.assert_frame_equal(a, b)
        assert final["matured"] == 17


def test_metric_future_cutoff_and_serialization():
    x = np.arange(24).reshape(12, 2).astype(float)
    with pytest.raises(ValueError, match="mature"):
        KeyMetric().fit(x, np.ones(12), np.arange(12), 11)
    metric = KeyMetric().fit(x, np.ones(12), np.arange(12), 12)
    assert np.isfinite(metric.transform(x)).all()
    assert json.loads(json.dumps(metric.to_dict()))["fitted_through"] == 11


def test_shared_calibration_can_reject_spurious_mean():
    data = frame()
    data.actual = 0
    data.prediction = 10
    assert calibrate(data)["selected"]["alpha"] == 0


def test_partitions_and_constant_base_roundtrip(tmp_path):
    from test_model_contracts import windows

    data = windows(100)
    parts = partitions(data)
    assert all(a.label_time.max() < b.time.min() for a, b in itertools.pairwise(parts))
    base = S3Base("zero", 11)
    base.fit(data)
    later = windows(8, start=130)
    before = base.predict(later)
    base.save(tmp_path / "checkpoint")
    restored = S3Base.load(tmp_path / "checkpoint")
    pd.testing.assert_frame_equal(before, restored.predict(later))


def test_deployed_density_drives_utility():
    memory = Replay(ReplayConfig("utility", retention="utility"), "v1")
    row = frame(1).iloc[0].to_dict()
    memory.schedule(row, [1], 0, 0.2, [])
    memory.advance(3 * DAY)
    next_row = frame(1, 4).iloc[0].to_dict()
    memory.schedule(next_row, [1], 0.1, 0.2, [0])
    memory.advance(7 * DAY)
    assert memory.records[0]["utility"] > 0


def test_memory_fit_does_not_mutate_seed_state():
    cfg = ReplayConfig("x")
    state = seed_memory(frame(), np.ones((12, 1)), cfg, "v1", 20 * DAY)
    original = copy.deepcopy(state)
    run_replay(state, frame(8, 21), np.ones((8, 1)), cfg, 1, 1)
    assert state == original


@pytest.mark.parametrize("name", ["frem", "no_jepa", "ridge", "xgboost"])
def test_base_families_fit_causal_partitions_and_reload(name, tmp_path):
    from test_model_contracts import windows

    base = S3Base(name, 11)
    base.fit(windows(100))
    later = windows(8, start=130)
    original = base.predict(later)
    base.save(tmp_path / name)
    restored = S3Base.load(tmp_path / name, trust_joblib=True)
    pd.testing.assert_frame_equal(original, restored.predict(later))
    np.testing.assert_allclose(base.keys(later, True), restored.keys(later, True))
