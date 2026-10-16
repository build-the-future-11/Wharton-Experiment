import numpy as np
import pandas as pd
import pytest
from test_model_contracts import windows
from test_s3 import frame

from frem.s3 import DAY, run_replay, seed_memory
from frem.s4 import arm, scale_keys, select
from frem.s4_forecaster import S4Base, partitions


def test_distance_conversion_exact():
    keys = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scaled = scale_keys(keys, "euclidean_distance")
    assert np.mean((scaled[0] - scaled[1]) ** 2) == pytest.approx(2)
    np.testing.assert_array_equal(keys, scale_keys(keys, "old_distance"))


def test_recent_partitions_are_purged_and_exclude_old_dates():
    data = windows(200)
    parts = partitions(data)
    assert parts[0].time.min() == np.unique(data.time)[-96]
    for a, b in zip(parts, parts[1:]):
        assert a.label_time.max() < b.time.min()


def test_conservative_gate_rejects_null_correction():
    cfg = arm("euclidean_distance", 11)
    initial = seed_memory(frame(), np.ones((12, 2)), cfg, "v", 20 * DAY)
    valid = frame(24, 21)
    valid.actual = 0
    selection, _ = select(initial, valid, np.ones((24, 2)), cfg)
    assert selection["selected"]["weight"] == 0


def test_future_label_invariance():
    cfg = arm("euclidean_distance", 11)
    initial = seed_memory(frame(), scale_keys(np.ones((12, 2)), cfg.name), cfg, "v", 20 * DAY)
    stream = frame(8, 21)
    altered = stream.copy()
    altered.loc[4:, "actual"] = 1000
    keys = scale_keys(np.ones((8, 2)), cfg.name)
    a = run_replay(initial, stream, keys, cfg, 0.5, 0)
    b = run_replay(initial, altered, keys, cfg, 0.5, 0)
    pd.testing.assert_frame_equal(a.loc[:6, ["prediction", "scale"]], b.loc[:6, ["prediction", "scale"]])


def test_recent_base_reload(tmp_path):
    base = S4Base("zero", 11)
    base.fit(windows(150))
    later = windows(8, start=180)
    base.save(tmp_path / "base")
    restored = S4Base.load(tmp_path / "base")
    pd.testing.assert_frame_equal(base.predict(later), restored.predict(later))
