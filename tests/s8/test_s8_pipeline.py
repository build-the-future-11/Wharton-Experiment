"""Tiny generated integration fixture; engineering evidence, never market data."""

import json

import numpy as np

from successors.frem_s6_real.prepare import stamp
from successors.frem_s8.run import fit_seed, validate_scope


def fixture():
    rng = np.random.default_rng(81)

    def part(dates, train=False):
        t = np.repeat(dates, 3)
        past = t[:, None] + np.arange(-8, 0)[None, :] * 86400
        y = rng.normal(0, 0.01, len(t))
        out = {
            "assets": ["SPY", "TLT", "GLD"] * len(dates),
            "origin_times": t.tolist(),
            "maturity_times": (t + 7 * 86400).tolist(),
            "available_times": (t + 7 * 86400 + 3600).tolist(),
            "window_times": past.tolist(),
            "window_available": past.tolist(),
            "windows": rng.normal(0, 0.01, (len(t), 8, 1)).tolist(),
            "y": y.tolist(),
            "simple_returns": np.expm1(y).tolist(),
        }
        if train:
            future = t[:, None] + np.arange(1, 9)[None, :] * 86400
            out.update(
                future_times=future.tolist(),
                future_available=future.tolist(),
                future_windows=rng.normal(0, 0.01, (len(t), 8, 1)).tolist(),
            )
        return out

    train = stamp("2016-01-20") + np.arange(16) * 7 * 86400
    calibration = np.concatenate(
        [stamp("2017-02-01") + np.arange(12) * 7 * 86400, stamp("2018-02-01") + np.arange(12) * 7 * 86400]
    )
    evaluation = stamp("2020-02-01") + np.arange(8) * 7 * 86400
    return {
        "schema_version": "s7-variable-week-v1",
        "return_units": "log_return",
        "currency": "USD",
        "horizon_seconds": 7 * 86400,
        "protected_start": stamp("2025-01-01"),
        "fit_cutoff": stamp("2017-01-01"),
        "asset_universe": ["SPY", "TLT", "GLD"],
        "partitions": {
            "train": part(train, True),
            "calibration": part(calibration),
            "evaluation": part(evaluation),
        },
    }


def test_seed_flow_saves_complete_forecasts_and_strict_feedback(tmp_path):
    data = validate_scope(fixture())
    rows = fit_seed(data, tmp_path, 11)
    assert len(rows) == 8 * 15
    assert len({r["variant"] for r in rows}) == 15
    state = json.loads((tmp_path / "seed11/gate_state.json").read_text())
    assert all(r["available"] < r["used_at"] for r in state["local"]["updates"])
    assert len(state["local"]["pending"]) > 0
    assert (tmp_path / "seed11/fit_checkpoint.json").exists()
    assert (tmp_path / "seed11/online_checkpoint.json").exists()
