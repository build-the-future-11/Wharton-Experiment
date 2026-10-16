"""Conservative mean-residual selection for development-only S4."""

from dataclasses import replace

import numpy as np

from .s3 import ReplayConfig, balanced, run_replay


def arm(name, seed):
    base = ReplayConfig(name, seed=seed, mode="mean")
    if name == "off":
        return replace(base, mode="off")
    if name == "matched_random":
        return replace(base, retrieval="random")
    if name not in {"old_distance", "euclidean_distance"}:
        raise ValueError("Unknown S4 arm")
    return base


def scale_keys(keys, name):
    keys = np.asarray(keys, dtype=float)
    if keys.ndim != 2 or keys.shape[1] == 0 or not np.isfinite(keys).all():
        raise ValueError("Invalid S4 keys")
    return keys * np.sqrt(keys.shape[1]) if name == "euclidean_distance" else keys.copy()


def select(initial, validation, keys, config):
    dates = np.unique(validation.time)
    if len(dates) < 12:
        raise ValueError("S4 requires at least 12 validation decision dates")
    boundary = dates[len(dates) // 2]
    baseline = run_replay(initial, validation, keys, config, 0.0, 0.0)
    groups = [validation.time < boundary, validation.time >= boundary]
    candidates = []
    for weight in (0.0,) if config.mode == "off" else (0.0, 0.25, 0.5, 1.0):
        pred = run_replay(initial, validation, keys, config, weight, 0.0)
        gains = [balanced((baseline.nll - pred.nll)[m], pred.time[m], pred.asset_id[m]) for m in groups]
        candidates.append(
            {
                "weight": weight,
                "power": 0.0,
                "nll": balanced(pred.nll, pred.time, pred.asset_id),
                "half_gains": gains,
                "eligible": weight == 0 or min(gains) > 0.01,
            }
        )
    choice = min((c for c in candidates if c["eligible"]), key=lambda c: (c["nll"], c["weight"]))
    _, state = run_replay(initial, validation, keys, config, choice["weight"], 0.0, keep_state=True)
    return {"selected": choice, "candidates": candidates, "validation_boundary": int(boundary)}, state
