"""Predeclared input perturbations and chronological low-data controls."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

DAY_NS = 86_400_000_000_000


def recent_training_fraction(data: Any, fraction: float) -> Any:
    """Keep the most recent fraction of observed UTC dates, including whole days."""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Training fraction must lie in (0,1]")
    row_dates = np.asarray(data.time, dtype=np.int64) // DAY_NS
    dates = np.unique(row_dates)
    if not len(dates):
        raise ValueError("Training history is empty before low-data selection")
    keep = max(1, int(np.ceil(len(dates) * fraction)))
    return data.subset(np.flatnonzero(row_dates >= dates[-keep]))


def perturb_inputs(
    data: Any, training: Any, *, noise_std: float = 0.0, missing_probability: float = 0.0, seed: int = 0
) -> Any:
    """Simulated feature corruption; target and future target are untouched.

    Noise scale and missing replacements come only from the training distribution.
    Missingness indicators are updated for affected channels when recognizable.
    This is a corruption assay, not a simulation of an exchange outage.
    """
    if (
        not math.isfinite(noise_std)
        or not math.isfinite(missing_probability)
        or noise_std < 0
        or not 0 <= missing_probability < 1
    ):
        raise ValueError("Invalid corruption severity")
    if list(data.feature_names) != list(training.feature_names):
        raise ValueError("Train/evaluation feature schema mismatch")
    rng = np.random.default_rng(seed)
    out = copy.copy(data)
    out.x = np.asarray(data.x, dtype=np.float64).copy()
    if not len(training.x) or np.isinf(out.x).any():
        raise ValueError("Corruption requires nonempty training features and no infinite input values")
    features = list(data.feature_names)
    flat = np.asarray(training.x, dtype=float).reshape(-1, training.x.shape[-1])
    if np.isinf(flat).any():
        raise ValueError("Infinite training features cannot define corruption scales")
    all_missing = np.all(np.isnan(flat), axis=0)
    safe = flat.copy()
    safe[:, all_missing] = 0.0
    median = np.nanmedian(safe, axis=0)
    q25, q75 = np.nanquantile(safe, [0.25, 0.75], axis=0)
    width = np.where(all_missing, 1.0, np.maximum(q75 - q25, 1e-8))
    median[all_missing] = 0.0
    corrupted_counts: dict[str, int] = {}
    for j, name in enumerate(features):
        if "missing" in name.lower() or "mask" in name.lower():
            continue
        if noise_std:
            out.x[..., j] = out.x[..., j] + rng.normal(0.0, noise_std * width[j], out.x[..., j].shape)
        if missing_probability:
            missing = rng.random(out.x[..., j].shape) < missing_probability
            out.x[..., j][missing] = median[j]
            corrupted_counts[name] = int(missing.sum())
            for missing_name in (f"{name}_missing", f"missing_{name}", f"{name}__missing"):
                if missing_name in features:
                    out.x[..., features.index(missing_name)][missing] = 1.0
    out.metadata = {
        **copy.deepcopy(data.metadata),
        "input_perturbation": {
            "noise_std": float(noise_std),
            "missing_probability": float(missing_probability),
            "seed": int(seed),
            "scale_fit_source": "training contexts only",
            "feature_medians": median.tolist(),
            "feature_noise_widths": width.tolist(),
            "all_missing_training_features": [features[index] for index in np.flatnonzero(all_missing)],
            "all_missing_fallback": {"median": 0.0, "noise_width": 1.0},
            "synthetic_missing_counts": corrupted_counts,
            "targets_unchanged": True,
        },
    }
    return out


def held_out_assets(data: Any, names: list[str]) -> tuple[Any, Any]:
    if not names or len(set(names)) != len(names):
        raise ValueError("OOD asset names must be nonempty and unique")
    unknown = sorted(set(names) - set(data.asset_names))
    if unknown:
        raise ValueError(f"Unknown held-out asset identifiers: {unknown}")
    ids = [data.asset_names.index(name) for name in names]
    mask = np.isin(data.asset_id, ids)
    # An unused side may legitimately be empty: an unseen future instrument can
    # have no training rows, while an evaluation panel may contain only held-out
    # instruments. The matrix validates the side it actually consumes.
    return data.subset(np.flatnonzero(~mask)), data.subset(np.flatnonzero(mask))
