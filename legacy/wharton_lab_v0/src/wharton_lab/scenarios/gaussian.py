"""IID Gaussian scenario paths (research proxy only)."""

from __future__ import annotations

import numpy as np


def gaussian_scenario_paths(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    k = mean.shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, cov, size=(n_paths, n_steps))
    if draws.shape[-1] != k:
        raise ValueError("cov dimension mismatch")
    return draws
