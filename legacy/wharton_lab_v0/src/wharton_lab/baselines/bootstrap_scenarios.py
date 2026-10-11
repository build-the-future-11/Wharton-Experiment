"""Bootstrap resampled return paths."""

from __future__ import annotations

import numpy as np


def bootstrap_scenario_paths(
    returns: np.ndarray,
    *,
    n_paths: int = 100,
    n_steps: int = 5,
    seed: int = 0,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    n_hist, k = returns.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_hist, size=(n_paths, n_steps))
    return returns[idx]
