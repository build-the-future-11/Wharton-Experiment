"""Point-in-time safe lag features."""

from __future__ import annotations

import numpy as np


def lagged_return_features(returns: np.ndarray, lags: tuple[int, ...] = (1, 2, 5)) -> np.ndarray:
    """Stack lagged returns; row t uses only returns strictly before t."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    n, k = r.shape
    cols = []
    for lag in lags:
        pad = np.zeros((lag, k))
        lagged = np.vstack([pad, r[:-lag]]) if lag < n else np.zeros_like(r)
        cols.append(lagged)
    return np.hstack(cols)


def panel_cross_sectional_ranks(returns: np.ndarray) -> np.ndarray:
    """Cross-sectional rank of contemporaneous returns (for LTR-style inputs)."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    order = np.argsort(np.argsort(r, axis=1), axis=1).astype(float)
    denom = max(r.shape[1] - 1, 1)
    return order / denom
