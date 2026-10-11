"""EWMA covariance estimate."""

from __future__ import annotations

import numpy as np


def ewma_covariance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    n, k = returns.shape
    cov = np.cov(returns.T) if n > 1 else np.eye(k) * 1e-6
    for i in range(1, n):
        r = returns[i : i + 1].T
        cov = lam * cov + (1 - lam) * (r @ r.T)
    return cov
