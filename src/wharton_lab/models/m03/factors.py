"""Factor estimation and AR(1) forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_factors(returns_panel: np.ndarray, n_factors: int) -> tuple[np.ndarray, PCA]:
    """
    returns_panel: (T, N) cross-section of returns
    Returns factor series (T, K) and fitted PCA on demeaned returns.
    """
    R = np.asarray(returns_panel, dtype=float)
    R = R - np.nanmean(R, axis=1, keepdims=True)
    R = np.nan_to_num(R)
    k = int(min(n_factors, R.shape[0], R.shape[1]))
    if k < 1:
        k = 1
    pca = PCA(n_components=k, random_state=0)
    factors = pca.fit_transform(R)
    return factors, pca


def fit_ar1(series: np.ndarray) -> tuple[float, float]:
    """OLS AR(1): x_t = c + phi x_{t-1}. Returns (c, phi)."""
    x = np.asarray(series, dtype=float).ravel()
    if len(x) < 3:
        return 0.0, 0.0
    y = x[1:]
    z = x[:-1]
    X = np.column_stack([np.ones(len(y)), z])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def forecast_ar1(
    history: np.ndarray, c: float, phi: float, steps: int = 1
) -> np.ndarray:
    """Multi-step AR(1) forecast from last observed value."""
    history = np.asarray(history, dtype=float).ravel()
    if len(history) == 0:
        return np.zeros(steps)
    x = history[-1]
    out = []
    for _ in range(steps):
        x = c + phi * x
        out.append(x)
    return np.array(out, dtype=float)
