"""Rolling OLS factor exposures (historical only)."""

from __future__ import annotations

import numpy as np


def rolling_ols_exposure(
    asset_returns: np.ndarray,
    factor_returns: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    For each time t, beta_t from OLS of asset_returns[t-window:t] on factors.

    asset_returns: (T,) or (T, N)
    factor_returns: (T, F)
    Returns betas (T, N, F) or (T, F) for single asset.
    """
    factor_returns = np.asarray(factor_returns, dtype=float)
    asset_returns = np.asarray(asset_returns, dtype=float)
    if asset_returns.ndim == 1:
        asset_returns = asset_returns[:, None]
    T, N = asset_returns.shape
    F = factor_returns.shape[1]
    betas = np.full((T, N, F), np.nan, dtype=float)
    X_full = np.column_stack([np.ones(T), factor_returns])
    for t in range(window, T):
        sl = slice(t - window, t)
        X = X_full[sl]
        for n in range(N):
            y = asset_returns[sl, n]
            if np.any(~np.isfinite(y)) or np.any(~np.isfinite(X)):
                continue
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            betas[t, n, :] = coef[1:]
    return betas


def residualize_labels(
    future_returns: np.ndarray,
    betas: np.ndarray,
    realized_future_factors: np.ndarray,
) -> np.ndarray:
    """
    residual = future_return - sum_f beta_{t,f} * realized_future_factor_f

    betas aligned at decision time t; realized_future_factors shape (N, F) per date
    or (T, N, F) batched — here per-date (N, F) with beta (N, F).
    """
    future_returns = np.asarray(future_returns, dtype=float).ravel()
    betas = np.asarray(betas, dtype=float)
    realized_future_factors = np.asarray(realized_future_factors, dtype=float)
    pred = np.sum(betas * realized_future_factors, axis=-1)
    return future_returns - pred
