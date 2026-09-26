"""Causal regime labels from past returns/volatility only."""

from __future__ import annotations

import numpy as np


def trailing_volatility(
    returns: np.ndarray,
    window: int,
    expanding: bool = False,
) -> np.ndarray:
    """
    Trailing volatility using only past returns (causal).

    For row i, uses returns[max(0, i-window):i] — never includes return at i
    when used as a label for the same index; callers should align features
    one step ahead if needed.
    """
    returns = np.asarray(returns, dtype=float).ravel()
    n = len(returns)
    vol = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if expanding:
            start = 0
        else:
            start = max(0, i - window)
        hist = returns[start:i]
        if len(hist) >= 2:
            vol[i] = float(np.std(hist, ddof=1))
        elif len(hist) == 1:
            vol[i] = 0.0
    return vol


def causal_regime_ids(
    returns: np.ndarray,
    *,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
    expanding_vol: bool = False,
) -> np.ndarray:
    """
    Hard regimes from past-only vol and lagged return sign.

    Regime encoding:
      0 = low vol, non-negative lagged return
      1 = low vol, negative lagged return
      2 = high vol, non-negative lagged return
      3 = high vol, negative lagged return
    """
    returns = np.asarray(returns, dtype=float).ravel()
    vol = trailing_volatility(returns, vol_window, expanding=expanding_vol)
    lag_sign = np.sign(np.roll(returns, 1))
    lag_sign[0] = 0.0
    high_vol = vol >= vol_threshold
    neg = lag_sign < 0
    regime = np.zeros(len(returns), dtype=int)
    regime[high_vol & ~neg] = 2
    regime[high_vol & neg] = 3
    regime[~high_vol & neg] = 1
    return regime


def soft_regime_weights(
    returns: np.ndarray,
    *,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
    temperature: float = 5.0,
) -> np.ndarray:
    """Soft weights over four regimes (past-only vol and lagged sign)."""
    vol = trailing_volatility(returns, vol_window, expanding=False)
    lag = np.roll(returns, 1)
    lag[0] = 0.0
    n = len(returns)
    w = np.zeros((n, 4), dtype=float)
    for i in range(n):
        v = vol[i] if np.isfinite(vol[i]) else vol_threshold
        p_high = 1.0 / (1.0 + np.exp(-temperature * (v - vol_threshold)))
        p_neg = 1.0 / (1.0 + np.exp(-temperature * (-lag[i])))
        w[i, 0] = (1 - p_high) * (1 - p_neg)
        w[i, 1] = (1 - p_high) * p_neg
        w[i, 2] = p_high * (1 - p_neg)
        w[i, 3] = p_high * p_neg
        s = w[i].sum()
        if s > 0:
            w[i] /= s
        else:
            w[i, 0] = 1.0
    return w
