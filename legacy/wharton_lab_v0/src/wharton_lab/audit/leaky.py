"""Detect deliberately leaky feature/label alignment."""

from __future__ import annotations

from typing import Callable

import numpy as np


class LeakyFeatureError(ValueError):
    """Raised when features contain same-timestep label information."""


def feature_label_max_lag(features: np.ndarray, labels: np.ndarray) -> int:
    """
    Heuristic: if corr(feature[t], label[t]) >> corr(feature[t], label[t-1]),
    treat as zero-lag leak. Returns detected lag (0 = leaky).
    """
    f = np.asarray(features, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    n = min(len(f), len(y))
    if n < 5:
        return 1
    c0 = abs(np.corrcoef(f[:n], y[:n])[0, 1])
    c1 = abs(np.corrcoef(f[1:n], y[: n - 1])[0, 1]) if n > 5 else 0.0
    if c0 > 0.95 and c0 > c1 + 0.5:
        return 0
    return 1


def assert_no_feature_leak(features: np.ndarray, labels: np.ndarray) -> None:
    f = np.asarray(features, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    n = min(len(f), len(y))
    if n >= 3 and np.allclose(f[:n], y[:n], rtol=0, atol=1e-12):
        raise LeakyFeatureError("Features are identical to contemporaneous labels (leak).")
    lag = feature_label_max_lag(features, labels)
    if lag == 0:
        raise LeakyFeatureError("Features appear aligned with contemporaneous labels (leak).")


def assert_features_causal(
    build: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    returns: np.ndarray,
    *,
    probe_rows: tuple[int, ...] | None = None,
    bump: float = 1.0,
) -> None:
    """Structural check: bumping ``returns[t]`` must not change feature rows ``<= t``.

    ``build`` maps a (T, K) returns panel to ``(X, row_t)`` where ``row_t[i]`` is the
    returns row whose value is the label of feature row ``i``.
    """
    rets = np.asarray(returns, dtype=float)
    X0, row_t = build(rets)
    row_t = np.asarray(row_t, dtype=int)
    if probe_rows is None:
        probe_rows = tuple(int(row_t[i]) for i in np.linspace(0, len(row_t) - 1, 5).astype(int))
    for t in probe_rows:
        bumped = rets.copy()
        bumped[t] = bumped[t] + bump
        X1, row_t1 = build(bumped)
        mask = row_t <= t
        if not np.array_equal(row_t, np.asarray(row_t1, dtype=int)) or not np.allclose(
            X0[mask], X1[mask], rtol=0, atol=0
        ):
            raise LeakyFeatureError(f"Feature rows at or before t={t} depend on returns[t] (leak).")
