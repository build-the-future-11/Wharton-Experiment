"""Detect deliberately leaky feature/label alignment."""

from __future__ import annotations

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
