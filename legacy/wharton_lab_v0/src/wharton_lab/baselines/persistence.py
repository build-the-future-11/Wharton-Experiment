"""Last-value persistence forecast."""

from __future__ import annotations

import numpy as np


def persistence_forecast(y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
    y_history = np.asarray(y_history, dtype=float).ravel()
    if len(y_history) == 0:
        return np.zeros(horizon)
    last = y_history[-1]
    return np.full(horizon, last)
