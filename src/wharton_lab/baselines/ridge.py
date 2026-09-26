"""Ridge regression forecast."""

from __future__ import annotations

import numpy as np


def ridge_forecast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_test = np.asarray(X_test, dtype=float)
    n_feat = X_train.shape[1]
    w = np.linalg.solve(
        X_train.T @ X_train + alpha * np.eye(n_feat),
        X_train.T @ y_train,
    )
    return X_test @ w
