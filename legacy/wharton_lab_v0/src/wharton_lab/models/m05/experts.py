"""Linear and Gaussian experts with distinct parameterizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ExpertKind = Literal["linear", "gaussian"]


@dataclass
class ExpertSpec:
    expert_id: str
    kind: ExpertKind
    n_features: int


def init_weights(kind: ExpertKind, n_features: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "linear":
        w = rng.normal(scale=0.05, size=n_features + 1)
        return w.astype(np.float64)
    # gaussian: mean weights + log-var bias
    w = rng.normal(scale=0.05, size=n_features + 2)
    return w.astype(np.float64)


def expected_weight_dim(kind: ExpertKind, n_features: int) -> int:
    return n_features + 1 if kind == "linear" else n_features + 2


def predict_expert(kind: ExpertKind, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    need = expected_weight_dim(kind, X.shape[1])
    if weights.size != need:
        # Corrupt / mismatched pack — fall back to zeros bias prediction
        return np.zeros(X.shape[0], dtype=float)
    if kind == "linear":
        w, b = weights[:-1], weights[-1]
        return X @ w + b
    w, b, _logvar = weights[:-2], weights[-2], weights[-1]
    return X @ w + b


def expert_variance(kind: ExpertKind, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).ravel()
    if kind == "linear" or weights.size < 2:
        return np.full(X.shape[0], 1e-4, dtype=float)
    logvar = float(weights[-1])
    return np.full(X.shape[0], np.exp(np.clip(logvar, -10, 10)), dtype=float)


def fit_expert_local(
    kind: ExpertKind,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    steps: int = 5,
) -> np.ndarray:
    """Few-step SGD for smoke / online refinement."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    need = expected_weight_dim(kind, X.shape[1])
    w = np.asarray(weights, dtype=float).ravel().copy()
    if w.size != need:
        rng = np.random.default_rng(0)
        w = init_weights(kind, X.shape[1], rng)
    n = X.shape[0]
    for _ in range(steps):
        pred = predict_expert(kind, w, X)
        err = pred - y
        if kind == "linear":
            grad_w = (2.0 / n) * (X.T @ err)
            grad_b = (2.0 / n) * err.sum()
            w[:-1] -= lr * grad_w
            w[-1] -= lr * grad_b
        else:
            grad_w = (2.0 / n) * (X.T @ err)
            grad_b = (2.0 / n) * err.sum()
            w[:-2] -= lr * grad_w
            w[-2] -= lr * grad_b
            w[-1] -= lr * (2.0 / n) * (np.exp(np.clip(w[-1], -10, 10)) - np.var(err))
    return w
