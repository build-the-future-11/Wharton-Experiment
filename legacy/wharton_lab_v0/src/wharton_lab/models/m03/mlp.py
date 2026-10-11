"""Small numpy MLP for beta(characteristics)."""

from __future__ import annotations

import numpy as np


class LoadingMLP:
    """Maps characteristics (d,) to factor loadings (K,)."""

    def __init__(
        self,
        n_char: int,
        n_factors: int,
        hidden: int = 32,
        lr: float = 0.01,
        random_state: int = 42,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.K = n_factors
        self.W1 = rng.normal(scale=0.1, size=(n_char, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=0.1, size=(hidden, n_factors))
        self.b2 = np.zeros(n_factors)
        self.lr = lr

    def forward(self, C: np.ndarray) -> np.ndarray:
        C = np.asarray(C, dtype=float)
        if C.ndim == 1:
            C = C[None, :]
        h = np.tanh(C @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def fit(
        self,
        C: np.ndarray,
        targets: np.ndarray,
        epochs: int = 120,
    ) -> None:
        C = np.asarray(C, dtype=float)
        targets = np.asarray(targets, dtype=float)
        n = C.shape[0]
        for _ in range(epochs):
            pred = self.forward(C)
            err = pred - targets
            grad_W2 = (self._last_h(C).T @ err) / n
            grad_b2 = err.mean(axis=0)
            dh = err @ self.W2.T * (1 - np.tanh(C @ self.W1 + self.b1) ** 2)
            grad_W1 = (C.T @ dh) / n
            grad_b1 = dh.mean(axis=0)
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

    def _last_h(self, C: np.ndarray) -> np.ndarray:
        return np.tanh(C @ self.W1 + self.b1)

    def state_dict(self) -> dict:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "K": self.K,
            "lr": self.lr,
        }

    def load_state_dict(self, d: dict) -> None:
        self.W1 = d["W1"]
        self.b1 = d["b1"]
        self.W2 = d["W2"]
        self.b2 = d["b2"]
        self.K = d["K"]
        self.lr = d["lr"]
