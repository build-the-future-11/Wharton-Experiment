"""Pairwise logistic ranker (numpy, vectorized pair updates)."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


class PairwiseLogisticRanker:
    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        lr: float = 0.05,
        l2: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.W1 = rng.normal(scale=0.1, size=(n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr
        self.l2 = l2

    def _score(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def fit_pairs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        date_ids: np.ndarray,
        epochs: int = 80,
        max_pairs: int = 256,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng(0)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        date_ids = np.asarray(date_ids)
        unique_dates = np.unique(date_ids)
        for _ in range(epochs):
            for d in unique_dates:
                mask = date_ids == d
                Xd = X[mask]
                yd = y[mask]
                n = len(yd)
                if n < 2:
                    continue
                n_pairs = min(max_pairs, n * 2)
                i = rng.integers(0, n, size=n_pairs)
                j = rng.integers(0, n, size=n_pairs)
                same = i == j
                j[same] = (j[same] + 1) % n
                better = (yd[i] > yd[j]).astype(float)
                Xi, Xj = Xd[i], Xd[j]
                hi = np.tanh(Xi @ self.W1 + self.b1)
                hj = np.tanh(Xj @ self.W1 + self.b1)
                si = (hi @ self.W2 + self.b2).ravel()
                sj = (hj @ self.W2 + self.b2).ravel()
                p = _sigmoid(si - sj)
                g = (p - better)  # (n_pairs,)
                # dL/dsi = g, dL/dsj = -g
                dW2 = (hi * g[:, None] - hj * g[:, None]).sum(axis=0)
                db2 = float(g.sum() - (-g).sum())  # 2*sum(g) wait: sum(g)+sum(-g)=0 for bias from i and j
                db2 = float(g.sum() + (-g).sum())  # = 0 — correct for equal push
                # Actually bias: dsi/db2=1, dsj/db2=1 → contrib g*1 + (-g)*1 = 0. Skip.
                dhi = g[:, None] * self.W2[:, 0]
                dhj = (-g)[:, None] * self.W2[:, 0]
                dhi *= 1.0 - hi**2
                dhj *= 1.0 - hj**2
                dW1 = Xi.T @ dhi + Xj.T @ dhj
                db1 = dhi.sum(axis=0) + dhj.sum(axis=0)
                self.W2[:, 0] -= self.lr * dW2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            self.W1 *= 1 - self.lr * self.l2
            self.W2 *= 1 - self.lr * self.l2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._score(np.asarray(X, dtype=float))

    def state_dict(self) -> dict:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "lr": self.lr,
            "l2": self.l2,
        }

    def load_state_dict(self, d: dict) -> None:
        self.W1 = d["W1"]
        self.b1 = d["b1"]
        self.W2 = d["W2"]
        self.b2 = d["b2"]
        self.lr = d["lr"]
        self.l2 = d["l2"]
