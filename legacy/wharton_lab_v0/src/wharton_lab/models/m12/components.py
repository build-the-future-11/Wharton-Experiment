"""Composable submodules for M12."""

from __future__ import annotations

import numpy as np


class DynamicGraphState:
    """kNN correlation graph features (causal trailing window)."""

    def __init__(self, k: int = 3, window: int = 20):
        self.k = k
        self.window = window

    def encode(self, returns: np.ndarray) -> np.ndarray:
        returns = np.asarray(returns, dtype=float)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        n, d = returns.shape
        out = np.zeros((n, d))
        for i in range(n):
            sl = returns[max(0, i - self.window) : i]
            if len(sl) < 2:
                out[i] = returns[i]
                continue
            corr = np.corrcoef(sl.T)
            if not np.all(np.isfinite(corr)):
                out[i] = returns[i]
                continue
            np.fill_diagonal(corr, -np.inf)
            for j in range(d):
                nbrs = np.argsort(corr[j])[-self.k :]
                out[i, j] = float(np.mean(returns[i, nbrs]))
        return out


class EpisodicMemory:
    def __init__(self, slots: int = 8, dim: int = 4, seed: int = 0):
        self.slots = slots
        self.dim = dim
        self.memory = np.zeros((slots, dim))
        self.rng = np.random.default_rng(seed)

    def read(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=float).ravel()[: self.dim]
        if q.size < self.dim:
            q = np.pad(q, (0, self.dim - q.size))
        sims = self.memory @ q
        w = np.exp(sims - sims.max())
        w /= w.sum() + 1e-12
        return w @ self.memory

    def write(self, key: np.ndarray) -> None:
        k = np.asarray(key, dtype=float).ravel()[: self.dim]
        if k.size < self.dim:
            k = np.pad(k, (0, self.dim - k.size))
        idx = int(np.argmin(np.linalg.norm(self.memory - k, axis=1)))
        self.memory[idx] = 0.9 * self.memory[idx] + 0.1 * k


class LatentWorldModel:
    def __init__(self, dim: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(scale=0.1, size=(dim, dim))
        self.dim = dim

    def step(self, z: np.ndarray, obs: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).ravel()[: self.dim]
        o = np.asarray(obs, dtype=float).ravel()[: self.dim]
        if z.size < self.dim:
            z = np.pad(z, (0, self.dim - z.size))
        if o.size < self.dim:
            o = np.pad(o, (0, self.dim - o.size))
        return np.tanh(self.A @ z + 0.1 * o)
