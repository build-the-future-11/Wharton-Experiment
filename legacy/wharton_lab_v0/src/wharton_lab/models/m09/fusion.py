"""Explicit fusion of fast and slow representation streams."""

from __future__ import annotations

import numpy as np


class FusionModule:
    """Two-stream fusion: linear fast + slow then gated combination."""

    def __init__(self, fast_dim: int, slow_dim: int, hidden: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W_fast = rng.normal(scale=0.1, size=(fast_dim, hidden))
        self.W_slow = rng.normal(scale=0.1, size=(slow_dim, hidden))
        self.gate_w = rng.normal(scale=0.1, size=(hidden, 1))
        self.bias = np.zeros(hidden)

    def transform(self, fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
        fast = np.asarray(fast, dtype=float)
        slow = np.asarray(slow, dtype=float)
        h_f = np.tanh(fast @ self.W_fast)
        h_s = np.tanh(slow @ self.W_slow)
        gate = 1.0 / (1.0 + np.exp(-(h_f + h_s) @ self.gate_w))
        fused = gate * h_f + (1.0 - gate) * h_s + self.bias
        return fused

    def state_dict(self) -> dict:
        return {
            "W_fast": self.W_fast,
            "W_slow": self.W_slow,
            "gate_w": self.gate_w,
            "bias": self.bias,
        }

    def load_state_dict(self, d: dict) -> None:
        self.W_fast = np.asarray(d["W_fast"])
        self.W_slow = np.asarray(d["W_slow"])
        self.gate_w = np.asarray(d["gate_w"])
        self.bias = np.asarray(d["bias"])
