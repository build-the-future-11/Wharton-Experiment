"""Deterministic generated panels for engineering-only verification."""

from __future__ import annotations

import numpy as np

from frem_s12.data import ReturnPanel


def _months(start_year: int, start_month: int, count: int) -> np.ndarray:
    values = []
    year, month = start_year, start_month
    for _ in range(count):
        values.append(year * 100 + month)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return np.asarray(values, dtype=np.int64)


def generated_panel(*, family: str = "generated", months: int = 360, seed: int = 811) -> ReturnPanel:
    """Create a causal regime-switching six-dimensional panel for tests."""
    if months < 180:
        raise ValueError("generated panel requires at least 180 months")
    rng = np.random.default_rng(seed)
    loadings = np.asarray(
        [
            [1.0, -0.8, 0.3],
            [1.0, -0.2, 0.1],
            [1.0, 0.7, -0.2],
            [0.8, -0.7, 0.2],
            [0.8, -0.1, 0.0],
            [0.8, 0.8, -0.3],
        ],
        dtype=np.float64,
    )
    factors = np.zeros((months, 3), dtype=np.float64)
    returns = np.zeros((months, 6), dtype=np.float64)
    regime = 0
    for index in range(1, months):
        if rng.random() < 0.035:
            regime = 1 - regime
        persistence = 0.55 if regime == 0 else -0.15
        scale = 0.012 if regime == 0 else 0.030
        factors[index] = persistence * factors[index - 1] + rng.normal(0, scale, 3)
        seasonal = 0.002 * np.sin(2 * np.pi * index / 24)
        returns[index] = loadings @ factors[index] + seasonal + rng.normal(0, scale * 0.35, 6)
    return ReturnPanel(family, _months(1980, 1, months), returns)
