"""Myopic equal-weight controller."""

from __future__ import annotations

import numpy as np


def myopic_equal_weights(n_assets: int) -> np.ndarray:
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    return np.full(n_assets, 1.0 / n_assets)
