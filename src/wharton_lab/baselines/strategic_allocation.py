"""Static strategic allocation baselines."""

from __future__ import annotations

import numpy as np


def equal_weight_targets(n_assets: int) -> np.ndarray:
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")
    return np.full(n_assets, 1.0 / n_assets)
