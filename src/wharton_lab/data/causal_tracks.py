"""Repaired (causal) feature builders: feature row t uses only returns strictly before t.

The legacy loaders in ``orchestration/datasets.py`` are kept unchanged so historical
receipts stay reproducible; they are non-causal (see protocol/DECISIONS.md D-050/D-051).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wharton_lab.data.synthetic import SyntheticWorldConfig, WorldKind, generate_world

FEATURE_NAMES: tuple[str, ...] = ("lag1", "mean_lag5", "vol_lag20", "xs_mean_lag1")


@dataclass(frozen=True)
class CausalPanel:
    X: np.ndarray
    y: np.ndarray
    row_t: np.ndarray
    source: str


def causal_return_features(
    returns: np.ndarray,
    *,
    target_col: int = 0,
    vol_window: int = 20,
    mean_window: int = 5,
    source: str = "returns",
) -> CausalPanel:
    """Label is ``returns[t, target_col]``; every feature is computed from rows < t."""
    rets = np.asarray(returns, dtype=float)
    if rets.ndim != 2:
        raise ValueError(f"returns must be 2D, got {rets.shape}")
    r = rets[:, target_col]
    start = max(vol_window, mean_window, 1)
    if len(r) <= start:
        raise ValueError("series too short for causal features")
    rows_x = []
    rows_t = []
    for t in range(start, len(r)):
        rows_x.append(
            [
                r[t - 1],
                r[t - mean_window : t].mean(),
                r[t - vol_window : t].std(ddof=1),
                rets[t - 1].mean(),
            ]
        )
        rows_t.append(t)
    row_t = np.asarray(rows_t, dtype=int)
    return CausalPanel(np.asarray(rows_x, dtype=float), r[row_t], row_t, source)


def causal_synthetic_panel(seed: int, n_steps: int, n_assets: int = 5) -> CausalPanel:
    world = generate_world(
        SyntheticWorldConfig(kind=WorldKind.SIGNAL, n_steps=n_steps, seed=seed, n_assets=n_assets)
    )
    return causal_return_features(world.returns, source=f"synthetic_signal_seed_{seed}")


def legacy_synthetic_panel(seed: int, n_steps: int, n_assets: int = 5) -> CausalPanel:
    """Legacy Track C features ``[latent_t, latent_{t-1}]`` — contemporaneous leak, for audit only."""
    world = generate_world(
        SyntheticWorldConfig(kind=WorldKind.SIGNAL, n_steps=n_steps, seed=seed, n_assets=n_assets)
    )
    return CausalPanel(
        world.features, world.labels, np.arange(len(world.labels)), f"legacy_leaky_seed_{seed}"
    )
