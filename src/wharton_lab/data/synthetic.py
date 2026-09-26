"""Track C controlled synthetic worlds for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class WorldKind(str, Enum):
    SIGNAL = "signal"
    COV_ROTATION = "cov_rotation"
    REGIMES = "regimes"
    FAT_TAILS = "fat_tails"
    NULL = "null"


@dataclass
class SyntheticWorldConfig:
    n_steps: int = 252
    n_assets: int = 5
    seed: int = 0
    kind: WorldKind = WorldKind.SIGNAL
    signal_strength: float = 0.15
    regime_switch_prob: float = 0.02
    fat_tail_df: float = 4.0


@dataclass
class SyntheticWorld:
    returns: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    regime_ids: np.ndarray
    meta: dict


def _iid_gaussian(n_steps: int, n_assets: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(scale=0.01, size=(n_steps, n_assets))


def generate_world(cfg: SyntheticWorldConfig) -> SyntheticWorld:
    rng = np.random.default_rng(cfg.seed)
    n, k = cfg.n_steps, cfg.n_assets
    regime_ids = np.zeros(n, dtype=int)

    if cfg.kind == WorldKind.NULL:
        rets = _iid_gaussian(n, k, rng)
        feat = np.column_stack([rets.mean(axis=1), rets.std(axis=1)])
        labels = rets[:, 0]
        return SyntheticWorld(rets, feat, labels, regime_ids, {"kind": cfg.kind.value})

    if cfg.kind == WorldKind.SIGNAL:
        latent = rng.normal(size=n)
        rets = np.zeros((n, k))
        for j in range(k):
            beta = cfg.signal_strength * (1 + 0.1 * j)
            noise = rng.normal(scale=0.01, size=n)
            rets[:, j] = beta * latent + noise
        feat = np.column_stack([latent, np.roll(latent, 1)])
        feat[0, 1] = 0.0
        labels = rets[:, 0]
        return SyntheticWorld(rets, feat, labels, regime_ids, {"kind": cfg.kind.value})

    if cfg.kind == WorldKind.COV_ROTATION:
        rets = np.zeros((n, k))
        chol_a = np.linalg.cholesky(_random_spd(k, rng, scale=0.015**2))
        chol_b = np.linalg.cholesky(_random_spd(k, rng, scale=0.025**2))
        for i in range(n):
            chol = chol_a if (i // 63) % 2 == 0 else chol_b
            z = rng.normal(size=k)
            rets[i] = chol @ z
        feat = np.column_stack([rets.mean(axis=1), np.linalg.norm(rets, axis=1)])
        labels = rets[:, 0]
        return SyntheticWorld(rets, feat, labels, regime_ids, {"kind": cfg.kind.value})

    if cfg.kind == WorldKind.REGIMES:
        rets = np.zeros((n, k))
        state = 0
        for i in range(n):
            if i > 0 and rng.random() < cfg.regime_switch_prob:
                state = 1 - state
            regime_ids[i] = state
            scale = 0.008 if state == 0 else 0.02
            rets[i] = rng.normal(scale=scale, size=k)
        feat = np.column_stack([regime_ids.astype(float), rets.mean(axis=1)])
        labels = rets[:, 0]
        return SyntheticWorld(rets, feat, labels, regime_ids, {"kind": cfg.kind.value})

    if cfg.kind == WorldKind.FAT_TAILS:
        df = max(cfg.fat_tail_df, 2.1)
        scale = np.sqrt((df - 2) / df) * 0.01
        rets = rng.standard_t(df, size=(n, k)) * scale
        feat = np.column_stack([rets.mean(axis=1), np.abs(rets).mean(axis=1)])
        labels = rets[:, 0]
        return SyntheticWorld(rets, feat, labels, regime_ids, {"kind": cfg.kind.value})

    raise ValueError(f"Unknown world kind: {cfg.kind}")


def _random_spd(dim: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    a = rng.normal(size=(dim, dim))
    cov = a @ a.T / dim
    cov += np.eye(dim) * scale
    return cov


def panel_from_world(
    world: SyntheticWorld,
    *,
    feature_lag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal features from lagged returns; label is next-step first asset return."""
    rets = world.returns
    n = len(rets)
    lagged = np.zeros_like(rets)
    lagged[feature_lag:] = rets[:-feature_lag]
    X = np.hstack([world.features, lagged])
    y = np.roll(rets[:, 0], -1)
    y[-1] = np.nan
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]
