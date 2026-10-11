"""Stress-conditioned latent-world simulation and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn

from .worlds import LatentWorldModel, LatentWorldOutput


@dataclass(frozen=True)
class StressCondition:
    volatility: float = 0.0
    liquidity: float = 0.0
    interest_rate: float = 0.0
    currency: float = 0.0
    equity_drawdown: float = 0.0
    correlation_breakdown: float = 0.0
    regime_transition: float = 0.0
    contagion: float = 0.0

    def tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = torch.tensor(
            [
                self.volatility,
                self.liquidity,
                self.interest_rate,
                self.currency,
                self.equity_drawdown,
                self.correlation_breakdown,
                self.regime_transition,
                self.contagion,
            ],
            device=device,
            dtype=dtype,
        )
        if not torch.isfinite(values).all():
            raise ValueError("Stress condition must be finite")
        return values


class CounterfactualLatentSimulator(nn.Module):
    def __init__(self, latent_dim: int, worlds: int = 8, horizon: int = 5, interpreters: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_projection = nn.Sequential(
            nn.Linear(8, latent_dim), nn.Tanh(), nn.Linear(latent_dim, latent_dim)
        )
        self.worlds = LatentWorldModel(latent_dim, worlds, horizon, interpreters)

    def forward(
        self,
        latent: torch.Tensor,
        condition: StressCondition | torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> LatentWorldOutput:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim or not torch.isfinite(latent).all():
            raise ValueError("Expected finite latent [batch, latent_dim]")
        raw = (
            condition.tensor(device=latent.device, dtype=latent.dtype)
            if isinstance(condition, StressCondition)
            else condition.to(device=latent.device, dtype=latent.dtype)
        )
        if raw.ndim == 1:
            raw = raw.unsqueeze(0).expand(latent.shape[0], -1)
        if raw.shape != (latent.shape[0], 8) or not torch.isfinite(raw).all():
            raise ValueError("Condition must have eight values per batch row")
        conditioned = latent + self.condition_projection(raw)
        return self.worlds(conditioned, generator=generator)


def conditional_coverage(
    observed: np.ndarray,
    scenario_draws: np.ndarray,
    *,
    confidence: float = 0.90,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    actual = np.asarray(observed, dtype=float)
    draws = np.asarray(scenario_draws, dtype=float)
    if draws.ndim != 2 or actual.shape != (draws.shape[0],):
        raise ValueError("Observed values and scenario rows must align")
    if not np.isfinite(actual).all() or not np.isfinite(draws).all() or not 0 < confidence < 1:
        raise ValueError("Invalid coverage inputs")
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(draws, [tail, 1 - tail], axis=1)
    covered = (actual >= lower) & (actual <= upper)
    result = {"overall": float(covered.mean()), "expected": confidence}
    if groups is not None:
        labels = np.asarray(groups)
        if labels.shape != actual.shape:
            raise ValueError("groups must align with observed values")
        for label in np.unique(labels):
            result[f"group:{label}"] = float(covered[labels == label].mean())
    return result


def tail_event_diagnostics(
    observed: np.ndarray, scenario_draws: np.ndarray, *, threshold: float
) -> dict[str, float]:
    actual = np.asarray(observed, dtype=float)
    draws = np.asarray(scenario_draws, dtype=float)
    if draws.ndim != 2 or actual.shape != (draws.shape[0],):
        raise ValueError("Observed values and scenario rows must align")
    predicted = (draws <= threshold).mean(axis=1)
    realized = (actual <= threshold).astype(float)
    return {
        "predicted_tail_rate": float(predicted.mean()),
        "realized_tail_rate": float(realized.mean()),
        "tail_brier": float(np.mean((predicted - realized) ** 2)),
    }


def synthetic_real_distinguishability(
    real_latents: np.ndarray, synthetic_latents: np.ndarray, *, seed: int = 11
) -> float:
    real = np.asarray(real_latents, dtype=float)
    synthetic = np.asarray(synthetic_latents, dtype=float)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("Real and synthetic latents must share feature width")
    if min(len(real), len(synthetic)) < 4 or not np.isfinite(real).all() or not np.isfinite(synthetic).all():
        raise ValueError("At least four finite samples per class are required")
    features = np.vstack([real, synthetic])
    labels = np.concatenate([np.zeros(len(real)), np.ones(len(synthetic))])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    split = max(2, int(0.7 * len(labels)))
    train, test = order[:split], order[split:]
    if len(np.unique(labels[test])) < 2:
        test = order[-max(4, len(labels) // 3) :]
        train = np.setdiff1d(order, test, assume_unique=False)
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(features[train], labels[train])
    probability = model.predict_proba(features[test])[:, 1]
    return float(roc_auc_score(labels[test], probability))
