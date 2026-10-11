"""Prespecified classical, learned, and direct-distribution controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import SequenceEncoder
from .probability import GaussianComponent
from .reservoir import FixedReservoir


@dataclass(frozen=True)
class ClassicalForecast:
    mean: np.ndarray
    covariance: np.ndarray


class ZeroHistoricalRisk:
    def fit(self, returns: np.ndarray) -> ZeroHistoricalRisk:
        values = np.asarray(returns, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("returns must be (dates, assets)")
        self.covariance_ = np.cov(values, rowvar=False)
        return self

    def predict(self) -> ClassicalForecast:
        return ClassicalForecast(np.zeros(len(self.covariance_)), self.covariance_.copy())


class RollingMeanShrinkage:
    def __init__(self, shrinkage: float = 0.25) -> None:
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must lie in [0, 1]")
        self.shrinkage = shrinkage

    def fit(self, returns: np.ndarray) -> RollingMeanShrinkage:
        values = np.asarray(returns, dtype=float)
        self.mean_ = values.mean(axis=0)
        sample = np.cov(values, rowvar=False)
        diagonal = np.diag(np.diag(sample))
        self.covariance_ = (1 - self.shrinkage) * sample + self.shrinkage * diagonal
        return self

    def predict(self) -> ClassicalForecast:
        return ClassicalForecast(self.mean_.copy(), self.covariance_.copy())


class RidgeForecast:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = alpha

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> RidgeForecast:
        x, y = np.asarray(features, dtype=float), np.asarray(outcomes, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("Ridge arrays do not align")
        augmented = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(augmented.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coefficients_ = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
        residual = y - augmented @ self.coefficients_
        covariance = np.cov(residual, rowvar=False)
        self.covariance_ = np.atleast_2d(covariance) + np.eye(y.shape[1]) * 1e-8
        return self

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=float)
        augmented = np.column_stack([np.ones(len(x)), x])
        return augmented @ self.coefficients_, self.covariance_.copy()


class ReservoirRidgeForecast:
    def __init__(
        self, input_dim: int, state_dim: int = 32, alpha: float = 1.0, seed: int = 11
    ) -> None:
        self.reservoir = FixedReservoir(input_dim, state_dim, leak=0.2, seed=seed)
        self.ridge = RidgeForecast(alpha)

    def fit(self, sequences: torch.Tensor, outcomes: np.ndarray) -> ReservoirRidgeForecast:
        with torch.no_grad():
            state = self.reservoir(sequences).cpu().numpy()
        self.ridge.fit(state, outcomes)
        return self

    def predict(self, sequences: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            state = self.reservoir(sequences).cpu().numpy()
        return self.ridge.predict(state)


class SupervisedGRU(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, assets: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, assets * 2)
        self.assets = assets

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, state = self.gru(sequence)
        raw = self.head(state[-1]).reshape(len(sequence), self.assets, 2)
        return raw[..., 0], torch.nn.functional.softplus(raw[..., 1]) + 1e-4


class DLinear(nn.Module):
    def __init__(self, context_length: int, assets: int) -> None:
        super().__init__()
        self.trend = nn.Linear(context_length, 1)
        self.remainder = nn.Linear(context_length, 1)
        self.assets = assets

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        if returns.shape[-2:] != (self.assets, self.trend.in_features):
            raise ValueError("DLinear expects (batch, assets, context)")
        trend = torch.nn.functional.avg_pool1d(
            returns.reshape(-1, 1, returns.shape[-1]), kernel_size=5, stride=1, padding=2
        ).reshape_as(returns)
        return (self.trend(trend) + self.remainder(returns - trend)).squeeze(-1)


class PatchTransformer(nn.Module):
    def __init__(self, input_dim: int = 18, model_dim: int = 32, patch: int = 16) -> None:
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(input_dim * patch, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, 4, 64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(model_dim, 2)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        usable = sequence.shape[1] // self.patch * self.patch
        patches = sequence[:, -usable:].reshape(len(sequence), -1, self.patch * sequence.shape[-1])
        state = self.encoder(self.embed(patches)).mean(dim=1)
        raw = self.head(state)
        return raw[:, 0], torch.nn.functional.softplus(raw[:, 1]) + 1e-4


class TemporalJEPAAdaptation(nn.Module):
    """Dataset adaptation of TS-JEPA/CF-JEPA, not a benchmark reproduction."""

    def __init__(self, variant: str, input_dim: int = 18, latent_dim: int = 32) -> None:
        super().__init__()
        if variant not in {"ts_jepa", "cf_jepa"}:
            raise ValueError("variant must be ts_jepa or cf_jepa")
        self.variant = variant
        self.context_encoder = SequenceEncoder(input_dim, 64, latent_dim)
        self.target_encoder = SequenceEncoder(input_dim, 64, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 8, 64), nn.GELU(), nn.Linear(64, latent_dim)
        )
        self.horizon_embedding = nn.Embedding(3, 8)
        self.return_head = nn.Linear(latent_dim, 2)
        if variant == "cf_jepa":
            for parameter in self.context_encoder.parameters():
                parameter.requires_grad_(False)

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, horizon_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        online = self.context_encoder(context)
        with torch.no_grad():
            target = self.target_encoder(future)
        horizon = self.horizon_embedding.weight[horizon_index].expand(len(context), -1)
        prediction = self.predictor(torch.cat([online, horizon], dim=-1))
        raw_return = self.return_head(prediction)
        return prediction, target, torch.nn.functional.softplus(raw_return[:, 1]) + 1e-4


class DirectCorrelatedMixture(nn.Module):
    """Primary mechanism-off control using the same 200-d branch inputs."""

    def __init__(
        self, input_dim: int = 200, hidden_dim: int = 160, branches: int = 3, factors: int = 8
    ) -> None:
        super().__init__()
        self.branches = branches
        self.factors = factors
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 2 + factors),
                )
                for _ in range(branches)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim * branches, 64), nn.GELU(), nn.Linear(64, branches)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self, branch_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[list[GaussianComponent]]]:
        if branch_inputs.ndim != 4 or branch_inputs.shape[2] != self.branches:
            raise ValueError("Expected (batch, assets, branches, features)")
        summaries = branch_inputs.mean(dim=1).flatten(start_dim=1)
        logits = self.gate(summaries)
        components: list[list[GaussianComponent]] = []
        for batch in range(len(branch_inputs)):
            per_batch = []
            for branch, head in enumerate(self.heads):
                raw = head(branch_inputs[batch, :, branch])
                per_batch.append(
                    GaussianComponent(
                        raw[:, 0],
                        torch.nn.functional.softplus(raw[:, 1]).square() + 1e-6,
                        0.5 * torch.tanh(raw[:, 2:]),
                    )
                )
            components.append(per_batch)
        return logits, components


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_match_error(candidate: nn.Module, control: nn.Module) -> float:
    left, right = trainable_parameter_count(candidate), trainable_parameter_count(control)
    return abs(left - right) / max(left, 1)
