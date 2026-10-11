"""Reservoir-Engram JEPA with latent imagination and dual objectives."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.stats import t as scipy_student_t
from torch import nn

from .engram import SpectralEngramMemory
from .reservoir import StableReservoir
from .spectral import CausalFourierBank
from .worlds import LatentWorldModel, LatentWorldOutput


@dataclass(frozen=True)
class REJEPAConfig:
    input_dim: int
    reservoir_dim: int = 32
    latent_dim: int = 24
    spectral_bands: int = 4
    worlds: int = 8
    world_horizon: int = 5
    interpreters: int = 3
    reservoir_spectral_norm: float = 0.85
    reservoir_leak: float = 0.35
    student_df: float = 5.0
    ema_decay: float = 0.99
    use_world_model: bool = True
    use_engram: bool = True
    use_fourier_key: bool = True
    separate_heads: bool = True
    use_reservoir: bool = True
    use_fourier: bool = True
    use_risk_head: bool = True
    adaptive_gate: bool = True
    max_horizon: int = 20
    downside_quantile_level: float = 0.05

    def __post_init__(self) -> None:
        if (
            min(
                self.input_dim,
                self.reservoir_dim,
                self.latent_dim,
                self.spectral_bands,
                self.worlds,
                self.world_horizon,
                self.interpreters,
                self.max_horizon,
            )
            < 1
        ):
            raise ValueError("All dimensions must be positive")
        if not 2.0 < self.student_df:
            raise ValueError("Student-t degrees of freedom must exceed two")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if not 0.0 < self.downside_quantile_level < 0.5:
            raise ValueError("downside_quantile_level must be in (0, 0.5)")


@dataclass
class REJEPAOutput:
    context_latent: torch.Tensor
    target_latent: torch.Tensor | None
    predicted_latent: torch.Tensor
    return_location: torch.Tensor
    return_scale: torch.Tensor
    loss_probability: torch.Tensor
    downside_magnitude: torch.Tensor
    risk_volatility: torch.Tensor
    downside_quantile: torch.Tensor
    drawdown_exposure: torch.Tensor
    world_output: LatentWorldOutput
    retrieved_engram: torch.Tensor
    retrieved_identifiers: tuple[int, ...]


class REJEPA(nn.Module):
    def __init__(self, config: REJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.spectral = CausalFourierBank(config.input_dim, config.spectral_bands)
        self.reservoir = StableReservoir(
            config.input_dim,
            config.reservoir_dim,
            config.reservoir_spectral_norm,
            config.reservoir_leak,
        )
        encoder_input = config.input_dim + config.reservoir_dim + self.spectral.output_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(encoder_input, 2 * config.latent_dim),
            nn.LayerNorm(2 * config.latent_dim),
            nn.SiLU(),
            nn.Linear(2 * config.latent_dim, config.latent_dim),
        )
        self.target_encoder = deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.world_model = LatentWorldModel(
            config.latent_dim, config.worlds, config.world_horizon, config.interpreters
        )
        self.horizon_embedding = nn.Embedding(config.max_horizon + 1, config.latent_dim)
        self.key_projection = nn.Linear(config.latent_dim + self.spectral.output_dim, config.latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * config.latent_dim, 2 * config.latent_dim),
            nn.LayerNorm(2 * config.latent_dim),
            nn.SiLU(),
            nn.Linear(2 * config.latent_dim, config.latent_dim),
        )
        self.growth_expert = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim), nn.SiLU(), nn.Linear(config.latent_dim, 2)
        )
        self.protection_expert = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim), nn.SiLU(), nn.Linear(config.latent_dim, 4)
        )
        self.joint_expert = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim), nn.SiLU(), nn.Linear(config.latent_dim, 6)
        )
        quantile = float(scipy_student_t.ppf(config.downside_quantile_level, df=config.student_df))
        if not math.isfinite(quantile):
            raise ValueError("Student-t downside quantile is not finite")
        self.register_buffer(
            "_downside_quantile_multiplier",
            torch.tensor(quantile, dtype=torch.float64),
            persistent=False,
        )

    def _encode(self, sequence: torch.Tensor, encoder: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[-1] != self.config.input_dim:
            raise ValueError("Expected sequence [batch, time, input_dim]")
        if sequence.shape[1] < 2:
            raise ValueError("Sequences require at least two time steps")
        if not torch.isfinite(sequence).all():
            raise ValueError("Sequence contains non-finite values")
        reservoir = (
            self.reservoir(sequence)[:, -1, :]
            if self.config.use_reservoir
            else torch.zeros(
                sequence.shape[0],
                self.config.reservoir_dim,
                device=sequence.device,
                dtype=sequence.dtype,
            )
        )
        spectral = (
            self.spectral(sequence)
            if self.config.use_fourier
            else torch.zeros(
                sequence.shape[0],
                self.spectral.output_dim,
                device=sequence.device,
                dtype=sequence.dtype,
            )
        )
        features = torch.cat([sequence[:, -1, :], reservoir, spectral], dim=-1)
        return encoder(features), spectral

    def forward(
        self,
        context: torch.Tensor,
        future_target: torch.Tensor | None = None,
        memory: SpectralEngramMemory | None = None,
        as_of_ns: int | None = None,
        generator: torch.Generator | None = None,
        horizon: int | torch.Tensor | None = None,
    ) -> REJEPAOutput:
        context_latent, spectral = self._encode(context, self.context_encoder)
        target_latent = None
        if future_target is not None:
            with torch.no_grad():
                target_latent, _ = self._encode(future_target, self.target_encoder)
        if horizon is None:
            horizon_index = torch.full(
                (context_latent.shape[0],),
                self.config.world_horizon,
                device=context_latent.device,
                dtype=torch.long,
            )
        elif isinstance(horizon, int):
            horizon_index = torch.full(
                (context_latent.shape[0],), horizon, device=context_latent.device, dtype=torch.long
            )
        else:
            raw_horizon = horizon.to(device=context_latent.device)
            if not torch.isfinite(raw_horizon).all() or (
                raw_horizon.is_floating_point() and not torch.equal(raw_horizon, raw_horizon.round())
            ):
                raise ValueError("horizon values must be finite integers")
            horizon_index = raw_horizon.to(dtype=torch.long)
        if horizon_index.shape != (context_latent.shape[0],):
            raise ValueError("horizon must be a scalar or one value per batch row")
        if (horizon_index < 1).any() or (horizon_index > self.config.max_horizon).any():
            raise ValueError("horizon is outside the configured range")
        conditioned_latent = context_latent + self.horizon_embedding(horizon_index)
        if self.config.use_world_model:
            world_output = self.world_model(conditioned_latent, generator=generator)
            if not self.config.adaptive_gate:
                uniform_gate = torch.full_like(world_output.gate, 1.0 / self.config.interpreters)
                world_output = LatentWorldOutput(
                    trajectories=world_output.trajectories,
                    interpretations=world_output.interpretations,
                    gate=uniform_gate,
                    combined=world_output.interpretations.mean(dim=1),
                )
        else:
            batch, latent = conditioned_latent.shape
            trajectories = conditioned_latent[:, None, None, :].expand(
                batch, self.config.worlds, self.config.world_horizon, latent
            )
            interpretations = conditioned_latent[:, None, :].expand(batch, self.config.interpreters, latent)
            gate = torch.full(
                (batch, self.config.interpreters),
                1.0 / self.config.interpreters,
                device=context_latent.device,
                dtype=context_latent.dtype,
            )
            world_output = LatentWorldOutput(
                trajectories=trajectories,
                interpretations=interpretations,
                gate=gate,
                combined=conditioned_latent,
            )
        key_spectral = spectral if self.config.use_fourier_key else torch.zeros_like(spectral)
        query = self.key_projection(torch.cat([conditioned_latent, key_spectral], dim=-1))
        if memory is None or not self.config.use_engram:
            retrieved = torch.zeros_like(context_latent)
            identifiers: tuple[int, ...] = ()
        else:
            if as_of_ns is None:
                raise ValueError("as_of_ns is required when memory is supplied")
            retrieved, identifiers = memory.retrieve(query, as_of_ns)
        predicted_latent = self.fusion(
            torch.cat(
                [
                    conditioned_latent,
                    world_output.combined,
                    retrieved,
                ],
                dim=-1,
            )
        )
        if self.config.separate_heads:
            growth = self.growth_expert(predicted_latent)
            protection = self.protection_expert(predicted_latent)
        else:
            joint = self.joint_expert(predicted_latent)
            growth = joint[:, :2]
            protection = joint[:, 2:]
        return_location = growth[:, 0]
        return_scale = F.softplus(growth[:, 1]) + 1e-4
        if self.config.use_risk_head:
            loss_probability = torch.sigmoid(protection[:, 0])
            downside_magnitude = F.softplus(protection[:, 1])
            risk_volatility = F.softplus(protection[:, 2]) + 1e-4
            drawdown_exposure = F.softplus(protection[:, 3])
        else:
            loss_probability = torch.full_like(return_location, 0.5)
            downside_magnitude = torch.zeros_like(return_location)
            risk_volatility = torch.zeros_like(return_location)
            drawdown_exposure = torch.zeros_like(return_location)
        downside_quantile = return_location + return_scale * self._downside_quantile_multiplier.to(
            device=return_scale.device, dtype=return_scale.dtype
        )
        return REJEPAOutput(
            context_latent=context_latent,
            target_latent=target_latent,
            predicted_latent=predicted_latent,
            return_location=return_location,
            return_scale=return_scale,
            loss_probability=loss_probability,
            downside_magnitude=downside_magnitude,
            risk_volatility=risk_volatility,
            downside_quantile=downside_quantile,
            drawdown_exposure=drawdown_exposure,
            world_output=world_output,
            retrieved_engram=retrieved,
            retrieved_identifiers=identifiers,
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = self.config.ema_decay
        for target, online in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target.mul_(decay).add_(online, alpha=1.0 - decay)


def student_t_nll(
    target: torch.Tensor, location: torch.Tensor, scale: torch.Tensor, df: float
) -> torch.Tensor:
    if not df > 2 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Student-t scoring requires df > 2 and finite positive scale")
    standardized = (target - location) / scale
    constant = (
        torch.lgamma(torch.tensor((df + 1.0) / 2.0, device=target.device, dtype=target.dtype))
        - torch.lgamma(torch.tensor(df / 2.0, device=target.device, dtype=target.dtype))
        - 0.5 * math.log(df * math.pi)
    )
    log_density = constant - torch.log(scale) - ((df + 1.0) / 2.0) * torch.log1p(standardized.square() / df)
    return -log_density


def rejepa_loss(
    output: REJEPAOutput,
    realized_return: torch.Tensor,
    student_df: float = 5.0,
    downside_threshold: float = 0.0,
    latent_weight: float = 1.0,
    growth_weight: float = 1.0,
    protection_weight: float = 1.0,
    diversity_floor: float = 0.05,
) -> dict[str, torch.Tensor]:
    if output.target_latent is None:
        raise ValueError("future_target is required for training loss")
    if realized_return.shape != output.return_location.shape:
        raise ValueError("realized_return has the wrong shape")
    if not torch.isfinite(realized_return).all():
        raise ValueError("realized_return contains non-finite values")
    nll = student_t_nll(realized_return, output.return_location, output.return_scale, student_df).mean()
    downside = (realized_return < downside_threshold).to(realized_return.dtype)
    protection = F.binary_cross_entropy(output.loss_probability, downside)
    realized_magnitude = torch.relu(downside_threshold - realized_return)
    tail = F.smooth_l1_loss(output.downside_magnitude, realized_magnitude)
    volatility = F.smooth_l1_loss(output.risk_volatility, realized_return.abs())
    drawdown = F.smooth_l1_loss(output.drawdown_exposure, torch.relu(-realized_return))
    latent = F.smooth_l1_loss(output.predicted_latent, output.target_latent)
    endpoints = output.world_output.trajectories[:, :, -1, :]
    if endpoints.shape[1] > 1:
        pairwise = torch.cdist(endpoints, endpoints)
        upper = torch.triu(
            torch.ones(
                endpoints.shape[1],
                endpoints.shape[1],
                device=endpoints.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        distances = pairwise[:, upper].mean()
        diversity = torch.relu(torch.as_tensor(diversity_floor, device=distances.device) - distances)
    else:
        diversity = torch.as_tensor(diversity_floor, device=endpoints.device, dtype=endpoints.dtype)
    context_std = output.context_latent.std(dim=0, unbiased=False).mean()
    collapse = torch.relu(torch.as_tensor(0.05, device=context_std.device) - context_std)
    total = (
        latent_weight * latent
        + growth_weight * nll
        + protection_weight * (protection + tail + volatility + drawdown)
        + diversity
        + collapse
    )
    return {
        "total": total,
        "latent": latent,
        "return_nll": nll,
        "loss_classification": protection,
        "downside_magnitude": tail,
        "volatility": volatility,
        "drawdown_exposure": drawdown,
        "world_diversity": diversity,
        "collapse": collapse,
    }
