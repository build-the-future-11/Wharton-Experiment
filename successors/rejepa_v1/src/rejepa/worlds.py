"""Stochastic latent-world simulation with development-learned interpreters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LatentWorldOutput:
    trajectories: torch.Tensor
    interpretations: torch.Tensor
    gate: torch.Tensor
    combined: torch.Tensor


class LatentWorldModel(nn.Module):
    def __init__(self, latent_dim: int, worlds: int, horizon: int, interpreters: int) -> None:
        super().__init__()
        if min(latent_dim, worlds, horizon, interpreters) < 1:
            raise ValueError("Latent-world dimensions must be positive")
        self.latent_dim = latent_dim
        self.worlds = worlds
        self.horizon = horizon
        self.interpreters_count = interpreters
        self.transition = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.SiLU(),
            nn.Linear(2 * latent_dim, 2 * latent_dim),
        )
        summary_dim = 3 * latent_dim
        self.interpreters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(summary_dim, 2 * latent_dim), nn.SiLU(), nn.Linear(2 * latent_dim, latent_dim)
                )
                for _ in range(interpreters)
            ]
        )
        self.gate = nn.Linear(latent_dim, interpreters)

    def forward(
        self,
        context: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> LatentWorldOutput:
        if context.ndim != 2 or context.shape[-1] != self.latent_dim:
            raise ValueError("Expected context [batch, latent_dim]")
        state = context.unsqueeze(1).expand(-1, self.worlds, -1)
        paths = []
        for _ in range(self.horizon):
            mean_delta, raw_scale = self.transition(state).chunk(2, dim=-1)
            scale = F.softplus(raw_scale).clamp(max=1.0) + 1e-4
            noise = torch.randn(state.shape, generator=generator, device=state.device, dtype=state.dtype)
            state = F.layer_norm(state + mean_delta + scale * noise, (self.latent_dim,))
            paths.append(state)
        trajectories = torch.stack(paths, dim=2)
        endpoint = trajectories[:, :, -1, :]
        summary = torch.cat(
            [
                trajectories.mean(dim=(1, 2)),
                endpoint.mean(dim=1),
                endpoint.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        interpretations = torch.stack([expert(summary) for expert in self.interpreters], dim=1)
        gate = torch.softmax(self.gate(context), dim=-1)
        combined = (gate.unsqueeze(-1) * interpretations).sum(dim=1)
        return LatentWorldOutput(trajectories, interpretations, gate, combined)
