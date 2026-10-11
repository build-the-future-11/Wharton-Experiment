"""Fixed multiscale reservoirs normalized by exact largest singular value."""

from __future__ import annotations

import math

import torch
from torch import nn


class FixedReservoir(nn.Module):
    def __init__(
        self, input_dim: int, state_dim: int, leak: float, seed: int, norm: float = 0.9
    ) -> None:
        super().__init__()
        if not 0 < leak <= 1 or not 0 < norm <= 0.9:
            raise ValueError("Invalid contractive reservoir settings")
        generator = torch.Generator().manual_seed(seed)
        recurrent = torch.randn(state_dim, state_dim, generator=generator) / math.sqrt(state_dim)
        recurrent = recurrent * (norm / torch.linalg.svdvals(recurrent).max())
        input_weight = (
            torch.rand(state_dim, input_dim, generator=generator) * 0.4 - 0.2
        ) / math.sqrt(input_dim)
        self.register_buffer("recurrent", recurrent)
        self.register_buffer("input_weight", input_weight)
        self.leak = float(leak)
        self.norm = float(norm)

    @property
    def contraction_bound(self) -> float:
        return 1.0 - self.leak + self.leak * self.norm

    def forward(self, sequence: torch.Tensor, initial: torch.Tensor | None = None) -> torch.Tensor:
        if sequence.ndim < 3:
            raise ValueError("Expected (..., length, features)")
        leading = sequence.shape[:-2]
        flattened = sequence.reshape(-1, sequence.shape[-2], sequence.shape[-1])
        if initial is None:
            state = torch.zeros(
                flattened.shape[0],
                self.recurrent.shape[0],
                device=sequence.device,
                dtype=sequence.dtype,
            )
        else:
            state = initial.reshape(flattened.shape[0], -1).to(sequence)
        for step in flattened.unbind(dim=1):
            proposal = torch.tanh(step @ self.input_weight.T + state @ self.recurrent.T)
            state = (1.0 - self.leak) * state + self.leak * proposal
        return state.reshape(*leading, -1)


class MultiscaleReservoir(nn.Module):
    def __init__(self, input_dim: int = 18, state_dim: int = 32, seed: int = 11) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                FixedReservoir(input_dim, state_dim, leak, seed + index)
                for index, leak in enumerate((0.05, 0.20, 0.80))
            ]
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.stack([branch(sequence) for branch in self.branches], dim=-2)
