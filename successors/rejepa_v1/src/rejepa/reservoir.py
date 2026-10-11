"""Fixed contractive reservoir for causal fading-memory state."""

from __future__ import annotations

import math

import torch
from torch import nn


class StableReservoir(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        spectral_norm: float = 0.85,
        leak: float = 0.35,
        seed: int = 11,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim) < 1:
            raise ValueError("Reservoir dimensions must be positive")
        if not 0.0 < spectral_norm < 1.0:
            raise ValueError("spectral_norm must be in (0, 1)")
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        recurrent = torch.randn(hidden_dim, hidden_dim, generator=generator) / math.sqrt(hidden_dim)
        recurrent = recurrent * (spectral_norm / torch.linalg.matrix_norm(recurrent, ord=2))
        input_weight = torch.randn(hidden_dim, input_dim, generator=generator) / math.sqrt(input_dim)
        bias = torch.randn(hidden_dim, generator=generator) * 0.05
        self.register_buffer("recurrent", recurrent)
        self.register_buffer("input_weight", input_weight)
        self.register_buffer("bias", bias)
        self.hidden_dim = hidden_dim
        self.spectral_norm = spectral_norm
        self.leak = leak

    @property
    def contraction_bound(self) -> float:
        return (1.0 - self.leak) + self.leak * self.spectral_norm

    def forward(self, sequence: torch.Tensor, initial: torch.Tensor | None = None) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_weight.shape[1]:
            raise ValueError("Expected [batch, time, input_dim]")
        if not torch.isfinite(sequence).all():
            raise ValueError("Reservoir input must be finite")
        batch = sequence.shape[0]
        state = (
            torch.zeros(batch, self.hidden_dim, device=sequence.device, dtype=sequence.dtype)
            if initial is None
            else initial
        )
        if state.shape != (batch, self.hidden_dim):
            raise ValueError("Initial reservoir state has the wrong shape")
        states = []
        recurrent = self.recurrent.to(dtype=sequence.dtype)
        input_weight = self.input_weight.to(dtype=sequence.dtype)
        bias = self.bias.to(dtype=sequence.dtype)
        for step in sequence.unbind(dim=1):
            proposal = torch.tanh(step @ input_weight.T + state @ recurrent.T + bias)
            state = (1.0 - self.leak) * state + self.leak * proposal
            states.append(state)
        return torch.stack(states, dim=1)
