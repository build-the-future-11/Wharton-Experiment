"""Causal Hann-window Fourier channels with an explicit real-coordinate map."""

from __future__ import annotations

import math

import torch
from torch import nn


def real_fourier_matrix(
    length: int, bins: int, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    if length < 2 or not 0 < bins <= length // 2:
        raise ValueError("Invalid Fourier matrix dimensions")
    positions = torch.arange(length, dtype=dtype)
    hann = 0.5 * (1.0 - torch.cos(2.0 * math.pi * positions / (length - 1)))
    rows = []
    for frequency in range(1, bins + 1):
        phase = 2.0 * math.pi * frequency * positions / length
        rows.append(hann * torch.cos(phase) / math.sqrt(length))
        rows.append(-hann * torch.sin(phase) / math.sqrt(length))
    return torch.stack(rows)


class CausalSpectralChannels(nn.Module):
    """Return all complex bins plus the three learned 32-d band projections."""

    def __init__(self, length: int = 128, bins: int = 32, latent_dim: int = 32) -> None:
        super().__init__()
        self.length = length
        self.bins = bins
        matrix = real_fourier_matrix(length, bins, dtype=torch.float32)
        self.register_buffer("real_matrix", matrix)
        self.project = nn.Sequential(nn.Linear(2 * bins, 64), nn.GELU(), nn.Linear(64, latent_dim))
        masks = torch.zeros(3, 2 * bins)
        for branch, (start, stop) in enumerate(((1, 4), (5, 16), (17, 32))):
            masks[branch, 2 * (start - 1) : 2 * stop] = 1.0
        self.register_buffer("band_masks", masks)

    def transform(self, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if returns.shape[-1] != self.length:
            raise ValueError(f"Expected context length {self.length}")
        centered = returns - returns.mean(dim=-1, keepdim=True)
        standardized = centered / returns.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        coordinates = torch.einsum("kl,...l->...k", self.real_matrix, standardized)
        complex_bins = torch.complex(coordinates[..., 0::2], coordinates[..., 1::2])
        return coordinates, complex_bins

    def forward(self, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates, complex_bins = self.transform(returns)
        branches = torch.stack(
            [self.project(coordinates * mask) for mask in self.band_masks], dim=-2
        )
        return branches, complex_bins


def finite_difference_gradient_error(
    module: CausalSpectralChannels, values: torch.Tensor, *, epsilon: float = 1e-6
) -> float:
    x = values.detach().to(torch.float64).requires_grad_(True)
    matrix = module.real_matrix.to(torch.float64)
    centered = x - x.mean()
    standardized = centered / x.std(unbiased=False).clamp_min(1e-6)
    transformed = matrix @ standardized
    weights = torch.linspace(0.2, 1.1, len(transformed), dtype=torch.float64)
    loss = (transformed * weights).sum()
    analytic = torch.autograd.grad(loss, x)[0]
    numeric = torch.zeros_like(x)
    for index in range(len(x)):
        plus = x.detach().clone()
        minus = x.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon

        def objective(value: torch.Tensor) -> torch.Tensor:
            value = value - value.mean()
            value = value / value.std(unbiased=False).clamp_min(1e-6)
            return ((matrix @ value) * weights).sum()

        numeric[index] = (objective(plus) - objective(minus)) / (2 * epsilon)
    denominator = torch.maximum(analytic.abs(), numeric.abs()).clamp_min(1e-8)
    return float(((analytic - numeric).abs() / denominator).max())
