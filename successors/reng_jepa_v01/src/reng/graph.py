"""Signed trailing-correlation graph used only for representation messages."""

from __future__ import annotations

import torch
from torch import nn


def signed_correlation_adjacency(
    returns: torch.Tensor,
    observed: torch.Tensor,
    *,
    min_pairs: int = 50,
    threshold: float = 0.2,
    scale: float = 0.8,
) -> torch.Tensor:
    """Build row-|A|-normalized adjacency for (batch, assets, lookback)."""
    if returns.shape != observed.shape or returns.ndim != 3:
        raise ValueError("returns and observed must be (batch, assets, lookback)")
    if observed.dtype is not torch.bool:
        raise ValueError("observed must be a boolean tensor")

    # Pairwise sufficient statistics avoid the Python batch/asset loops without
    # materializing a (batch, asset, asset, lookback) overlap tensor.
    _, assets, _ = returns.shape
    mask = observed.to(returns.dtype)
    values = torch.where(
        observed, returns, torch.zeros((), dtype=returns.dtype, device=returns.device)
    )

    def transpose(tensor):
        return tensor.transpose(-1, -2)

    pair_count = mask @ transpose(mask)
    pair_sum = values @ transpose(mask)
    pair_square_sum = values.square() @ transpose(mask)
    pair_cross_sum = values @ transpose(values)
    safe_count = pair_count.clamp_min(1.0)
    covariance = pair_cross_sum - pair_sum * transpose(pair_sum) / safe_count
    variance = (pair_square_sum - pair_sum.square() / safe_count).clamp_min(0.0)
    denominator = torch.sqrt(variance * transpose(variance)).clamp_min(1e-12)
    correlation = covariance / denominator
    eligible = (pair_count >= min_pairs) & (correlation.abs() >= threshold)
    adjacency = torch.where(eligible, scale * correlation, torch.zeros_like(correlation))
    identity = torch.eye(assets, dtype=returns.dtype, device=returns.device).expand_as(adjacency)
    adjacency = adjacency * (1.0 - identity) + identity
    return adjacency / adjacency.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)


class GraphContext(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, latent: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.shape != (latent.shape[0], latent.shape[1], latent.shape[1]):
            raise ValueError("Graph and latent asset dimensions differ")
        message = torch.einsum("bij,bjd->bid", adjacency, latent)
        return torch.nn.functional.gelu(self.projection(message))
