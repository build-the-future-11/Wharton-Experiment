"""Torch encoders and predictors for FI-JEPA."""

from __future__ import annotations

from wharton_lab.models.torch_utils import require_torch

torch = require_torch()
nn = torch.nn


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, h = self.gru(x)
        return self.proj(h.squeeze(0))


class TargetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h.squeeze(0))


class HorizonPredictor(nn.Module):
    def __init__(self, latent_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


class AuxHead(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.return_head = nn.Linear(latent_dim, 1)
        self.vol_head = nn.Linear(latent_dim, 1)

    def forward(self, z):
        ret = self.return_head(z).squeeze(-1)
        vol = torch.nn.functional.softplus(self.vol_head(z)).squeeze(-1)
        return ret, vol


def ema_update(target: nn.Module, online: nn.Module, m: float) -> None:
    with torch.no_grad():
        for pt, po in zip(target.parameters(), online.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1.0 - m)


def vicreg_loss(z: torch.Tensor, coeff: float = 0.1) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    var_loss = torch.mean(torch.relu(1.0 - std))
    n, d = z.shape
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / d
    return coeff * (var_loss + cov_loss)
