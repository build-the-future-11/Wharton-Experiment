"""Finite-grid latent SPDE with exact conditional linear Gaussian transitions.

dZ = (D_row L_row + D_col L_col - kappa I) Z dt + f(z0) dt + sigma dW.
The forcing is frozen at the input for each transition. This is a dimensionless
latent prior, not an identified physical law or a fitted financial-price SDE.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def ring_laplacian(n):
    eye = torch.eye(n, dtype=torch.float64)
    return 2 * eye - eye.roll(1, 0) - eye.roll(-1, 0)


class StochasticField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        rows, columns = config.field_rows, config.latent // config.field_rows
        lr, qr = torch.linalg.eigh(ring_laplacian(rows))
        lc, qc = torch.linalg.eigh(ring_laplacian(columns))
        # Positive Laplacian spectrum; lattice spacing is one latent coordinate.
        self.register_buffer("basis", torch.kron(qr.contiguous(), qc.contiguous()).float())
        self.register_buffer("row_spectrum", lr.clamp_min(0).repeat_interleave(columns).float())
        self.register_buffer("col_spectrum", lc.clamp_min(0).repeat(rows).float())
        self.raw_diffusion = nn.Parameter(torch.full((2,), -2.0))
        self.raw_damping = nn.Parameter(torch.tensor(-2.0))
        self.raw_noise = nn.Parameter(torch.tensor(-1.0))
        self.forcing = nn.Linear(config.latent, config.latent)

    def coefficients(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite latent time increment required")
        diffusion = F.softplus(self.raw_diffusion) + 1e-5
        rate = (
            diffusion[0] * self.row_spectrum
            + diffusion[1] * self.col_spectrum
            + F.softplus(self.raw_damping)
            + 1e-4
        )
        decay = torch.exp(-rate * dt)
        integral = -torch.expm1(-rate * dt) / rate
        variance = (F.softplus(self.raw_noise) + 1e-4).square() * (
            -torch.expm1(-2 * rate * dt) / (2 * rate)
        )
        return decay, integral, variance

    def forward(self, z, dt=1.0):
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            decay, integral, variance = self.coefficients(dt)
            force = torch.tanh(self.forcing(z))
            spectral = (z @ self.basis) * decay + (force @ self.basis) * integral
            return spectral @ self.basis.T, variance

    def latent_nll(self, target, mean, variance):
        with torch.autocast(device_type=target.device.type, enabled=False):
            residual = (target.float() - mean.float()) @ self.basis
            return (
                0.5 * (math.log(2 * math.pi) + variance.log() + residual.square() / variance).mean()
            )

    def sample(self, z, dt=1.0, count=1):
        if type(count) is not int or count < 1:
            raise ValueError("positive sample count required")
        mean, variance = self(z, dt)
        noise = torch.randn(count, *mean.shape, device=mean.device) * variance.sqrt()
        return mean[None] + noise @ self.basis.T
