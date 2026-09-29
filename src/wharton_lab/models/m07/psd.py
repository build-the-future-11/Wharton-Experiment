"""PSD covariance parameterization."""

from __future__ import annotations

from wharton_lab.models.torch_utils import require_torch

torch = require_torch()
nn = torch.nn


class PSDCovHead(nn.Module):
    """Cholesky-factor covariance: Σ = L L^T."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.tril_idx = torch.tril_indices(n, n)
        self.params = nn.Parameter(torch.zeros(n * (n + 1) // 2))

    def forward(self) -> torch.Tensor:
        L = torch.zeros(self.n, self.n, device=self.params.device, dtype=self.params.dtype)
        L[self.tril_idx[0], self.tril_idx[1]] = self.params
        diag = torch.arange(self.n)
        L[diag, diag] = torch.nn.functional.softplus(L[diag, diag]) + 1e-3
        return L @ L.T


def softplus_eigen_cov(factors: torch.Tensor, log_eigs: torch.Tensor) -> torch.Tensor:
    """Alternative: Σ = U diag(softplus(λ)) U^T with orthogonal U from QR."""
    q, _ = torch.linalg.qr(factors)
    eigs = torch.nn.functional.softplus(log_eigs) + 1e-4
    return q @ torch.diag(eigs) @ q.T


class ConditionalPSDCovHead(nn.Module):
    """A distinct positive-definite covariance for each encoded context."""
    def __init__(self, n: int, latent_dim: int) -> None:
        super().__init__()
        self.n = n
        self.linear = nn.Linear(latent_dim, n * (n + 1) // 2)
        self.register_buffer("row", torch.tril_indices(n, n)[0])
        self.register_buffer("col", torch.tril_indices(n, n)[1])
        self.register_buffer("eye", torch.eye(n))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        values = self.linear(context)
        L = values.new_zeros((len(context), self.n, self.n))
        L[:, self.row, self.col] = values
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        positive = torch.nn.functional.softplus(diag) + 1e-4
        L = L * (1 - self.eye) + torch.diag_embed(positive)
        return L @ L.transpose(-1, -2)
