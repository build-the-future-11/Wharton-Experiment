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
