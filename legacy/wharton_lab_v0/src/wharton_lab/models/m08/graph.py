"""Trailing shrinkage-correlation graphs (history only)."""

from __future__ import annotations

import numpy as np


def shrinkage_correlation(returns: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    """
    returns: (time, nodes) — build correlation from history only.
    Ledoit-style diagonal shrinkage toward identity.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be (time, nodes)")
    t, n = x.shape
    if t < 2:
        return np.eye(n)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(t - 1, 1)
    std = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    target = np.eye(n)
    return (1.0 - shrinkage) * corr + shrinkage * target


def adjacency_from_correlation(corr: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    A = np.abs(corr.copy())
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    # row-normalize for message passing
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-12] = 1.0
    return A / row_sum


def shuffle_edges_control(adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute off-diagonal edges as negative control."""
    A = adj.copy()
    n = A.shape[0]
    flat = A[~np.eye(n, dtype=bool)]
    perm = rng.permutation(flat.size)
    shuffled = flat[perm]
    B = np.zeros_like(A)
    B[~np.eye(n, dtype=bool)] = shuffled
    row_sum = B.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-12] = 1.0
    return B / row_sum
