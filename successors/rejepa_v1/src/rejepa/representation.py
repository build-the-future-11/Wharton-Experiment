"""JEPA representation-collapse and downstream-probe diagnostics."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def latent_diagnostics(latents: torch.Tensor, *, collapse_floor: float = 0.05) -> dict[str, float]:
    if latents.ndim != 2 or latents.shape[0] < 2 or not torch.isfinite(latents).all():
        raise ValueError("Finite latent matrix with at least two rows is required")
    centered = latents - latents.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = latents.var(dim=0, unbiased=False)
    normalized = singular / singular.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(normalized * torch.log(normalized.clamp_min(1e-12))).sum())
    covariance = centered.T @ centered / latents.shape[0]
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return {
        "mean_feature_variance": float(variance.mean()),
        "minimum_feature_variance": float(variance.min()),
        "effective_rank": float(effective_rank),
        "off_diagonal_covariance_rms": float(torch.sqrt(off_diagonal.square().mean())),
        "collapsed_fraction": float((variance < collapse_floor**2).float().mean()),
    }


def downstream_ridge_probe(
    train_latent: np.ndarray,
    train_target: np.ndarray,
    test_latent: np.ndarray,
    test_target: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict[str, float]:
    x_train = np.asarray(train_latent, dtype=float)
    y_train = np.asarray(train_target, dtype=float)
    x_test = np.asarray(test_latent, dtype=float)
    y_test = np.asarray(test_target, dtype=float)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train and test latent widths must match")
    if y_train.shape != (len(x_train),) or y_test.shape != (len(x_test),):
        raise ValueError("Probe targets must align")
    model = Ridge(alpha=alpha).fit(x_train, y_train)
    prediction = model.predict(x_test)
    return {
        "mse": float(np.mean((prediction - y_test) ** 2)),
        "r2": float(r2_score(y_test, prediction)),
    }
