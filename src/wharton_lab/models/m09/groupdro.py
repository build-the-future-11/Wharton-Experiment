"""GroupDRO-style reweighting over training groups."""

from __future__ import annotations

import numpy as np


def group_losses(residuals: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    losses = np.zeros(n_groups, dtype=float)
    counts = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        mask = group_ids == g
        if mask.any():
            losses[g] = float(np.mean(residuals[mask] ** 2))
            counts[g] = int(mask.sum())
        else:
            losses[g] = 0.0
    return losses


def groupdro_weights(
    group_losses_vec: np.ndarray,
    q: np.ndarray,
    *,
    eta: float = 0.5,
) -> np.ndarray:
    """Exponentiated gradient update on group weights (train-time only)."""
    q = np.asarray(q, dtype=float)
    q = q / (q.sum() + 1e-12)
    losses = np.asarray(group_losses_vec, dtype=float)
    logits = np.log(q + 1e-12) + eta * losses
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()
