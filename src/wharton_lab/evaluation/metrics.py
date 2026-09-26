"""Forecast and decision metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, quantile: float) -> float:
    y = np.asarray(y, dtype=float).ravel()
    q_pred = np.asarray(q_pred, dtype=float).ravel()
    err = y - q_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1) * err)))


def spearman_ic(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return float("nan")
    corr, _ = stats.spearmanr(y[mask], pred[mask])
    return float(corr)


def energy_score(samples: np.ndarray, y_true: np.ndarray) -> float:
    """
    Multivariate energy score (Gneiting & Raftery style).

    samples: (n_samples, n_draws, d) or (n_draws, d)
    y_true: (n_samples, d) or (d,)
    """
    samples = np.asarray(samples, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if samples.ndim == 2:
        samples = samples[np.newaxis, ...]
    if y_true.ndim == 1:
        y_true = y_true[np.newaxis, ...]
    n, m, d = samples.shape
    term1 = np.mean(np.linalg.norm(samples - y_true[:, np.newaxis, :], axis=-1), axis=1)
    if m < 2:
        term2 = np.zeros(n)
    else:
        pairwise = []
        for i in range(m):
            for j in range(i + 1, m):
                pairwise.append(np.linalg.norm(samples[:, i, :] - samples[:, j, :], axis=-1))
        term2 = np.mean(np.stack(pairwise, axis=0), axis=0)
    es = term1 - 0.5 * term2
    return float(np.mean(es))


def projector_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    projector: np.ndarray,
) -> float:
    """Error after linear projection (e.g. factor alignment)."""
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    P = np.asarray(projector, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    pred_p = P @ (np.linalg.pinv(P) @ predicted)
    act_p = P @ (np.linalg.pinv(P) @ actual)
    return float(np.mean((pred_p - act_p) ** 2))


def funding_shortfall(
    cash_paths: np.ndarray,
    obligations: np.ndarray,
    *,
    alpha: float = 0.95,
) -> float:
    """
    CVaR-style funding shortfall: max(obligation - cash, 0) aggregated.

    cash_paths: (n_scenarios, n_steps)
    obligations: (n_steps,) broadcastable
    """
    cash_paths = np.asarray(cash_paths, dtype=float)
    obligations = np.asarray(obligations, dtype=float)
    shortfall = np.maximum(obligations - cash_paths, 0.0)
    per_scenario = shortfall.sum(axis=-1)
    q = np.quantile(per_scenario, alpha)
    tail = per_scenario[per_scenario >= q]
    if len(tail) == 0:
        return float(q)
    return float(np.mean(tail))
