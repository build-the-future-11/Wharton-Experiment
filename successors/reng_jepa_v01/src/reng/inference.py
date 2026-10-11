"""Frozen analytic marginals, growth shrinkage, and market scenario sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import MemberOutput
from .probability import GaussianComponent, marginal_mixture_quantile, sample_market_mixture


@dataclass(frozen=True)
class AnalyticForecast:
    quantiles: np.ndarray
    quantile_levels: np.ndarray
    log_return_mean: np.ndarray
    simple_return_mean: np.ndarray
    growth_mean: np.ndarray
    disagreement_covariance: np.ndarray
    growth_world_gap: np.ndarray


def analytic_forecast(
    outputs: tuple[MemberOutput, ...],
    *,
    batch_index: int = 0,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
) -> AnalyticForecast:
    if not outputs:
        raise ValueError("At least one ensemble member is required")
    assets = outputs[0].return_mean.shape[-1]
    weights, means, variances = [], [], []
    for output in outputs:
        branch_weights = torch.softmax(output.gate_logits[batch_index], dim=-1).cpu().numpy()
        for branch in range(len(branch_weights)):
            weights.append(float(branch_weights[branch]) / len(outputs))
            means.append(output.return_mean[batch_index, branch].detach().cpu().numpy())
            factor_variance = output.return_factors[batch_index, branch].square().sum(dim=-1)
            variances.append(
                (output.return_diagonal[batch_index, branch] + factor_variance)
                .detach()
                .cpu()
                .numpy()
            )
    weights_array = np.asarray(weights)
    means_array = np.asarray(means)
    variances_array = np.asarray(variances)
    q_values = np.empty((assets, len(quantiles)), dtype=float)
    for asset in range(assets):
        for q_index, level in enumerate(quantiles):
            q_values[asset, q_index] = marginal_mixture_quantile(
                level, weights_array, means_array[:, asset], variances_array[:, asset]
            )
    log_mean = np.sum(weights_array[:, None] * means_array, axis=0)
    simple_mean = np.sum(
        weights_array[:, None] * (np.exp(means_array + 0.5 * variances_array) - 1.0), axis=0
    )
    growth = np.stack(
        [output.growth_mean[batch_index].detach().cpu().numpy() for output in outputs]
    )
    growth_mean = growth.mean(axis=0)
    covariance = np.atleast_2d(np.cov(growth, rowvar=False, ddof=1))
    if len(outputs) == 1:
        covariance = np.zeros((assets, assets), dtype=float)
    return AnalyticForecast(
        quantiles=q_values,
        quantile_levels=np.asarray(quantiles),
        log_return_mean=log_mean,
        simple_return_mean=simple_mean,
        growth_mean=growth_mean,
        disagreement_covariance=covariance,
        growth_world_gap=np.abs(growth_mean - simple_mean),
    )


def calibrated_growth_signal(
    forecast: AnalyticForecast,
    calibration_uncertainty_p90: float,
    *,
    coverage_90: float,
    representation_collapsed: bool,
    reference_mean: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    reference = np.asarray(reference_mean, dtype=float)
    if reference.shape != forecast.growth_mean.shape or calibration_uncertainty_p90 <= 0:
        raise ValueError("Invalid growth calibration contract")
    current_uncertainty = float(np.mean(np.diag(forecast.disagreement_covariance)))
    fallback = coverage_90 < 0.80 or representation_collapsed
    beta = (
        0.0 if fallback else 0.5 / (1.0 + (current_uncertainty / calibration_uncertainty_p90) ** 2)
    )
    signal = reference + beta * (forecast.growth_mean - reference)
    return signal, beta, fallback


def sample_ensemble_scenarios(
    outputs: tuple[MemberOutput, ...],
    *,
    batch_index: int = 0,
    count_per_member: int = 1024,
    seed: int = 11,
) -> np.ndarray:
    draws = []
    for member_index, output in enumerate(outputs):
        weights = torch.softmax(output.gate_logits[batch_index], dim=-1)
        components = [
            GaussianComponent(
                output.return_mean[batch_index, branch],
                output.return_diagonal[batch_index, branch],
                output.return_factors[batch_index, branch],
            )
            for branch in range(output.return_mean.shape[1])
        ]
        draws.append(
            sample_market_mixture(weights, components, count_per_member, seed=seed + member_index)
        )
    log_returns = torch.cat(draws, dim=0).detach().cpu().numpy()
    return np.expm1(log_returns)


def empirical_block_scenarios(
    daily_simple_returns: np.ndarray,
    *,
    horizon: int,
    count: int = 1024,
    expected_block: int = 20,
    seed: int = 11,
) -> np.ndarray:
    history = np.asarray(daily_simple_returns, dtype=float)
    if history.ndim != 2 or len(history) < max(horizon, expected_block):
        raise ValueError("Insufficient synchronized fitting history")
    if np.any(history <= -1) or not np.isfinite(history).all():
        raise ValueError("Empirical scenario history is invalid")
    rng = np.random.default_rng(seed)
    scenarios = np.empty((count, history.shape[1]), dtype=float)
    restart = 1.0 / expected_block
    for scenario in range(count):
        indices = np.empty(horizon, dtype=int)
        indices[0] = rng.integers(len(history))
        for step in range(1, horizon):
            if rng.random() < restart:
                indices[step] = rng.integers(len(history))
            else:
                indices[step] = (indices[step - 1] + 1) % len(history)
        scenarios[scenario] = np.prod(1.0 + history[indices], axis=0) - 1.0
    return scenarios
