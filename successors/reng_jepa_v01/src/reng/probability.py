"""Low-rank Gaussian mixture mathematics and deterministic calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import ndtr


@dataclass(frozen=True)
class GaussianComponent:
    mean: torch.Tensor
    diagonal: torch.Tensor
    factors: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.shape != self.diagonal.shape:
            raise ValueError("Mean and diagonal must share shape")
        if self.factors.shape[:-1] != self.mean.shape:
            raise ValueError("Factor leading dimensions must match mean")
        if torch.any(self.diagonal <= 0):
            raise ValueError("Gaussian diagonal variance must be positive")


def low_rank_gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    diagonal: torch.Tensor,
    factors: torch.Tensor,
    *,
    jitters: tuple[float, ...] = (1e-6, 1e-5, 1e-4),
) -> tuple[torch.Tensor, float]:
    """Woodbury/determinant-lemma log density in float64."""
    value64, mean64 = value.to(torch.float64), mean.to(torch.float64)
    diagonal64, factors64 = diagonal.to(torch.float64), factors.to(torch.float64)
    if value64.shape != mean64.shape or diagonal64.shape != mean64.shape:
        raise ValueError("Value, mean, and diagonal must share shape")
    if factors64.shape[:-1] != mean64.shape:
        raise ValueError("Factor dimensions do not match")
    residual = value64 - mean64
    rank = factors64.shape[-1]
    eye = torch.eye(rank, dtype=torch.float64, device=value.device)
    for jitter in jitters:
        variance = diagonal64 + jitter
        inverse = variance.reciprocal()
        small = eye + factors64.mT @ (inverse.unsqueeze(-1) * factors64)
        chol, info = torch.linalg.cholesky_ex(small)
        if int(info.max()) != 0:
            continue
        projected = factors64.mT @ (inverse * residual)
        correction = torch.cholesky_solve(projected.unsqueeze(-1), chol).squeeze(-1)
        quadratic = (residual.square() * inverse).sum() - (projected * correction).sum()
        logdet = variance.log().sum() + 2.0 * torch.diagonal(chol).log().sum()
        dimension = value64.numel()
        return -0.5 * (dimension * math.log(2 * math.pi) + logdet + quadratic), jitter
    raise FloatingPointError("Low-rank Gaussian factorization failed at maximum jitter")


def dense_gaussian_log_prob(
    value: torch.Tensor, mean: torch.Tensor, diagonal: torch.Tensor, factors: torch.Tensor
) -> torch.Tensor:
    covariance = (
        torch.diag(diagonal.to(torch.float64))
        + factors.to(torch.float64) @ factors.to(torch.float64).mT
    )
    distribution = torch.distributions.MultivariateNormal(
        mean.to(torch.float64), covariance_matrix=covariance
    )
    return distribution.log_prob(value.to(torch.float64))


def mixture_log_prob(
    value: torch.Tensor,
    logits: torch.Tensor,
    components: list[GaussianComponent],
    *,
    covariance_multiplier: float = 1.0,
) -> tuple[torch.Tensor, float]:
    if logits.shape[-1] != len(components):
        raise ValueError("Mixture weight/component mismatch")
    log_weights = torch.log_softmax(logits, dim=-1)
    terms = []
    largest_jitter = 0.0
    for index, component in enumerate(components):
        log_prob, jitter = low_rank_gaussian_log_prob(
            value,
            component.mean,
            component.diagonal * covariance_multiplier,
            component.factors * math.sqrt(covariance_multiplier),
        )
        largest_jitter = max(largest_jitter, jitter)
        terms.append(log_weights[index].to(torch.float64) + log_prob)
    return torch.logsumexp(torch.stack(terms), dim=0), largest_jitter


def marginal_mixture_cdf(
    x: np.ndarray | float,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)[..., None]
    return np.sum(weights * ndtr((x_array - means) / np.sqrt(variances)), axis=-1)


def marginal_mixture_quantile(
    quantile: float,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    *,
    tolerance: float = 1e-7,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("Quantile must lie in (0, 1)")
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    means, variances = np.asarray(means, dtype=float), np.asarray(variances, dtype=float)
    if weights.ndim != 1 or means.shape != weights.shape or variances.shape != weights.shape:
        raise ValueError("Marginal mixture arrays must be one-dimensional and aligned")
    if (variances <= 0).any():
        raise ValueError("Variances must be positive")
    spread = np.sqrt(variances)
    lower = float(np.min(means - 12 * spread))
    upper = float(np.max(means + 12 * spread))
    while upper - lower > tolerance:
        midpoint = 0.5 * (lower + upper)
        if marginal_mixture_cdf(midpoint, weights, means, variances) < quantile:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def sample_market_mixture(
    weights: torch.Tensor,
    components: list[GaussianComponent],
    count: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Sample one branch identity per complete cross-asset market scenario."""
    if count <= 0:
        raise ValueError("Scenario count must be positive")
    generator = torch.Generator(device=weights.device).manual_seed(seed)
    branch = torch.multinomial(weights, count, replacement=True, generator=generator)
    dimension = components[0].mean.numel()
    scenarios = torch.empty(count, dimension, dtype=components[0].mean.dtype, device=weights.device)
    for index, component in enumerate(components):
        chosen = torch.nonzero(branch == index, as_tuple=False).flatten()
        if len(chosen) == 0:
            continue
        epsilon = torch.randn(
            len(chosen),
            dimension,
            generator=generator,
            device=weights.device,
            dtype=scenarios.dtype,
        )
        factor_noise = torch.randn(
            len(chosen),
            component.factors.shape[-1],
            generator=generator,
            device=weights.device,
            dtype=scenarios.dtype,
        )
        draws = (
            component.mean
            + epsilon * component.diagonal.sqrt()
            + factor_noise @ component.factors.mT
        )
        scenarios[chosen] = draws
    return scenarios


def calibrate_grid(
    component_logits: np.ndarray,
    component_means: np.ndarray,
    component_variances: np.ndarray,
    outcomes: np.ndarray,
    *,
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0),
    multipliers: tuple[float, ...] = (0.75, 1.0, 1.5, 2.0, 3.0),
) -> tuple[float, float, float]:
    """Select by marginal return NLL; deterministic ties prefer (1, 1)."""
    logits = np.asarray(component_logits, dtype=float)
    means = np.asarray(component_means, dtype=float)
    variances = np.asarray(component_variances, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    candidates: list[tuple[float, float, float, float]] = []
    for temperature in temperatures:
        shifted = logits / temperature
        shifted -= shifted.max(axis=-1, keepdims=True)
        weights = np.exp(shifted)
        weights /= weights.sum(axis=-1, keepdims=True)
        for multiplier in multipliers:
            density = np.sum(
                weights
                * np.exp(-0.5 * (y[..., None] - means) ** 2 / (variances * multiplier))
                / np.sqrt(2 * np.pi * variances * multiplier),
                axis=-1,
            )
            nll = float(-np.log(np.clip(density, 1e-300, None)).mean())
            tie_distance = abs(math.log(temperature)) + abs(math.log(multiplier))
            candidates.append((nll, tie_distance, temperature, multiplier))
    nll, _, temperature, multiplier = min(candidates)
    return float(temperature), float(multiplier), float(nll)
