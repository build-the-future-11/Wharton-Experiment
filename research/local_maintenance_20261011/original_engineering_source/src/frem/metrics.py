"""Distributional scores and explicit, nonoverlapping portfolio accounting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import t as student_t


def _df_array(df: float | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    degrees = np.broadcast_to(np.asarray(df, dtype=float), shape)
    if not np.all(np.isfinite(degrees) & (degrees > 2)):
        raise ValueError("Student-t degrees of freedom must be finite and greater than two")
    return degrees


def student_t_nll(
    actual: np.ndarray, location: np.ndarray, scale: np.ndarray, df: float | np.ndarray = 5.0
) -> np.ndarray:
    y, mu, sigma = np.broadcast_arrays(
        np.asarray(actual, float), np.asarray(location, float), np.asarray(scale, float)
    )
    nu = _df_array(df, y.shape)
    if not np.all(np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)):
        raise ValueError("Student-t score requires finite actual/location and positive finite scale")
    z = (y - mu) / sigma
    return (
        np.log(sigma)
        + 0.5 * np.log(nu * np.pi)
        + gammaln(nu / 2)
        - gammaln((nu + 1) / 2)
        + ((nu + 1) / 2) * np.log1p(z * z / nu)
    )


def student_t_crps(
    actual: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    df: float | np.ndarray = 5.0,
    quadrature_nodes: int = 96,
) -> np.ndarray:
    """Deterministic quadrature of 2*integral(pinball_loss), a CRPS approximation.

    This avoids an incorrect Gaussian formula for a Student-t predictive density.
    Accuracy in very heavy tails should be checked when execution is authorized.
    """
    if quadrature_nodes < 16:
        raise ValueError("Use at least sixteen CRPS quadrature nodes")
    y, mu, sigma = np.broadcast_arrays(
        np.asarray(actual, float), np.asarray(location, float), np.asarray(scale, float)
    )
    _ = student_t_nll(y, mu, sigma, df)
    nu = _df_array(df, y.shape)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    tau = (nodes + 1) / 2
    quantiles = mu[..., None] + sigma[..., None] * student_t.ppf(tau, nu[..., None])
    error = y[..., None] - quantiles
    pinball = np.maximum(tau * error, (tau - 1) * error)
    return np.sum(pinball * weights, axis=-1)


def probability_calibration(probability: np.ndarray, event: np.ndarray, bins: int = 10) -> dict[str, Any]:
    p, event = np.asarray(probability, float), np.asarray(event, float)
    if p.ndim != 1 or p.shape != event.shape or not p.size or bins < 2:
        raise ValueError("Calibration requires equal nonempty arrays and at least two bins")
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)) or not np.all(np.isin(event, [0, 1])):
        raise ValueError("Invalid probabilities or binary events")
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    table = []
    ece = 0.0
    for index in range(bins):
        selected = assignment == index
        count = int(selected.sum())
        predicted = float(p[selected].mean()) if count else None
        observed = float(event[selected].mean()) if count else None
        if count:
            ece += count / len(p) * abs(predicted - observed)
        table.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": predicted,
                "event_frequency": observed,
            }
        )
    return {
        "brier_score": float(np.mean((p - event) ** 2)),
        "ece": float(ece),
        "bins": table,
        "ece_is_not_a_proper_score": True,
    }


def distributional_scores(
    actual: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    df: float | np.ndarray = 5.0,
    interval_levels: Sequence[float] = (0.5, 0.8, 0.95),
) -> pd.DataFrame:
    y, mu, sigma = np.asarray(actual, float), np.asarray(location, float), np.asarray(scale, float)
    if not (y.ndim == mu.ndim == sigma.ndim == 1 and y.shape == mu.shape == sigma.shape):
        raise ValueError("Distributional score arrays must be equally sized one-dimensional arrays")
    nll = student_t_nll(y, mu, sigma, df)
    nu = _df_array(df, y.shape)
    p_up = student_t.cdf(mu / sigma, df=nu)
    event = (y > 0).astype(float)
    result = pd.DataFrame(
        {
            "nll": nll,
            "crps": student_t_crps(y, mu, sigma, nu),
            "squared_error": (y - mu) ** 2,
            "absolute_error": np.abs(y - mu),
            "p_up": p_up,
            "brier_up": (p_up - event) ** 2,
            "pit": student_t.cdf((y - mu) / sigma, df=nu),
            "direction_correct": ((mu > 0) == (y > 0)).astype(float),
            "predictive_std": sigma * np.sqrt(nu / (nu - 2)),
        }
    )
    for level in interval_levels:
        if not 0 < level < 1:
            raise ValueError("Interval levels must be strictly between zero and one")
        alpha = 1 - level
        radius = student_t.ppf(1 - alpha / 2, df=nu) * sigma
        lower, upper = mu - radius, mu + radius
        suffix = f"{level:g}"
        result[f"coverage_{suffix}"] = ((y >= lower) & (y <= upper)).astype(float)
        result[f"interval_width_{suffix}"] = upper - lower
        result[f"interval_score_{suffix}"] = (
            upper - lower + 2 / alpha * (lower - y) * (y < lower) + 2 / alpha * (y - upper) * (y > upper)
        )
    return result


def apply_execution_costs(
    gross_return: np.ndarray,
    entry_cost_fraction: np.ndarray | float,
    exit_cost_fraction: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Costs quoted as fractions of wealth immediately before each cost event.

    net wealth = (1-entry_cost)*(1+gross_return)*(1-exit_cost). Bps must first be
    divided by 10,000; these fractions are not percentages or turnover measures.
    """
    gross, entry, exit_cost = np.broadcast_arrays(
        np.asarray(gross_return, float),
        np.asarray(entry_cost_fraction, float),
        np.asarray(exit_cost_fraction, float),
    )
    if not np.all(np.isfinite(gross) & (gross > -1)):
        raise ValueError("Gross simple returns must be finite and greater than -1")
    if not np.all(
        np.isfinite(entry)
        & np.isfinite(exit_cost)
        & (entry >= 0)
        & (entry < 1)
        & (exit_cost >= 0)
        & (exit_cost < 1)
    ):
        raise ValueError("Execution cost fractions must lie in [0,1)")
    return (1 - entry) * (1 + gross) * (1 - exit_cost) - 1


def portfolio_return_metrics(
    net_returns: np.ndarray, periods_per_year: float, risk_free_returns: np.ndarray | None = None
) -> dict[str, Any]:
    """Metrics for an explicitly regular, disjoint realized return series.

    ``periods_per_year`` is a protocol input. Overlapping H-period prediction
    labels must never be passed here as if they were independent daily returns.
    """
    returns = np.asarray(net_returns, float)
    if returns.ndim != 1 or not len(returns):
        raise ValueError("Portfolio returns must be a nonempty one-dimensional series")
    if not np.all(np.isfinite(returns) & (returns > -1)) or periods_per_year <= 0:
        raise ValueError("Returns must be finite and >-1; periods_per_year must be positive")
    rf = np.zeros_like(returns) if risk_free_returns is None else np.asarray(risk_free_returns, float)
    if rf.shape != returns.shape or not np.isfinite(rf).all():
        raise ValueError("Risk-free returns must be finite and aligned")
    wealth = np.cumprod(1 + returns)
    path = np.r_[1.0, wealth]
    peak = np.maximum.accumulate(path)
    drawdown = path / peak - 1
    excess = returns - rf
    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else np.nan
    sharpe = (
        float(np.mean(excess) / std * np.sqrt(periods_per_year)) if np.isfinite(std) and std > 0 else None
    )
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    years = len(returns) / periods_per_year
    log_growth = float(np.log1p(returns).sum())
    cagr = float(np.expm1(log_growth / years))
    return {
        "periods": len(returns),
        "periods_per_year": float(periods_per_year),
        "total_return": float(wealth[-1] - 1),
        "geometric_annual_return": cagr,
        "annualized_volatility": float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
        if len(returns) > 1
        else None,
        "annualized_sharpe": sharpe,
        "annualized_sortino": float(np.mean(excess) / downside * np.sqrt(periods_per_year))
        if downside > 0
        else None,
        "max_drawdown": float(drawdown.min()),
        "mean_log_return": float(np.log1p(returns).mean()),
        "worst_return": float(returns.min()),
        "best_return": float(returns.max()),
        "positive_fraction": float((returns > 0).mean()),
        "annualization_assumption": "equally spaced nonoverlapping periods; serial dependence remains",
    }
