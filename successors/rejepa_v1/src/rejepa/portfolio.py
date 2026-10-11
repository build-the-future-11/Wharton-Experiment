"""Research-only international allocation with separate growth and protection signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class InternationalInstrument:
    asset_id: str
    market: str
    currency: str
    eligible: bool
    point_in_time_verified: bool
    max_weight: float


@dataclass(frozen=True)
class AllocationPolicy:
    base_currency: str = "USD"
    min_cash: float = 0.05
    max_single_security: float = 0.20
    max_non_base_currency: float = 0.35
    variance_aversion: float = 2.0
    loss_aversion: float = 2.0


@dataclass(frozen=True)
class ConstrainedInstrument:
    asset_id: str
    sector: str
    country: str
    currency: str
    eligible: bool
    point_in_time_verified: bool
    max_weight: float
    minimum_average_daily_value: float = 0.0


@dataclass(frozen=True)
class ScenarioAllocationPolicy:
    min_cash: float = 0.05
    max_single_security: float = 0.20
    max_sector: float = 0.35
    max_country: float = 0.50
    max_non_base_currency: float = 0.35
    base_currency: str = "USD"
    cvar_alpha: float = 0.95
    cvar_aversion: float = 2.0
    turnover_penalty: float = 0.002
    uncertainty_penalty: float = 0.5
    maximum_loss_proxy: float = 0.25
    stale_after_days: int = 7

    def __post_init__(self) -> None:
        if not 0 <= self.min_cash < 1 or not 0 < self.cvar_alpha < 1:
            raise ValueError("Invalid cash or CVaR configuration")
        caps = (self.max_single_security, self.max_sector, self.max_country)
        if min(caps) <= 0 or max(caps) > 1:
            raise ValueError("Security, sector and country caps must be positive")
        if not 0 <= self.max_non_base_currency <= 1:
            raise ValueError("Non-base-currency cap must be nonnegative")
        penalties = (self.cvar_aversion, self.turnover_penalty, self.uncertainty_penalty)
        if not np.isfinite((*caps, self.max_non_base_currency, *penalties, self.maximum_loss_proxy)).all():
            raise ValueError("Allocation policy values must be finite")
        if min(penalties) < 0:
            raise ValueError("Penalties must be nonnegative")
        if not 0 <= self.maximum_loss_proxy <= 1 or self.stale_after_days < 0:
            raise ValueError("Invalid loss proxy or staleness limit")


def objective_weights(
    year: int, reserve_funded_ratio: float, reserve_year: int = 2033
) -> tuple[float, float]:
    """Predeclared pressure toward protection as the liability date approaches."""
    if not np.isfinite(reserve_funded_ratio) or reserve_funded_ratio < 0:
        raise ValueError("reserve_funded_ratio must be finite and nonnegative")
    years_left = max(reserve_year - year, 0)
    urgency = 1.0 / (1.0 + years_left)
    shortfall = max(1.0 - reserve_funded_ratio, 0.0)
    protection = min(0.9, 0.35 + 0.35 * urgency + 0.30 * shortfall)
    return 1.0 - protection, protection


def allocate_dual_objective(
    expected_returns: np.ndarray,
    loss_scores: np.ndarray,
    covariance: np.ndarray,
    instruments: list[InternationalInstrument],
    policy: AllocationPolicy,
    year: int,
    reserve_funded_ratio: float,
) -> dict[str, object]:
    expected_returns = np.asarray(expected_returns, dtype=float)
    loss_scores = np.asarray(loss_scores, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    count = len(instruments)
    if expected_returns.shape != (count,) or loss_scores.shape != (count,):
        raise ValueError("Signal arrays must align with instruments")
    if covariance.shape != (count, count) or not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("Covariance must be a symmetric square matrix")
    if (
        not np.isfinite(expected_returns).all()
        or not np.isfinite(loss_scores).all()
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("Allocation inputs must be finite")
    if (loss_scores < 0).any():
        raise ValueError("Loss scores must be nonnegative")
    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError("Covariance must be positive semidefinite")
    if any(not instrument.eligible or not instrument.point_in_time_verified for instrument in instruments):
        return {"status": "blocked", "weights": None, "reason": "unverified or ineligible instrument"}
    growth_weight, protection_weight = objective_weights(year, reserve_funded_ratio)
    risky_budget = 1.0 - policy.min_cash
    non_base = np.asarray(
        [0.0 if instrument.currency == policy.base_currency else 1.0 for instrument in instruments]
    )
    bounds = [(0.0, min(policy.max_single_security, instrument.max_weight)) for instrument in instruments]

    def objective(weights: np.ndarray) -> float:
        growth = expected_returns @ weights
        variance = weights @ covariance @ weights
        loss = loss_scores @ weights
        return float(
            -growth_weight * growth
            + protection_weight * (policy.variance_aversion * variance + policy.loss_aversion * loss)
        )

    constraints = [
        {"type": "ineq", "fun": lambda weights: risky_budget - weights.sum()},
        {"type": "ineq", "fun": lambda weights: policy.max_non_base_currency - non_base @ weights},
    ]
    initial = np.full(count, min(risky_budget / max(count, 1), policy.max_single_security))
    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return {"status": "fallback_cash", "weights": np.zeros(count), "cash": 1.0, "reason": result.message}
    weights = np.clip(result.x, 0.0, None)
    if weights.sum() > risky_budget + 1e-7 or non_base @ weights > policy.max_non_base_currency + 1e-7:
        return {"status": "infeasible", "weights": None, "reason": "post-solve constraint failure"}
    return {
        "status": "optimized_research_only",
        "weights": weights,
        "cash": float(1.0 - weights.sum()),
        "growth_weight": growth_weight,
        "protection_weight": protection_weight,
    }


def optimize_scenario_cvar(
    expected_returns: np.ndarray,
    scenario_returns: np.ndarray,
    uncertainty: np.ndarray,
    current_weights: np.ndarray,
    instruments: list[ConstrainedInstrument],
    policy: ScenarioAllocationPolicy,
    *,
    data_age_days: int,
) -> dict[str, object]:
    """Solve the empirical Rockafellar-Uryasev CVaR allocation as a linear program."""
    mean = np.asarray(expected_returns, dtype=float)
    scenarios = np.asarray(scenario_returns, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    previous = np.asarray(current_weights, dtype=float)
    count = len(instruments)
    if mean.shape != (count,) or uncertainty.shape != (count,) or previous.shape != (count,):
        raise ValueError("Asset vectors must align with instruments")
    if scenarios.ndim != 2 or scenarios.shape[1] != count or scenarios.shape[0] < 2:
        raise ValueError("scenario_returns must be [scenarios, assets]")
    arrays = np.concatenate([mean, uncertainty, previous, scenarios.ravel()])
    if not np.isfinite(arrays).all() or (uncertainty < 0).any() or (previous < 0).any():
        raise ValueError("Optimization inputs must be finite and nonnegative where required")
    if previous.sum() > 1 + 1e-9:
        raise ValueError("current_weights cannot exceed total wealth")
    if data_age_days < 0:
        raise ValueError("data_age_days must be nonnegative")
    if data_age_days > policy.stale_after_days:
        return {"status": "fallback_cash", "weights": np.zeros(count), "cash": 1.0, "reason": "stale data"}
    if any(not item.eligible or not item.point_in_time_verified for item in instruments):
        return {"status": "blocked", "weights": None, "reason": "ineligible or unverified instrument"}
    loss_proxy = np.maximum(-scenarios, 0).max(axis=0)
    if (loss_proxy > 1).any():
        raise ValueError("Scenario losses below -100 percent are invalid for long-only assets")
    asset_bounds = [(0.0, min(policy.max_single_security, item.max_weight)) for item in instruments]
    if any(not np.isfinite(bound[1]) or bound[1] < 0 for bound in asset_bounds):
        raise ValueError("Instrument maximum weights must be finite and nonnegative")
    risky_budget = 1 - policy.min_cash
    scenario_count = scenarios.shape[0]
    eta_index = count
    excess_start = count + 1
    turnover_start = excess_start + scenario_count
    variable_count = turnover_start + count

    objective = np.zeros(variable_count)
    objective[:count] = -mean + policy.uncertainty_penalty * uncertainty
    objective[eta_index] = policy.cvar_aversion
    objective[excess_start:turnover_start] = policy.cvar_aversion / ((1 - policy.cvar_alpha) * scenario_count)
    objective[turnover_start:] = policy.turnover_penalty

    rows: list[np.ndarray] = []
    limits: list[float] = []

    def add_constraint(asset_coefficients: np.ndarray, limit: float) -> None:
        row = np.zeros(variable_count)
        row[:count] = asset_coefficients
        rows.append(row)
        limits.append(float(limit))

    add_constraint(np.ones(count), risky_budget)
    add_constraint(loss_proxy, policy.maximum_loss_proxy)
    for field, cap in (("sector", policy.max_sector), ("country", policy.max_country)):
        labels = sorted({getattr(item, field) for item in instruments})
        for label in labels:
            mask = np.array([1.0 if getattr(item, field) == label else 0.0 for item in instruments])
            add_constraint(mask, cap)
    non_base = np.array([0.0 if item.currency == policy.base_currency else 1.0 for item in instruments])
    add_constraint(non_base, policy.max_non_base_currency)

    for index, scenario in enumerate(scenarios):
        row = np.zeros(variable_count)
        row[:count] = -scenario
        row[eta_index] = -1.0
        row[excess_start + index] = -1.0
        rows.append(row)
        limits.append(0.0)
    for index in range(count):
        positive = np.zeros(variable_count)
        positive[index] = 1.0
        positive[turnover_start + index] = -1.0
        rows.append(positive)
        limits.append(float(previous[index]))
        negative = np.zeros(variable_count)
        negative[index] = -1.0
        negative[turnover_start + index] = -1.0
        rows.append(negative)
        limits.append(float(-previous[index]))

    bounds = [*asset_bounds, (None, None)]
    bounds.extend([(0.0, None)] * scenario_count)
    bounds.extend([(0.0, None)] * count)
    result = linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(limits),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        return {
            "status": "fallback_cash",
            "weights": np.zeros(count),
            "cash": 1.0,
            "reason": str(result.message),
        }
    weights = np.clip(result.x[:count], 0, None)
    tolerance = 1e-7
    if (
        weights.sum() > risky_budget + tolerance
        or loss_proxy @ weights > policy.maximum_loss_proxy + tolerance
        or non_base @ weights > policy.max_non_base_currency + tolerance
    ):
        return {"status": "infeasible", "weights": None, "reason": "post-solve constraint failure"}
    eta = float(result.x[eta_index])
    excess = result.x[excess_start:turnover_start]
    cvar = eta + float(excess.sum()) / ((1 - policy.cvar_alpha) * scenario_count)
    return {
        "status": "optimized_research_only",
        "weights": weights,
        "cash": float(1 - weights.sum()),
        "expected_return": float(mean @ weights),
        "cvar": cvar,
        "turnover": float(np.abs(weights - previous).sum()),
        "uncertainty_exposure": float(uncertainty @ weights),
        "solver": "scipy.optimize.linprog-highs",
    }


def minimum_variance_weights(covariance: np.ndarray, *, max_weight: float = 1.0) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance must be square")
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("Covariance must be finite and symmetric")
    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError("Covariance must be positive semidefinite")
    count = len(covariance)
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        np.full(count, 1 / count),
        method="SLSQP",
        bounds=[(0, max_weight)] * count,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return np.asarray(result.x)
