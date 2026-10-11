"""Scenario-CVaR research allocation and exact post-fee accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class AllocationPolicy:
    alpha: float = 0.95
    risk_aversion: float = 2.0
    disagreement_penalty: float = 1.0
    max_risky_weight: float = 0.20
    turnover_cap: float = 0.40
    transaction_cost: float = 0.001
    feasibility_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1 or self.risk_aversion < 0 or self.disagreement_penalty < 0:
            raise ValueError("Invalid risk settings")
        if not 0 < self.max_risky_weight <= 1 or not 0 <= self.turnover_cap <= 2:
            raise ValueError("Invalid allocation bounds")
        if not 0 <= self.transaction_cost < 1:
            raise ValueError("Invalid transaction cost")


@dataclass(frozen=True)
class AllocationResult:
    status: str
    weights: np.ndarray | None
    objective: float | None
    max_cvar: float | None
    turnover: float | None
    message: str
    residual: float | None


def empirical_cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    loss = np.asarray(losses, dtype=float)
    if loss.ndim != 1 or len(loss) == 0 or not np.isfinite(loss).all():
        raise ValueError("Losses must be a finite vector")
    threshold = np.quantile(loss, alpha, method="higher")
    tail = loss[loss >= threshold]
    return float(tail.mean())


def _feasibility_residual(
    weights: np.ndarray,
    previous: np.ndarray,
    eligible: np.ndarray,
    policy: AllocationPolicy,
) -> float:
    errors = [abs(weights.sum() - 1.0), max(0.0, -weights.min())]
    errors.append(max(0.0, weights[:-1].max(initial=0.0) - policy.max_risky_weight))
    errors.append(float(np.abs(weights - previous).sum() - policy.turnover_cap))
    errors.append(float(weights[~eligible].max(initial=0.0)))
    return max(errors)


def optimize_allocation(
    expected_simple_return: np.ndarray,
    scenario_families: dict[str, np.ndarray],
    previous_weights: np.ndarray,
    eligible: np.ndarray,
    disagreement_covariance: np.ndarray,
    *,
    policy: AllocationPolicy | None = None,
) -> AllocationResult:
    policy = policy or AllocationPolicy()
    mu = np.asarray(expected_simple_return, dtype=float)
    previous = np.asarray(previous_weights, dtype=float)
    allowed = np.asarray(eligible, dtype=bool)
    covariance = np.asarray(disagreement_covariance, dtype=float)
    count = len(mu)
    if (
        previous.shape != (count,)
        or allowed.shape != (count,)
        or covariance.shape != (count, count)
    ):
        raise ValueError("Allocation inputs have inconsistent dimensions")
    if not allowed[-1]:
        raise ValueError("Cash instrument must be eligible")
    if not np.isfinite(mu).all() or not np.isfinite(covariance).all():
        raise ValueError("Allocation inputs must be finite")
    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError("Disagreement covariance must be positive semidefinite")
    families = {name: np.asarray(values, dtype=float) for name, values in scenario_families.items()}
    if not families or any(
        values.ndim != 2 or values.shape[1] != count for values in families.values()
    ):
        raise ValueError("Every scenario family must be (scenarios, instruments)")

    def objective(weights: np.ndarray) -> float:
        cvars = [empirical_cvar(-(values @ weights), policy.alpha) for values in families.values()]
        cost = policy.transaction_cost * np.abs(weights - previous).sum()
        disagreement = float(np.sqrt(max(weights @ covariance @ weights, 0.0) + 1e-12))
        return -(
            mu @ weights
            - policy.risk_aversion * max(cvars)
            - cost
            - policy.disagreement_penalty * disagreement
        )

    bounds = [
        (0.0, policy.max_risky_weight if index < count - 1 else 1.0)
        if allowed[index]
        else (0.0, 0.0)
        for index in range(count)
    ]
    constraints = [
        {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        {
            "type": "ineq",
            "fun": lambda weights: policy.turnover_cap - np.abs(weights - previous).sum(),
        },
    ]
    start = previous.copy()
    if _feasibility_residual(start, previous, allowed, policy) > policy.feasibility_tolerance:
        start = np.zeros(count)
        start[-1] = 1.0
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": policy.feasibility_tolerance, "maxiter": 1000},
    )
    if result.success:
        residual = _feasibility_residual(result.x, previous, allowed, policy)
        if residual <= policy.feasibility_tolerance:
            cvar = max(
                empirical_cvar(-(values @ result.x), policy.alpha) for values in families.values()
            )
            return AllocationResult(
                "OPTIMAL",
                result.x,
                -float(result.fun),
                cvar,
                float(np.abs(result.x - previous).sum()),
                result.message,
                residual,
            )
    previous_residual = _feasibility_residual(previous, previous, allowed, policy)
    if previous_residual <= policy.feasibility_tolerance:
        cvar = max(
            empirical_cvar(-(values @ previous), policy.alpha) for values in families.values()
        )
        return AllocationResult(
            "HOLD_PRETRADE",
            previous.copy(),
            -objective(previous),
            cvar,
            0.0,
            f"Optimization failed; retained feasible pretrade portfolio: {result.message}",
            previous_residual,
        )
    return AllocationResult(
        "INFEASIBLE",
        None,
        None,
        None,
        None,
        f"No feasible portfolio; mandatory constraints were not relaxed: {result.message}",
        None,
    )


def exact_transaction_fee(
    wealth: float,
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    rates: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> float:
    if wealth < 0:
        raise ValueError("Wealth cannot be negative")
    target, previous, cost = map(np.asarray, (target_weights, previous_weights, rates))
    if target.shape != previous.shape or target.shape != cost.shape or (cost < 0).any():
        raise ValueError("Fee vectors do not align")

    def residual(fee: float) -> float:
        return float((cost * np.abs((wealth - fee) * target - wealth * previous)).sum() - fee)

    lower, upper = 0.0, wealth
    if residual(lower) <= 0:
        return 0.0
    if residual(upper) > 0:
        raise ValueError("Transaction fees exceed available wealth")
    while upper - lower > tolerance * max(wealth, 1.0):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    fee = 0.5 * (lower + upper)
    if not 0 <= fee <= wealth:
        raise AssertionError("Fee bisection left valid bracket")
    return fee


def execute_rebalance(
    wealth: float,
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    rates: np.ndarray,
) -> dict[str, object]:
    fee = exact_transaction_fee(wealth, target_weights, previous_weights, rates)
    post_fee = wealth - fee
    holdings = post_fee * np.asarray(target_weights, dtype=float)
    if holdings.sum() > wealth + 1e-8 or fee < 0:
        raise AssertionError("Fees created money")
    return {"fee": fee, "post_fee_wealth": post_fee, "holdings": holdings}


def international_simple_return(
    local_return: np.ndarray, fx_start: np.ndarray, fx_end: np.ndarray
) -> np.ndarray:
    local, start, end = map(
        lambda value: np.asarray(value, dtype=float), (local_return, fx_start, fx_end)
    )
    if (
        local.shape != start.shape
        or local.shape != end.shape
        or (start <= 0).any()
        or (end <= 0).any()
    ):
        raise ValueError("International return inputs are invalid")
    return (1.0 + local) * end / start - 1.0


def portfolio_metrics(
    simple_returns: np.ndarray,
    *,
    periods_per_year: float,
    cash_returns: np.ndarray | None = None,
    downside_target: float = 0.0,
) -> dict[str, float]:
    values = np.asarray(simple_returns, dtype=float)
    cash = np.zeros_like(values) if cash_returns is None else np.asarray(cash_returns, dtype=float)
    if values.ndim != 1 or cash.shape != values.shape or periods_per_year <= 0:
        raise ValueError("Invalid return series")
    wealth = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(wealth)
    drawdown = wealth / peaks - 1.0
    excess = values - cash
    volatility = values.std(ddof=1) * np.sqrt(periods_per_year) if len(values) > 1 else 0.0
    sharpe = (
        excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)
        if len(values) > 1 and excess.std(ddof=1) > 0
        else np.nan
    )
    downside = np.minimum(values - downside_target, 0.0)
    downside_deviation = np.sqrt(np.mean(downside.square())) * np.sqrt(periods_per_year)
    sortino = (
        (values.mean() - downside_target) * periods_per_year / downside_deviation
        if downside_deviation > 0
        else np.nan
    )
    return {
        "net_compound_growth": float(wealth[-1] - 1.0),
        "mean_return": float(values.mean()),
        "annualized_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "maximum_drawdown": float(drawdown.min()),
        "loss_cvar_95": empirical_cvar(-values, 0.95),
    }


def equal_weight_feasible(eligible: np.ndarray) -> np.ndarray:
    allowed = np.asarray(eligible, dtype=bool)
    if allowed.ndim != 1 or not allowed.any():
        raise ValueError("At least one instrument must be eligible")
    weights = np.zeros(len(allowed), dtype=float)
    weights[allowed] = 1.0 / allowed.sum()
    return weights


def fixed_60_35_5_reference(
    categories: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Research-only reference; unavailable sleeves are assigned to cash."""
    category = np.asarray(categories, dtype=object)
    allowed = np.asarray(eligible, dtype=bool)
    if category.shape != allowed.shape or category[-1] != "cash" or not allowed[-1]:
        raise ValueError("Last instrument must be eligible cash")
    weights = np.zeros(len(category), dtype=float)
    sleeves = {
        "us_equity": 0.60 * 0.70,
        "international_equity": 0.60 * 0.30,
        "usd_fixed_income": 0.35,
    }
    cash = 0.05
    for name, weight in sleeves.items():
        indices = np.flatnonzero((category == name) & allowed)
        if len(indices):
            weights[indices] = weight / len(indices)
        else:
            cash += weight
    weights[-1] = cash
    return weights


def minimum_variance_weights(
    covariance: np.ndarray,
    eligible: np.ndarray,
    *,
    max_risky_weight: float = 0.20,
) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    allowed = np.asarray(eligible, dtype=bool)
    count = len(allowed)
    if cov.shape != (count, count) or np.linalg.eigvalsh(cov).min() < -1e-8:
        raise ValueError("Covariance must be aligned and positive semidefinite")
    bounds = [
        (0.0, max_risky_weight if index < count - 1 else 1.0) if allowed[index] else (0.0, 0.0)
        for index in range(count)
    ]
    result = minimize(
        lambda weights: float(weights @ cov @ weights),
        equal_weight_feasible(allowed),
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance baseline failed: {result.message}")
    return result.x


def project_to_feasible_set(
    target: np.ndarray,
    eligible: np.ndarray,
    *,
    max_risky_weight: float = 0.20,
) -> np.ndarray:
    """Prespecified minimum-distance projection for an infeasible baseline."""
    desired = np.asarray(target, dtype=float)
    allowed = np.asarray(eligible, dtype=bool)
    bounds = [
        (0.0, max_risky_weight if index < len(desired) - 1 else 1.0)
        if allowed[index]
        else (0.0, 0.0)
        for index in range(len(desired))
    ]
    result = minimize(
        lambda weights: float(np.square(weights - desired).sum()),
        equal_weight_feasible(allowed),
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Feasible-set projection failed: {result.message}")
    return result.x
