"""The six fixed benchmark rules, using the same visible holdings and fees."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from frem_s10.arrays import FloatArray, frozen_f64
from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ContractError, NumericalError
from frem_s10.identities import scientific_digest
from frem_s10.portfolio.actions import Action, Valuation, grid_weights
from frem_s10.portfolio.policy import (
    PolicyDecision,
    decision_from_evaluation,
    evaluate_actions,
    rank_candidates,
)

BENCHMARK_NAMES = (
    "cash",
    "equal100",
    "equal80",
    "buy_hold100",
    "inverse_vol80",
    "shrink_min_variance80",
)


@dataclass(frozen=True, slots=True)
class ProjectionCertificate:
    """Capped-simplex target and its independent numerical KKT certificate."""

    weights: FloatArray
    multiplier: float
    sum_residual: float
    kkt_residual: float
    iterations: int


def project_capped_simplex(target: FloatArray) -> ProjectionCertificate:
    """Project an .8-risk target onto the fixed .4 cap without renormalization."""
    z = frozen_f64(target, shape=(3,), name="inverse_volatility_target")
    if np.any(z < 0) or abs(float(np.sum(z)) - 0.8) > 1e-12:
        raise ContractError("inverse-volatility target must be nonnegative and sum to .8")
    lower, upper = float(np.min(z)) - 0.4, float(np.max(z))
    weights = np.zeros(3, dtype=np.float64)
    multiplier = 0.0
    residual = float("inf")
    iterations = 0
    for _ in range(100):
        iterations += 1
        multiplier = (lower + upper) * 0.5
        weights = np.clip(z - multiplier, 0.0, 0.4)
        residual = float(np.sum(weights)) - 0.8
        if abs(residual) <= 1e-12:
            break
        if residual > 0:
            lower = multiplier
        else:
            upper = multiplier
    else:
        raise NumericalError("capped-simplex projection failed its sum tolerance")
    gradient = weights - z + multiplier
    violations = np.where(weights == 0.0, np.maximum(-gradient, 0.0), np.abs(gradient))
    violations = np.where(weights == 0.4, np.maximum(gradient, 0.0), violations)
    kkt = float(np.max(violations))
    if kkt > 1e-10 or np.any(weights < 0) or np.any(weights > 0.4):
        raise NumericalError("capped-simplex projection failed its KKT tolerance")
    return ProjectionCertificate(
        frozen_f64(weights, shape=(3,), name="projected_weights"),
        multiplier,
        residual,
        kkt,
        iterations,
    )


def unconditional_covariance(forecast: ForecastDistribution) -> FloatArray:
    """Compute population covariance of the matured unconditional atom law.

    The caller is responsible for chronology and selecting U. A nonuniform law
    is rejected here so contextual probabilities cannot enter these controls.
    """
    if not isinstance(forecast, ForecastDistribution):
        raise ContractError("typed unconditional forecast required")
    count = len(forecast.probabilities)
    if not np.allclose(forecast.probabilities, 1.0 / count, rtol=0.0, atol=1e-12):
        raise ContractError("benchmarks require the uniform unconditional atom law")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            returns = forecast.r
            centered = returns - np.mean(returns, axis=0)
            covariance = centered.T @ centered / count
        except FloatingPointError as error:
            raise NumericalError("unconditional covariance arithmetic failed") from error
    if not np.all(np.isfinite(covariance)):
        raise NumericalError("unconditional covariance is not finite")
    return frozen_f64(covariance, shape=(3, 3), name="unconditional_covariance")


def _fixed_decision(
    action: Action,
    valuation: Valuation,
    forecast: ForecastDistribution,
    *,
    one_way_cost: float,
    name: str,
    constrained: bool,
) -> PolicyDecision:
    evaluation = evaluate_actions(valuation, forecast, (action,), one_way_cost=one_way_cost)
    return decision_from_evaluation(
        (action,),
        evaluation,
        evaluation.nominal,
        selected=0,
        criterion=name,
        direction="fixed",
        tie_counts=(1, 1),
        constrained=constrained,
        hold_index=0 if action.is_hold else None,
        cash_index=0 if np.all(action.weights == 0) else None,
    )


def _inverse_volatility(
    valuation: Valuation,
    forecast: ForecastDistribution,
    *,
    one_way_cost: float,
) -> PolicyDecision:
    covariance = unconditional_covariance(forecast)
    volatility = np.maximum(np.sqrt(np.diag(covariance)), 1e-4)
    inverse = 1.0 / volatility
    target = 0.8 * inverse / float(np.sum(inverse))
    projection = project_capped_simplex(target)
    decision = _fixed_decision(
        Action(projection.weights),
        valuation,
        forecast,
        one_way_cost=one_way_cost,
        name="inverse_vol80",
        constrained=True,
    )
    certificate = replace(
        decision.certificate,
        statistic_id=scientific_digest([[float(v) for v in row] for row in covariance]),
        projection_multiplier=projection.multiplier,
        projection_sum_residual=projection.sum_residual,
        projection_kkt_residual=projection.kkt_residual,
    )
    return replace(decision, certificate=certificate)


def _minimum_variance(
    valuation: Valuation,
    forecast: ForecastDistribution,
    *,
    one_way_cost: float,
) -> PolicyDecision:
    covariance = unconditional_covariance(forecast)
    shrink = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
    targets = grid_weights(total_risky=0.8)
    actions = tuple(Action(row) for row in targets)
    evaluation = evaluate_actions(valuation, forecast, actions, one_way_cost=one_way_cost)
    with np.errstate(over="raise", invalid="raise"):
        try:
            variances = np.sum((targets @ shrink) * targets, axis=1)
        except FloatingPointError as error:
            raise NumericalError("benchmark variance arithmetic failed") from error
    if not np.all(np.isfinite(variances)) or np.any(variances < -1e-12):
        raise NumericalError("benchmark variances are invalid")
    turnovers = np.asarray([s.turnover for s in evaluation.solutions], dtype=np.float64)
    selected, objective_ties, turnover_ties = rank_candidates(
        variances,
        turnovers,
        actions,
        nav=valuation.nav,
        maximize=False,
        prefer_hold=False,
        objective_tolerance=0.0,
    )
    decision = decision_from_evaluation(
        actions,
        evaluation,
        variances,
        selected=selected,
        criterion="shrink_min_variance80",
        direction="min",
        tie_counts=(objective_ties, turnover_ties),
        constrained=True,
    )
    return replace(
        decision,
        certificate=replace(
            decision.certificate,
            statistic_id=scientific_digest([[float(v) for v in row] for row in shrink]),
        ),
    )


def select_benchmark(
    name: str,
    valuation: Valuation,
    unconditional_forecast: ForecastDistribution,
    *,
    one_way_cost: float,
    initial_entry: bool,
) -> PolicyDecision:
    """Select a fixed benchmark action; final liquidation belongs to the ledger.

    Buy-and-hold preserves actual holdings and accumulated dividend cash after
    its one equal-dollar entry. Broad 100% controls have no .4/.8 policy caps.
    """
    if not isinstance(name, str) or name not in BENCHMARK_NAMES:
        raise ContractError("unknown benchmark")
    if not isinstance(initial_entry, bool) or not isinstance(valuation, Valuation):
        raise ContractError("typed valuation and boolean initial_entry required")
    if name == "inverse_vol80":
        return _inverse_volatility(valuation, unconditional_forecast, one_way_cost=one_way_cost)
    if name == "shrink_min_variance80":
        return _minimum_variance(valuation, unconditional_forecast, one_way_cost=one_way_cost)
    is_hold = name == "buy_hold100" and not initial_entry
    if is_hold:
        weights = valuation.weights
    elif name == "cash":
        weights = np.zeros(3, dtype=np.float64)
    else:
        weights = np.full(3, (0.8 if name == "equal80" else 1.0) / 3.0, dtype=np.float64)
    return _fixed_decision(
        Action(weights, is_hold=is_hold),
        valuation,
        unconditional_forecast,
        one_way_cost=one_way_cost,
        name=name,
        constrained=name in {"cash", "equal80"},
    )
