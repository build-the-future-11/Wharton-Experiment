"""Fee-aware finite-action CRRA decisions with an explicit optimum certificate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from frem_s10.arrays import FloatArray, frozen_f64
from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ContractError, NumericalError
from frem_s10.identities import JSONValue, scientific_digest
from frem_s10.math.fees import (
    FeeSolution,
    keep_holdings_solution,
    solve_rebalances,
    validate_cost,
)
from frem_s10.math.utility import crra_utility, robust_objective
from frem_s10.portfolio.actions import Action, Valuation, enumerate_actions

OBJECTIVE_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class DecisionCertificate:
    """Evidence of exhaustive selection over a declared finite candidate set.

    The gap is relative to this finite set, never a continuous-action claim.
    For a fixed benchmark, the singleton certificate only verifies its rule.
    """

    action_set_id: str
    candidate_count: int
    selected_index: int
    criterion: str
    direction: str
    best_objective: float
    selected_objective: float
    optimality_gap: float
    objective_tie_count: int
    turnover_tie_count: int
    max_asset_weight: float
    total_risky_weight: float
    constraint_residual: float
    statistic_id: str | None = None
    projection_multiplier: float | None = None
    projection_sum_residual: float | None = None
    projection_kkt_residual: float | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Chosen action, accounting solution, utility diagnostics and certificate."""

    action: Action
    fee_solution: FeeSolution
    objective: float
    nominal_utility: float
    robust_utility: float
    worst_utility: float
    worst_atom_index: int
    hold_objective: float | None
    cash_objective: float | None
    certificate: DecisionCertificate
    forecast_id: str


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    """Aligned immutable utility and fee evaluations of an ordered action set."""

    solutions: tuple[FeeSolution, ...]
    nominal: FloatArray
    robust: FloatArray
    worst: FloatArray
    worst_atoms: tuple[int, ...]
    forecast_id: str


def evaluate_actions(
    valuation: Valuation,
    forecast: ForecastDistribution,
    actions: tuple[Action, ...],
    *,
    one_way_cost: float,
) -> ActionEvaluation:
    """Evaluate liquidation utility without booking hypothetical exit fees.

    The exact hold path bypasses target-weight reconstruction so it preserves
    actual dollars and dividend cash. The robust minimum includes zero-mass
    support atoms, and exit costs apply to price gross only.
    """
    if not isinstance(valuation, Valuation) or not isinstance(forecast, ForecastDistribution):
        raise ContractError("typed valuation and forecast are required")
    if not isinstance(actions, tuple) or not actions or any(not isinstance(a, Action) for a in actions):
        raise ContractError("a nonempty tuple of actions is required")
    validate_cost(one_way_cost)
    if any(a.is_hold and not np.array_equal(a.weights, valuation.weights) for a in actions):
        raise ContractError("hold action must identify actual current weights")
    targets = np.stack([a.weights for a in actions])
    batch = solve_rebalances(valuation.risky_dollars, valuation.cash, targets, one_way_cost=one_way_cost)
    solutions = tuple(
        keep_holdings_solution(valuation.risky_dollars, valuation.cash) if a.is_hold else batch.solution(i)
        for i, a in enumerate(actions)
    )
    risky = np.stack([solution.risky_dollars for solution in solutions])
    cash = np.asarray([solution.cash for solution in solutions], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            terminal = (1.0 - one_way_cost) * forecast.g + forecast.d
            factors = (cash[:, None] + risky @ terminal.T) / valuation.nav
        except FloatingPointError as error:
            raise NumericalError("terminal liquidation arithmetic failed") from error
    # BLAS kernels do not consistently honor NumPy's floating-point traps.
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise NumericalError("terminal liquidation factors must be finite and positive")
    if one_way_cost == 0.0:
        # Enforce the exact no-return accounting identity rather than create
        # fictitious utility from a target-weight round trip in binary64.
        flat = np.all((forecast.g == 1.0) & (forecast.d == 0.0), axis=1)
        factors[:, flat] = 1.0
    utilities = crra_utility(factors)
    nominal = robust_objective(utilities, forecast.probabilities, rho=0.0)
    robust = robust_objective(utilities, forecast.probabilities, rho=0.05)
    worst = frozen_f64(np.min(utilities, axis=1), shape=(len(actions),), name="worst_utilities")
    return ActionEvaluation(
        solutions,
        nominal,
        robust,
        worst,
        tuple(int(i) for i in np.argmin(utilities, axis=1)),
        forecast.record_id,
    )


def rank_candidates(
    objectives: FloatArray,
    turnovers: FloatArray,
    actions: tuple[Action, ...],
    *,
    nav: float,
    maximize: bool = True,
    prefer_hold: bool = True,
    objective_tolerance: float = OBJECTIVE_TOLERANCE,
) -> tuple[int, int, int]:
    """Apply global objective, hold, turnover and lexicographic tie sets.

    Comparing every candidate with the global optimum prevents order-dependent
    tolerance chains. Returned counts retain both tie-set sizes.
    """
    if not actions or any(not isinstance(action, Action) for action in actions):
        raise ContractError("candidate actions must be nonempty")
    values = frozen_f64(objectives, shape=(len(actions),), name="objectives")
    trades = frozen_f64(turnovers, shape=(len(actions),), name="turnovers")
    if not math.isfinite(nav) or nav <= 0 or np.any(trades < 0):
        raise ContractError("positive NAV and nonnegative turnovers required")
    if not isinstance(maximize, bool) or not isinstance(prefer_hold, bool):
        raise ContractError("ranking switches must be boolean")
    if (
        isinstance(objective_tolerance, bool)
        or not math.isfinite(objective_tolerance)
        or objective_tolerance < 0
    ):
        raise ContractError("objective tolerance must be finite and nonnegative")
    best = float(np.max(values) if maximize else np.min(values))
    tied = np.flatnonzero(np.abs(values - best) <= objective_tolerance)
    hold = [int(i) for i in tied if actions[int(i)].is_hold]
    if prefer_hold and hold:
        if len(hold) != 1:
            raise ContractError("at most one exact hold candidate is permitted")
        return hold[0], len(tied), 1
    least_trade = float(np.min(trades[tied]))
    trade_tied = [int(i) for i in tied if trades[int(i)] - least_trade <= 1e-10 * max(1.0, nav)]
    selected = min(trade_tied, key=lambda i: tuple(float(w) for w in actions[i].weights))
    return selected, len(tied), len(trade_tied)


def decision_from_evaluation(
    actions: tuple[Action, ...],
    evaluation: ActionEvaluation,
    objectives: FloatArray,
    *,
    selected: int,
    criterion: str,
    direction: str,
    tie_counts: tuple[int, int],
    hold_index: int | None = None,
    cash_index: int | None = None,
    constrained: bool = True,
) -> PolicyDecision:
    """Retain chosen diagnostics and the exact ordered candidate identity."""
    values = frozen_f64(objectives, shape=(len(actions),), name="selection_objectives")
    if direction not in {"max", "min", "fixed"}:
        raise ContractError("unknown decision direction")
    action = actions[selected]
    best = float(np.min(values) if direction == "min" else np.max(values))
    objective = float(values[selected])
    gap = objective - best if direction == "min" else best - objective
    targets: list[JSONValue] = [
        {"weights": [float(w) for w in a.weights], "is_hold": a.is_hold} for a in actions
    ]
    max_asset = float(np.max(action.weights))
    total = float(np.sum(action.weights))
    residual = max(0.0, max_asset - (0.4 if constrained else 1.0), total - (0.8 if constrained else 1.0))
    certificate = DecisionCertificate(
        scientific_digest(targets),
        len(actions),
        selected,
        criterion,
        direction,
        best,
        objective,
        gap,
        *tie_counts,
        max_asset,
        total,
        residual,
    )
    return PolicyDecision(
        action,
        evaluation.solutions[selected],
        objective,
        float(evaluation.nominal[selected]),
        float(evaluation.robust[selected]),
        float(evaluation.worst[selected]),
        evaluation.worst_atoms[selected],
        None if hold_index is None else float(values[hold_index]),
        None if cash_index is None else float(values[cash_index]),
        certificate,
        evaluation.forecast_id,
    )


def select_policy(
    valuation: Valuation,
    forecast: ForecastDistribution,
    *,
    one_way_cost: float,
    robust: bool = True,
    include_hold: bool = True,
) -> PolicyDecision:
    """Select the frozen 609-point policy plus its optional exact feasible hold."""
    if not isinstance(valuation, Valuation) or not isinstance(robust, bool):
        raise ContractError("typed valuation and boolean robust switch required")
    registry = enumerate_actions(valuation, include_hold=include_hold)
    evaluation = evaluate_actions(valuation, forecast, registry.actions, one_way_cost=one_way_cost)
    objectives = evaluation.robust if robust else evaluation.nominal
    turnovers = np.asarray([solution.turnover for solution in evaluation.solutions], dtype=np.float64)
    selected, objective_ties, turnover_ties = rank_candidates(
        objectives,
        turnovers,
        registry.actions,
        nav=valuation.nav,
    )
    return decision_from_evaluation(
        registry.actions,
        evaluation,
        objectives,
        selected=selected,
        criterion="robust_utility" if robust else "nominal_utility",
        direction="max",
        tie_counts=(objective_ties, turnover_ties),
        hold_index=registry.hold_index,
        cash_index=registry.cash_index,
    )
