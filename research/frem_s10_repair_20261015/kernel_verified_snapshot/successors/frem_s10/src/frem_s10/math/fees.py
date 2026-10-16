"""Self-financing proportional fees with certified float64 bisection.

The target is a fraction of post-fee NAV. Costs apply to actual risky
dollars traded, never to dividend cash or an imaginary periodic exit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError, NumericalError
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeeSolution:
    """A rebalance and its dollar accounting/root certificate."""

    post_fee_nav: float
    risky_dollars: FloatArray
    cash: float
    fee: float
    turnover: float
    residual: float
    bracket_width: float
    iterations: int
    target_weights: FloatArray

    def __post_init__(self) -> None:
        """Own immutable arrays and reject invalid scalar cost certificates."""
        object.__setattr__(
            self, "risky_dollars", frozen_f64(self.risky_dollars, shape=(3,), name="risky_dollars")
        )
        object.__setattr__(
            self, "target_weights", frozen_f64(self.target_weights, shape=(3,), name="target_weights")
        )
        for name in ("post_fee_nav", "cash", "fee", "turnover", "residual", "bracket_width"):
            if not math.isfinite(getattr(self, name)):
                raise NumericalError(f"nonfinite fee certificate {name}")
        if self.post_fee_nav <= 0 or self.cash < -1e-10 * max(1.0, self.post_fee_nav):
            raise NumericalError("invalid post-fee wealth or cash")
        if self.fee < 0 or self.turnover < 0 or self.bracket_width < 0:
            raise NumericalError("negative cost certificate")


@dataclass(frozen=True, slots=True)
class FeeBatch:
    """Equivalent certificates for a finite matrix of target weights."""

    post_fee_nav: FloatArray
    risky_dollars: FloatArray
    cash: FloatArray
    fees: FloatArray
    turnovers: FloatArray
    residuals: FloatArray
    bracket_widths: FloatArray
    iterations: int
    targets: FloatArray

    def __post_init__(self) -> None:
        """Own each aligned immutable batch certificate array."""
        targets = frozen_f64(self.targets, shape=(None, 3), name="targets")
        object.__setattr__(self, "targets", targets)
        count = len(targets)
        for name in ("post_fee_nav", "cash", "fees", "turnovers", "residuals", "bracket_widths"):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(count,), name=name))
        object.__setattr__(
            self, "risky_dollars", frozen_f64(self.risky_dollars, shape=(count, 3), name="risky_dollars")
        )

    def solution(self, index: int) -> FeeSolution:
        """Return one immutable target's certificate without solving again."""
        return FeeSolution(
            float(self.post_fee_nav[index]),
            self.risky_dollars[index],
            float(self.cash[index]),
            float(self.fees[index]),
            float(self.turnovers[index]),
            float(self.residuals[index]),
            float(self.bracket_widths[index]),
            self.iterations,
            self.targets[index],
        )


def validate_valuation(holdings: FloatArray, cash: float) -> tuple[FloatArray, float]:
    """Validate nonnegative USD holdings/cash and return owned holdings and NAV."""
    values = frozen_f64(holdings, shape=(3,), name="current_risky_dollars")
    if np.any(values < 0) or isinstance(cash, bool) or not math.isfinite(cash) or cash < 0:
        raise ContractError("holdings and cash must be finite nonnegative USD")
    nav = float(np.sum(values)) + cash
    if not math.isfinite(nav) or nav <= 0:
        raise ContractError("NAV must be finite and strictly positive")
    return values, nav


def validate_cost(one_way_cost: float) -> None:
    """Require a finite proportional fee below one for root uniqueness."""
    if isinstance(one_way_cost, bool) or not math.isfinite(one_way_cost) or not 0 <= one_way_cost < 1:
        raise ContractError("one_way_cost must be finite and in [0, 1)")


def solve_rebalances(
    holdings: FloatArray,
    cash: float,
    targets: FloatArray,
    *,
    one_way_cost: float,
    max_iterations: int = 100,
) -> FeeBatch:
    """Solve q+c*sum(abs(q*w-h))=V for every target, including broad benchmarks.

    Targets are nonnegative and sum to at most one. Both the root residual
    and remaining bracket must be <=1e-10*max(1,V). A failure is explicit.
    """
    current, nav = validate_valuation(holdings, cash)
    validate_cost(one_way_cost)
    weights = frozen_f64(targets, shape=(None, 3), name="target_weights")
    if len(weights) == 0 or np.any(weights < 0) or np.any(np.sum(weights, axis=1) > 1 + 1e-12):
        raise ContractError("targets must be nonempty long-only weight vectors summing to at most one")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or not 1 <= max_iterations <= 100
    ):
        raise ContractError("max_iterations must be an integer from 1 to 100")
    count = len(weights)
    lower = np.zeros(count, dtype=np.float64)
    upper = np.full(count, nav, dtype=np.float64)
    # An exact zero-trade target has the known root q=V. Preserve it exactly;
    # bisection roundoff must not create fictitious losses on an all-cash path.
    no_trade = np.all(nav * weights == current, axis=1)
    lower[no_trade] = nav
    q = upper.copy()
    tolerance = 1e-10 * max(1.0, nav)
    iterations = 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if one_way_cost == 0:
                lower = q.copy()
            else:
                for iteration in range(1, max_iterations + 1):
                    iterations = iteration
                    q = (lower + upper) * 0.5
                    turnover = np.sum(np.abs(q[:, None] * weights - current), axis=1)
                    residual = q + one_way_cost * turnover - nav
                    lower = np.where(residual <= 0, q, lower)
                    upper = np.where(residual >= 0, q, upper)
                    if np.all(upper - lower <= tolerance) and np.all(np.abs(residual) <= tolerance):
                        break
                else:
                    raise NumericalError("fee root did not satisfy bracket and accounting tolerances")
            risky = q[:, None] * weights
            cash_after = q * (1.0 - np.sum(weights, axis=1))
            turnovers = np.sum(np.abs(risky - current), axis=1)
            fees = one_way_cost * turnovers
            residuals = q + fees - nav
        except FloatingPointError as error:
            raise NumericalError("nonfinite rebalance arithmetic") from error
    if np.any(np.abs(residuals) > tolerance) or np.any(upper - lower > tolerance):
        raise NumericalError("uncertified fee root")
    if np.any(cash_after < -tolerance):
        raise NumericalError("negative cash after rebalance")
    # Only an arithmetic roundoff below the documented monetary tolerance.
    cash_after = np.maximum(cash_after, 0.0)
    return FeeBatch(q, risky, cash_after, fees, turnovers, residuals, upper - lower, iterations, weights)


def solve_rebalance(
    holdings: FloatArray,
    cash: float,
    target_weights: FloatArray,
    *,
    one_way_cost: float,
    max_iterations: int = 100,
) -> FeeSolution:
    """Certify a single target-weight rebalance using the same batch kernel."""
    target = frozen_f64(target_weights, shape=(3,), name="target_weights")
    return solve_rebalances(
        holdings,
        cash,
        target[None, :],
        one_way_cost=one_way_cost,
        max_iterations=max_iterations,
    ).solution(0)


def keep_holdings_solution(holdings: FloatArray, cash: float) -> FeeSolution:
    """Return the exact zero-trade action, preserving dollar holdings bitwise."""
    current, nav = validate_valuation(holdings, cash)
    return FeeSolution(nav, current, cash, 0.0, 0.0, 0.0, 0.0, 0, current / nav)


def solve_fee_root(
    nav: float,
    current_risky: FloatArray,
    target_weights: FloatArray,
    *,
    one_way_cost: float,
) -> FeeSolution:
    """Explicit NAV convenience entry point; inconsistent NAV is rejected."""
    current = frozen_f64(current_risky, shape=(3,), name="current_risky_dollars")
    if not math.isfinite(nav) or nav <= 0 or float(np.sum(current)) > nav:
        raise ContractError("NAV must cover current risky dollars")
    return solve_rebalance(current, nav - float(np.sum(current)), target_weights, one_way_cost=one_way_cost)
