"""Constrained, cost-aware research allocations with identical-rule fallbacks.

Only hypothetical target weights are returned.  There is no broker client,
credential use, order submission, hidden data retrieval, or implicit backtest.
Weights are fractions of *post-transaction-cost* NAV.  Turnover and ADV limits
are measured using traded dollars divided by the pre-trade NAV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .mandate import Instrument, Mandate, scenario_probabilities, utc_timestamp
from .risk import (
    RegimeInputs,
    RegimePolicy,
    choose_regime,
    portfolio_risk_summary,
    validate_covariance,
    weighted_cvar,
)


@dataclass(frozen=True)
class PortfolioConfig:
    risk_aversion: float = 5.0
    cvar_penalty: float = 0.0
    cvar_confidence: float = 0.95
    cost_bps: float = 10.0
    buy_cost_bps: Mapping[str, float] = field(default_factory=dict)
    sell_cost_bps: Mapping[str, float] = field(default_factory=dict)
    turnover_penalty: float = 0.001
    max_iterations: int = 300
    tolerance: float = 1e-9
    constraint_tolerance: float = 1e-7
    regime_policy: RegimePolicy = field(default_factory=RegimePolicy)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> PortfolioConfig:
        values = dict(config)
        if isinstance(values.get("regime_policy"), Mapping):
            values["regime_policy"] = RegimePolicy(**values["regime_policy"])
        return cls(**values)

    def validate(self) -> None:
        for name in ("risk_aversion", "cvar_penalty", "turnover_penalty"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0 < self.cvar_confidence < 1:
            raise ValueError("cvar_confidence must be strictly between zero and one")
        for bps in (self.cost_bps, *self.buy_cost_bps.values(), *self.sell_cost_bps.values()):
            if not np.isfinite(bps) or not 0 <= bps < 1000:
                raise ValueError("declared per-side trading costs must lie in [0,1000) basis points")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not 0 < self.tolerance < 1e-2 or not 0 < self.constraint_tolerance < 1e-2:
            raise ValueError("solver and constraint tolerances must be positive and below 0.01")
        policy = self.regime_policy
        if not 0 <= policy.defensive_recession_probability <= policy.cash_recession_probability <= 1:
            raise ValueError("recession policy thresholds must be ordered in [0,1]")
        if not -1 <= policy.cash_drawdown <= policy.defensive_drawdown <= 0:
            raise ValueError("drawdown policy thresholds must be ordered in [-1,0]")
        if (
            not 0 <= policy.defensive_cash_floor <= 1
            or not np.isfinite(policy.defensive_risk_multiplier)
            or policy.defensive_risk_multiplier < 1
        ):
            raise ValueError("invalid defensive cash preference or risk multiplier")
        if (
            not np.isfinite(policy.defensive_annualized_volatility)
            or policy.defensive_annualized_volatility < 0
        ):
            raise ValueError("invalid defensive volatility threshold")


@dataclass
class AllocationResult:
    status: str
    weights: dict[str, float] | None
    cash_weight: float | None
    regime: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _instruments(values: Sequence[Instrument] | pd.DataFrame) -> list[Instrument]:
    if isinstance(values, pd.DataFrame):
        records = values.to_dict(orient="records")
        return [Instrument.from_mapping(value) for value in records]
    return [value if isinstance(value, Instrument) else Instrument.from_mapping(value) for value in values]


class _PortfolioProblem:
    """Shared constraints used by optimization, fallbacks and lot-rounding checks."""

    def __init__(
        self,
        instruments: list[Instrument],
        mandate: Mandate,
        as_of: Any,
        nav: float,
        current: np.ndarray,
        reserve_cash: float,
        config: PortfolioConfig,
    ):
        self.instruments, self.mandate, self.config = instruments, mandate, config
        self.nav, self.current, self.reserve_cash = nav, current, reserve_cash
        self.ids = [value.asset_id for value in instruments]
        self.exclusions = {value.asset_id: value.issues(as_of, mandate) for value in instruments}
        self.upper = np.array(
            [
                0.0
                if self.exclusions[value.asset_id]
                else min(
                    float(mandate.max_asset_weight), 1.0 if value.max_weight is None else value.max_weight
                )
                for value in instruments
            ]
        )
        self.buy_rates = np.array(
            [config.buy_cost_bps.get(value.asset_id, config.cost_bps) / 10000 for value in instruments]
        )
        self.sell_rates = np.array(
            [config.sell_cost_bps.get(value.asset_id, config.cost_bps) / 10000 for value in instruments]
        )
        self.adv = np.array(
            [
                float(value.average_daily_value_base)
                if value.average_daily_value_base is not None
                and np.isfinite(value.average_daily_value_base)
                and value.average_daily_value_base > 0
                else 0.0
                for value in instruments
            ]
        )
        self.untradable = np.array(
            [not isinstance(value.tradable, (bool, np.bool_)) or not value.tradable for value in instruments]
        )
        self.foreign = np.array(
            [
                0.0
                if self.exclusions[value.asset_id]
                else 1 - value.currency_exposures.get(mandate.base_currency, 0)
                if value.currency_exposures
                else float(value.currency != mandate.base_currency)
                for value in instruments
            ]
        )
        self.groups: list[tuple[str, np.ndarray, float | None, float | None]] = []
        for prefix in ("market", "sector", "asset_class"):
            caps = getattr(mandate, f"{prefix}_weight_caps")
            floors = getattr(mandate, f"{prefix}_weight_floors")
            for key in sorted(caps.keys() | floors.keys()):
                exposure_field = f"{prefix}_exposures"
                exposure = np.array(
                    [
                        0.0
                        if self.exclusions[value.asset_id]
                        else getattr(value, exposure_field).get(key, 0)
                        if hasattr(value, exposure_field) and getattr(value, exposure_field)
                        else float(getattr(value, prefix) == key)
                        for value in instruments
                    ]
                )
                self.groups.append((f"{prefix}:{key}", exposure, caps.get(key), floors.get(key)))

    def state(self, weights: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Solve self-financing alpha + cost(alpha*w - current) = 1.

        alpha is post-cost/pre-trade NAV. With per-side rates below 10% and
        long-only weights summing to at most one this fixed point contracts.
        No cash is created by computing weights before deducting fees.
        """
        # Bracketing remains valid for infeasible trial points generated by a
        # constrained solver. A fixed-point contraction need not hold there.
        low, high = 0.0, 1.0
        for _ in range(48):
            alpha = (low + high) / 2
            delta = alpha * weights - self.current
            cost = float(self.buy_rates @ np.maximum(delta, 0) + self.sell_rates @ np.maximum(-delta, 0))
            residual = alpha + cost - 1.0
            if not np.isfinite(residual):
                raise ValueError("transaction-cost arithmetic is nonfinite")
            if residual > 0:
                high = alpha
            else:
                low = alpha
        alpha = (low + high) / 2
        delta = alpha * weights - self.current
        cash_amount = alpha * (1.0 - float(weights.sum())) * self.nav
        return alpha, delta, cash_amount

    def inequalities(self, weights: np.ndarray, *, named: bool = False) -> np.ndarray | dict[str, float]:
        alpha, delta, cash_amount = self.state(weights)
        cash_weight = 1 - float(weights.sum())
        m = self.mandate
        constraints = {
            "minimum_cash": cash_weight - float(m.min_cash_weight),
            "maximum_cash": float(m.max_cash_weight) - cash_weight,
            "cash_reserve_amount": (cash_amount - self.reserve_cash) / self.nav,
            "gross_traded_fraction": float(m.max_turnover) - float(np.abs(delta).sum()),
            "foreign_currency_weight": float(m.max_foreign_currency_weight) - float(weights @ self.foreign),
        }
        for index, value in enumerate(self.instruments):
            constraints[f"long_only:{value.asset_id}"] = float(weights[index])
            constraints[f"security_cap:{value.asset_id}"] = float(self.upper[index] - weights[index])
            constraints[f"trade_ADV:{value.asset_id}"] = float(m.max_trade_adv_fraction) * self.adv[
                index
            ] / self.nav - abs(float(delta[index]))
            constraints[f"position_ADV:{value.asset_id}"] = float(m.max_position_adv_days) * self.adv[
                index
            ] / self.nav - alpha * float(weights[index])
            if value.foreign_buy_headroom_base is not None and np.isfinite(value.foreign_buy_headroom_base):
                constraints[f"foreign_buy_headroom:{value.asset_id}"] = (
                    value.foreign_buy_headroom_base / self.nav - max(float(delta[index]), 0)
                )
        for key, fractions, cap, floor in self.groups:
            exposure = float(weights @ fractions)
            if cap is not None:
                constraints[f"cap:{key}"] = float(cap) - exposure
            if floor is not None:
                constraints[f"floor:{key}"] = exposure - float(floor)
        return constraints if named else np.asarray(list(constraints.values()))

    def equalities(self, weights: np.ndarray) -> np.ndarray:
        _, delta, _ = self.state(weights)
        return delta[self.untradable & (self.current > 0)]

    def check(self, weights: np.ndarray) -> dict[str, Any]:
        if weights.shape != self.current.shape or not np.all(np.isfinite(weights)):
            return {"feasible": False, "violations": {"weights": "shape mismatch or nonfinite value"}}
        named = self.inequalities(weights, named=True)
        assert isinstance(named, dict)
        tolerance = self.config.constraint_tolerance
        violations = {
            k: (-float(v) if np.isfinite(v) else "nonfinite constraint slack")
            for k, v in named.items()
            if not np.isfinite(v) or v < -tolerance
        }
        equality = self.equalities(weights)
        if len(equality) and np.max(np.abs(equality)) > tolerance:
            violations["untradable_held_positions"] = float(np.max(np.abs(equality)))
        alpha, delta, cash_amount = self.state(weights)
        # An untradable instrument with zero holdings may not be newly purchased.
        untradable_trade = np.abs(delta[self.untradable])
        if len(untradable_trade) and untradable_trade.max() > tolerance:
            violations["untradable_instrument_change"] = float(untradable_trade.max())
        return {
            "feasible": not violations,
            "violations": violations,
            "nav_retention_after_costs": alpha,
            "estimated_cost": (1 - alpha) * self.nav,
            "post_cost_nav": alpha * self.nav,
            "cash_amount": cash_amount,
            "gross_traded_fraction": float(np.abs(delta).sum()),
            "signed_trade_values": dict(zip(self.ids, (delta * self.nav).tolist(), strict=True)),
            "constraint_slack": named,
        }


def optimize_portfolio(
    expected_returns: Sequence[float] | None,
    covariance: Sequence[Sequence[float]],
    instruments: Sequence[Instrument] | pd.DataFrame,
    mandate: Mandate,
    *,
    as_of: Any,
    nav: float,
    current_weights: Sequence[float] | None = None,
    scenarios: Sequence[Sequence[float]] | None = None,
    scenario_probabilities: Sequence[float] | None = None,
    cash_return: float = 0.0,
    reserve_cash: float = 0.0,
    decision_context: RegimeInputs | None = None,
    config: PortfolioConfig | None = None,
) -> AllocationResult:
    """Create a verified research allocation or return blocked/infeasible/no_action.

    ``expected_returns``, covariance and scenario columns must be aligned with
    the instrument sequence and already use the SAME SIMPLE-return horizon and
    base currency. Current weights use pre-trade NAV; output weights use NAV
    after estimated transaction costs. Missing current holdings are not assumed
    to be all-cash: pass an explicit zero vector for a new all-cash portfolio.
    """
    cfg = config or PortfolioConfig()
    diagnostics: dict[str, Any] = {"research_only": True, "solver_attempts": [], "as_of": str(as_of)}
    mandate_issues = mandate.validate(as_of, require_allocation=True)
    if mandate_issues:
        diagnostics["blocking_reasons"] = mandate_issues
        return AllocationResult("blocked", None, None, "no_action", diagnostics)
    try:
        cfg.validate()
        master = _instruments(instruments)
        if not master or len({value.asset_id for value in master}) != len(master):
            raise ValueError("instrument IDs must be nonempty and unique")
        n = len(master)
        context = decision_context or RegimeInputs()
        if expected_returns is None and context.forecast_available:
            raise ValueError(
                "missing forecasts require an explicit forecast_available=False fallback declaration"
            )
        mu = np.zeros(n) if expected_returns is None else np.asarray(expected_returns, dtype=float)
        if mu.shape != (n,) or not np.all(np.isfinite(mu)) or np.any(mu < -1):
            raise ValueError(
                "forecast vector must contain finite SIMPLE expected returns >= -1 in instrument order"
            )
        cov = validate_covariance(covariance, n)
        if not np.isfinite(nav) or nav <= 0 or not np.isfinite(reserve_cash) or reserve_cash < 0:
            raise ValueError("NAV must be positive and cash reserve nonnegative, both finite")
        if not np.isfinite(cash_return) or cash_return < -1:
            raise ValueError("cash return must be finite and >= -1 over the same horizon")
        if current_weights is None:
            raise ValueError(
                "current holdings are required; explicitly pass zeros for an initial all-cash portfolio"
            )
        current = np.asarray(current_weights, dtype=float)
        if (
            current.shape != (n,)
            or not np.all(np.isfinite(current))
            or np.any(current < 0)
            or current.sum() > 1 + cfg.constraint_tolerance
        ):
            raise ValueError("current weights must be aligned, finite, nonnegative and sum to at most one")
        scenario_matrix = None if scenarios is None else np.asarray(scenarios, dtype=float)
        probabilities = None
        if scenario_matrix is not None:
            if (
                scenario_matrix.ndim != 2
                or scenario_matrix.shape[1] != n
                or not np.all(np.isfinite(scenario_matrix))
                or np.any(scenario_matrix < -1)
            ):
                raise ValueError(
                    "scenario matrix must be finite simple returns [scenarios,assets], each >= -1"
                )
            probabilities = _validated_probabilities(len(scenario_matrix), scenario_probabilities)
        elif scenario_probabilities is not None or cfg.cvar_penalty > 0:
            raise ValueError("CVaR penalty/probabilities require an explicit aligned scenario matrix")
        if context.inputs_available_at is not None and utc_timestamp(
            context.inputs_available_at
        ) > utc_timestamp(as_of):
            raise ValueError("regime inputs are not available at the decision timestamp")
        if (
            any(
                x is not None
                for x in (context.recession_probability, context.drawdown, context.annualized_volatility)
            )
            and context.inputs_available_at is None
        ):
            raise ValueError("supplied regime observations need an inputs_available_at timestamp")
        regime, reasons = choose_regime(context, cfg.regime_policy)
        diagnostics.update(
            {
                "regime_reasons": reasons,
                "config": asdict(cfg),
                "cost_assumption": "Per-side costs are declared research assumptions unless replaced with dated verified schedules.",
            }
        )
        if regime == "no_action":
            return AllocationResult("no_action", None, None, regime, diagnostics)
        problem = _PortfolioProblem(master, mandate, as_of, float(nav), current, float(reserve_cash), cfg)
        # An untradable instrument with no position is fixed at zero.
        problem.upper[problem.untradable & (current == 0)] = 0
    except (ValueError, TypeError, OverflowError) as exc:
        diagnostics["blocking_reasons"] = [str(exc)]
        return AllocationResult("blocked", None, None, "no_action", diagnostics)
    diagnostics["instrument_exclusions"] = {k: v for k, v in problem.exclusions.items() if v}
    diagnostics["hard_constraints"] = (
        "All optimizer and fallback candidates use the same dated mandate, eligibility, reserve, costs, liquidity and turnover checks."
    )

    risk_multiplier = cfg.regime_policy.defensive_risk_multiplier if regime == "defensive" else 1.0

    def objective(weights: np.ndarray, mode: str) -> float:
        alpha, delta, _ = problem.state(weights)
        cash = 1 - float(weights.sum())
        variance = float(alpha * alpha * (weights @ cov @ weights))
        turnover = float(np.abs(delta).sum())
        cvar = 0.0
        if cfg.cvar_penalty and scenario_matrix is not None:
            scenario_return = alpha * (1 + scenario_matrix @ weights + cash * cash_return) - 1
            cvar = weighted_cvar(-scenario_return, cfg.cvar_confidence, probabilities)
        if mode == "minimum_variance":
            return (
                cfg.risk_aversion * variance
                + cfg.cvar_penalty * cvar
                + cfg.turnover_penalty * turnover
                + (1 - alpha)
            )
        if mode == "cash":
            # Cash preference remains subject to maximum cash and all group floors.
            return -cash + cfg.risk_aversion * variance + cfg.turnover_penalty * turnover + (1 - alpha)
        expected = alpha * (1 + float(weights @ mu) + cash * cash_return) - 1
        preference = 0.0
        if regime == "defensive":
            preference = 0.10 * max(cfg.regime_policy.defensive_cash_floor - cash, 0) ** 2
        return (
            -expected
            + cfg.risk_aversion * risk_multiplier * variance
            + cfg.cvar_penalty * cvar
            + cfg.turnover_penalty * turnover
            + preference
        )

    bounds = [(0.0, float(x)) for x in problem.upper]
    constraints: list[dict[str, Any]] = [{"type": "ineq", "fun": problem.inequalities}]
    if np.any(problem.untradable & (current > 0)):
        constraints.append({"type": "eq", "fun": problem.equalities})
    eligible = problem.upper > 0
    equal = np.zeros(len(master))
    if eligible.any():
        desired_risky = min(1 - float(mandate.min_cash_weight), max(1 - float(mandate.max_cash_weight), 0.5))
        equal[eligible] = min(desired_risky / eligible.sum(), 1)
        equal = np.minimum(equal, problem.upper)
    seeds = [np.minimum(current, problem.upper), equal, np.zeros(len(master))]
    unique_seeds: list[np.ndarray] = []
    for seed in seeds:
        if not any(np.array_equal(seed, other) for other in unique_seeds):
            unique_seeds.append(seed)

    def attempt(mode: str) -> np.ndarray | None:
        best: tuple[float, np.ndarray] | None = None
        for seed_index, seed in enumerate(unique_seeds):
            try:
                solved = minimize(
                    lambda weights: objective(weights, mode),
                    seed,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": cfg.max_iterations, "ftol": cfg.tolerance, "disp": False},
                )
                check = problem.check(np.asarray(solved.x, dtype=float))
                diagnostics["solver_attempts"].append(
                    {
                        "mode": mode,
                        "seed_index": seed_index,
                        "success": bool(solved.success),
                        "message": str(solved.message),
                        "iterations": int(getattr(solved, "nit", 0)),
                        "postcheck_feasible": check["feasible"],
                        "violations": check["violations"],
                    }
                )
                if solved.success and check["feasible"]:
                    value = objective(solved.x, mode)
                    if best is None or value < best[0]:
                        best = (value, np.asarray(solved.x, dtype=float))
            except (ValueError, FloatingPointError, RuntimeError, OverflowError) as exc:
                diagnostics["solver_attempts"].append(
                    {"mode": mode, "seed_index": seed_index, "success": False, "message": str(exc)}
                )
        return None if best is None else best[1]

    initial_mode = (
        "cash" if regime == "cash" else "main" if context.forecast_available else "minimum_variance"
    )
    weights = attempt(initial_mode)
    status = "optimized" if context.forecast_available else "fallback_minimum_variance"
    if weights is None and initial_mode != "minimum_variance":
        weights = attempt("minimum_variance")
        status = "fallback_minimum_variance"
    if weights is None:
        for fallback_name, candidate in (("cash", np.zeros(len(master))), ("hold", current)):
            checked = problem.check(candidate)
            diagnostics["solver_attempts"].append(
                {"mode": fallback_name, "success": checked["feasible"], "violations": checked["violations"]}
            )
            if checked["feasible"]:
                weights, status = candidate.copy(), f"fallback_{fallback_name}"
                break
    if weights is None:
        diagnostics["blocking_reasons"] = [
            "No candidate satisfies all original constraints. No fallback allocation is certified."
        ]
        return AllocationResult("infeasible", None, None, regime, diagnostics)
    final = problem.check(weights)
    if not final["feasible"]:
        diagnostics["blocking_reasons"] = ["Final independent constraint check failed"]
        diagnostics["final_check"] = final
        return AllocationResult("infeasible", None, None, regime, diagnostics)
    diagnostics["final_check"] = final
    diagnostics["expected_simple_return_net_cost"] = (
        float(final["nav_retention_after_costs"] * (1 + weights @ mu + (1 - weights.sum()) * cash_return) - 1)
        if context.forecast_available
        else None
    )
    diagnostics["variance_at_forecast_horizon"] = float(
        final["nav_retention_after_costs"] ** 2 * weights @ cov @ weights
    )
    if scenario_matrix is not None:
        diagnostics["scenario_risk"] = portfolio_risk_summary(
            weights,
            scenario_matrix,
            cash_return=cash_return,
            confidence=cfg.cvar_confidence,
            probabilities=probabilities,
            nav_retention=final["nav_retention_after_costs"],
        )
    diagnostics["implementation_limitations"] = [
        "Continuous research weights; use round_allocation_to_lots and recheck before any executable-lot interpretation.",
        "Scenario values and cost/liquidity estimates are assumptions; trade fills and funding outcomes are not guaranteed.",
        "Country/sector exposure classifications, approved instruments and rule versions must be verified at each decision.",
    ]
    return AllocationResult(
        status,
        dict(zip(problem.ids, [float(x) for x in weights], strict=True)),
        float(1 - weights.sum()),
        regime,
        diagnostics,
    )


def _validated_probabilities(count: int, supplied: Sequence[float] | None) -> np.ndarray:
    # Separate name avoids shadowing the optimize_portfolio parameter.
    return scenario_probabilities(count, supplied)


def validate_allocation(
    weights: Sequence[float],
    instruments: Sequence[Instrument] | pd.DataFrame,
    mandate: Mandate,
    *,
    as_of: Any,
    nav: float,
    current_weights: Sequence[float],
    reserve_cash: float = 0.0,
    config: PortfolioConfig | None = None,
) -> dict[str, Any]:
    """Check an external proposed allocation using the exact optimizer restrictions."""
    cfg = config or PortfolioConfig()
    cfg.validate()
    issues = mandate.validate(as_of, require_allocation=True)
    if issues:
        return {"feasible": False, "violations": {"mandate": issues}}
    master = _instruments(instruments)
    current = np.asarray(current_weights, dtype=float)
    if (
        not master
        or len({value.asset_id for value in master}) != len(master)
        or current.shape != (len(master),)
        or not np.all(np.isfinite(current))
        or np.any(current < 0)
        or current.sum() > 1 + cfg.constraint_tolerance
    ):
        raise ValueError("valid aligned current holdings are required")
    if not np.isfinite(nav) or nav <= 0 or not np.isfinite(reserve_cash) or reserve_cash < 0:
        raise ValueError("positive NAV and nonnegative cash reserve are required")
    problem = _PortfolioProblem(master, mandate, as_of, nav, current, reserve_cash, cfg)
    problem.upper[problem.untradable & (current == 0)] = 0
    return problem.check(np.asarray(weights, dtype=float))


def round_allocation_to_lots(
    allocation: AllocationResult,
    instruments: Sequence[Instrument] | pd.DataFrame,
    mandate: Mandate,
    *,
    as_of: Any,
    nav: float,
    current_quantities: Sequence[float],
    reserve_cash: float = 0.0,
    config: PortfolioConfig | None = None,
) -> dict[str, Any]:
    """Floor target quantities to allowed lots and certify the resulting cash/weights.

    Rounding may break allocation floors, turnover, or reserve constraints. In
    that case the returned status is infeasible, with quantities withheld.
    These are hypothetical target quantities, not submitted trade instructions.
    """
    if allocation.weights is None:
        return {"status": "blocked", "quantities": None, "reason": "no verified continuous allocation"}
    cfg = config or PortfolioConfig()
    cfg.validate()
    master = _instruments(instruments)
    if set(allocation.weights) != {value.asset_id for value in master}:
        raise ValueError("allocation IDs and instrument master must match exactly")
    quantities = np.asarray(current_quantities, dtype=float)
    if quantities.shape != (len(master),) or not np.all(np.isfinite(quantities)) or np.any(quantities < 0):
        raise ValueError("current quantities must be finite, nonnegative and aligned")
    prices = np.asarray([value.price_base for value in master], dtype=float)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0) or not np.isfinite(nav) or nav <= 0:
        raise ValueError("finite positive base-currency prices and NAV are required")
    current_weights = quantities * prices / nav
    continuous = np.array([allocation.weights[value.asset_id] for value in master])
    continuous_check = validate_allocation(
        continuous,
        master,
        mandate,
        as_of=as_of,
        nav=nav,
        current_weights=current_weights,
        reserve_cash=reserve_cash,
        config=cfg,
    )
    if not continuous_check["feasible"]:
        return {"status": "infeasible", "quantities": None, "constraint_check": continuous_check}
    lots = np.array([1 if value.odd_lots_allowed else value.lot_size for value in master], dtype=float)
    target_value = continuous * continuous_check["post_cost_nav"]
    target = np.floor(target_value / (prices * lots)) * lots
    for index, value in enumerate(master):
        if not value.tradable:
            target[index] = quantities[index]
    delta_values = (target - quantities) * prices
    buy_rates = np.array([cfg.buy_cost_bps.get(value.asset_id, cfg.cost_bps) / 10000 for value in master])
    sell_rates = np.array([cfg.sell_cost_bps.get(value.asset_id, cfg.cost_bps) / 10000 for value in master])
    fees = float(buy_rates @ np.maximum(delta_values, 0) + sell_rates @ np.maximum(-delta_values, 0))
    post_nav = nav - fees
    if post_nav <= 0:
        return {"status": "infeasible", "quantities": None, "reason": "costs exhaust NAV"}
    rounded_weights = target * prices / post_nav
    checked = validate_allocation(
        rounded_weights,
        master,
        mandate,
        as_of=as_of,
        nav=nav,
        current_weights=current_weights,
        reserve_cash=reserve_cash,
        config=cfg,
    )
    if not checked["feasible"]:
        return {
            "status": "infeasible",
            "quantities": None,
            "constraint_check": checked,
            "reason": "Lot rounding violates at least one original constraint; do not silently relax it.",
        }
    return {
        "status": "research_lots_validated",
        "quantities": dict(zip([value.asset_id for value in master], target.tolist(), strict=True)),
        "weights": dict(zip([value.asset_id for value in master], rounded_weights.tolist(), strict=True)),
        "cash_amount": float(post_nav - target @ prices),
        "estimated_cost": fees,
        "constraint_check": checked,
        "research_only": True,
    }
