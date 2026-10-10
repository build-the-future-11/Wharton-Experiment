"""Opt-in, self-financing M11 successor; frozen ``mpc.py`` is unchanged.

One open-loop weight sequence is shared by every scenario. A period rebalances
at post-cost wealth, earns asset returns, liquidates proportionally if cash is
insufficient, then pays the liability and any previous arrears exactly once.
No borrowing or external funding is introduced. The primary objective is the
empirical CVaR of each scenario's maximum outstanding arrears.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from wharton_lab.competition.stress import rebalance
from wharton_lab.models.m11.mpc import PortfolioState


@dataclass(frozen=True)
class CashLedgerV2:
    cash: np.ndarray
    holdings: np.ndarray
    wealth: np.ndarray
    payments: np.ndarray
    arrears: np.ndarray
    fees: np.ndarray
    turnover: np.ndarray
    asset_gains: np.ndarray


@dataclass(frozen=True)
class MPCResultV2:
    first_action_weights: np.ndarray
    full_horizon_weights: np.ndarray
    success: bool
    message: str
    objective: float
    ledger: CashLedgerV2


def _inputs(state, scenario_returns, obligations, cost_bps):
    returns = np.asarray(scenario_returns, dtype=np.float64)
    due = np.asarray(obligations, dtype=np.float64)
    holdings = np.asarray(state.holdings, dtype=np.float64)
    if returns.ndim != 3 or not all(returns.shape):
        raise ValueError("scenario_returns must be nonempty [scenarios, horizon, assets]")
    if due.shape != (returns.shape[1],) or holdings.shape != (returns.shape[2],):
        raise ValueError("obligations and holdings must match the horizon and assets")
    if (not np.isfinite(returns).all() or np.any(returns < -1)
            or not np.isfinite(due).all() or np.any(due < 0)
            or not np.isfinite(holdings).all() or np.any(holdings < 0)
            or not np.isfinite(state.cash) or state.cash < 0
            or not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000):
        raise ValueError("inputs must be finite, long-only, with simple returns >= -1 and costs < 10000 bps")
    wealth = float(state.cash + holdings.sum())
    if not np.isfinite(wealth):
        raise ValueError("initial wealth is not representable")
    return returns, due, holdings


def simulate_cash_ledger_v2(
    *, state: PortfolioState, scenario_returns: np.ndarray,
    obligations: np.ndarray, weights: np.ndarray, cost_bps: float = 0.0,
) -> CashLedgerV2:
    """Retain every period of every scenario, including unpaid liabilities.

    Weights are risky-asset fractions of wealth after rebalancing fees; their
    unused fraction is cash. Arrays have shape [scenario, period], except
    holdings [scenario, period, asset]. Arrears are carried, not counted as a
    new scheduled liability. ``sum(payments)+last_arrears == sum(obligations)``.
    """
    returns, due, initial_holdings = _inputs(state, scenario_returns, obligations, cost_bps)
    n, horizon, assets = returns.shape
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (horizon, assets) or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(weights.sum(axis=1) > 1 + 1e-12)):
        raise ValueError("weights must be finite [horizon, assets], nonnegative, with row sums <= 1")
    cash = np.zeros((n, horizon))
    holdings = np.zeros((n, horizon, assets))
    wealth, payments, arrears, fees, turnover, gains = [np.zeros((n, horizon)) for _ in range(6)]
    rate = cost_bps / 10000.0
    for s in range(n):
        h = initial_holdings.copy()
        c, unpaid = float(state.cash), 0.0
        for t in range(horizon):
            h, c, fee, traded = rebalance(h, c, weights[t], cost_bps)
            gains[s, t] = float(h @ returns[s, t])
            h = h * (1 + returns[s, t])
            current_due = float(due[t] + unpaid)
            holding_value = float(h.sum())
            sale = min(holding_value, max(0.0, current_due - c) / (1 - rate))
            sale_fee = rate * sale
            if holding_value > 0:
                h *= max(0.0, 1.0 - sale / holding_value)
            c += sale - sale_fee
            paid = min(c, current_due)
            c = max(0.0, c - paid)
            unpaid = max(0.0, current_due - paid)
            cash[s, t], holdings[s, t] = c, h
            wealth[s, t] = c + float(h.sum())
            payments[s, t], arrears[s, t] = paid, unpaid
            fees[s, t], turnover[s, t] = fee + sale_fee, traded + sale
    ledger = CashLedgerV2(cash, holdings, wealth, payments, arrears, fees, turnover, gains)
    if not all(np.isfinite(x).all() for x in vars(ledger).values()):
        raise ArithmeticError("non-finite cash ledger; the scenario cannot be reported as valid")
    return ledger


def empirical_cvar_v2(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Average exactly the upper (1-alpha) empirical probability mass.

    Fractional mass at the boundary is retained; ties at an empirical quantile
    must not expand the requested tail or underweight the worst outcomes.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 1 or not losses.size or not np.isfinite(losses).all():
        raise ValueError("losses must be a nonempty finite vector")
    if not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    scale = float(np.max(np.abs(losses)))
    if scale == 0:
        return 0.0
    ordered = np.sort(losses / scale)[::-1]
    mass = (1 - alpha) * len(ordered)
    whole = int(np.floor(mass))
    fractional = mass - whole
    total = float(ordered[:whole].sum())
    if fractional > 0 and whole < len(ordered):
        total += fractional * float(ordered[whole])
    # A normalized weighted average cannot exceed the observed loss range.
    tail_mean = float(np.clip(total / mass, ordered[-1], ordered[0]))
    return tail_mean * scale


def solve_mpc_v2(
    *, state: PortfolioState, scenario_returns: np.ndarray, obligations: np.ndarray,
    cvar_alpha: float = 0.95, max_weight: float = 0.4, tcost_bps: float = 5.0,
    min_cash_buffer: float = 0.0, terminal_wealth_weight: float = 0.0,
    maxiter: int = 100,
) -> MPCResultV2:
    """Optimize the declared v2 objective without fallback investment advice.

    Cash is feasible even when assets*max_weight < 1. The optional nonnegative
    terminal-wealth reward is explicit and defaults to zero. A failed feasible
    optimization returns success=False with its message and ledger; infeasible
    controls raise. It never invents a 100% allocation to asset zero.
    """
    returns, due, _ = _inputs(state, scenario_returns, obligations, tcost_bps)
    _, horizon, assets = returns.shape
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if not np.isfinite(min_cash_buffer) or not 0 <= min_cash_buffer <= 1:
        raise ValueError("min_cash_buffer must be in [0, 1]")
    if not np.isfinite(terminal_wealth_weight) or terminal_wealth_weight < 0:
        raise ValueError("terminal_wealth_weight must be finite and nonnegative")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    empirical_cvar_v2(np.zeros(returns.shape[0]), cvar_alpha)
    allowed = 1.0 - min_cash_buffer

    def ledger_at(flat):
        return simulate_cash_ledger_v2(
            state=state, scenario_returns=returns, obligations=due,
            weights=flat.reshape(horizon, assets), cost_bps=tcost_bps,
        )

    def objective(flat):
        ledger = ledger_at(flat)
        return (empirical_cvar_v2(ledger.arrears.max(axis=1), cvar_alpha)
                - terminal_wealth_weight * float(ledger.wealth[:, -1].mean()))

    # One linear inequality per period; the same controls serve all scenarios.
    constraints = {"type": "ineq", "fun": lambda flat: allowed - flat.reshape(horizon, assets).sum(axis=1)}
    initial = np.zeros(horizon * assets)
    if allowed == 0:
        ledger = ledger_at(initial)
        return MPCResultV2(initial[:assets], initial.reshape(horizon, assets), True,
                           "Cash-only constraint; no optimization required", objective(initial), ledger)
    # SLSQP may evaluate infeasible trial points. Parameterize row exposure by
    # a cash-inclusive simplex projection for those trials, while retaining
    # the optimizer's original constraints and validating its final controls.
    def trial_objective(flat):
        trial = np.clip(flat.reshape(horizon, assets), 0.0, max_weight)
        sums = trial.sum(axis=1, keepdims=True)
        trial = trial * np.minimum(1.0, allowed / np.maximum(sums, 1e-300))
        return objective(trial.ravel())

    result = minimize(trial_objective, initial, method="SLSQP",
                      bounds=[(0.0, max_weight)] * initial.size, constraints=constraints,
                      options={"maxiter": int(maxiter), "ftol": 1e-10, "disp": False})
    attempted = np.asarray(result.x, dtype=float).reshape(horizon, assets)
    feasible = bool(np.isfinite(attempted).all() and np.all(attempted >= -1e-9)
                    and np.all(attempted <= max_weight + 1e-9)
                    and np.all(attempted.sum(axis=1) <= allowed + 1e-9))
    if not feasible:
        raise ArithmeticError(f"MPC optimizer returned infeasible controls: {result.message}")
    # Remove numerical boundary noise only; do not disguise optimizer failure.
    weights = np.clip(attempted, 0.0, max_weight)
    sums = weights.sum(axis=1, keepdims=True)
    weights *= np.minimum(1.0, allowed / np.maximum(sums, 1e-300))
    ledger = ledger_at(weights.ravel())
    return MPCResultV2(weights[0].copy(), weights, bool(result.success), str(result.message),
                       objective(weights.ravel()), ledger)
