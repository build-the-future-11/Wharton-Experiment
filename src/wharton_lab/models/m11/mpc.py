"""Receding-horizon allocation (QP-lite / SLSQP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from wharton_lab.evaluation.metrics import funding_shortfall


@dataclass
class PortfolioState:
    holdings: np.ndarray
    cash: float


@dataclass
class MPCResult:
    first_action_weights: np.ndarray
    full_horizon_weights: np.ndarray
    success: bool
    message: str
    objective: float


def _scenario_cash_paths(
    weights_seq: np.ndarray,
    scenario_returns: np.ndarray,
    initial_cash: float,
    initial_holdings: np.ndarray,
    obligations: np.ndarray,
    tcost: float,
) -> np.ndarray:
    """weights_seq: (H, k) — same first row enforced externally."""
    n_scen, H, k = scenario_returns.shape
    paths = np.zeros((n_scen, H))
    for s in range(n_scen):
        cash = float(initial_cash)
        h = np.asarray(initial_holdings, dtype=float).copy()
        port = cash + float(h.sum())
        for t in range(H):
            w = weights_seq[t]
            h = w * port
            cash = port - float(h.sum())
            r = scenario_returns[s, t]
            port = cash + float((h * (1 + r)).sum())
            cash = port - float(h.sum())
            port -= tcost * float(np.sum(np.abs(np.diff(np.r_[0, w]))))
            cash = max(cash - float(obligations[t]), cash - float(obligations[t]))
            paths[s, t] = cash - float(obligations[t])
    return paths


def solve_mpc(
    *,
    state: PortfolioState,
    scenario_returns: np.ndarray,
    obligations: np.ndarray,
    horizon: int,
    cvar_alpha: float,
    max_weight: float,
    tcost_bps: float,
) -> MPCResult:
    scenario_returns = np.asarray(scenario_returns, dtype=float)
    obligations = np.asarray(obligations, dtype=float).ravel()
    k = scenario_returns.shape[-1]
    if obligations.shape[0] != horizon:
        raise ValueError("obligations length must match horizon")

    n_vars = horizon * k
    w0 = np.full(n_vars, 1.0 / k)

    def unpack(w_flat: np.ndarray) -> np.ndarray:
        return w_flat.reshape(horizon, k)

    tcost = tcost_bps * 1e-4

    def objective(w_flat: np.ndarray) -> float:
        w_seq = unpack(w_flat)
        cash_paths = _scenario_cash_paths(
            w_seq,
            scenario_returns,
            state.cash,
            state.holdings,
            obligations,
            tcost,
        )
        fs = funding_shortfall(cash_paths, obligations, alpha=cvar_alpha)
        return fs + 0.01 * float(np.sum(w_flat**2))

    bounds = [(0.0, max_weight)] * n_vars
    constraints = []

    def _sum_slice(w_flat: np.ndarray, start: int, end: int) -> float:
        return float(np.sum(w_flat[start:end]) - 1.0)

    for t in range(horizon):
        start, end = t * k, (t + 1) * k
        constraints.append(
            {"type": "eq", "fun": lambda w, s=start, e=end: _sum_slice(w, s, e)}
        )

    # Non-anticipativity: first-period weights equal across scenarios (single w_0)
    # Enforced by sharing variables — one weight vector per time, not per scenario.

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 15, "ftol": 1e-3, "disp": False},
    )
    w_seq = unpack(res.x)
    return MPCResult(
        first_action_weights=w_seq[0],
        full_horizon_weights=w_seq,
        success=bool(res.success),
        message=str(res.message),
        objective=float(res.fun),
    )
