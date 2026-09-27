"""Self-financing cash-flow stress model. All scenario inputs are assumptions.

Monthly order: rebalance using available wealth and known obligation, pay trading
costs, earn asset returns, pay the obligation once. Unpaid amounts are reported
and carried as arrears; no borrowing, external contributions, or forced rescue.
"""
from __future__ import annotations

import numpy as np


def rebalance(holdings, cash: float, weights, cost_bps: float):
    holdings = np.asarray(holdings, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (holdings.ndim != 1 or holdings.shape != weights.shape
            or not np.all(np.isfinite(holdings)) or not np.all(np.isfinite(weights))
            or not np.isfinite(cash) or cash < 0 or np.any(holdings < 0)
            or np.any(weights < 0) or weights.sum() > 1 + 1e-12
            or not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000):
        raise ValueError("invalid long-only portfolio or cost")
    wealth = float(cash + holdings.sum())
    cost_rate = cost_bps / 10000
    # Solve V_after + cost * sum(|w * V_after - current_holdings|) = V_before.
    lo, hi = 0.0, wealth
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid + cost_rate * np.abs(weights * mid - holdings).sum() > wealth:
            hi = mid
        else:
            lo = mid
    target = weights * ((lo + hi) / 2)
    turnover = float(np.abs(target - holdings).sum())
    fee = cost_rate * turnover
    remaining_cash = wealth - target.sum() - fee
    if remaining_cash < -1e-10:
        raise ArithmeticError("rebalance overspent available cash")
    return target, max(0.0, float(remaining_cash)), fee, turnover


def policy_weights(policy, wealth, current_due, n_assets, reserve_months,
                   planning_inflation, remaining_months):
    if policy == "fully_invested":
        cash_fraction = 0.0
    elif policy == "fixed_cash_20pct":
        cash_fraction = 0.2
    elif policy == "liability_reserve":
        months = min(reserve_months, remaining_months)
        factor = (1 + planning_inflation) ** (1 / 12)
        reserve = current_due * sum(factor ** j for j in range(months))
        cash_fraction = min(1.0, reserve / wealth) if wealth > 0 else 1.0
    else:
        raise ValueError(f"unknown policy: {policy}")
    return np.full(n_assets, (1 - cash_fraction) / n_assets)


def simulate(returns, obligations, *, policy="liability_reserve", cost_bps=10,
             initial_wealth=1.0, reserve_months=12, planning_inflation=0.025):
    returns = np.asarray(returns, dtype=float)
    obligations = np.asarray(obligations, dtype=float)
    if (returns.ndim != 2 or not all(returns.shape) or obligations.shape != (len(returns),)
            or not np.all(np.isfinite(returns)) or np.any(returns < -1)
            or not np.all(np.isfinite(obligations)) or np.any(obligations < 0)
            or not np.isfinite(initial_wealth) or initial_wealth <= 0
            or not isinstance(reserve_months, int) or reserve_months < 1
            or not np.isfinite(planning_inflation) or planning_inflation <= -1):
        raise ValueError("invalid scenario inputs")
    cash = float(initial_wealth)
    holdings = np.zeros(returns.shape[1])
    arrears = 0.0
    unit_value, peak, max_dd = 1.0, 1.0, 0.0
    records = []
    for t, (r, obligation) in enumerate(zip(returns, obligations)):
        wealth_start = cash + float(holdings.sum())
        due = float(obligation + arrears)
        weights = policy_weights(policy, wealth_start, due, len(holdings),
                                 reserve_months, planning_inflation, len(returns) - t)
        holdings, cash, rebalance_fee, turnover = rebalance(holdings, cash, weights, cost_bps)
        holdings *= 1 + r
        # Sell proportionally when cash is insufficient. Fee applies to sale value.
        sale = min(float(holdings.sum()), max(0.0, due - cash) / (1 - cost_bps / 10000))
        liquidation_fee = sale * cost_bps / 10000
        if holdings.sum() > 0:
            holdings *= 1 - sale / holdings.sum()
        cash += sale - liquidation_fee
        paid = min(cash, due)
        cash = max(0.0, cash - paid)
        arrears = max(0.0, due - paid)
        wealth_end = cash + float(holdings.sum())
        fee = rebalance_fee + liquidation_fee
        # Time-weighted, net-of-cost drawdown excludes the external withdrawal.
        net_return = (wealth_end + paid) / wealth_start - 1 if wealth_start > 0 else 0.0
        unit_value *= 1 + net_return
        peak = max(peak, unit_value)
        max_dd = max(max_dd, 1 - unit_value / peak)
        records.append(dict(month=t + 1, wealth_start=wealth_start,
                            obligation=float(obligation), due_with_arrears=due,
                            paid=paid, arrears=arrears, cash=cash, wealth_end=wealth_end,
                            fees=fee, turnover=turnover + sale,
                            net_return=net_return, unit_value=unit_value))
    total_due = float(obligations.sum())
    paid = sum(r["paid"] for r in records)
    return {
        "policy": policy, "cost_bps": cost_bps,
        "terminal_wealth": records[-1]["wealth_end"],
        "total_obligations": total_due, "total_paid": paid,
        "unpaid_at_end": arrears,
        "funded_fraction": paid / total_due if total_due else 1.0,
        "first_shortfall_month": next((r["month"] for r in records if r["arrears"] > 1e-10), None),
        "max_drawdown_ex_withdrawals": max_dd,
        "total_fees": sum(r["fees"] for r in records),
        "total_turnover": sum(r["turnover"] for r in records),
        "trajectory": records,
    }


def scenario_inputs(config, scenario):
    months = config["months"]
    annual = np.asarray(scenario["annual_asset_returns"], dtype=float)
    returns = np.tile((1 + annual) ** (1 / 12) - 1, (months, 1))
    for shock in scenario.get("shocks", []):
        returns[shock["month"] - 1] = shock["asset_returns"]
    inflation = (1 + scenario["annual_inflation"]) ** (1 / 12)
    obligations = (config["monthly_obligation"] * scenario["obligation_multiplier"]
                   * inflation ** np.arange(months))
    return returns, obligations
