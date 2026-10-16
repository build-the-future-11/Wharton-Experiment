"""Continuous concave allocation with a declared committed holding horizon.

The objective is a log-moment ranking surrogate. It is not an expected-simple-
return forecast, and diagonal risk and constant multiweek coefficients are
explicit approximations. Actual holdings drift and pay self-financing fees.
"""

import math

import numpy as np

from successors.frem_portfolio_v1.engine import rebalance


def objective(weights, current, mean, variance, horizon, fee):
    w, h, mu, v = (np.asarray(x, float) for x in (weights, current, mean, variance))
    return float(horizon * np.sum(mu * w - 4 * v * w**2) - fee * np.abs(w - h).sum() - fee * w.sum())


def continuous_target(current, mean, variance, *, horizon=4, fee=0.001, budget=0.8, cap=1 / 3):
    h, mu, v = (np.asarray(x, float) for x in (current, mean, variance))
    if (
        h.ndim != 1
        or not len(h)
        or mu.shape != h.shape
        or v.shape != h.shape
        or not all(np.isfinite(x).all() for x in (h, mu, v))
        or (h < 0).any()
        or h.sum() > 1 + 1e-10
        or (v <= 0).any()
        or type(horizon) is not int
        or horizon < 1
        or not all(math.isfinite(x) for x in (fee, budget, cap))
        or not 0 <= fee < 1
        or not 0 <= budget <= 1
        or not 0 < cap <= 1
    ):
        raise ValueError("invalid continuous allocation inputs")

    def at(multiplier):
        b = horizon * mu - fee - multiplier
        above = (b - fee) / (8 * horizon * v)
        below = (b + fee) / (8 * horizon * v)
        return np.clip(np.where(above > h, above, np.where(below < h, below, h)), 0, cap)

    result = at(0)
    if result.sum() > budget:
        low, high = 0.0, 1.0
        while at(high).sum() > budget:
            high *= 2
        for _ in range(80):
            mid = (low + high) / 2
            if at(mid).sum() > budget:
                low = mid
            else:
                high = mid
        result = at(high)
    return result


def holding_policy(periods, cost_bps, *, horizon=4):
    if not periods or type(horizon) is not int or horizon < 1:
        raise ValueError("nonempty periods and positive integer holding horizon required")
    rate = float(cost_bps) / 10000
    if not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("invalid cost")
    names = [f.asset for f in periods[0]["forecasts"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("invalid asset universe")
    holdings, cash, rows = dict.fromkeys(names, 0.0), 100000.0, []
    previous_end = None
    peak, drawdown = cash, 0.0
    for step, p in enumerate(periods):
        if (
            p["time"] >= p["end"]
            or (previous_end is not None and p["time"] != previous_end)
            or p.get("available_at", p["time"]) > p["time"]
            or p["currency"] != "USD"
            or [f.asset for f in p["forecasts"]] != names
            or set(p["returns"]) != set(names)
        ):
            raise ValueError("continuous, synchronized, available USD observations required")
        before = cash + sum(holdings.values())
        current = np.array([holdings[a] / before if before else 0 for a in names])
        decision = None
        if step % horizon == 0 and before:
            mu = np.array([f.location for f in p["forecasts"]])
            var = np.array([f.variance for f in p["forecasts"]])
            remaining = min(horizon, len(periods) - step)
            target = continuous_target(current, mu, var, horizon=remaining, fee=rate)
            decision = {
                "horizon": remaining,
                "current": current.tolist(),
                "target": target.tolist(),
                "mean": mu.tolist(),
                "variance": var.tolist(),
                "objective": objective(target, current, mu, var, remaining, rate),
                "hold_objective": objective(current, current, mu, var, remaining, rate),
                "cash_objective": objective(np.zeros(len(names)), current, mu, var, remaining, rate),
            }
            holdings, cash, trade = rebalance(holdings, cash, dict(zip(names, target, strict=True)), rate)
        else:
            trade = {"fee": 0.0, "notional": 0.0, "nav": before}
        weights = {a: holdings[a] / trade["nav"] if trade["nav"] else 0 for a in names}
        contributions = {}
        for a in names:
            ret = p["returns"][a]
            if not math.isfinite(ret) or ret < -1:
                raise ValueError("invalid realized return")
            contributions[a] = holdings[a] * ret
            holdings[a] *= 1 + ret
        after = cash + sum(holdings.values())
        peak = max(peak, after)
        drawdown = max(drawdown, 1 - after / peak)
        rows.append(
            {
                "time": p["time"],
                "end": p["end"],
                "nav_start": before,
                "nav_end": after,
                "net_factor": after / before if before else 1.0,
                "fee": trade["fee"],
                "notional": trade["notional"],
                "weights": weights,
                "cash": cash,
                "holdings": dict(holdings),
                "contributions": contributions,
                "decision": decision,
            }
        )
        previous_end = p["end"]
    liquidation = rate * sum(holdings.values())
    terminal = cash + sum(holdings.values()) - liquidation
    rows[-1]["liquidation_fee"] = liquidation
    rows[-1]["nav_end"] = terminal
    rows[-1]["net_factor"] = terminal / rows[-1]["nav_start"] if rows[-1]["nav_start"] else 1.0
    return {
        "policy": f"s8_h{horizon}",
        "terminal_wealth": terminal,
        "net_return": terminal / 100000 - 1,
        "max_drawdown": max(drawdown, 1 - terminal / peak),
        "fees": sum(r["fee"] for r in rows) + liquidation,
        "traded_notional": sum(r["notional"] for r in rows) + sum(holdings.values()),
        "liquidation_fee": liquidation,
        "periods": rows,
    }
