"""Synchronous common-currency accounting and a fixed, unoptimized signal policy.

Student-t log returns have no finite exponential moment. We therefore use their
location only as a ranking signal, never call exp(mu + variance/2) an expected
simple return, and evaluate against actual simple returns in the ledger.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Forecast:
    asset: str
    location: float
    variance: float

    def __post_init__(self):
        if not self.asset or finite(self.variance, "variance") <= 0:
            raise ValueError("Forecast needs asset and positive variance")
        finite(self.location, "location")

    @classmethod
    def student(cls, asset, location, scale, df):
        scale, df = finite(scale, "scale"), finite(df, "df")
        if scale <= 0 or df <= 2:
            raise ValueError("Student scale >0 and df >2 required")
        return cls(asset, finite(location, "location"), scale * scale * df / (df - 2))


def ensemble(families):
    """Moment-match the log-return mixture without choosing a favorable seed."""
    if not families or not families[0]:
        raise ValueError("Empty ensemble")
    names = [f.asset for f in families[0]]
    if len(set(names)) != len(names) or any([f.asset for f in fs] != names for fs in families):
        raise ValueError("Ensemble asset order mismatch")
    result = []
    for column in zip(*families, strict=True):
        mu = statistics.fmean(f.location for f in column)
        var = statistics.fmean(f.variance + (f.location - mu) ** 2 for f in column)
        result.append(Forecast(column[0].asset, mu, var))
    return result


def target_weights(forecasts, *, anchor=0.8, risk_aversion=8.0, cap=0.15, eligible=None):
    """80% equal-weight anchor, up to 20% risk-adjusted positive signal, rest cash.

    Cap is raised to 1/n for a small research basket. This is a declared research
    convention, not a Laura mandate. Capped active weight stays cash; there is no
    hidden reallocation that could push another name through its cap.
    """
    if not forecasts or not 0 <= anchor <= 1 or risk_aversion <= 0 or not 0 < cap <= 1:
        raise ValueError("Invalid policy parameters")
    for value in (anchor, risk_aversion, cap):
        finite(value, "policy")
    names = [f.asset for f in forecasts]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate asset")
    eligible = set(names) if eligible is None else set(eligible)
    if not eligible <= set(names):
        raise ValueError("Unknown eligible asset")
    n = len(names)
    effective_cap = max(cap, 1 / n)
    raw = [min(1.0, max(0.0, f.location) / (risk_aversion * f.variance)) for f in forecasts]
    total = sum(v for f, v in zip(forecasts, raw, strict=True) if f.asset in eligible)
    return {
        f.asset: min(effective_cap, anchor / n + (1 - anchor) * v / max(1, total))
        if f.asset in eligible
        else 0.0
        for f, v in zip(forecasts, raw, strict=True)
    }


def rebalance(holdings, cash, weights, fee, *, no_trade_band=0.0):
    """Solve A + fee*sum(abs(w_i*A - h_i)) = pretrade NAV.

    Holdings are dollar values at a common execution instant. Cash is the
    remaining post-fee wealth. This supports full default without taking log(0).
    The optional band applies to aggregate risky-asset L1 weight change.
    """
    cash, fee = finite(cash, "cash"), finite(fee, "fee")
    if cash < 0 or not 0 <= fee < 1 or not 0 <= no_trade_band <= 2:
        raise ValueError("Invalid fee/cash/band")
    if set(holdings) != set(weights):
        raise ValueError("Exact asset universe required")
    for v in holdings.values():
        if finite(v, "holding") < 0:
            raise ValueError("Short holdings unsupported")
    for v in weights.values():
        if finite(v, "weight") < 0:
            raise ValueError("Negative weight")
    if sum(weights.values()) > 1 + 1e-12:
        raise ValueError("Leverage unsupported")
    nav = cash + sum(holdings.values())
    if nav == 0:
        return dict(holdings), 0.0, {"fee": 0.0, "notional": 0.0, "nav": 0.0}
    difference = sum(abs(weights[k] - holdings[k] / nav) for k in holdings)
    if difference <= no_trade_band:
        return dict(holdings), cash, {"fee": 0.0, "notional": 0.0, "nav": nav}
    low, high = 0.0, nav
    for _ in range(80):
        after = (low + high) / 2
        notional = sum(abs(after * weights[k] - holdings[k]) for k in holdings)
        if after + fee * notional > nav:
            high = after
        else:
            low = after
    after = (low + high) / 2
    values = {k: weights[k] * after for k in weights}
    notional = sum(abs(values[k] - holdings[k]) for k in weights)
    charged = fee * notional
    cash_after = after - sum(values.values())
    if abs(nav - after - charged) > 1e-8 * max(1, nav) or cash_after < -1e-8:
        raise ArithmeticError("Self-financing residual")
    return values, max(0.0, cash_after), {"fee": charged, "notional": notional, "nav": after}


def backtest(periods, *, policy="signal", cost_bps=10, initial_wealth=100000, no_trade_band=0.02):
    """Run complete nonoverlapping monthly periods; outcomes never enter policy.

    A period carries time, end, forecasts, returns, currency and optional
    available_at/eligible. Missing returns fail; -100% removes future tradability.
    This ledger assumes synchronous research execution, no settlement/fill claim.
    """
    if not periods or policy not in {"signal", "equal_weight", "cash", "buy_hold"}:
        raise ValueError("Unknown policy or empty history")
    fee = finite(cost_bps, "cost_bps") / 10000
    initial_wealth = finite(initial_wealth, "wealth")
    if initial_wealth <= 0 or not 0 <= fee < 1:
        raise ValueError("Invalid initial capital or cost")
    names = [f.asset for f in periods[0]["forecasts"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Empty/duplicate universe")
    currency = periods[0]["currency"]
    holdings, cash = dict.fromkeys(names, 0.0), initial_wealth
    dead, rows = set(), []
    previous_end, peak, max_drawdown = None, initial_wealth, 0.0
    total_fees = total_notional = 0.0
    for step, period in enumerate(periods):
        t, end = period["time"], period["end"]
        if t >= end or (previous_end is not None and t < previous_end):
            raise ValueError("Overlapping/backward intervals")
        if period.get("available_at", t) > t:
            raise ValueError("Forecast not available at execution")
        if period["currency"] != currency:
            raise ValueError("Mixed currencies need a verified FX ledger")
        if [f.asset for f in period["forecasts"]] != names or set(period["returns"]) != set(names):
            raise ValueError("Incomplete asset grid")
        eligible = set(period.get("eligible", names)) - dead
        if not eligible <= set(names):
            raise ValueError("Unknown eligible asset")
        before = cash + sum(holdings.values())
        if policy == "signal":
            weights = target_weights(period["forecasts"], eligible=eligible)
        elif policy in {"equal_weight", "buy_hold"}:
            weights = {k: 1 / len(names) if k in eligible else 0.0 for k in names}
        else:
            weights = dict.fromkeys(names, 0.0)
        if policy == "buy_hold" and step:
            trade = {"fee": 0.0, "notional": 0.0, "nav": before}
        else:
            holdings, cash, trade = rebalance(
                holdings,
                cash,
                weights,
                fee,
                no_trade_band=no_trade_band if policy == "signal" else 0.0,
            )
        executed_weights = {k: holdings[k] / trade["nav"] if trade["nav"] else 0 for k in names}
        for k in names:
            ret = finite(period["returns"][k], "realized return")
            if ret < -1:
                raise ValueError("Return below total loss")
            holdings[k] *= 1 + ret
            if ret == -1:
                dead.add(k)
        cash_return = finite(period.get("cash_return", 0.0), "cash return")
        if cash_return < -1:
            raise ValueError("Cash return below -100%")
        cash *= 1 + cash_return
        after = cash + sum(holdings.values())
        total_fees += trade["fee"]
        total_notional += trade["notional"]
        peak = max(peak, after)
        max_drawdown = max(max_drawdown, 1 - after / peak)
        rows.append(
            {
                "time": t,
                "end": end,
                "nav_start": before,
                "nav_end": after,
                "fee": trade["fee"],
                "notional": trade["notional"],
                "weights": executed_weights,
                "cash": cash,
                "holdings": dict(holdings),
                "net_factor": after / before if before else 1.0,
            }
        )
        previous_end = end
    liquidation = fee * sum(holdings.values())
    terminal = cash + sum(holdings.values()) - liquidation
    total_fees += liquidation
    total_notional += sum(holdings.values())
    rows[-1]["liquidation_fee"] = liquidation
    rows[-1]["nav_end"] = terminal
    rows[-1]["net_factor"] = terminal / rows[-1]["nav_start"] if rows[-1]["nav_start"] else 1.0
    max_drawdown = max(max_drawdown, 1 - terminal / peak)
    return {
        "policy": policy,
        "currency": currency,
        "terminal_wealth": terminal,
        "net_return": terminal / initial_wealth - 1,
        "max_drawdown": max_drawdown,
        "fees": total_fees,
        "traded_notional": total_notional,
        "periods": rows,
        "defaults": sorted(dead),
    }


def relative_wealth(candidate, benchmark):
    denominator = finite(benchmark, "benchmark wealth")
    candidate = finite(candidate, "candidate wealth")
    if denominator <= 0 or candidate < 0:
        raise ValueError("Relative wealth requires positive benchmark and nonnegative candidate")
    return candidate / denominator - 1


def paired_interval(candidate, benchmark, *, block=3, repetitions=2000, seed=1901):
    """Conditional path bootstrap, not an independent-policy reproduction."""
    if len(candidate) != len(benchmark) or block < 1 or repetitions < 100:
        raise ValueError("Invalid bootstrap inputs")
    if len(candidate) < 2 * block:
        return {"status": "INSUFFICIENT_DATES", "dates": len(candidate), "block": block}
    if any(x <= 0 or not math.isfinite(x) for x in [*candidate, *benchmark]):
        return {"status": "UNDEFINED_LOG_INCREMENT_AFTER_RUIN"}
    increments = [math.log(a) - math.log(b) for a, b in zip(candidate, benchmark, strict=True)]
    rng, n = random.Random(seed), len(increments)
    sampled = []
    for _ in range(repetitions):
        positions = []
        while len(positions) < n:
            start = rng.randrange(n)
            positions.extend((start + k) % n for k in range(block))
        sampled.append(math.expm1(sum(increments[i] for i in positions[:n])))
    sampled.sort()
    return {
        "status": "EXPLORATORY_CONDITIONAL_PATH",
        "block": block,
        "dates": n,
        "repetitions": repetitions,
        "seed": seed,
        "estimate": math.expm1(sum(increments)),
        "interval": [sampled[int(0.025 * repetitions)], sampled[int(0.975 * repetitions) - 1]],
        "not_joint_or_future_coverage": True,
    }


def laura_path(annual_returns, *, reserve_discount=0.0, flexibility_buffer=None):
    """Beginning-year contributions; value just before the 2033 first payment."""
    if len(annual_returns) != 6:
        raise ValueError("Six annual returns for 2027-2032 required")
    d = finite(reserve_discount, "reserve discount")
    if d <= -1:
        raise ValueError("Invalid discount")
    wealth, history = 0.0, []
    for year, value in zip(range(2027, 2033), annual_returns, strict=True):
        value = finite(value, "annual return")
        if value < -1:
            raise ValueError("Annual return below total loss")
        wealth += 300000 if year == 2027 else 150000 if year == 2028 else 0
        wealth *= 1 + value
        history.append({"year": year, "ending_wealth": wealth})
    reserve = sum(50000 / (1 + d) ** k for k in range(10))
    if flexibility_buffer is not None and finite(flexibility_buffer, "buffer") < 0:
        raise ValueError("Negative flexibility buffer")
    return {
        "wealth_before_2033_payment": wealth,
        "reserve_annuity_due": reserve,
        "reserve_shortfall": max(0.0, reserve - wealth),
        "surplus_before_buffer": wealth - reserve,
        "facility_capacity": None
        if flexibility_buffer is None
        else max(0.0, wealth - reserve - flexibility_buffer),
        "funding_probability": None,
        "assumed_reserve_discount": d,
        "assumptions_not_current_yields": True,
        "history": history,
    }


def external_forecasts(payload, *, decision_time, assets, currency, horizon):
    """Explicit adapter for the user's future framework; reject unavailable data.

    Timestamp attestation must be backed by source receipts; this parser cannot
    certify truthfulness or turn an exploratory forecast into a current mandate.
    """

    def timestamp(value):
        result = datetime.fromisoformat(value)
        if result.tzinfo is None:
            raise ValueError("Timezone-qualified timestamps required")
        return result

    decision = timestamp(decision_time)
    if payload.get("currency") != currency or payload.get("horizon") != horizon:
        raise ValueError("Forecast currency/horizon mismatch")
    if not payload.get("source_sha256") or len(payload["source_sha256"]) != 64:
        raise ValueError("Source identity required")
    if any(c not in "0123456789abcdef" for c in payload["source_sha256"]):
        raise ValueError("Malformed source hash")
    for key in ("trained_through", "max_input_available_at", "generated_at"):
        if timestamp(payload[key]) > decision:
            raise ValueError(f"Unavailable future information: {key}")
    rows = payload["forecasts"]
    if len(rows) != len(assets) or {r["asset"] for r in rows} != set(assets):
        raise ValueError("Exact forecast universe required")
    mapped = {r["asset"]: Forecast.student(r["asset"], r["location"], r["scale"], r["df"]) for r in rows}
    return [mapped[a] for a in assets]
