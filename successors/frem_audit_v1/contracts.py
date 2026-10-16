"""Exact small mathematical references, not a forecast or investment model."""

import math
from dataclasses import dataclass


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ReturnInterval:
    start: float
    end: float
    available: float
    simple_return: float

    def __post_init__(self):
        for key in ("start", "end", "available", "simple_return"):
            finite(getattr(self, key), key)
        if self.start >= self.end or self.available < self.end or self.simple_return < -1:
            raise ValueError("invalid return interval")


def compound(intervals, *, start, end, decision_time):
    """Compound exactly covering, continuous, already-mature simple returns."""
    start, end, decision_time = [finite(v, "time") for v in (start, end, decision_time)]
    if not intervals or start >= end:
        raise ValueError("nonempty positive horizon required")
    cursor, gross = start, 1.0
    for row in intervals:
        if row.start != cursor or row.available >= decision_time:
            raise ValueError("gap, overlap or unmatured outcome")
        gross *= 1 + row.simple_return
        if not math.isfinite(gross):
            raise ValueError("compounded return overflow")
        cursor = row.end
    if cursor != end:
        raise ValueError("returns do not cover requested horizon")
    return gross - 1


def horizon_log_variance(variance, autocovariances, periods):
    """Stationary sum variance with every relevant lag supplied explicitly.

    Caller must establish stationarity and covariance validity; this formula
    does not prove either, estimate future volatility, or scale arbitrary data.
    """
    variance = finite(variance, "variance")
    if type(periods) is not int or periods < 1 or variance < 0:
        raise ValueError("invalid variance or horizon")
    if len(autocovariances) != periods - 1:
        raise ValueError("all horizon lags must be supplied")
    ac = [finite(x, "autocovariance") for x in autocovariances]
    if any(abs(x) > variance for x in ac):
        raise ValueError("autocovariance exceeds marginal variance")
    result = periods * variance + 2 * math.fsum((periods - k) * x for k, x in enumerate(ac, 1))
    if result < -1e-12:
        raise ValueError("invalid negative horizon variance")
    return max(result, 0.0)


def finite_scenario_moments(rows, probabilities):
    """Joint moments of a finite simple-return scenario distribution.

    The scenarios and their calibration are supplied assumptions. Nothing here
    converts a Student-t log-return density into expected simple returns.
    """
    if not rows or not rows[0] or len(rows) != len(probabilities):
        raise ValueError("nonempty aligned scenarios required")
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise ValueError("inconsistent scenario dimension")
    p = [finite(v, "probability") for v in probabilities]
    x = [[finite(v, "simple return") for v in row] for row in rows]
    if any(v < 0 for v in p) or abs(math.fsum(p) - 1) > 1e-12:
        raise ValueError("probabilities must be nonnegative and sum to one")
    if any(v < -1 for row in x for v in row):
        raise ValueError("unlevered asset simple returns cannot be below -1")
    mean = [math.fsum(q * row[i] for q, row in zip(p, x)) for i in range(dimension)]
    covariance = [
        [math.fsum(q * (row[i] - mean[i]) * (row[j] - mean[j]) for q, row in zip(p, x))
         for j in range(dimension)]
        for i in range(dimension)
    ]
    if not all(math.isfinite(v) for v in mean + [v for row in covariance for v in row]):
        raise ValueError("scenario moments overflow")
    return mean, covariance


def self_financing_target(holdings, cash, target, fee):
    """Exact scalar root for long-only post-fee target weights.

    This is a mathematical accounting reference, not a policy selector.
    The fee convention matches proportional risky-asset turnover.
    """
    if not holdings or len(holdings) != len(target):
        raise ValueError("aligned nonempty holdings and targets required")
    holdings = [finite(x, "holding") for x in holdings]
    target = [finite(x, "weight") for x in target]
    cash, fee = finite(cash, "cash"), finite(fee, "fee")
    if min(holdings + target + [cash]) < 0 or math.fsum(target) > 1 + 1e-12 or not 0 <= fee < 1:
        raise ValueError("long-only cash-funded weights and fee in [0,1) required")
    nav = cash + math.fsum(holdings)

    def residual(post):
        return post + fee * math.fsum(abs(post * w - h) for w, h in zip(target, holdings)) - nav

    lo, hi = 0.0, nav
    for _ in range(100):
        mid = (lo + hi) / 2
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid
    post = (lo + hi) / 2
    assets = [post * w for w in target]
    cost = fee * math.fsum(abs(a - h) for a, h in zip(assets, holdings))
    remaining = nav - cost - math.fsum(assets)
    if remaining < -1e-9:
        raise ValueError("numerical cash deficit")
    return {"assets": assets, "cash": max(remaining, 0.0), "fee": cost, "post_fee_nav": post}
