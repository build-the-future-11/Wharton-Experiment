"""Independent likelihood reconstruction, date-cluster inference, self-financing ledgers."""

import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata, t


def score_law(raw, y, df):
    if not np.isfinite(df) or df <= 2 or np.asarray(y).ndim != 1:
        raise ValueError("finite-variance Student law and outcome vector required")
    weights = raw["weights"]
    if weights.ndim != 3 or weights.shape[1] != 3:
        raise ValueError("three objective gates required")
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.allclose(weights.sum(-1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("invalid positive mixture gates")
    mu, scale, w = raw["mu"], raw["scale"], raw["weights"][:, 0]
    if mu.ndim != 2 or mu.shape != scale.shape or mu.shape != w.shape or mu.shape[0] != len(y):
        raise ValueError("law and outcome shapes differ")
    if any(not np.isfinite(x).all() for x in (mu, scale, w, y)):
        raise ValueError("nonfinite law")
    if np.any(scale <= 0):
        raise ValueError("invalid positive mixture law")
    nll = -logsumexp(np.log(w) + t.logpdf((y[:, None] - mu) / scale, df) - np.log(scale), axis=1)
    loss_probability = (w * t.cdf(-mu / scale, df)).sum(1)
    pit = (w * t.cdf((y[:, None] - mu) / scale, df)).sum(1)
    return {
        "nll": nll,
        "brier": (loss_probability - (y < 0)) ** 2,
        "coverage90": (pit >= 0.05) & (pit <= 0.95),
        "pit": pit,
    }


def canonical_mean(values):
    """A scalar mean with a fixed Python-float accumulation contract."""
    vector = [float(value) for value in np.asarray(values).reshape(-1)]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("canonical mean requires finite nonempty values")
    return math.fsum(vector) / len(vector)


def date_nll(nll, times):
    """Aggregate a retained score vector by its ordered decision-date clusters."""
    nll, times = np.asarray(nll), np.asarray(times)
    if nll.ndim != 1 or times.ndim != 1 or nll.shape != times.shape:
        raise ValueError("date score vector and timestamps must be aligned")
    dates = np.unique(times)
    return np.array([canonical_mean(nll[times == date]) for date in dates], dtype=np.float64)


def summary_from_scores(raw, y, times, scores):
    """Build metrics from a saved, independently checked target-runtime score vector."""
    required = {"nll", "brier", "coverage90"}
    if set(scores) != required:
        raise ValueError("saved score artifact has an unexpected schema")
    nll = np.asarray(scores["nll"], dtype=np.float64)
    brier = np.asarray(scores["brier"], dtype=np.float64)
    coverage = np.asarray(scores["coverage90"])
    if (
        any(value.shape != np.asarray(y).shape for value in (nll, brier, coverage))
        or not np.isfinite(nll).all()
        or not np.isfinite(brier).all()
        or coverage.dtype != np.bool_
    ):
        raise ValueError("saved score artifact contains invalid vectors")
    date_nll, ics = [], []
    for date in np.unique(times):
        ix = times == date
        date_nll.append(canonical_mean(nll[ix]))
        a, b = rankdata(raw["mean"][ix]), rankdata(y[ix])
        if np.std(a) > 0 and np.std(b) > 0:
            ics.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "date_mean_nll": canonical_mean(date_nll),
        "mae": canonical_mean(abs(np.asarray(y) - raw["mean"])),
        "brier": canonical_mean(brier),
        "coverage90": canonical_mean(coverage.astype(np.float64)),
        "rank_ic": canonical_mean(ics) if ics else None,
        "defined_ic_dates": len(ics),
        "dates": len(date_nll),
        "rows": len(y),
    }


def summary(raw, y, times, df):
    """Convenience scorer for ephemeral callers; persisted runs use score artifacts."""
    scores = score_law(raw, y, df)
    return summary_from_scores(
        raw, y, times, {key: scores[key] for key in ("nll", "brier", "coverage90")}
    )


def paired_bootstrap(difference, block, repetitions, seed=1901):
    if type(block) is not int or block < 1 or type(repetitions) is not int or repetitions < 1:
        raise ValueError("positive integer block size and repetition count required")
    d = np.asarray(difference, dtype=np.float64)
    if d.ndim != 1 or not np.isfinite(d).all() or len(d) < 2 * block or block < 1:
        raise ValueError("paired inference requires finite dates and at least two declared blocks")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        starts = rng.integers(0, len(d), size=(len(d) + block - 1) // block)
        indices = ((starts[:, None] + np.arange(block)) % len(d)).ravel()[: len(d)]
        draws[i] = d[indices].mean()
    estimate = float(d.mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    p = (1 + int(np.sum(np.abs(draws - estimate) >= abs(estimate)))) / (repetitions + 1)
    return {
        "difference": estimate,
        "ci95": [float(low), float(high)],
        "p_two_sided": p,
        "dates": len(d),
        "block": block,
        "repetitions": repetitions,
    }


def holm(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid p-values")
    order = np.argsort(p, kind="stable")
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return out


def capped_simplex(vector, cap):
    """Euclidean projection onto 0 <= w <= cap, sum(w) <= 1 (cash remainder)."""
    vector = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(vector).all() or vector.ndim != 1 or not 0 < cap <= 1:
        raise ValueError("invalid allocation vector/cap")
    clipped = np.clip(vector, 0, cap)
    if clipped.sum() <= 1:
        return clipped
    low, high = float(vector.min() - cap), float(vector.max())
    for _ in range(100):
        mid = (low + high) / 2
        if np.clip(vector - mid, 0, cap).sum() > 1:
            low = mid
        else:
            high = mid
    return np.clip(vector - high, 0, cap)


def allocate(mean, covariance, previous, cap, risk_aversion, fee):
    """Proximal projected risk/cost objective on a bounded cash-inclusive simplex.

    L1 turnover is handled with an exact separable prox plus a scalar multiplier
    for the sum constraint; no gradient through a nondifferentiable sign heuristic.
    """
    mean, previous = np.asarray(mean, dtype=np.float64), np.asarray(previous, dtype=np.float64)
    if (
        mean.ndim != 1
        or not len(mean)
        or previous.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(previous).all()
        or np.any(previous < 0)
        or previous.sum() > 1 + 1e-10
        or not np.isfinite(risk_aversion)
        or risk_aversion <= 0
        or not 0 <= fee < 1
    ):
        raise ValueError("invalid forecast, holdings or risk/cost scale")
    n = len(mean)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (n, n) or not np.isfinite(covariance).all():
        raise ValueError("invalid covariance")
    covariance = (covariance + covariance.T) / 2
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    step = 1 / max(risk_aversion * np.linalg.eigvalsh(covariance).max(), 1e-6)
    weights = capped_simplex(previous, cap)
    residual = float("inf")
    for _iteration in range(2000):
        center = weights + step * (mean - risk_aversion * covariance @ weights)

        def prox(multiplier, center=center):
            delta = center - step * multiplier - previous
            return np.clip(
                previous + np.sign(delta) * np.maximum(abs(delta) - step * fee, 0), 0, cap
            )

        new = prox(0)
        if new.sum() > 1:
            low, high = 0.0, max(float(np.max(center)) / step + fee + 1, 1.0)
            for _ in range(80):
                mid = (low + high) / 2
                if prox(mid).sum() > 1:
                    low = mid
                else:
                    high = mid
            new = prox(high)
        residual = float(np.max(abs(new - weights)))
        weights = new
        if residual <= 1e-9:
            break
    if residual > 1e-7:
        raise RuntimeError("portfolio optimizer did not converge")
    return weights, {"iterations": _iteration + 1, "residual": residual}


def rebalance(previous, target, returns, fee):
    """Solve the transaction-fee fixed point in pre-trade NAV units."""
    previous, target, returns = [
        np.asarray(x, dtype=np.float64) for x in (previous, target, returns)
    ]
    if not (previous.shape == target.shape == returns.shape) or target.ndim != 1:
        raise ValueError("ledger dimensions differ")
    if any(not np.isfinite(x).all() for x in (previous, target, returns)):
        raise ValueError("nonfinite ledger input")
    if (
        np.any(previous < 0)
        or np.any(target < 0)
        or previous.sum() > 1 + 1e-10
        or target.sum() > 1 + 1e-10
        or np.any(returns < -1)
        or not 0 <= fee < 1
    ):
        raise ValueError("ledger requires unlevered long-only weights and simple total returns")
    low, high = 0.0, 1.0
    for _ in range(100):
        q = (low + high) / 2
        equation = q + fee * np.abs(q * target - previous).sum() - 1
        if equation > 0:
            high = q
        else:
            low = q
    q = (low + high) / 2
    gross = 1 + float(target @ returns)
    if gross <= 0:
        return 0.0, np.zeros_like(target), float(1 - q)
    return q * gross, target * (1 + returns) / gross, float(1 - q)


def portfolio(samples, raw, config, policy="model", cost_multiplier=1):
    ix = samples.test
    a = len(samples.panel.assets)
    # All assets share a complete calendar by construction. Shrink using training only.
    train = samples.y[samples.train].reshape(-1, a)
    covariance = np.atleast_2d(np.cov(train, rowvar=False))
    covariance = 0.9 * covariance + 0.1 * np.diag(np.diag(covariance)) + np.eye(a) * 1e-8
    means = raw["mean"].reshape(-1, a)
    realized = samples.y[ix].reshape(-1, a)
    dates = samples.time[ix].reshape(-1, a)[:, 0]
    ends = samples.end[ix].reshape(-1, a)[:, 0]
    previous, wealth, last_end = np.zeros(a), 1.0, -1
    fee = config.cost_bps * cost_multiplier / 10000
    ledger = []
    for i, date in enumerate(dates):
        if date < last_end:
            continue
        if policy == "model":
            target, diagnostics = allocate(
                means[i], covariance, previous, config.max_weight, config.risk_aversion, fee
            )
        elif policy == "equal_weight":
            target, diagnostics = np.full(a, min(1 / a, config.max_weight)), {}
        elif policy == "cash":
            target, diagnostics = np.zeros(a), {}
        else:
            raise ValueError("unknown policy")
        before = previous.copy()
        factor, previous, paid = rebalance(before, target, realized[i], fee)
        wealth *= factor
        ledger.append(
            {
                "time": int(date),
                "end": int(ends[i]),
                "previous": before.tolist(),
                "target": target.tolist(),
                "returns": realized[i].tolist(),
                "post": previous.tolist(),
                "factor": factor,
                "cost_fraction": paid,
                "wealth": wealth,
                "optimizer": diagnostics,
            }
        )
        last_end = ends[i]
        if wealth == 0:
            break
    liquidation, _, _ = rebalance(previous, np.zeros(a), np.zeros(a), fee)
    wealth *= liquidation
    path = np.array([1.0, *[r["wealth"] for r in ledger], wealth])
    drawdown = float(np.max(1 - path / np.maximum.accumulate(path)))
    return {
        "policy": policy,
        "cost_bps": config.cost_bps * cost_multiplier,
        "ledger": ledger,
        "liquidation_factor": liquidation,
        "ending_wealth": wealth,
        "max_drawdown": drawdown,
        "cash_return_assumption": 0.0,
        "covariance": covariance.tolist(),
        "max_weight": config.max_weight,
        "risk_aversion": config.risk_aversion,
        "data_purpose": samples.panel.provenance["purpose"],
        "scope": "Idealized boundary execution, constant proportional fees, zero cash yield; adjusted-close proxy is not investable performance",
    }
