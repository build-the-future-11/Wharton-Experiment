"""Conditional, long-only S6 scenario planning; no executable advice or causality claim.

All returns here are *simple* returns. Student-t shocks are sampled directly in
simple-return space and floored at total loss. They are never exponentiated or
presented as calibrated probabilities. The planner makes one initial allocation
and holds it over every simulated period. It does not choose trades with future
scenario information. Its optimum is only over the evaluated finite candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (ndim is not None and result.ndim != ndim) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _utc(value: str | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-qualified")
    return value.astimezone(UTC)


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def convert_to_base_returns(local_returns: Any, fx_returns: Any) -> np.ndarray:
    """Convert aligned simple returns; FX is base-currency units per local unit.

    The caller must align each asset and FX observation to the same interval.
    A local total loss remains exactly -1, including after FX appreciation.
    Scalars may broadcast; otherwise the arrays must have the same shape so an
    accidental outer product cannot masquerade as synchronized currency data.
    """
    local = _finite_array(local_returns, "local_returns")
    fx = _finite_array(fx_returns, "fx_returns")
    if local.ndim and fx.ndim and local.shape != fx.shape:
        raise ValueError("non-scalar local and FX returns must have matching shapes")
    if np.any(local < -1) or np.any(fx < -1):
        raise ValueError("simple returns cannot be below -100%")
    result = (1.0 + local) * (1.0 + fx) - 1.0
    if not np.all(np.isfinite(result)):
        raise ValueError("currency conversion overflowed")
    return result


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: np.ndarray
    mean: np.ndarray
    rows_used: int
    available_from: str
    available_through: str
    decision_time: str
    shrinkage: float


def trailing_shrinkage_covariance(
    returns: Any,
    available_at: Sequence[str | datetime],
    *,
    decision_time: str | datetime,
    lookback: int = 60,
    shrinkage: float = 0.2,
    observed_at: Sequence[str | datetime] | None = None,
) -> CovarianceEstimate:
    """Fit a trailing covariance using only fully available, complete return rows.

    Shrinkage is toward the sample diagonal, not a tuned correlation matrix.
    Future values are never inspected, validated, imputed or summarized. The
    caller supplies genuine availability metadata; this function cannot certify
    a data provider's point-in-time provenance. Missing selected rows fail closed.
    """
    # Inspect shape without parsing outcome numbers. A protected row may even
    # contain an opaque string sentinel; only selected mature rows are cast.
    values = np.asarray(returns)
    if values.ndim != 2 or values.shape[1] < 1 or len(available_at) != len(values):
        raise ValueError("returns must be [row, asset] with one availability per row")
    lookback = _positive_integer(lookback, "lookback")
    shrinkage = _unit_interval(shrinkage, "shrinkage")
    cutoff = _utc(decision_time)
    times = [_utc(item) for item in available_at]
    if any(left >= right for left, right in pairwise(times)):
        raise ValueError("availability timestamps must be strictly increasing")
    if observed_at is not None:
        if len(observed_at) != len(values):
            raise ValueError("observed_at must align with returns")
        outcomes = [_utc(item) for item in observed_at]
        if any(outcome > available for outcome, available in zip(outcomes, times)):
            raise ValueError("a return cannot be available before its outcome is observed")
        if any(left >= right for left, right in pairwise(outcomes)):
            raise ValueError("outcome timestamps must be strictly increasing")
    selected = [index for index, stamp in enumerate(times) if stamp < cutoff][-lookback:]
    if len(selected) < 2:
        raise ValueError("at least two mature training rows are required")
    history = np.asarray(values[selected], dtype=np.float64)
    if not np.all(np.isfinite(history)) or np.any(history < -1):
        raise ValueError("selected training returns must be finite and at least -1")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = history.mean(axis=0)
        centered = history - mean
        sample = centered.T @ centered / (len(history) - 1)
        covariance = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("training covariance estimate overflowed")
    return CovarianceEstimate(
        covariance=covariance,
        mean=mean,
        rows_used=len(history),
        available_from=times[selected[0]].isoformat(),
        available_through=times[selected[-1]].isoformat(),
        decision_time=cutoff.isoformat(),
        shrinkage=shrinkage,
    )


@dataclass(frozen=True)
class ScenarioPaths:
    simple_returns: np.ndarray
    assumptions: dict[str, Any]
    currency: str | None = None
    asset_ids: tuple[str, ...] | None = None


def _scenario_metadata(
    assets: int, currency: str | None, asset_ids: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if currency is not None and (not isinstance(currency, str) or len(currency) != 3
                                 or not currency.isascii() or not currency.isalpha()
                                 or not currency.isupper()):
        raise ValueError("currency must be a three-letter uppercase currency code")
    ids = tuple(asset_ids) if asset_ids is not None else None
    if ids is not None and (len(ids) != assets
                            or any(not isinstance(item, str) or not item for item in ids)
                            or len(set(ids)) != assets):
        raise ValueError("asset_ids must be unique nonempty names matching the asset axis")
    return ids


def sample_joint_paths(
    location: Any,
    covariance: Any,
    *,
    horizon: int | None = None,
    paths: int = 512,
    df: float = 5.0,
    jump_probability: float = 0.02,
    jump_mean: float = -0.08,
    jump_scale: float = 0.03,
    seed: int = 11,
    currency: str | None = None,
    asset_ids: Sequence[str] | None = None,
) -> ScenarioPaths:
    """Sample joint heavy-tail simple-return paths with shared jump occurrence.

    ``location`` is [asset] repeated over ``horizon``, or [horizon, asset].
    Covariance is the pre-floor, pre-jump Student-t covariance; df must exceed 2.
    A common chi-square scale creates within-period multivariate t dependence.
    Jump indicators are shared across assets, with independent normal jump sizes.
    Periods are conditionally independent. Flooring at -1 introduces a default
    mass, which is absorbing. These operations alter the mean/covariance; input
    locations are NOT asserted to equal post-transformation expected returns.
    """
    loc = _finite_array(location, "location")
    paths = _positive_integer(paths, "paths")
    if loc.ndim == 1:
        horizon = _positive_integer(horizon if horizon is not None else 1, "horizon")
        loc = np.broadcast_to(loc, (horizon, len(loc)))
    elif loc.ndim == 2:
        if horizon is not None and _positive_integer(horizon, "horizon") != len(loc):
            raise ValueError("horizon does not match location rows")
        horizon = len(loc)
    else:
        raise ValueError("location must be [asset] or [horizon, asset]")
    if not horizon or not loc.shape[1]:
        raise ValueError("scenario horizon and asset count must be nonzero")
    if np.any(loc < -1):
        raise ValueError("simple-return locations cannot be below -1")
    if not np.isfinite(df) or df <= 2:
        raise ValueError("df must exceed 2 for the supplied covariance to exist")
    jump_probability = _unit_interval(jump_probability, "jump_probability")
    if not np.isfinite(jump_mean) or not np.isfinite(jump_scale) or jump_scale < 0:
        raise ValueError("jump parameters must be finite and jump_scale nonnegative")
    assets = loc.shape[1]
    ids = _scenario_metadata(assets, currency, asset_ids)
    cov = _finite_array(covariance, "covariance")
    if cov.shape == (assets, assets):
        cov = np.broadcast_to(cov, (horizon, assets, assets))
    if cov.shape != (horizon, assets, assets):
        raise ValueError("covariance must be [asset, asset] or [horizon, asset, asset]")
    roots = []
    for matrix in cov:
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if np.min(eigenvalues) < -tolerance:
            raise ValueError("covariance must be positive semidefinite")
        roots.append(eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0)))
    rng = np.random.default_rng(seed)
    result = np.empty((paths, horizon, assets), dtype=np.float64)
    defaulted = np.zeros((paths, assets), dtype=bool)
    for step, root in enumerate(roots):
        gaussian = rng.standard_normal((paths, assets)) @ root.T
        scale = np.sqrt((df - 2.0) / rng.chisquare(df, (paths, 1)))
        jumps = (rng.random((paths, 1)) < jump_probability) * (
            jump_mean + jump_scale * rng.standard_normal((paths, assets)))
        period = np.maximum(loc[step] + gaussian * scale + jumps, -1.0)
        period[defaulted] = 0.0
        defaulted |= period == -1.0
        result[:, step] = period
    if not np.all(np.isfinite(result)):
        raise ValueError("scenario generation overflowed")
    return ScenarioPaths(
        simple_returns=result,
        assumptions={
            "distribution": "multivariate Student-t simple shocks plus shared Bernoulli jumps",
            "df": float(df), "jump_probability": jump_probability,
            "jump_mean": float(jump_mean), "jump_scale": float(jump_scale),
            "seed": int(seed), "periods_conditionally_independent": True,
            "loss_floor": -1.0, "default_absorbing": True,
            "covariance_scope": "pre-floor and pre-jump t shocks",
            "location_scope": "pre-floor and pre-jump location, not post-transform mean",
            "probability_status": "conditional simulation assumptions; not calibrated",
            "currency_scope": "all asset returns must already share the named currency",
        },
        currency=currency, asset_ids=ids,
    )


def sample_forecast_paths(
    forecast: Any,
    correlation: Any,
    *,
    lower_log: float,
    upper_log: float,
    horizon: int,
    paths: int = 512,
    seed: int = 11,
    currency: str | None = None,
    asset_ids: Sequence[str] | None = None,
) -> ScenarioPaths:
    """Bridge an S6 log-return t mixture to explicitly bounded joint scenarios.

    ``forecast`` is an object or mapping with mean, scale, df, weights matrices
    [asset, expert]. Scale is Student-t dispersion, not standard deviation.
    Mixture weights must be identical across assets; df must be identical across
    assets within each expert. Each scenario-period draws one shared expert,
    correlated normal shocks and one shared chi-square scale, then clips log
    returns to the caller's fixed support before ``expm1``. Periods repeat the
    same conditional one-period forecast; this is not a learned horizon model.

    Clipping changes the unbounded t mixture and introduces boundary atoms.
    Therefore this is a bounded scenario approximation, not a finite exponential
    moment of an unbounded Student-t and not calibrated outcome probabilities.
    All assets must already have common-currency log-return semantics.
    """
    fields = []
    for name in ("mean", "scale", "df", "weights"):
        value = forecast[name] if isinstance(forecast, Mapping) else getattr(forecast, name)
        fields.append(_finite_array(value, f"forecast {name}", 2))
    means, scales, degrees, weights = fields
    if 0 in means.shape or any(field.shape != means.shape for field in fields[1:]):
        raise ValueError("forecast fields must share a nonempty [asset, expert] shape")
    if np.any(scales <= 0) or np.any(degrees <= 2) or np.any(weights < 0):
        raise ValueError("forecast requires positive scales, df > 2, nonnegative weights")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError("forecast expert weights must sum to one per asset")
    if not np.array_equal(weights, np.broadcast_to(weights[0], weights.shape)):
        raise ValueError("joint mixture requires identical expert weights across assets")
    if not np.array_equal(degrees, np.broadcast_to(degrees[0], degrees.shape)):
        raise ValueError("joint mixture requires identical df across assets within each expert")
    horizon = _positive_integer(horizon, "horizon")
    paths = _positive_integer(paths, "paths")
    bounds = _finite_array([lower_log, upper_log], "log support bounds", 1)
    if bounds[0] >= bounds[1]:
        raise ValueError("lower_log must be strictly below upper_log")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        simple_bounds = np.expm1(bounds)
    if simple_bounds[0] <= -1 or not np.all(np.isfinite(simple_bounds)):
        raise ValueError("log support must produce finite simple returns strictly above -1")
    assets, experts = means.shape
    ids = _scenario_metadata(assets, currency, asset_ids)
    corr = _finite_array(correlation, "correlation", 2)
    if corr.shape != (assets, assets):
        raise ValueError("correlation must match the asset axis")
    if (not np.allclose(corr, corr.T, atol=1e-12, rtol=0)
            or not np.allclose(np.diag(corr), 1, atol=1e-12, rtol=0)):
        raise ValueError("correlation must be symmetric with unit diagonal")
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    if np.min(eigenvalues) < -1e-12 * max(1.0, float(np.max(np.abs(eigenvalues)))):
        raise ValueError("correlation must be positive semidefinite")
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    # The accepted tiny sum tolerance is purely numerical. The shared vector
    # must be normalized for categorical sampling, and its actual values are
    # recorded explicitly in the assumptions.
    mixture_weights = weights[0] / weights[0].sum()
    rng = np.random.default_rng(seed)
    result = np.empty((paths, horizon, assets), dtype=np.float64)
    component_counts = np.zeros(experts, dtype=np.int64)
    clipped_low = clipped_high = 0
    for step in range(horizon):
        components = rng.choice(experts, size=paths, p=mixture_weights)
        component_counts += np.bincount(components, minlength=experts)
        chosen_df = degrees[0, components]
        gaussian = rng.standard_normal((paths, assets)) @ root.T
        t_scale = np.sqrt(chosen_df / rng.chisquare(chosen_df))[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            log_returns = means[:, components].T + scales[:, components].T * gaussian * t_scale
        if not np.all(np.isfinite(log_returns)):
            raise ValueError("log-return scenario generation overflowed")
        clipped_low += int(np.count_nonzero(log_returns < lower_log))
        clipped_high += int(np.count_nonzero(log_returns > upper_log))
        result[:, step] = np.expm1(np.clip(log_returns, lower_log, upper_log))
    return ScenarioPaths(
        simple_returns=result,
        assumptions={
            "distribution": "bounded log-return Student-t mixture approximation",
            "lower_log": float(lower_log), "upper_log": float(upper_log),
            "mixture_weights": mixture_weights.tolist(),
            "expert_df": degrees[0].tolist(), "component_counts": component_counts.tolist(),
            "clipped_below": clipped_low, "clipped_above": clipped_high,
            "seed": int(seed), "shared_expert_per_period": True,
            "periods_conditionally_independent": True,
            "horizon_scope": "repeated current one-period conditional forecast",
            "scale_scope": "Student-t dispersion; standard deviation differs",
            "correlation_scope": "within-expert t scale correlation; clipping alters dependence",
            "probability_status": "conditional simulation assumptions; not calibrated",
            "expectation_scope": "bounded approximation; no unbounded t exponential expectation",
            "support_scope": "explicit clipping creates atoms at the two support bounds",
            "currency_scope": "all input forecasts must already share the named currency",
        },
        currency=currency, asset_ids=ids,
    )


def empirical_cvar(losses: Any, alpha: float = 0.95) -> float:
    """Exact upper-tail CVaR of an equally weighted finite empirical distribution.

    The boundary scenario receives fractional weight when n*(1-alpha) is not
    integral. This equals min_eta eta + mean(max(loss-eta,0))/(1-alpha).
    """
    values = _finite_array(losses, "losses", 1)
    if not len(values) or not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("nonempty losses and alpha in [0, 1) are required")
    ordered = np.sort(values)[::-1]
    mass = len(values) * (1.0 - alpha)
    whole = min(int(np.floor(mass)), len(values))
    fractional = mass - whole
    # Scale before summing: a finite empirical CVaR need not overflow merely
    # because several very large losses are present.
    total = float(np.sum(ordered[:whole] / mass))
    if whole < len(values):
        total += (fractional / mass) * float(ordered[whole])
    if not np.isfinite(total):
        raise ValueError("empirical CVaR overflowed")
    return total


def self_financing_rebalance(
    weights: Any, current_weights: Any, cost_bps: float = 10.0,
) -> tuple[float, float]:
    """Return post-fee NAV fraction and traded pre-fee-NAV fraction.

    For risky post-fee weights w, existing risky dollars h, fee k, solve
    q + k*sum(abs(q*w-h)) = 1. Cash is q*(1-sum(w)); it is never borrowed.
    The root is unique because k < 1 and sum(w) <= 1.
    """
    weights = _finite_array(weights, "weights", 1)
    previous = _finite_array(current_weights, "current_weights", 1)
    if weights.shape != previous.shape or not len(weights):
        raise ValueError("weights and current_weights must match")
    if (np.any(weights < 0) or np.any(previous < 0)
            or weights.sum() > 1 + 1e-12 or previous.sum() > 1 + 1e-12):
        raise ValueError("weights must be long-only with sum at most one")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be in [0, 10000)")
    rate = cost_bps / 10000.0
    if rate == 0 or np.array_equal(weights, previous):
        return 1.0, float(np.abs(weights - previous).sum())
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if mid + rate * float(np.abs(mid * weights - previous).sum()) > 1:
            high = mid
        else:
            low = mid
    fraction = (low + high) / 2
    turnover = float(np.abs(fraction * weights - previous).sum())
    return fraction, turnover


@dataclass(frozen=True)
class PortfolioPlan:
    weights: np.ndarray
    cash_weight: float
    post_cost_wealth_fraction: float
    turnover: float
    utility: float
    mean_terminal_wealth: float
    cvar_loss: float
    mean_terminal_shortfall: float
    no_trade_utility: float
    selected_candidate: str
    candidate_summaries: tuple[dict[str, Any], ...]
    terminal_wealth: np.ndarray
    assumptions: dict[str, Any]


def plan_portfolio(
    scenarios: ScenarioPaths | Any,
    current_weights: Any,
    *,
    candidates: Any | None = None,
    max_weight: float | Sequence[float] = 0.35,
    min_cash: float = 0.05,
    turnover_cap: float = 1.0,
    cost_bps: float = 10.0,
    cash_returns: Any = 0.0,
    cvar_alpha: float = 0.95,
    cvar_penalty: float = 1.0,
    shortfall_penalty: float = 1.0,
    terminal_target: float = 1.0,
    random_candidates: int = 64,
    seed: int = 11,
) -> PortfolioPlan:
    """Choose a finite candidate by multi-period log wealth net of costs and risk.

    Utility = mean(log terminal NAV) - cvar_penalty*CVaR(max(1-NAV,0))
              - shortfall_penalty*mean(max(terminal_target-NAV,0)).
    NAV starts at one. Cash accrues the supplied simple return each period.
    The candidate set always contains exact no-trade. Caps constrain initial
    post-fee weights only; future market drift can breach them. Holdings are
    marked, not terminally liquidated. Default remains total loss. No scenario
    path controls subsequent trades, and no global optimum is claimed.
    """
    values = scenarios.simple_returns if isinstance(scenarios, ScenarioPaths) else scenarios
    returns = _finite_array(values, "scenarios", 3)
    if 0 in returns.shape or np.any(returns < -1):
        raise ValueError("scenarios must be nonempty and simple returns at least -1")
    n_paths, horizon, assets = returns.shape
    previous = _finite_array(current_weights, "current_weights", 1)
    if previous.shape != (assets,) or np.any(previous < 0):
        raise ValueError("current_weights must match assets and be nonnegative")
    min_cash = _unit_interval(min_cash, "min_cash")
    caps = np.broadcast_to(_finite_array(max_weight, "max_weight"), (assets,)).copy()
    if np.any(caps < 0) or np.any(caps > 1):
        raise ValueError("max_weight must be in [0,1] for every asset")
    if not np.isfinite(turnover_cap) or not 0 <= turnover_cap <= 2:
        raise ValueError("turnover_cap must be in [0,2] of initial NAV")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be in [0,10000)")
    if not np.isfinite(cvar_alpha) or not 0 <= cvar_alpha < 1:
        raise ValueError("cvar_alpha must be in [0,1)")
    penalties = np.asarray([cvar_penalty, shortfall_penalty, terminal_target], dtype=float)
    if not np.all(np.isfinite(penalties)) or np.any(penalties < 0):
        raise ValueError("penalties and terminal_target must be finite and nonnegative")
    if (isinstance(random_candidates, (bool, np.bool_))
            or not isinstance(random_candidates, (int, np.integer)) or random_candidates < 0):
        raise ValueError("random_candidates must be a nonnegative integer")
    budget = 1.0 - min_cash
    if np.any(previous > caps + 1e-12) or previous.sum() > budget + 1e-12:
        raise ValueError("current weights violate initial caps; mandate repair is a separate action")
    cash = _finite_array(cash_returns, "cash_returns")
    if cash.ndim == 0:
        cash = np.full((n_paths, horizon), float(cash))
    elif cash.shape == (horizon,):
        cash = np.broadcast_to(cash, (n_paths, horizon))
    elif cash.shape != (n_paths, horizon):
        raise ValueError("cash_returns must be scalar, [horizon], or [path,horizon]")
    if np.any(cash < -1):
        raise ValueError("cash simple returns cannot be below -1")
    with np.errstate(over="ignore", invalid="ignore"):
        asset_growth = np.prod(1.0 + returns, axis=1)
        cash_growth = np.prod(1.0 + cash, axis=1)
    if not np.all(np.isfinite(asset_growth)) or not np.all(np.isfinite(cash_growth)):
        raise ValueError("multi-period compounding overflowed")

    def feasible(weights: np.ndarray) -> bool:
        return bool(np.all(weights >= 0) and np.all(weights <= caps + 1e-12)
                    and weights.sum() <= budget + 1e-12)

    def bound(weights: np.ndarray) -> np.ndarray:
        bounded = np.minimum(np.maximum(weights, 0.0), caps)
        total = bounded.sum()
        return bounded * (budget / total) if total > budget else bounded

    proposed: list[tuple[str, np.ndarray]] = [("no_trade", previous.copy())]
    if candidates is not None:
        provided = _finite_array(candidates, "candidates", 2)
        if provided.shape[1] != assets:
            raise ValueError("candidate columns must match assets")
        proposed.extend((f"provided_{index}", row.copy()) for index, row in enumerate(provided))
    proposed.extend([("cash", np.zeros(assets)),
                     ("equal_weight", bound(np.full(assets, budget / assets)))])
    for asset in range(assets):
        concentrated = np.zeros(assets)
        concentrated[asset] = min(caps[asset], budget)
        proposed.append((f"single_{asset}", concentrated))
    rng = np.random.default_rng(seed)
    for index in range(random_candidates):
        weights = bound(rng.dirichlet(np.ones(assets)) * budget * rng.random())
        proposed.append((f"random_{index}", weights))
    # Fixed initial interpolations give turnover-limited candidates without a
    # path-dependent optimizer. The exact turnover cap is still checked below.
    for name, weights in list(proposed[1:]):
        for fraction in (0.25, 0.5, 0.75):
            proposed.append((f"{name}_blend_{fraction}", previous + fraction * (weights - previous)))

    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_terminal: np.ndarray | None = None
    best_weights: np.ndarray | None = None
    seen: set[bytes] = set()
    for name, weights in proposed:
        identity = weights.tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        if not feasible(weights):
            summaries.append({"name": name, "status": "infeasible_initial_weights"})
            continue
        remaining, turnover = self_financing_rebalance(weights, previous, cost_bps)
        if turnover > turnover_cap + 1e-12:
            summaries.append({"name": name, "status": "turnover_cap", "turnover": turnover})
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            terminal = remaining * (asset_growth @ weights + cash_growth * (1 - weights.sum()))
        if np.any(terminal < 0) or not np.all(np.isfinite(terminal)):
            raise ValueError("candidate accounting produced invalid terminal wealth")
        log_utility = float(np.log(terminal).mean()) if np.all(terminal > 0) else -np.inf
        cvar = empirical_cvar(np.maximum(1 - terminal, 0), cvar_alpha)
        # Divide before summing so finite large NAVs do not overflow a mean's
        # intermediate sum. Overflow in risk penalties is a numerical failure,
        # distinct from the mathematically valid -inf log score at ruin.
        shortfall = float(np.sum(np.maximum(terminal_target - terminal, 0) / n_paths))
        mean_wealth = float(np.sum(terminal / n_paths))
        risk_charge = float(cvar_penalty) * cvar + float(shortfall_penalty) * shortfall
        if not np.isfinite(shortfall) or not np.isfinite(mean_wealth) or not np.isfinite(risk_charge):
            raise ValueError("candidate objective overflowed")
        utility = log_utility - risk_charge
        if np.isfinite(log_utility) and not np.isfinite(utility):
            raise ValueError("candidate objective overflowed")
        record = {
            "name": name, "status": "evaluated", "utility": utility,
            "mean_log_terminal_wealth": log_utility,
            "mean_terminal_wealth": mean_wealth, "cvar_loss": cvar,
            "mean_terminal_shortfall": shortfall, "turnover": turnover,
            "post_cost_wealth_fraction": remaining,
        }
        summaries.append(record)
        # Strict comparison retains no-trade when candidates tie, including ruin.
        if best is None or utility > best["utility"]:
            best, best_terminal, best_weights = record, terminal.copy(), weights.copy()
    if best is None or best_terminal is None or best_weights is None:
        raise RuntimeError("no feasible candidate; no-trade invariant violated")
    baseline = next(item for item in summaries if item["name"] == "no_trade")
    return PortfolioPlan(
        weights=best_weights, cash_weight=float(1 - best_weights.sum()),
        post_cost_wealth_fraction=best["post_cost_wealth_fraction"],
        turnover=best["turnover"], utility=best["utility"],
        mean_terminal_wealth=best["mean_terminal_wealth"], cvar_loss=best["cvar_loss"],
        mean_terminal_shortfall=best["mean_terminal_shortfall"],
        no_trade_utility=baseline["utility"], selected_candidate=best["name"],
        candidate_summaries=tuple(summaries), terminal_wealth=best_terminal,
        assumptions={
            "solver": "finite candidate search; no global optimum guarantee",
            "horizon": horizon, "scenario_count": n_paths,
            "rebalancing": "one initial trade then buy and hold; no scenario foresight",
            "constraint_scope": "initial post-fee weights and actual initial traded notional",
            "initial_nav": 1.0, "terminal_nav": "marked holdings, no terminal liquidation",
            "cost_bps": float(cost_bps), "cvar_alpha": float(cvar_alpha),
            "cvar_penalty": float(cvar_penalty), "shortfall_penalty": float(shortfall_penalty),
            "terminal_target": float(terminal_target), "liability_scope": "terminal target only",
            "cash_returns": "explicit supplied scenario rates; default zero",
            "currency": scenarios.currency if isinstance(scenarios, ScenarioPaths) else None,
            "probability_status": "conditional equal-weight scenarios; not calibrated",
        },
    )
