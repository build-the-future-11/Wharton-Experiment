"""Explicit scenario assumptions, covariance, tail risk and tactical regime gates.

There are no implicit downloads, model fitting calls or forecasts in this
module.  Random scenario generation only occurs when its named function is
called by a later authorized run.  All optimizer scenarios are simple returns
in one base currency and over one common executable horizon.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mandate import Instrument, scenario_probabilities, weighted_quantile


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: np.ndarray
    observations: int
    shrinkage: float
    eigenvalue_floor: float
    dropped_rows: int = 0


def validate_covariance(
    covariance: Sequence[Sequence[float]], n_assets: int | None = None, *, tolerance: float = 1e-9
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("covariance must be a nonempty square matrix")
    if n_assets is not None and matrix.shape != (n_assets, n_assets):
        raise ValueError("covariance does not align with forecast/instrument order")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance contains missing or infinite values")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0):
        raise ValueError("covariance is not symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    if np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError("covariance is not positive semidefinite")
    return matrix


def estimate_covariance(
    returns: Sequence[Sequence[float]],
    *,
    shrinkage: float = 0.10,
    min_observations: int = 20,
    eigenvalue_floor: float = 1e-10,
    drop_missing_rows: bool = False,
) -> CovarianceEstimate:
    """Diagonal shrinkage of a synchronized, already causal return matrix.

    This estimates covariance at the supplied return horizon; it does not
    silently annualize or assume that overlapping returns are independent.
    Complete-row deletion, when requested, preserves cross-asset alignment.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("returns must have [observations, assets] shape")
    if not 0 <= shrinkage <= 1 or not np.isfinite(shrinkage):
        raise ValueError("shrinkage must lie in [0,1]")
    if not isinstance(min_observations, int) or min_observations < 2:
        raise ValueError("min_observations must be an integer at least two")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive and finite")
    valid = np.all(np.isfinite(values), axis=1)
    dropped = int((~valid).sum())
    if dropped and not drop_missing_rows:
        raise ValueError(
            "missing return observations; align timestamps or explicitly opt into complete-row deletion"
        )
    values = values[valid]
    if len(values) < min_observations:
        raise ValueError(f"need at least {min_observations} complete synchronized observations")
    sample = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    shrunk = (1.0 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    matrix = (eigenvectors * np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
    matrix = 0.5 * (matrix + matrix.T)
    return CovarianceEstimate(matrix, len(values), float(shrinkage), float(eigenvalue_floor), dropped)


def base_currency_simple_returns(
    local_returns: Sequence[float] | float, fx_returns: Sequence[float] | float
) -> np.ndarray:
    """FX is base-currency units per one local unit; 1+R_base=(1+R_local)(1+R_FX)."""
    local, fx = np.broadcast_arrays(
        np.asarray(local_returns, dtype=float), np.asarray(fx_returns, dtype=float)
    )
    if (
        not np.all(np.isfinite(local))
        or not np.all(np.isfinite(fx))
        or np.any(local < -1)
        or np.any(fx <= -1)
    ):
        raise ValueError("finite returns required: asset >= -1 and FX > -1")
    return (1.0 + local) * (1.0 + fx) - 1.0


def gaussian_log_to_simple_moments(
    log_mean: Sequence[float], log_covariance: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert explicitly Gaussian joint log returns; never use for log Student-t.

    A log Student-t variable has no finite positive exponential moment.  Such
    forecasts require an explicitly bounded/calibrated empirical distribution
    before a finite expected arithmetic return is used by the optimizer.
    """
    mean = np.asarray(log_mean, dtype=float)
    covariance = validate_covariance(log_covariance, len(mean))
    if mean.ndim != 1 or not np.all(np.isfinite(mean)):
        raise ValueError("log means must be a finite vector")
    with np.errstate(over="raise", invalid="raise"):
        try:
            gross_means = np.exp(mean + 0.5 * np.diag(covariance))
            simple_covariance = np.outer(gross_means, gross_means) * np.expm1(covariance)
        except FloatingPointError as exc:
            raise ValueError("lognormal moments overflow under supplied assumptions") from exc
    return gross_means - 1, simple_covariance


@dataclass(frozen=True)
class ScenarioSet:
    returns: np.ndarray
    probabilities: np.ndarray
    asset_ids: tuple[str, ...]
    base_currency: str
    horizon_years: float
    origin: str
    assumptions: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.returns.ndim != 2 or self.returns.shape[1] != len(self.asset_ids):
            raise ValueError("scenario columns must match asset_ids exactly")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("scenario asset IDs must be unique")
        if not np.all(np.isfinite(self.returns)) or np.any(self.returns < -1):
            raise ValueError("simple-return scenarios must be finite and >= -1")
        scenario_probabilities(len(self.returns), self.probabilities)
        if not self.base_currency or not np.isfinite(self.horizon_years) or self.horizon_years <= 0:
            raise ValueError("base currency and a positive common horizon are required")


def stochastic_return_scenarios(
    annual_drift: Sequence[float],
    annual_covariance: Sequence[Sequence[float]],
    *,
    asset_ids: Sequence[str],
    base_currency: str,
    horizon_years: float,
    n_scenarios: int,
    seed: int,
    degrees_of_freedom: float | None = 5.0,
    max_abs_log_return: float = 3.0,
    annual_jump_intensity: float = 0.0,
    jump_mean: float = -0.10,
    jump_scale: float = 0.10,
) -> ScenarioSet:
    """Bounded SDE-inspired return scenarios, generated only on explicit call.

    The Gaussian branch uses GBM log-increment arithmetic.  The Student-t branch
    replaces the Gaussian shock by a common-scale multivariate t increment;
    it is a discrete stress approximation, not an exact diffusion solution.
    Shared Poisson jumps are an optional systemic shock.  The explicit log cap
    produces finite arithmetic moments and is reported because it changes tails.
    Drift is a supplied assumption, not an inferred profitable signal.
    """
    drift = np.asarray(annual_drift, dtype=float)
    if drift.ndim != 1 or not np.all(np.isfinite(drift)):
        raise ValueError("annual drift must be a finite vector")
    cov = validate_covariance(annual_covariance, len(drift))
    if len(asset_ids) != len(drift):
        raise ValueError("asset_ids must align with annual drift")
    if not isinstance(n_scenarios, int) or n_scenarios < 2:
        raise ValueError("n_scenarios must be an integer at least two")
    for name, number in (("horizon_years", horizon_years), ("max_abs_log_return", max_abs_log_return)):
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if degrees_of_freedom is not None and (not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2):
        raise ValueError("Student-t degrees of freedom must exceed two for finite increment covariance")
    if (
        not np.isfinite(annual_jump_intensity)
        or annual_jump_intensity < 0
        or not np.isfinite(jump_mean)
        or not np.isfinite(jump_scale)
        or jump_scale < 0
    ):
        raise ValueError("jump intensity/scale must be nonnegative and jump mean finite")
    rng = np.random.default_rng(seed)
    eigenvalues, eigenvectors = np.linalg.eigh(cov * horizon_years)
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    shock = rng.standard_normal((n_scenarios, len(drift))) @ factor.T
    if degrees_of_freedom is not None:
        chi_square = rng.chisquare(degrees_of_freedom, n_scenarios)
        shock *= np.sqrt((degrees_of_freedom - 2) / chi_square)[:, None]
    increments = (drift - 0.5 * np.diag(cov)) * horizon_years + shock
    if annual_jump_intensity:
        counts = rng.poisson(annual_jump_intensity * horizon_years, n_scenarios)
        jumps = counts * jump_mean + np.sqrt(counts) * jump_scale * rng.standard_normal(n_scenarios)
        increments += jumps[:, None]
    clipped = np.abs(increments) > max_abs_log_return
    increments = np.clip(increments, -max_abs_log_return, max_abs_log_return)
    result = ScenarioSet(
        returns=np.expm1(increments),
        probabilities=np.full(n_scenarios, 1.0 / n_scenarios),
        asset_ids=tuple(str(x) for x in asset_ids),
        base_currency=base_currency,
        horizon_years=float(horizon_years),
        origin="hypothetical_stochastic_scenarios",
        assumptions={
            "seed": seed,
            "degrees_of_freedom": degrees_of_freedom,
            "annual_drift": drift.tolist(),
            "annual_covariance": cov.tolist(),
            "max_abs_log_return": max_abs_log_return,
            "clipped_increment_count": int(clipped.sum()),
            "annual_jump_intensity": annual_jump_intensity,
            "jump_mean": jump_mean,
            "jump_scale": jump_scale,
            "distribution_note": "Finite-step, bounded scenario approximation; not a no-arbitrage theorem, calibrated probability or verified return forecast.",
        },
    )
    result.validate()
    return result


def weighted_cvar(
    losses: Sequence[float], confidence: float = 0.95, probabilities: Sequence[float] | None = None
) -> float:
    """Average worst (1-confidence) probability mass, including fractional atoms.

    Fractional tail weighting matters for finite weighted scenarios. A plain
    mean of losses above a quantile can use the wrong tail mass at ties.
    """
    loss = np.asarray(losses, dtype=float)
    if loss.ndim != 1 or not np.all(np.isfinite(loss)):
        raise ValueError("losses must be a finite vector")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("CVaR confidence must be strictly between zero and one")
    p = scenario_probabilities(len(loss), probabilities)
    order = np.argsort(-loss, kind="stable")
    loss, p = loss[order], p[order]
    tail = 1.0 - confidence
    mass_before = np.cumsum(p) - p
    used = np.minimum(p, np.maximum(tail - mass_before, 0.0))
    return float(used @ loss / tail)


def bond_total_return_from_shock(
    modified_duration: float,
    convexity: float,
    rate_change: float,
    *,
    credit_spread_change: float = 0.0,
    spread_duration: float | None = None,
    carry: float = 0.0,
) -> float:
    """Local second-order bond approximation; yields/spreads are decimal changes.

    Inputs require dated bond duration/convexity and carry over the same horizon.
    Large shocks, defaults, calls, changing duration and coupon reinvestment can
    invalidate this approximation and require instrument cashflow repricing.
    """
    values = (modified_duration, convexity, rate_change, credit_spread_change, carry)
    if not all(np.isfinite(x) for x in values) or modified_duration < 0 or convexity < 0:
        raise ValueError("finite shocks/carry and nonnegative duration/convexity are required")
    spread_duration = modified_duration if spread_duration is None else float(spread_duration)
    if not np.isfinite(spread_duration) or spread_duration < 0:
        raise ValueError("spread duration must be nonnegative and finite")
    value = (
        carry
        - modified_duration * rate_change
        - spread_duration * credit_spread_change
        + 0.5 * convexity * (rate_change + credit_spread_change) ** 2
    )
    if value < -1:
        raise ValueError("duration/convexity approximation implies return below -100%; reprice cashflows")
    return float(value)


@dataclass(frozen=True)
class StressScenario:
    name: str
    equity_return: float = 0.0
    rate_change: float = 0.0
    credit_spread_change: float = 0.0
    bond_carry: float = 0.0
    fx_returns: Mapping[str, float] = field(default_factory=dict)
    market_returns: Mapping[str, float] = field(default_factory=dict)
    sector_returns: Mapping[str, float] = field(default_factory=dict)
    instrument_returns: Mapping[str, float] = field(default_factory=dict)
    description: str = "Hypothetical scenario; no forecast probability assigned."


def stress_return_matrix(
    instruments: Sequence[Instrument], scenarios: Sequence[StressScenario], *, base_currency: str
) -> np.ndarray:
    """Compose local asset shocks with FX; explicit instrument values take precedence."""
    if not instruments or not scenarios:
        raise ValueError("instruments and scenarios must be nonempty")
    output = np.empty((len(scenarios), len(instruments)))
    for row, scenario in enumerate(scenarios):
        for column, instrument in enumerate(instruments):
            if instrument.asset_id in scenario.instrument_returns:
                local = scenario.instrument_returns[instrument.asset_id]
            elif instrument.asset_class in {"bond", "bond_etf", "treasury", "fixed_income"}:
                if instrument.modified_duration is None or instrument.convexity is None:
                    raise ValueError(f"{instrument.asset_id}: bond shock requires duration and convexity")
                local = bond_total_return_from_shock(
                    instrument.modified_duration,
                    instrument.convexity,
                    scenario.rate_change,
                    credit_spread_change=scenario.credit_spread_change,
                    spread_duration=instrument.spread_duration,
                    carry=scenario.bond_carry,
                )
            else:
                local = scenario.instrument_returns.get(
                    instrument.asset_id,
                    scenario.market_returns.get(
                        instrument.market,
                        scenario.sector_returns.get(instrument.sector, scenario.equity_return),
                    ),
                )
            if instrument.currency == base_currency:
                fx = 0.0
            elif instrument.currency in scenario.fx_returns:
                fx = scenario.fx_returns[instrument.currency]
            else:
                raise ValueError(f"{scenario.name}: missing explicit FX shock for {instrument.currency}")
            output[row, column] = float(base_currency_simple_returns(local, fx))
    return output


@dataclass(frozen=True)
class RegimeInputs:
    recession_probability: float | None = None
    drawdown: float | None = None
    annualized_volatility: float | None = None
    calibration_valid: bool = True
    data_valid: bool = True
    forecast_available: bool = True
    inputs_available_at: str | None = None


@dataclass(frozen=True)
class RegimePolicy:
    defensive_recession_probability: float = 0.60
    cash_recession_probability: float = 0.85
    defensive_drawdown: float = -0.10
    cash_drawdown: float = -0.20
    defensive_annualized_volatility: float = 0.30
    defensive_cash_floor: float = 0.30
    defensive_risk_multiplier: float = 2.0


def choose_regime(inputs: RegimeInputs, policy: RegimePolicy | None = None) -> tuple[str, list[str]]:
    """A transparent declared decision policy, not a recession-prediction model."""
    policy = policy or RegimePolicy()
    reasons: list[str] = []
    for name in ("data_valid", "calibration_valid", "forecast_available"):
        if not isinstance(getattr(inputs, name), (bool, np.bool_)):
            return "no_action", [f"{name} must be an explicit Boolean"]
    # Validate every supplied observation before considering any threshold.
    # An early cash preference must not hide an invalid later observation.
    for name, value, lower, upper in (
        ("recession probability", inputs.recession_probability, 0, 1),
        ("drawdown", inputs.drawdown, -1, 0),
        ("annualized volatility", inputs.annualized_volatility, 0, None),
    ):
        if value is not None:
            try:
                invalid = not np.isfinite(value) or value < lower or (upper is not None and value > upper)
            except (ValueError, TypeError):
                invalid = True
            if invalid:
                return "no_action", [f"invalid supplied {name}"]
    if not inputs.data_valid:
        return "no_action", ["input data quality gate failed"]
    if not inputs.calibration_valid:
        return "no_action", ["forecast calibration gate failed"]
    if not inputs.forecast_available:
        reasons.append("forecast unavailable: use the constrained risk-only fallback")
    if inputs.recession_probability is not None:
        p = inputs.recession_probability
        if not np.isfinite(p) or not 0 <= p <= 1:
            return "no_action", ["invalid supplied recession probability"]
        if p >= policy.cash_recession_probability:
            return "cash", ["supplied recession probability crossed the declared cash preference threshold"]
        if p >= policy.defensive_recession_probability:
            reasons.append("supplied recession probability crossed the declared defensive threshold")
    if inputs.drawdown is not None:
        dd = inputs.drawdown
        if not np.isfinite(dd) or not -1 <= dd <= 0:
            return "no_action", ["drawdown must be a finite signed loss in [-1,0]"]
        if dd <= policy.cash_drawdown:
            return "cash", ["observed drawdown crossed the declared cash preference threshold"]
        if dd <= policy.defensive_drawdown:
            reasons.append("observed drawdown crossed the declared defensive threshold")
    if inputs.annualized_volatility is not None:
        vol = inputs.annualized_volatility
        if not np.isfinite(vol) or vol < 0:
            return "no_action", ["invalid supplied annualized volatility"]
        if vol >= policy.defensive_annualized_volatility:
            reasons.append("supplied annualized volatility crossed the declared defensive threshold")
    return ("defensive", reasons) if reasons else ("normal", ["no declared defensive threshold was crossed"])


def portfolio_risk_summary(
    weights: Sequence[float],
    scenarios: Sequence[Sequence[float]],
    *,
    cash_return: float = 0.0,
    confidence: float = 0.95,
    probabilities: Sequence[float] | None = None,
    nav_retention: float = 1.0,
) -> dict[str, float]:
    risky = np.asarray(weights, dtype=float)
    values = np.asarray(scenarios, dtype=float)
    if risky.ndim != 1 or values.ndim != 2 or values.shape[1] != len(risky):
        raise ValueError("portfolio weights and scenario columns must align")
    if np.any(risky < 0) or risky.sum() > 1 + 1e-9 or not np.all(np.isfinite(risky)):
        raise ValueError("long-only unlevered weights are required")
    if not np.all(np.isfinite(values)) or np.any(values < -1):
        raise ValueError("scenarios must be finite simple returns >= -1")
    if not np.isfinite(cash_return) or cash_return < -1 or not 0 < nav_retention <= 1:
        raise ValueError("invalid cash return or after-cost NAV retention")
    p = scenario_probabilities(len(values), probabilities)
    outcomes = nav_retention * (1 + values @ risky + (1 - risky.sum()) * cash_return) - 1
    losses = -outcomes
    return {
        "scenario_mean_return": float(p @ outcomes),
        "scenario_var_loss": float(weighted_quantile(losses, confidence, p)[0]),
        "scenario_cvar_loss": weighted_cvar(losses, confidence, p),
        "worst_scenario_return": float(outcomes.min()),
        "scenario_loss_probability": float(p[outcomes < 0].sum()),
        "confidence": confidence,
    }
