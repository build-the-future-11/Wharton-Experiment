"""Explicit forecast-to-portfolio assumptions for one synchronous decision.

This bridge never fits a model, reads realized outcomes into a decision, fetches
market data or places orders. Scenario generation and optimization occur only
when its functions are explicitly called. A bounded Gaussian-copula scenario
model is an additional allocation assumption, not the original log Student-t
distribution and not a learned stochastic differential equation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from .artifacts import object_digest
from .mandate import Instrument, Mandate, scenario_probabilities
from .portfolio import PortfolioConfig, optimize_portfolio
from .risk import RegimeInputs, validate_covariance

LOG_RETURN_SPACE = "cumulative_base_currency_log_total_return"
SIMPLE_RETURN_SPACE = "simple"


def _timestamp(value: Any, field_name: str) -> pd.Timestamp:
    """Preserve nanosecond integer timestamps; reject naive/float time claims."""
    if isinstance(value, (bool, float, np.floating)) or value is None:
        raise ValueError(
            f"{field_name} needs an integer UTC nanosecond timestamp or timezone-qualified date-time"
        )
    try:
        if isinstance(value, (int, np.integer)):
            stamp = pd.Timestamp(int(value), unit="ns", tz="UTC")
        else:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                raise ValueError(f"{field_name} needs an explicit UTC offset")
            stamp = stamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc
    if pd.isna(stamp):
        raise ValueError(f"{field_name} is missing")
    return stamp


def _nonempty_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must state a nonempty source or assumption")


@dataclass(frozen=True)
class ReturnContract:
    """Attested common units, asset order and executable forecast interval.

    ``inputs_available_at`` must be the latest availability time of every
    forecast/risk input represented by the request. It is an input assertion,
    not independent verification that an upstream provider used no future data.
    """

    return_space: str
    base_currency: str
    asset_ids: Sequence[str]
    forecast_time: str | int
    inputs_available_at: str | int
    horizon_bars: int
    entry_time: str | int
    exit_time: str | int
    common_execution_horizon_attested: bool
    timing_source: str
    input_provenance: str

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ReturnContract:
        return cls(**dict(values))

    def validate(
        self, *, as_of: Any, asset_ids: Sequence[str], base_currency: str, expected_return_space: str
    ) -> None:
        if self.return_space != expected_return_space:
            raise ValueError(f"return_space must explicitly be {expected_return_space!r}")
        if not self.base_currency or self.base_currency != base_currency:
            raise ValueError("return contract base currency differs from the mandate")
        if isinstance(self.asset_ids, (str, bytes)) or not self.asset_ids:
            raise ValueError("asset_ids must be a nonempty ordered sequence")
        if any(not isinstance(x, str) or not x for x in self.asset_ids):
            raise ValueError("return contract asset IDs must be nonempty stable strings")
        if len(set(self.asset_ids)) != len(self.asset_ids) or list(self.asset_ids) != list(asset_ids):
            raise ValueError("return contract asset order must exactly match the instrument master")
        if (
            isinstance(self.horizon_bars, bool)
            or not isinstance(self.horizon_bars, (int, np.integer))
            or self.horizon_bars < 1
        ):
            raise ValueError("horizon_bars must be an explicit positive integer")
        if self.common_execution_horizon_attested is not True:
            raise ValueError("a synchronous common executable horizon must be explicitly attested")
        _nonempty_text(self.timing_source, "timing_source")
        _nonempty_text(self.input_provenance, "input_provenance")
        decision = _timestamp(as_of, "as_of")
        forecast = _timestamp(self.forecast_time, "forecast_time")
        available = _timestamp(self.inputs_available_at, "inputs_available_at")
        entry, exit_time = _timestamp(self.entry_time, "entry_time"), _timestamp(self.exit_time, "exit_time")
        if forecast != decision:
            raise ValueError("this bridge requires one common forecast time exactly equal to as_of")
        if available > decision:
            raise ValueError("forecast or risk input was unavailable at the decision time")
        if not decision < entry < exit_time:
            raise ValueError("require decision < common entry < common exit")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedScenarioConfig:
    """Finite scenario policy; all statistical choices must be declared.

    Correlation is the Gaussian latent copula correlation, not the resulting
    simple-return correlation. No correlation is estimated from evaluation data.
    """

    n_scenarios: int
    seed: int
    lower_log_return: float | Sequence[float]
    upper_log_return: float | Sequence[float]
    dependence: str
    assumption_rationale: str
    assumptions_available_at: str | int
    assumption_source: str
    correlation: Sequence[Sequence[float]] | None = None
    correlation_available_at: str | int | None = None
    correlation_source: str | None = None
    probabilities: Sequence[float] | None = None
    weighting_rationale: str | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> BoundedScenarioConfig:
        return cls(**dict(values))

    def validated_arrays(
        self, n_assets: int, as_of: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            isinstance(self.n_scenarios, bool)
            or not isinstance(self.n_scenarios, int)
            or not 2 <= self.n_scenarios <= 100_000
        ):
            raise ValueError("n_scenarios must be an integer between 2 and 100000")
        if n_assets < 1 or n_assets * self.n_scenarios > 2_000_000:
            raise ValueError("bounded bridge budget permits at most 2000000 scenario-asset values")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("scenario seed must be an explicit nonnegative integer")
        _nonempty_text(self.assumption_rationale, "assumption_rationale")
        _nonempty_text(self.assumption_source, "assumption_source")
        decision = _timestamp(as_of, "as_of")
        if _timestamp(self.assumptions_available_at, "assumptions_available_at") > decision:
            raise ValueError("the clipping/dependence policy was chosen after the decision")
        bounds = []
        for name in ("lower_log_return", "upper_log_return"):
            raw = np.asarray(getattr(self, name), dtype=float)
            if raw.ndim == 0:
                raw = np.full(n_assets, float(raw))
            if raw.shape != (n_assets,) or not np.all(np.isfinite(raw)):
                raise ValueError(f"{name} must be a finite scalar or aligned asset vector")
            bounds.append(raw)
        lower, upper = bounds
        if np.any(lower >= upper):
            raise ValueError("each lower log-return bound must be strictly below its upper bound")
        # Leave ample finite-float room for exponentiation and covariance squares.
        if np.any(lower <= np.log(np.finfo(float).tiny)) or np.any(
            upper >= 0.5 * np.log(np.finfo(float).max) - 5
        ):
            raise ValueError("log-return bounds exceed safe finite-moment arithmetic limits")
        if self.dependence == "independent":
            if any(
                value is not None
                for value in (self.correlation, self.correlation_available_at, self.correlation_source)
            ):
                raise ValueError("independence cannot simultaneously specify a correlation matrix/source")
            correlation = np.eye(n_assets)
        elif self.dependence == "gaussian_copula":
            if self.correlation is None or self.correlation_available_at is None:
                raise ValueError("Gaussian copula requires an explicitly supplied dated correlation matrix")
            _nonempty_text(self.correlation_source, "correlation_source")
            if _timestamp(self.correlation_available_at, "correlation_available_at") > decision:
                raise ValueError("copula correlation was unavailable at the decision timestamp")
            correlation = validate_covariance(self.correlation, n_assets)
            if not np.allclose(np.diag(correlation), 1, atol=1e-10, rtol=0) or np.any(
                np.abs(correlation) > 1 + 1e-10
            ):
                raise ValueError("copula correlation needs a unit diagonal and entries within [-1,1]")
        else:
            raise ValueError("dependence must explicitly be 'gaussian_copula' or 'independent'")
        if self.probabilities is not None:
            _nonempty_text(self.weighting_rationale, "weighting_rationale")
        probabilities = scenario_probabilities(self.n_scenarios, self.probabilities)
        return lower, upper, correlation, probabilities


@dataclass(frozen=True)
class ForecastScenarioBundle:
    simple_returns: np.ndarray
    probabilities: np.ndarray
    expected_simple_returns: np.ndarray
    simple_return_covariance: np.ndarray
    forecasts: list[dict[str, Any]]
    assumptions: dict[str, Any]


def _master(values: Sequence[Instrument] | pd.DataFrame) -> list[Instrument]:
    records = values.to_dict(orient="records") if isinstance(values, pd.DataFrame) else list(values)
    return [value if isinstance(value, Instrument) else Instrument.from_mapping(value) for value in records]


def _aligned_forecasts(frame: pd.DataFrame, contract: ReturnContract) -> pd.DataFrame:
    required = {"asset", "time", "prediction", "scale", "student_df"}
    if not isinstance(frame, pd.DataFrame) or required - set(frame) or len(frame) == 0:
        raise ValueError(f"forecasts need nonempty columns {sorted(required)}")
    # Realized actual/label columns are deliberately not selected or inspected.
    selected = frame[list(required)].copy()
    if selected["asset"].isna().any() or any(not isinstance(x, str) or not x for x in selected["asset"]):
        raise ValueError("forecast asset must contain stable string IDs, not model-local integer asset_id")
    if selected["asset"].duplicated().any() or set(selected["asset"]) != set(contract.asset_ids):
        raise ValueError(
            "exactly one forecast per requested asset is required; seed/ensemble choices must be made explicitly"
        )
    forecast_time = _timestamp(contract.forecast_time, "forecast_time")
    if any(_timestamp(value, "forecast CSV time") != forecast_time for value in selected["time"]):
        raise ValueError("forecast rows do not share the declared decision timestamp")
    for field in ("entry_time", "exit_time"):
        if field in frame:
            expected = _timestamp(getattr(contract, field), field)
            if any(_timestamp(value, f"forecast CSV {field}") != expected for value in frame[field]):
                raise ValueError(f"forecast {field} conflicts with the declared common executable horizon")
    if "model" in frame and frame["model"].nunique(dropna=False) != 1:
        raise ValueError("one decision cannot silently combine forecasts from different model variants")
    if "seed" in frame and frame["seed"].nunique(dropna=False) != 1:
        raise ValueError("select one declared forecast path or an explicitly constructed ensemble artifact")
    if "df" in frame and not np.array_equal(frame["df"].to_numpy(), frame["student_df"].to_numpy()):
        raise ValueError("df and student_df disagree")
    selected = selected.set_index("asset").loc[list(contract.asset_ids)].reset_index()
    for field in ("prediction", "scale", "student_df"):
        selected[field] = pd.to_numeric(selected[field], errors="raise")
        if not np.all(np.isfinite(selected[field].to_numpy(float))):
            raise ValueError(f"forecast {field} is nonfinite")
    if (selected["scale"] <= 0).any() or (selected["student_df"] <= 2).any():
        raise ValueError("positive Student-t scale and df>2 are required by the fitted forecast contract")
    selected["time"] = forecast_time.isoformat()
    return selected


def bounded_forecast_scenarios(
    forecasts: pd.DataFrame,
    contract: ReturnContract,
    config: BoundedScenarioConfig,
    *,
    as_of: Any,
    asset_ids: Sequence[str],
    base_currency: str,
) -> ForecastScenarioBundle:
    """Create a finite probability model from explicitly clipped log forecasts.

    Draw Z~N(0,C), U=Phi(Z), Q_i=t_df_i^-1(U_i), and
    R_i=expm1(clip(mu_i + scale_i*Q_i, lower_i, upper_i)).
    Clipping creates boundary point masses; this is winsorization, not a
    conditional/truncated Student-t distribution. Weighted moments are exactly
    those of the retained finite scenario set, with no Bessel correction.
    """
    contract.validate(
        as_of=as_of, asset_ids=asset_ids, base_currency=base_currency, expected_return_space=LOG_RETURN_SPACE
    )
    aligned = _aligned_forecasts(forecasts, contract)
    lower, upper, correlation, probabilities = config.validated_arrays(len(asset_ids), as_of)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    rng = np.random.default_rng(config.seed)
    latent = rng.standard_normal((config.n_scenarios, len(asset_ids))) @ factor.T
    uniforms_raw = norm.cdf(latent)
    epsilon = np.finfo(float).eps
    uniforms = np.clip(uniforms_raw, epsilon, 1 - epsilon)
    df = aligned["student_df"].to_numpy(float)
    scale = aligned["scale"].to_numpy(float)
    location = aligned["prediction"].to_numpy(float)
    with np.errstate(over="raise", invalid="raise"):
        try:
            raw_log = location + scale * student_t.ppf(uniforms, df=df)
            if not np.all(np.isfinite(raw_log)):
                raise ValueError(
                    "unbounded log-scenario arithmetic is nonfinite under supplied forecast parameters"
                )
            below, above = raw_log < lower, raw_log > upper
            bounded_log = np.clip(raw_log, lower, upper)
            simple = np.expm1(bounded_log)
            mean = probabilities @ simple
            centered = simple - mean
            covariance = (centered.T * probabilities) @ centered
        except FloatingPointError as exc:
            raise ValueError("finite-scenario moment calculation overflowed") from exc
    covariance = validate_covariance(0.5 * (covariance + covariance.T), len(asset_ids))
    if not np.all(np.isfinite(mean)) or np.any(mean < -1):
        raise ValueError("invalid simple-return scenario expectation")
    assumptions = {
        "method": "Gaussian copula with winsorized marginal log Student-t quantiles",
        "seed": config.seed,
        "n_scenarios": config.n_scenarios,
        "return_space": SIMPLE_RETURN_SPACE,
        "base_currency": base_currency,
        "asset_ids": list(asset_ids),
        "horizon_bars": contract.horizon_bars,
        "entry_time": _timestamp(contract.entry_time, "entry_time").isoformat(),
        "exit_time": _timestamp(contract.exit_time, "exit_time").isoformat(),
        "dependence": config.dependence,
        "gaussian_latent_correlation": correlation.tolist(),
        "correlation_roundoff_eigenvalues_clipped": int(np.sum(eigenvalues < 0)),
        "correlation_source": config.correlation_source,
        "correlation_available_at": config.correlation_available_at,
        "lower_log_return": lower.tolist(),
        "upper_log_return": upper.tolist(),
        "lower_clipped_count_by_asset": dict(zip(asset_ids, below.sum(axis=0).tolist(), strict=True)),
        "upper_clipped_count_by_asset": dict(zip(asset_ids, above.sum(axis=0).tolist(), strict=True)),
        "clipped_scenario_count": int(np.any(below | above, axis=1).sum()),
        "cdf_numerical_epsilon": epsilon,
        "cdf_endpoint_clamp_count": int(np.sum(uniforms != uniforms_raw)),
        "scenario_weighting": "uniform Monte Carlo"
        if config.probabilities is None
        else "explicitly supplied probability weights",
        "weighting_rationale": config.weighting_rationale,
        "assumption_rationale": config.assumption_rationale,
        "assumption_source": config.assumption_source,
        "assumptions_available_at": config.assumptions_available_at,
        "covariance_convention": "weighted finite-distribution second central moment; no annualization or Bessel correction",
        "limitations": [
            "Log Student-t forecasts do not have finite exponential moments. These moments belong only to the explicitly winsorized finite scenario distribution.",
            "The copula or independence assumption is supplied, not learned or certified by this bridge.",
            "Latent Gaussian correlation is not a simple-return correlation or Student-t copula, and implies no calibrated joint tail risk.",
            "Clipping creates endpoint probability mass and can materially change expected returns and tail risk.",
            "Finite scenarios, clipping choices and assumed probabilities do not establish real-world funding certainty.",
        ],
    }
    return ForecastScenarioBundle(
        simple, probabilities, mean, covariance, aligned.to_dict(orient="records"), assumptions
    )


def decide_from_forecasts(
    forecasts: pd.DataFrame,
    instruments: Sequence[Instrument] | pd.DataFrame,
    mandate: Mandate,
    *,
    contract: ReturnContract | Mapping[str, Any],
    scenario_config: BoundedScenarioConfig | Mapping[str, Any],
    as_of: Any,
    nav: float,
    current_weights: Sequence[float] | None,
    cash_return: float = 0.0,
    reserve_cash: float = 0.0,
    decision_context: RegimeInputs | None = None,
    config: PortfolioConfig | None = None,
) -> dict[str, Any]:
    """One research allocation with retained scenario evidence; fail closed.

    Current holdings are mandatory. `reserve_cash` is a separately justified
    immediate reserve floor, not an automatic ten-year liability guarantee.
    Realized CSV outcome columns are excluded from scenario construction.
    """
    result: dict[str, Any] = {
        "status": "blocked",
        "research_only": True,
        "allocation": None,
        "execution_scope": "one synchronous hypothetical allocation; no orders, realized P&L, settlement engine or asynchronous holdings ledger",
    }
    try:
        contract = contract if isinstance(contract, ReturnContract) else ReturnContract.from_dict(contract)
        sc = (
            scenario_config
            if isinstance(scenario_config, BoundedScenarioConfig)
            else BoundedScenarioConfig.from_dict(scenario_config)
        )
        master = _master(instruments)
        ids = [value.asset_id for value in master]
        contract.validate(
            as_of=as_of,
            asset_ids=ids,
            base_currency=mandate.base_currency,
            expected_return_space=LOG_RETURN_SPACE,
        )
        result["return_contract"] = contract.to_dict()
        issues = mandate.validate(as_of, require_allocation=True)
        if issues:
            result["blocking_reasons"] = issues
            return result
        if current_weights is None:
            raise ValueError(
                "current holdings are required; explicitly supply zeros for an initially all-cash portfolio"
            )
        holdings = np.asarray(current_weights, dtype=float)
        if (
            holdings.shape != (len(ids),)
            or not np.all(np.isfinite(holdings))
            or np.any(holdings < 0)
            or holdings.sum() > 1 + 1e-9
        ):
            raise ValueError("current holdings must be aligned finite unlevered weights")
        if decision_context is not None and not decision_context.forecast_available:
            raise ValueError("use the simple-array allocation risk-only path for an unavailable forecast")
        bundle = bounded_forecast_scenarios(
            forecasts, contract, sc, as_of=as_of, asset_ids=ids, base_currency=mandate.base_currency
        )
        allocation = optimize_portfolio(
            bundle.expected_simple_returns,
            bundle.simple_return_covariance,
            master,
            mandate,
            as_of=as_of,
            nav=nav,
            current_weights=holdings,
            scenarios=bundle.simple_returns,
            scenario_probabilities=bundle.probabilities,
            cash_return=cash_return,
            reserve_cash=reserve_cash,
            decision_context=decision_context,
            config=config,
        )
        scenario_record = {
            "asset_ids": ids,
            "returns": bundle.simple_returns.tolist(),
            "probabilities": bundle.probabilities.tolist(),
        }
        result.update(
            status=allocation.status,
            allocation=allocation.to_dict(),
            retained_forecasts=bundle.forecasts,
            scenario_assumptions=bundle.assumptions,
            scenario_moments={
                "asset_ids": ids,
                "expected_simple_returns": bundle.expected_simple_returns.tolist(),
                "simple_return_covariance": bundle.simple_return_covariance.tolist(),
            },
            scenarios=scenario_record,
            scenario_sha256=object_digest(scenario_record),
            mandate_sha256=object_digest(mandate.to_dict()),
            instrument_master_sha256=object_digest([asdict(x) for x in master]),
        )
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result["blocking_reasons"] = [str(exc)]
    return result
