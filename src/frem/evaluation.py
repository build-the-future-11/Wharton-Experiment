"""Evaluation of saved forecasts only. This module never reads model future_x."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .metrics import distributional_scores, portfolio_return_metrics, probability_calibration


def _time_ns(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"{name} contains missing timestamps")
    if pd.api.types.is_integer_dtype(series):
        return series.to_numpy(dtype=np.int64)
    if pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{name} cannot use float timestamps; nanosecond precision would be lost")
    for value in series:
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError(f"{name} requires timezone-qualified timestamps")
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def validate_predictions(frame: pd.DataFrame, require_actual: bool = True) -> pd.DataFrame:
    required = {"time", "label_time", "asset_id", "prediction", "scale", "actual"}
    if required - set(frame):
        raise ValueError(f"Prediction frame missing {sorted(required - set(frame))}")
    if not len(frame):
        raise ValueError("Prediction frame is empty")
    result = frame.copy()
    for name in ("asset_id", "seed"):
        if name in result:
            numeric = pd.to_numeric(result[name], errors="raise")
            values = numeric.to_numpy(float)
            if not np.all(np.isfinite(values) & (values >= 0) & (values == np.floor(values))):
                raise ValueError(f"{name} must contain finite nonnegative integer identifiers")
            if any(int(value) > np.iinfo(np.int64).max for value in numeric):
                raise ValueError(f"{name} exceeds the supported integer identifier range")
            result[name] = numeric.astype("int64")
    for name in ("time", "label_time", "entry_time", "exit_time"):
        if name in result:
            result[name] = _time_ns(result[name], name)
    keys = ["time", "asset_id"] + (["seed"] if "seed" in result else [])
    if result.duplicated(keys).any():
        raise ValueError("Duplicate prediction decision/asset/seed keys")
    if (result["label_time"] <= result["time"]).any():
        raise ValueError("Labels must mature strictly after a forecast")
    for name in ("prediction", "scale"):
        result[name] = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(result[name].to_numpy(float)).all():
            raise ValueError(f"Nonfinite {name}; retain the failed run instead of dropping rows")
    if (result["scale"] <= 0).any():
        raise ValueError("Predictive scale must be positive")
    result["actual"] = pd.to_numeric(result["actual"], errors="coerce")
    if require_actual and not np.isfinite(result["actual"].to_numpy(float)).all():
        raise ValueError("Actual outcomes are missing/nonfinite; unlabeled forecasts cannot be evaluated")
    if "entry_time" in result and (result["entry_time"] <= result["time"]).any():
        raise ValueError("An executed entry cannot be at or before the decision")
    if ("entry_time" in result) != ("exit_time" in result):
        raise ValueError("Execution timestamps require both entry_time and exit_time")
    if {"entry_time", "exit_time"}.issubset(result) and (
        (result["exit_time"] <= result["entry_time"]).any()
        or (result["label_time"] < result["exit_time"]).any()
    ):
        raise ValueError("Invalid executable label interval")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def score_predictions(
    frame: pd.DataFrame, student_df: float | None = 5.0, interval_levels: Sequence[float] = (0.5, 0.8, 0.95)
) -> pd.DataFrame:
    """Return every input row plus scores; never silently delete failed forecasts."""
    predictions = validate_predictions(frame)
    if "student_df" in predictions or "df" in predictions:
        column = "student_df" if "student_df" in predictions else "df"
        degrees = pd.to_numeric(predictions[column], errors="raise").to_numpy(float)
        if "student_df" in predictions and "df" in predictions:
            other = pd.to_numeric(predictions["df"], errors="raise").to_numpy(float)
            if not np.allclose(degrees, other, rtol=0, atol=1e-12):
                raise ValueError("Conflicting student_df and df columns")
    elif student_df is not None:
        degrees = float(student_df)
    else:
        raise ValueError("A saved student_df column or explicit evaluation df is required")
    scores = distributional_scores(
        predictions["actual"].to_numpy(),
        predictions["prediction"].to_numpy(),
        predictions["scale"].to_numpy(),
        degrees,
        interval_levels,
    )
    duplicate_scores = set(scores) - {"predictive_std"}
    if duplicate_scores.intersection(predictions.columns):
        raise ValueError(
            "Frame already contains score columns; supply original predictions to avoid stale scores"
        )
    for name in scores:
        predictions[name] = scores[name].to_numpy()
    return predictions


def evaluate_predictions(
    frame: pd.DataFrame, student_df: float | None = 5.0, interval_levels: Sequence[float] = (0.5, 0.8, 0.95)
) -> dict[str, Any]:
    scores = score_predictions(frame, student_df, interval_levels)
    if "seed" in scores and scores["seed"].nunique() > 1:
        raise ValueError(
            "Evaluate each run separately; average matched per-row scores across seeds for inference"
        )
    dates = pd.to_datetime(scores["time"], utc=True).dt.normalize()
    numeric_scores = ["nll", "crps", "squared_error", "absolute_error", "brier_up", "direction_correct"]
    for level in interval_levels:
        suffix = f"{level:g}"
        numeric_scores.extend([f"coverage_{suffix}", f"interval_width_{suffix}", f"interval_score_{suffix}"])
    per_asset_date = (
        scores.assign(utc_date=dates)
        .groupby(["utc_date", "asset_id"])[numeric_scores + ["prediction", "actual"]]
        .mean()
    )
    date_means = per_asset_date.groupby(level="utc_date").mean()
    rank_correlations: list[float] = []
    undefined_rank_dates = 0
    for _, cross_section in per_asset_date.groupby(level="utc_date"):
        # Rank each asset's date-mean forecast and outcome with average ties.
        # Constant forecasts or tiny universes have an undefined rank statistic.
        if (
            len(cross_section) < 3
            or cross_section["prediction"].nunique() < 2
            or cross_section["actual"].nunique() < 2
        ):
            undefined_rank_dates += 1
            continue
        ranked = cross_section[["prediction", "actual"]].rank(method="average")
        rank_correlations.append(float(ranked["prediction"].corr(ranked["actual"])))
    calibration = probability_calibration(
        scores["p_up"].to_numpy(), (scores["actual"].to_numpy() > 0).astype(float)
    )
    pit_hist, pit_edges = np.histogram(scores["pit"], bins=np.linspace(0, 1, 11))
    intervals = {}
    for level in interval_levels:
        suffix = f"{level:g}"
        intervals[suffix] = {
            "nominal": float(level),
            "coverage": float(date_means[f"coverage_{suffix}"].mean()),
            "mean_width": float(date_means[f"interval_width_{suffix}"].mean()),
            "mean_interval_score": float(date_means[f"interval_score_{suffix}"].mean()),
        }
    return {
        "rows": len(scores),
        "utc_dates": len(date_means),
        "mean_nll": float(date_means["nll"].mean()),
        "nll": float(date_means["nll"].mean()),
        "mean_crps": float(date_means["crps"].mean()),
        "mse": float(date_means["squared_error"].mean()),
        "mae": float(date_means["absolute_error"].mean()),
        "brier_up": float(date_means["brier_up"].mean()),
        "direction_accuracy": float(date_means["direction_correct"].mean()),
        "pooled_mean_nll": float(scores["nll"].mean()),
        "primary_weighting": "equal UTC-date weight, equal asset weight within date",
        "calibration": calibration,
        "intervals": intervals,
        "calibration_weighting": "reliability bins and PIT counts pool individual observations",
        "rank_correlation": {
            "mean_over_defined_dates": float(np.mean(rank_correlations)) if rank_correlations else None,
            "defined_dates": len(rank_correlations),
            "undefined_dates": undefined_rank_dates,
            "definition": "Spearman correlation of per-asset date-mean forecasts and outcomes; at least three assets and nonconstant ranks required",
            "independent_sample_p_value": None,
        },
        "pit_histogram": {
            "counts": pit_hist.tolist(),
            "edges": pit_edges.tolist(),
            "independent_sample_p_value": None,
        },
        "crps_method": "96-node deterministic quantile quadrature",
        "forecast_units": "cumulative base-currency log total return",
        "distribution_note": "scale is Student-t scale, not standard deviation; no finite exponential moment",
        "performance_claim": "scores describe supplied observations only; they do not establish future profitability",
    }


def build_execution_ledger(
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    entry_cost_bps: float = 10.0,
    exit_cost_bps: float = 10.0,
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Account for explicit long-only decisions as nonoverlapping round trips.

    Weights specify fractions after entry costs and sum to at most one; remainder
    is cash. Each interval liquidates risky holdings, so there is no hidden carry
    or turnover netting. Entry and exit must align across every selected asset.
    No optimizer, inference, data fetching or broker call occurs here.
    """
    forecasts = validate_predictions(predictions)
    if "seed" in forecasts and forecasts["seed"].nunique() > 1:
        raise ValueError("A ledger needs one predeclared portfolio decision, not multiple seed paths")
    if {"entry_time", "exit_time"} - set(forecasts):
        raise ValueError("Execution ledger needs actual entry_time and exit_time label metadata")
    if {"time", "asset_id", "weight"} - set(weights):
        raise ValueError("Weights need time, asset_id, weight")
    if not len(weights):
        raise ValueError("Explicit weights are required")
    if not (0 <= entry_cost_bps < 10000 and 0 <= exit_cost_bps < 10000):
        raise ValueError("Cost inputs are nonnegative bps less than 10,000")
    w = weights.copy()
    w["time"] = _time_ns(w["time"], "weights.time")
    if w.duplicated(["time", "asset_id"]).any():
        raise ValueError("Duplicate weight keys")
    weight_values = pd.to_numeric(w["weight"], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(weight_values) & (weight_values >= 0)):
        raise ValueError("Research ledger supports finite long-only weights")
    w["weight"] = weight_values
    if "weight_known_at" in w and np.any(
        _time_ns(w["weight_known_at"], "weight_known_at") > w["time"].to_numpy()
    ):
        raise ValueError("Portfolio weights became known after their purported decision")
    joined = w.merge(forecasts, on=["time", "asset_id"], how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("A requested portfolio weight has no matching executable return")
    records = []
    previous_exit: int | None = None
    for decision, rows in joined.groupby("time", sort=True):
        if rows["entry_time"].nunique() != 1 or rows["exit_time"].nunique() != 1:
            raise ValueError("Asynchronous asset intervals need a synchronized price/holding ledger adapter")
        entry, exit_time = int(rows["entry_time"].iloc[0]), int(rows["exit_time"].iloc[0])
        if previous_exit is not None and entry < previous_exit:
            raise ValueError(
                "Overlapping H-horizon labels cannot be compounded as consecutive portfolio returns"
            )
        previous_exit = exit_time
        fractions = rows["weight"].to_numpy(float)
        invested = float(fractions.sum())
        if invested > 1 + 1e-10:
            raise ValueError("Weights exceed one; leveraged portfolios are not supported by this ledger")
        invested = min(1.0, invested)
        if cash_returns is None:
            cash_return = 0.0
        else:
            if decision not in cash_returns.index:
                raise ValueError("Missing explicit cash return for a decision interval")
            cash_return = float(cash_returns.loc[decision])
            if not np.isfinite(cash_return) or cash_return <= -1:
                raise ValueError("Invalid cash interval return")
        asset_gross = np.exp(rows["actual"].to_numpy(float))
        if not np.isfinite(asset_gross).all():
            raise ValueError("Observed log returns overflow simple-return accounting")
        risky_terminal = float(np.dot(fractions, asset_gross))
        gross_wealth = risky_terminal + (1 - invested) * (1 + cash_return)
        alpha = 1.0 / (1.0 + entry_cost_bps / 1e4 * invested)
        entry_cost = 1.0 - alpha
        exit_notional = alpha * risky_terminal
        exit_cost = exit_cost_bps / 1e4 * exit_notional
        net_wealth = alpha * gross_wealth - exit_cost
        if net_wealth <= 0:
            raise ValueError(
                "Costs/returns consume all wealth; report insolvency instead of hiding the interval"
            )
        records.append(
            {
                "time": int(decision),
                "entry_time": entry,
                "exit_time": exit_time,
                "label_time": int(rows["label_time"].max()),
                "gross_return": gross_wealth - 1,
                "net_return": net_wealth - 1,
                "cash_weight": 1 - invested,
                "entry_cost_fraction": entry_cost,
                "exit_cost_fraction_of_initial": exit_cost,
                "turnover": alpha * invested + exit_notional,
                "assets_held": int(np.sum(fractions > 0)),
                "cash_return": cash_return,
            }
        )
    ledger = pd.DataFrame(records)
    ledger.attrs["accounting"] = "nonoverlapping, long-only, fully liquidated round trips"
    ledger.attrs["cash_return_assumption"] = "zero" if cash_returns is None else "explicit interval series"
    ledger.attrs["entry_cost_bps"] = entry_cost_bps
    ledger.attrs["exit_cost_bps"] = exit_cost_bps
    return ledger


def evaluate_portfolio(
    ledger: pd.DataFrame,
    periods_per_year: float | None = None,
    risk_free_returns: np.ndarray | None = None,
    regular_periods: bool = False,
) -> dict[str, Any]:
    required = {"time", "entry_time", "exit_time", "gross_return", "net_return", "turnover"}
    if required - set(ledger) or not len(ledger):
        raise ValueError(f"Nonempty portfolio ledger requires {sorted(required)}")
    frame = ledger.copy()
    for name in ("time", "entry_time", "exit_time"):
        frame[name] = _time_ns(frame[name], name)
    order = np.argsort(frame["entry_time"].to_numpy(), kind="stable")
    frame = frame.iloc[order]
    if risk_free_returns is not None:
        risk_free_returns = np.asarray(risk_free_returns, dtype=float)
        if risk_free_returns.shape != (len(frame),):
            raise ValueError("Risk-free interval returns must align with the input ledger")
        risk_free_returns = risk_free_returns[order]
    if frame["entry_time"].duplicated().any() or (frame["entry_time"] <= frame["time"]).any():
        raise ValueError("Duplicate or contemporaneous execution entries")
    if (frame["exit_time"] <= frame["entry_time"]).any():
        raise ValueError("Exit precedes entry")
    if np.any(frame["entry_time"].to_numpy()[1:] < frame["exit_time"].to_numpy()[:-1]):
        raise ValueError("Ledger intervals overlap")
    for name in ("gross_return", "net_return", "turnover"):
        if not np.isfinite(frame[name].to_numpy(float)).all():
            raise ValueError(f"Nonfinite portfolio {name}")
    if (frame["net_return"] <= -1).any() or (frame["gross_return"] <= -1).any():
        raise ValueError("Wealth hit zero; an insolvency-specific report is required")
    if (frame["turnover"] < 0).any():
        raise ValueError("Turnover cannot be negative")
    returns = frame["net_return"].to_numpy(float)
    path = np.r_[1.0, np.cumprod(1 + returns)]
    report = {
        "periods": len(frame),
        "net_total_return": float(path[-1] - 1),
        "gross_total_return": float(np.prod(1 + frame["gross_return"].to_numpy(float)) - 1),
        "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
        "mean_turnover": float(frame["turnover"].mean()),
        "mean_net_return": float(returns.mean()),
        "mean_log_return": float(np.log1p(returns).mean()),
        "intervals_nonoverlapping": True,
        "annualized": False,
    }
    if periods_per_year is not None:
        if not regular_periods:
            raise ValueError(
                "Annualized metrics require regular_periods=True and an explicit protocol cadence"
            )
        report["annualized_metrics"] = portfolio_return_metrics(returns, periods_per_year, risk_free_returns)
        report["annualized"] = True
    elif risk_free_returns is not None:
        raise ValueError("Risk-free excess metrics need an explicit periodicity")
    return report


def preserve_failure(
    directory: str | Path, stage: str, error: Exception, context: dict[str, Any] | None = None
) -> Path:
    """Write an append-only failure artifact, never an empty success metric file."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "FAILED",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "context": context or {},
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    filename = target / f"failure-{uuid4().hex}.json"
    with filename.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")
    return filename
