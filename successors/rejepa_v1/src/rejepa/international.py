"""International return, currency and exposure analytics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def currency_adjusted_return(local_return: np.ndarray, fx_return_to_base: np.ndarray) -> np.ndarray:
    local = np.asarray(local_return, dtype=float)
    fx = np.asarray(fx_return_to_base, dtype=float)
    if local.shape != fx.shape or not np.isfinite(local).all() or not np.isfinite(fx).all():
        raise ValueError("Local and FX returns must be finite and aligned")
    return (1 + local) * (1 + fx) - 1


def aggregate_exposure(
    weights: np.ndarray, labels: Sequence[str], *, eligible: Sequence[bool] | None = None
) -> dict[str, float]:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(labels) != values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Weights and labels are invalid")
    if eligible is not None and (
        len(eligible) != values.size
        or any(not flag and value > 0 for flag, value in zip(eligible, values, strict=True))
    ):
        raise ValueError("Positive weight assigned to an ineligible instrument")
    result: dict[str, float] = {}
    for label, value in zip(labels, values, strict=True):
        result[label] = result.get(label, 0.0) + float(value)
    return result


def diversification_delta(
    domestic_returns: np.ndarray, global_returns: np.ndarray, global_weight: float
) -> dict[str, float]:
    domestic = np.asarray(domestic_returns, dtype=float)
    global_series = np.asarray(global_returns, dtype=float)
    if domestic.shape != global_series.shape or domestic.ndim != 1 or domestic.size < 2:
        raise ValueError("Return histories must be aligned vectors")
    if not np.isfinite(domestic).all() or not np.isfinite(global_series).all() or not 0 <= global_weight <= 1:
        raise ValueError("Invalid diversification inputs")
    mixed = (1 - global_weight) * domestic + global_weight * global_series
    return {
        "domestic_volatility": float(domestic.std(ddof=1)),
        "mixed_volatility": float(mixed.std(ddof=1)),
        "volatility_delta": float(mixed.std(ddof=1) - domestic.std(ddof=1)),
        "correlation": float(np.corrcoef(domestic, global_series)[0, 1]),
        "mixed_mean_return": float(mixed.mean()),
    }
