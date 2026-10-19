"""Checked finite probability operations; no implicit clipping or renormalization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError, NumericalError
from frem_s10.metadata import finite_float, strict_int


def validate_probabilities(
    values: ArrayLike, *, size: int | None = None, tolerance: float = 1e-12
) -> NDArray[np.float64]:
    """Return owned probabilities after verifying the declared simplex tolerance."""
    tolerance = finite_float(tolerance, name="probability_tolerance", minimum=0)
    if tolerance > 1e-12:
        raise ContractError("probability tolerance cannot exceed the declared 1e-12 bound")
    if size is not None:
        strict_int(size, name="probability_size", minimum=1)
    p = frozen_f64(values, shape=(size,), name="probabilities")
    if len(p) == 0 or np.any(p < 0) or abs(float(p.sum()) - 1.0) > tolerance:
        raise ContractError("probabilities must be nonnegative and sum to one")
    return p


def normalize_positive(values: ArrayLike) -> NDArray[np.float64]:
    """Normalize explicitly nonnegative weights, rejecting empty/zero mass."""
    weights = frozen_f64(values, shape=(None,), name="unnormalized weights")
    if len(weights) == 0 or np.any(weights < 0):
        raise ContractError("normalization requires nonnegative nonempty weights")
    maximum = float(weights.max())
    if maximum <= 0:
        raise NumericalError("normalization has zero mass")
    scaled = weights / maximum
    return validate_probabilities(scaled / float(scaled.sum()))


def softmax(log_weights: ArrayLike) -> NDArray[np.float64]:
    """Stable finite softmax; subtracting the maximum makes underflow benign."""
    values = frozen_f64(log_weights, shape=(None,), name="log weights")
    if len(values) == 0:
        raise ContractError("softmax needs at least one value")
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        weights = np.exp(values - float(values.max()))
    return normalize_positive(weights)


def recency_weights(
    origins_us: ArrayLike, *, decision_us: int, half_life_weeks: float = 52.0
) -> NDArray[np.float64]:
    """Exponential age weights using elapsed seven-day weeks, not record ranks."""
    strict_int(decision_us, name="decision_us")
    half_life_weeks = finite_float(half_life_weeks, name="half_life_weeks", strictly_positive=True)
    origins = np.asarray(origins_us)
    if origins.ndim != 1 or origins.dtype.kind not in "iu" or len(origins) == 0:
        raise ContractError("origin timestamps must be a nonempty integer vector")
    if half_life_weeks <= 0 or not np.isfinite(half_life_weeks):
        raise ContractError("half-life must be finite and positive")
    if np.any(origins >= decision_us):
        raise ContractError("recency timestamps must precede the decision")
    age = (decision_us - origins.astype(np.float64)) / (7 * 86400 * 1_000_000)
    return softmax(-np.log(2.0) * age / half_life_weeks)
