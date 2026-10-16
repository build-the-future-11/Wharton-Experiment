"""Stable CRRA and exact contamination utility over declared finite support."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError, NumericalError

FloatArray = NDArray[np.float64]


def crra_utility(factors: FloatArray, *, gamma: float = 4.0) -> FloatArray:
    """Evaluate normalized CRRA on strictly positive wealth factors.

    The production policy fixes gamma=4. The explicit gamma argument permits
    independent mathematical checks; no fitted preference is inferred.
    """
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim not in (1, 2) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ContractError("wealth factors must be finite positive vectors or matrices")
    if isinstance(gamma, bool) or not math.isfinite(gamma) or gamma <= 0:
        raise ContractError("gamma must be finite and strictly positive")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            logarithms = np.log(values)
            result = logarithms if gamma == 1 else np.expm1((1.0 - gamma) * logarithms) / (1.0 - gamma)
        except FloatingPointError as error:
            raise NumericalError("CRRA utility arithmetic failed") from error
    if not np.all(np.isfinite(result)):
        raise NumericalError("CRRA utility is not finite")
    return result


def robust_objective(utilities: FloatArray, probabilities: FloatArray, *, rho: float = 0.05) -> FloatArray:
    """Return exact (1-rho) expected utility + rho minimum over all atoms.

    Zero nominal probability never removes an atom from contamination support.
    A one-dimensional utility vector returns a one-element objective array.
    """
    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ContractError("utilities must be a finite action-by-atom array")
    probabilities = frozen_f64(probabilities, shape=(values.shape[1],), name="probabilities")
    if np.any(probabilities < 0) or abs(float(np.sum(probabilities)) - 1.0) > 1e-12:
        raise ContractError("probabilities must lie on the simplex")
    if isinstance(rho, bool) or not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ContractError("contamination mass must be in [0, 1]")
    result = (1.0 - rho) * (values @ probabilities) + rho * np.min(values, axis=1)
    if not np.all(np.isfinite(result)):
        raise NumericalError("nonfinite utility objective")
    return result
