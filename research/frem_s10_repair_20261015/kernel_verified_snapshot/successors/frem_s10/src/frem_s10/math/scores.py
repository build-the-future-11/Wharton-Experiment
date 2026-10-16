"""Exact finite-law scores and diagnostics in explicitly supplied coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError, NumericalError
from frem_s10.math.distances import pairwise_distances
from frem_s10.math.probabilities import validate_probabilities
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ScoreCoefficients:
    """Exact expert cross coefficients; expert axis is U, R, C."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own finite coefficient arrays and verify the required B symmetry."""
        object.__setattr__(self, "a", frozen_f64(self.a, shape=(3,), name="score A"))
        object.__setattr__(self, "b", frozen_f64(self.b, shape=(3, 3), name="score B"))
        scale = max(1.0, float(np.abs(self.b).max()))
        if float(np.abs(self.b - self.b.T).max()) > 1e-12 * scale:
            raise NumericalError("score B is asymmetric")

    def score(self, alpha: ArrayLike) -> float:
        """Evaluate the exact mixture score, rejecting unrepresentable arithmetic."""
        weights = validate_probabilities(alpha, size=3)
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = float(weights @ self.a - 0.5 * weights @ self.b @ weights)
            except FloatingPointError as error:
                raise NumericalError("mixture score arithmetic overflow") from error
        if not np.isfinite(result):
            raise NumericalError("nonfinite mixture score")
        return result


def energy_score(atoms: ArrayLike, probabilities: ArrayLike, outcome: ArrayLike) -> float:
    """Energy score on supplied standardized coordinates, including all pair terms."""
    z = frozen_f64(atoms, shape=(None, None), name="score atoms")
    y = frozen_f64(outcome, shape=(z.shape[1],), name="score outcome")
    p = validate_probabilities(probabilities, size=len(z))
    first = pairwise_distances(z, y[None, :])[:, 0]
    pair = pairwise_distances(z)
    score = float(p @ first - 0.5 * p @ pair @ p)
    if not np.isfinite(score):
        raise NumericalError("nonfinite energy score")
    return score


def score_coefficients(
    atoms: ArrayLike, expert_probabilities: ArrayLike, outcome: ArrayLike
) -> ScoreCoefficients:
    """A/B of the exact issued expert laws; caller supplies fixed scaled coordinates."""
    z = frozen_f64(atoms, shape=(None, 3), name="coefficient atoms")
    y = frozen_f64(outcome, shape=(3,), name="coefficient outcome")
    experts = frozen_f64(expert_probabilities, shape=(3, len(z)), name="expert probabilities")
    for row in experts:
        validate_probabilities(row, size=len(z))
    pair = pairwise_distances(z)
    distances = pairwise_distances(z, y[None, :])[:, 0]
    a = experts @ distances
    b = experts @ pair @ experts.T
    return ScoreCoefficients(a=a, b=b)


def marginal_crps(atoms: ArrayLike, probabilities: ArrayLike, outcome: ArrayLike) -> NDArray[np.float64]:
    """Per-coordinate CRPS; call with standardized or raw returns explicitly."""
    z = frozen_f64(atoms, shape=(None, 3), name="CRPS atoms")
    y = frozen_f64(outcome, shape=(3,), name="CRPS outcome")
    p = validate_probabilities(probabilities, size=len(z))
    result = [energy_score(z[:, a : a + 1], p, y[a : a + 1]) for a in range(3)]
    return frozen_f64(result, shape=(3,), name="marginal CRPS")


def mean_squared_error(atoms: ArrayLike, probabilities: ArrayLike, outcome: ArrayLike) -> float:
    """Return mean squared predictive-mean error in the supplied coordinate units."""
    z = frozen_f64(atoms, shape=(None, 3), name="MSE atoms")
    y = frozen_f64(outcome, shape=(3,), name="MSE outcome")
    p = validate_probabilities(probabilities, size=len(z))
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = float(np.mean((p @ z - y) ** 2))
        except FloatingPointError as error:
            raise NumericalError("mean squared error arithmetic overflow") from error
    if not np.isfinite(result):
        raise NumericalError("nonfinite mean squared error")
    return result


def variogram_score(atoms: ArrayLike, probabilities: ArrayLike, outcome: ArrayLike) -> float:
    """Equally weighted three-pair variogram score of order one half."""
    z = frozen_f64(atoms, shape=(None, 3), name="variogram atoms")
    y = frozen_f64(outcome, shape=(3,), name="variogram outcome")
    p = validate_probabilities(probabilities, size=len(z))
    with np.errstate(over="raise", invalid="raise"):
        try:
            terms = [
                (np.sqrt(abs(y[a] - y[b])) - float(p @ np.sqrt(np.abs(z[:, a] - z[:, b])))) ** 2
                for a, b in ((0, 1), (0, 2), (1, 2))
            ]
            result = float(np.mean(terms))
        except FloatingPointError as error:
            raise NumericalError("variogram score arithmetic overflow") from error
    if not np.isfinite(result):
        raise NumericalError("nonfinite variogram score")
    return result


def discrete_quantiles(values: ArrayLike, probabilities: ArrayLike, levels: ArrayLike) -> NDArray[np.float64]:
    """Smallest support value whose cumulative mass reaches each requested level."""
    x = frozen_f64(values, shape=(None,), name="quantile values")
    p = validate_probabilities(probabilities, size=len(x))
    q = frozen_f64(levels, shape=(None,), name="quantile levels")
    if np.any((q < 0) | (q > 1)):
        raise ContractError("quantile levels must be in [0,1]")
    order = np.argsort(x, kind="stable")
    positive = order[p[order] > 0]
    cumulative = np.cumsum(p[positive])
    cumulative[-1] = 1.0  # validated simplex; close only accumulated rounding
    indices = np.searchsorted(cumulative, q, side="left")
    return frozen_f64(x[positive[indices]], shape=q.shape, name="quantiles")


def loss_probabilities(atoms: ArrayLike, probabilities: ArrayLike) -> NDArray[np.float64]:
    """Three asset losses plus a separate equal-weight basket; zero is non-loss."""
    r = frozen_f64(atoms, shape=(None, 3), name="raw return atoms")
    p = validate_probabilities(probabilities, size=len(r))
    values = np.concatenate((p @ (r < 0), [float(p @ (r.mean(axis=1) < 0))]))
    return frozen_f64(values, shape=(4,), name="loss probabilities")


def calibration_bins(
    probabilities: ArrayLike, outcomes: ArrayLike
) -> tuple[dict[str, float | int | bool | None], ...]:
    """Summarize fixed ten-bin reliability without omitting sparse or empty bins."""
    p = frozen_f64(probabilities, shape=(None,), name="calibration probabilities")
    y = frozen_f64(outcomes, shape=p.shape, name="calibration events")
    if np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ContractError("calibration needs [0,1] probabilities and binary events")
    bins: list[dict[str, float | int | bool | None]] = []
    index = np.minimum(np.floor(p * 10).astype(np.int64), 9)
    for i in range(10):
        selected = index == i
        count = int(selected.sum())
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": count,
                "sparse": count < 10,
                "mean_probability": float(p[selected].mean()) if count else None,
                "frequency": float(y[selected].mean()) if count else None,
                "brier": float(np.mean((p[selected] - y[selected]) ** 2)) if count else None,
            }
        )
    return tuple(bins)
