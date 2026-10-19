"""Finite joint laws, proper scores, and paired block uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from frem_s12.errors import NumericalError


@dataclass(frozen=True, slots=True)
class Law:
    """Finite multivariate probability law."""

    atoms: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own arrays and validate the probability simplex."""
        atoms = np.array(self.atoms, dtype=np.float64, copy=True)
        probabilities = np.array(self.probabilities, dtype=np.float64, copy=True)
        if atoms.ndim != 2 or not len(atoms) or probabilities.shape != (len(atoms),):
            raise NumericalError("finite law shape is invalid")
        if not np.isfinite(atoms).all() or not np.isfinite(probabilities).all():
            raise NumericalError("finite law contains nonfinite values")
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
            raise NumericalError("finite law probabilities are invalid")
        probabilities /= probabilities.sum()
        atoms.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "probabilities", probabilities)


def combine_laws(laws: tuple[Law, ...], weights: ArrayLike) -> Law:
    """Create an exact finite mixture without a shared-support assumption."""
    alpha = np.asarray(weights, dtype=np.float64)
    if not laws or alpha.shape != (len(laws),) or np.any(alpha < 0):
        raise NumericalError("mixture weights are invalid")
    if not np.isclose(alpha.sum(), 1.0, atol=1e-12):
        raise NumericalError("mixture weights do not sum to one")
    dimensions = {law.atoms.shape[1] for law in laws}
    if len(dimensions) != 1:
        raise NumericalError("mixture law dimensions differ")
    return Law(
        np.concatenate([law.atoms for law in laws]),
        np.concatenate([weight * law.probabilities for weight, law in zip(alpha, laws, strict=True)]),
    )


def energy_components(
    laws: tuple[Law, ...], outcome: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact linear and cross-law terms for mixture energy score."""
    y = np.asarray(outcome, dtype=np.float64)
    if not laws or y.shape != (laws[0].atoms.shape[1],) or not np.isfinite(y).all():
        raise NumericalError("energy outcome is invalid")
    if any(law.atoms.shape[1] != len(y) for law in laws):
        raise NumericalError("energy law dimensions differ")
    linear = np.asarray(
        [law.probabilities @ np.linalg.norm(law.atoms - y, axis=1) for law in laws],
        dtype=np.float64,
    )
    cross = np.empty((len(laws), len(laws)), dtype=np.float64)
    for i, left in enumerate(laws):
        for j, right in enumerate(laws):
            distances = np.linalg.norm(left.atoms[:, None, :] - right.atoms[None, :, :], axis=2)
            cross[i, j] = left.probabilities @ distances @ right.probabilities
    if not np.isfinite(linear).all() or not np.isfinite(cross).all():
        raise NumericalError("energy components are nonfinite")
    return linear, cross


def mixture_energy_score(laws: tuple[Law, ...], weights: ArrayLike, outcome: ArrayLike) -> float:
    """Score a finite mixture using exact cross-expert terms."""
    alpha = np.asarray(weights, dtype=np.float64)
    if alpha.shape != (len(laws),) or np.any(alpha < 0) or not np.isclose(alpha.sum(), 1.0):
        raise NumericalError("energy mixture weights are invalid")
    linear, cross = energy_components(laws, outcome)
    value = float(alpha @ linear - 0.5 * alpha @ cross @ alpha)
    if not np.isfinite(value):
        raise NumericalError("energy score is nonfinite")
    return value


def energy_score(law: Law, outcome: ArrayLike) -> float:
    """Return the multivariate energy score of one finite law."""
    return mixture_energy_score((law,), np.ones(1, dtype=np.float64), outcome)


def energy_weight_gradient(
    laws: tuple[Law, ...], weights: ArrayLike, outcome: ArrayLike
) -> NDArray[np.float64]:
    """Return the exact derivative of mixture energy score by law weight."""
    alpha = np.asarray(weights, dtype=np.float64)
    linear, cross = energy_components(laws, outcome)
    gradient = linear - cross @ alpha
    if not np.isfinite(gradient).all():
        raise NumericalError("energy gradient is nonfinite")
    return gradient


def variogram_score(law: Law, outcome: ArrayLike, *, order: float = 0.5) -> float:
    """Return equal-pair-weight variogram score for a finite law."""
    y = np.asarray(outcome, dtype=np.float64)
    if y.shape != (law.atoms.shape[1],) or not 0 < order <= 2:
        raise NumericalError("variogram arguments are invalid")
    value = 0.0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            observed = abs(float(y[i] - y[j])) ** order
            forecast = float(law.probabilities @ np.abs(law.atoms[:, i] - law.atoms[:, j]) ** order)
            value += (observed - forecast) ** 2
    if not np.isfinite(value):
        raise NumericalError("variogram score is nonfinite")
    return value


def moving_block_interval(
    values: ArrayLike, *, block: int = 12, draws: int = 10_000, seed: int = 1912
) -> tuple[float, float]:
    """Return a circular moving-block percentile interval for a paired mean."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 2 * block or block < 2 or draws < 1000:
        raise NumericalError("block interval input is invalid")
    if not np.isfinite(data).all():
        raise NumericalError("block interval input is nonfinite")
    rng = np.random.default_rng(seed)
    count = int(np.ceil(len(data) / block))
    starts = rng.integers(0, len(data), size=(draws, count), endpoint=False)
    indices = (starts[:, :, None] + np.arange(block)) % len(data)
    samples = data[indices.reshape(draws, -1)[:, : len(data)]].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)
