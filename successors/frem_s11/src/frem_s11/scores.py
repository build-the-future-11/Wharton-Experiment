"""Finite-law scoring and deterministic paired uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from frem_s11.errors import NumericalError


@dataclass(frozen=True, slots=True)
class Law:
    """Finite joint return law."""

    atoms: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own arrays and enforce a finite probability simplex."""
        atoms = np.array(self.atoms, dtype=np.float64, copy=True)
        probabilities = np.array(self.probabilities, dtype=np.float64, copy=True)
        if atoms.ndim != 2 or atoms.shape[0] == 0 or probabilities.shape != (len(atoms),):
            raise NumericalError("finite law shape is invalid")
        if not np.isfinite(atoms).all() or not np.isfinite(probabilities).all():
            raise NumericalError("finite law contains nonfinite values")
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0):
            raise NumericalError("finite law probabilities are invalid")
        probabilities /= probabilities.sum()
        atoms.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "probabilities", probabilities)


def energy_score(law: Law, outcome: ArrayLike) -> float:
    """Return the multivariate energy score with every pair term retained."""
    y = np.asarray(outcome, dtype=np.float64)
    if y.shape != (law.atoms.shape[1],) or not np.isfinite(y).all():
        raise NumericalError("score outcome is invalid")
    first = np.linalg.norm(law.atoms - y, axis=1)
    pair = np.linalg.norm(law.atoms[:, None, :] - law.atoms[None, :, :], axis=2)
    value = float(law.probabilities @ first - 0.5 * law.probabilities @ pair @ law.probabilities)
    if not np.isfinite(value):
        raise NumericalError("energy score is nonfinite")
    return value


def combine_laws(laws: tuple[Law, ...], weights: ArrayLike) -> Law:
    """Form an exact finite mixture without assuming shared support."""
    alpha = np.asarray(weights, dtype=np.float64)
    if alpha.shape != (len(laws),) or np.any(alpha < 0) or not np.isclose(alpha.sum(), 1.0, atol=1e-12):
        raise NumericalError("mixture weights are invalid")
    dimensions = {law.atoms.shape[1] for law in laws}
    if len(dimensions) != 1:
        raise NumericalError("mixture laws have inconsistent dimensions")
    return Law(
        np.concatenate([law.atoms for law in laws]),
        np.concatenate([weight * law.probabilities for weight, law in zip(alpha, laws, strict=True)]),
    )


def moving_block_interval(
    differences: ArrayLike, *, block: int = 12, draws: int = 5000, seed: int = 1911
) -> tuple[float, float]:
    """Paired circular moving-block percentile interval for the mean difference."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 * block or not np.isfinite(values).all():
        raise NumericalError("paired interval input is too short or invalid")
    if draws < 1000 or block < 2:
        raise NumericalError("paired interval settings are invalid")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block))
    starts = rng.integers(0, len(values), size=(draws, blocks), endpoint=False)
    offsets = np.arange(block, dtype=np.int64)
    indices = (starts[:, :, None] + offsets) % len(values)
    sampled = values[indices.reshape(draws, -1)[:, : len(values)]].mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(low), float(high)
