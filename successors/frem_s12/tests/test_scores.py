from __future__ import annotations

import numpy as np
import pytest

from frem_s12.errors import NumericalError
from frem_s12.scores import (
    Law,
    combine_laws,
    energy_score,
    energy_weight_gradient,
    mixture_energy_score,
    moving_block_interval,
    variogram_score,
)


def law(offset: float) -> Law:
    return Law(np.asarray([[offset, 0.0], [offset + 1.0, 1.0]]), np.asarray([0.25, 0.75]))


def test_law_owns_and_freezes_arrays() -> None:
    atoms = np.asarray([[0.0], [1.0]])
    value = Law(atoms, np.asarray([0.5, 0.5]))
    atoms[0, 0] = 99
    assert value.atoms[0, 0] == 0
    assert not value.atoms.flags.writeable


@pytest.mark.parametrize("probabilities", [[-0.1, 1.1], [0.1, 0.8], [np.nan, np.nan]])
def test_law_rejects_invalid_probability(probabilities: list[float]) -> None:
    with pytest.raises(NumericalError):
        Law(np.asarray([[0.0], [1.0]]), np.asarray(probabilities))


def test_combination_matches_direct_score() -> None:
    laws = (law(0.0), law(2.0))
    weights = np.asarray([0.3, 0.7])
    outcome = np.asarray([0.5, -0.2])
    assert mixture_energy_score(laws, weights, outcome) == pytest.approx(
        energy_score(combine_laws(laws, weights), outcome), abs=1e-14
    )


def test_energy_gradient_matches_finite_difference_on_simplex() -> None:
    laws = (law(0.0), law(2.0))
    weights = np.asarray([0.4, 0.6])
    direction = np.asarray([1.0, -1.0])
    epsilon = 1e-6
    observed = (
        mixture_energy_score(laws, weights + epsilon * direction, [0.5, 0.2])
        - mixture_energy_score(laws, weights - epsilon * direction, [0.5, 0.2])
    ) / (2 * epsilon)
    expected = energy_weight_gradient(laws, weights, [0.5, 0.2]) @ direction
    assert observed == pytest.approx(expected, rel=1e-7, abs=1e-9)


def test_variogram_is_zero_for_exact_point_mass() -> None:
    outcome = np.asarray([1.0, 4.0, -2.0])
    assert variogram_score(Law(outcome[None, :], np.ones(1)), outcome) == 0.0


def test_block_interval_is_deterministic() -> None:
    values = np.linspace(-1, 1, 48)
    assert moving_block_interval(values, draws=1000) == moving_block_interval(values, draws=1000)
