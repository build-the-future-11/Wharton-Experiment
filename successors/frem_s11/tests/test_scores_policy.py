import numpy as np
import pytest

from frem_s11.policy import action_grid, realize
from frem_s11.scores import Law, energy_score, moving_block_interval


def test_energy_score_matches_two_atom_hand_calculation() -> None:
    law = Law(np.asarray([[0.0], [2.0]]), np.asarray([0.25, 0.75]))
    # E|X-y| = 1.5 and 0.5 E|X-X'| = 0.375 for y=0.
    assert energy_score(law, np.asarray([0.0])) == pytest.approx(1.125)


def test_policy_grid_and_realization_include_cost_and_weight_drift() -> None:
    grid = action_grid()
    assert np.allclose(grid.sum(axis=1), 1)
    assert np.all(grid[:, 1:] <= 0.5)
    cash = np.asarray([1.0, 0, 0, 0, 0, 0, 0])
    target = np.asarray([0.5, 0.5, 0, 0, 0, 0, 0])
    outcome = np.asarray([0.10, 0, 0, 0, 0, 0])
    factor, drifted = realize(target, outcome, cash, cost_bps=10)
    assert factor == pytest.approx(1.049475)
    assert drifted[1] == pytest.approx(0.55 / 1.05)
    assert drifted.sum() == pytest.approx(1.0)


def test_moving_block_interval_is_deterministic_and_paired() -> None:
    values = np.linspace(-0.2, 0.3, 60)
    first = moving_block_interval(values, block=12, draws=1000, seed=1911)
    second = moving_block_interval(values, block=12, draws=1000, seed=1911)
    assert first == second
    assert first[0] < values.mean() < first[1]
