import numpy as np

from frem_s11.policy import action_grid, realize


def test_action_grid_is_complete_and_constrained() -> None:
    grid = action_grid()
    assert len(np.unique(grid, axis=0)) == len(grid)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid[:, 1:] <= 0.5)
    assert np.all(np.isin(grid * 4, np.arange(5)))


def test_realization_carries_post_return_drift_into_next_decision() -> None:
    previous = np.asarray([1.0, 0, 0, 0, 0, 0, 0])
    action = np.asarray([0.5, 0.5, 0, 0, 0, 0, 0])
    outcome = np.asarray([1.0, 0, 0, 0, 0, 0])
    factor, drifted = realize(action, outcome, previous, cost_bps=10)
    assert factor == 1.49925
    np.testing.assert_allclose(drifted, np.asarray([1 / 3, 2 / 3, 0, 0, 0, 0, 0]))
    assert not drifted.flags.writeable
