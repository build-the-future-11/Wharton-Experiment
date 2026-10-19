from __future__ import annotations

import numpy as np
import pytest

from frem_s12.campaign import cells, registry, search_space, shard_cells
from frem_s12.policy import action_grid, choose_action, realize
from frem_s12.scores import Law


def test_campaign_registry_is_exact() -> None:
    assert len(cells()) == 120
    assert registry()["cell_count"] == 120
    assert len(search_space()) == 24


def test_shards_are_disjoint_and_complete() -> None:
    shards = [set(shard_cells(index, 7)) for index in range(7)]
    assert set.union(*shards) == set(cells())
    assert sum(len(value) for value in shards) == 120


def test_policy_grid_constraints() -> None:
    grid = action_grid()
    assert np.allclose(grid.sum(axis=1), 1)
    assert np.max(grid[:, 1:]) <= 0.5


def test_cash_is_selected_for_catastrophic_law() -> None:
    law = Law(np.full((4, 6), -0.9), np.full(4, 0.25))
    action = choose_action(law, np.asarray([1.0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(action, np.asarray([1.0, 0, 0, 0, 0, 0, 0]))


def test_realize_applies_turnover_cost() -> None:
    prior = np.asarray([1.0, 0, 0, 0, 0, 0, 0])
    action = np.asarray([0.5, 0.5, 0, 0, 0, 0, 0])
    factor, holdings = realize(action, np.zeros(6), prior, cost_bps=10)
    assert factor == pytest.approx(0.9995)
    assert holdings.sum() == pytest.approx(1.0)
