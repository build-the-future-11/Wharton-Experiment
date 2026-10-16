"""Independent finite-grid and exact-hold engineering checks."""

import math
from typing import cast

import numpy as np
import pytest
from frem_s10.errors import ContractError
from frem_s10.portfolio.actions import ActionSet, Valuation, enumerate_actions, grid_weights


def test_action_counts_and_exact_grid_constraints() -> None:
    """TEST-045: capped integer combinatorics independently implies 609/45."""
    # Complement forbidden total >=17 via y_i=8-x_i: sum(y)<=7.
    assert len(grid_weights()) == 9**3 - math.comb(10, 3) == 609
    # Sum(x)=16 implies sum(y)=8, where caps are automatic.
    assert len(grid_weights(total_risky=0.8)) == math.comb(10, 2) == 45
    weights = grid_weights()
    assert np.all(weights >= 0) and np.all(weights <= 0.4)
    assert np.all(weights.sum(axis=1) <= 0.8 + 1e-12)
    assert len({tuple(row) for row in weights}) == len(weights)
    np.testing.assert_array_equal(weights[0], np.zeros(3))


def test_feasible_hold_deduplicates_only_exact_grid_match() -> None:
    """TEST-044: exact keep-holdings metadata must survive deduplication."""
    matching = enumerate_actions(Valuation(np.array([20.0, 30.0, 10.0]), 40.0))
    assert len(matching.actions) == 609
    assert matching.hold_index is not None
    assert matching.actions[matching.hold_index].is_hold
    distinct = enumerate_actions(Valuation(np.array([23.0, 31.0, 9.0]), 37.0))
    assert len(distinct.actions) == 610
    assert distinct.hold_index == 609
    np.testing.assert_array_equal(distinct.actions[-1].weights, [0.23, 0.31, 0.09])
    breached = enumerate_actions(Valuation(np.array([41.0, 0.0, 0.0]), 59.0))
    assert breached.hold_index is None and len(breached.actions) == 609


def test_hold_switch_is_strict_boolean() -> None:
    """TEST-002: string/number switches must not alter a named ablation."""
    with pytest.raises(ContractError):
        enumerate_actions(Valuation(np.zeros(3), 100), include_hold=cast(bool, "false"))


def test_action_set_rejects_invalid_indices_and_cash() -> None:
    """TEST-002: invalid registry metadata must fail at construction."""
    valid = enumerate_actions(Valuation(np.zeros(3), 100))
    with pytest.raises(ContractError):
        ActionSet(valid.actions, -1, 0)
    with pytest.raises(ContractError):
        ActionSet(valid.actions, valid.hold_index, 1)


def test_public_grid_and_targets_are_immutable() -> None:
    """TEST-005: candidate arrays cannot mutate after identity creation."""
    action_set = enumerate_actions(Valuation(np.zeros(3), 100))
    for values in (grid_weights(), action_set.targets, Valuation(np.zeros(3), 100).weights):
        with pytest.raises(ValueError):
            values.setflags(write=True)
