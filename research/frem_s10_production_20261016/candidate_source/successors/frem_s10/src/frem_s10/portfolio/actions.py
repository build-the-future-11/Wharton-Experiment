"""Integer-grid actions and an exact feasible keep-holdings alternative."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError
from frem_s10.math.fees import validate_valuation
from frem_s10.metadata import strict_int

FloatArray = NDArray[np.float64]
WEIGHT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class Valuation:
    """Visible nonnegative risky dollars and cash at the admitted open quote."""

    risky_dollars: FloatArray
    cash: float

    def __post_init__(self) -> None:
        """Validate USD state and own its immutable numerical storage."""
        values, _ = validate_valuation(self.risky_dollars, self.cash)
        object.__setattr__(self, "risky_dollars", values)

    @property
    def nav(self) -> float:
        """Total pre-trade USD wealth."""
        return self.cash + float(np.sum(self.risky_dollars))

    @property
    def weights(self) -> FloatArray:
        """Current risky shares of total wealth, not future target weights."""
        return frozen_f64(self.risky_dollars / self.nav, shape=(3,), name="current_weights")


@dataclass(frozen=True, slots=True)
class Action:
    """A dimensionless target; is_hold preserves actual dollar holdings."""

    weights: FloatArray
    is_hold: bool = False

    def __post_init__(self) -> None:
        """Reject invalid targets and own an immutable weight vector."""
        values = frozen_f64(self.weights, shape=(3,), name="target_weights")
        if np.any(values < 0) or float(np.sum(values)) > 1 + WEIGHT_TOLERANCE:
            raise ContractError("an action must be long-only and unlevered")
        object.__setattr__(self, "weights", values)
        if not isinstance(self.is_hold, bool):
            raise ContractError("is_hold must be boolean")


@dataclass(frozen=True, slots=True)
class ActionSet:
    """Ordered candidate actions and an optional exact-hold index."""

    actions: tuple[Action, ...]
    hold_index: int | None
    cash_index: int

    def __post_init__(self) -> None:
        """Validate registry indices and their cash/hold semantics."""
        if not isinstance(self.actions, tuple) or not self.actions:
            raise ContractError("nonempty immutable action tuple required")
        if any(not isinstance(action, Action) for action in self.actions):
            raise ContractError("action registry contains a non-Action value")
        cash_index = strict_int(self.cash_index, name="cash_index", minimum=0)
        if cash_index >= len(self.actions) or np.any(self.actions[cash_index].weights != 0):
            raise ContractError("cash index must identify the zero-risk action")
        expected_hold = tuple(i for i, action in enumerate(self.actions) if action.is_hold)
        if self.hold_index is not None:
            strict_int(self.hold_index, name="hold_index", minimum=0)
        if expected_hold != (() if self.hold_index is None else (self.hold_index,)):
            raise ContractError("hold index must identify the sole exact-hold action")

    @property
    def targets(self) -> FloatArray:
        """Return the ordered action-by-three target matrix."""
        return frozen_f64(
            np.stack([action.weights for action in self.actions]),
            shape=(len(self.actions), 3),
            name="action_targets",
        )


def grid_weights(*, total_risky: float | None = None) -> FloatArray:
    """Enumerate exactly 609 policy or 45 fully allocated .8-risk targets."""
    if total_risky is not None and total_risky != 0.8:
        raise ContractError("only the fixed .8-risk constrained grid is supported")
    ticks = [row for row in product(range(9), repeat=3) if sum(row) <= 16]
    if total_risky is not None:
        ticks = [row for row in ticks if sum(row) == 16]
    return frozen_f64(np.asarray(ticks, dtype=np.float64) / 20.0, shape=(len(ticks), 3), name="grid_weights")


def feasible_policy_weights(weights: FloatArray) -> bool:
    """Check decision-time .4 asset/.8 total caps with CL-14 tolerance."""
    values = frozen_f64(weights, shape=(3,), name="weights")
    return bool(
        np.all(values >= 0)
        and np.all(values <= 0.4 + WEIGHT_TOLERANCE)
        and float(np.sum(values)) <= 0.8 + WEIGHT_TOLERANCE
    )


def enumerate_actions(valuation: Valuation, *, include_hold: bool = True) -> ActionSet:
    """Return the integer grid plus a distinct exact hold only when feasible.

    A hold that coincides bitwise with a grid point retains its special tie
    preference without adding a duplicate action.
    """
    if not isinstance(include_hold, bool):
        raise ContractError("include_hold must be boolean")
    targets = grid_weights()
    actions = [Action(row) for row in targets]
    hold_index: int | None = None
    current = valuation.weights
    if include_hold and feasible_policy_weights(current):
        matches = np.flatnonzero(np.all(targets == current, axis=1))
        if len(matches):
            hold_index = int(matches[0])
            actions[hold_index] = Action(current, is_hold=True)
        else:
            hold_index = len(actions)
            actions.append(Action(current, is_hold=True))
    return ActionSet(tuple(actions), hold_index, 0)
