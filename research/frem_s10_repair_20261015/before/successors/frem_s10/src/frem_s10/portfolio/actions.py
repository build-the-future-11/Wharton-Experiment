"""Integer-grid actions and an exact feasible keep-holdings alternative."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError
from frem_s10.math.fees import validate_valuation

FloatArray = NDArray[np.float64]
WEIGHT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class Valuation:
    """Visible nonnegative risky dollars and cash at the admitted open quote."""

    risky_dollars: FloatArray
    cash: float

    def __post_init__(self) -> None:
        values, _ = validate_valuation(self.risky_dollars, self.cash)
        object.__setattr__(self, "risky_dollars", values)

    @property
    def nav(self) -> float:
        """Total pre-trade USD wealth."""
        return self.cash + float(np.sum(self.risky_dollars))

    @property
    def weights(self) -> FloatArray:
        """Current risky shares of total wealth, not future target weights."""
        return self.risky_dollars / self.nav


@dataclass(frozen=True, slots=True)
class Action:
    """A dimensionless target; is_hold preserves actual dollar holdings."""

    weights: FloatArray
    is_hold: bool = False

    def __post_init__(self) -> None:
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

    @property
    def targets(self) -> FloatArray:
        """Return the ordered action-by-three target matrix."""
        return np.stack([action.weights for action in self.actions])


def grid_weights(*, total_risky: float | None = None) -> FloatArray:
    """Enumerate exactly 609 policy or 45 fully allocated .8-risk targets."""
    if total_risky is not None and total_risky != 0.8:
        raise ContractError("only the fixed .8-risk constrained grid is supported")
    ticks = [row for row in product(range(9), repeat=3) if sum(row) <= 16]
    if total_risky is not None:
        ticks = [row for row in ticks if sum(row) == 16]
    return np.asarray(ticks, dtype=np.float64) / 20.0


def feasible_policy_weights(weights: FloatArray) -> bool:
    """Check decision-time .4 asset/.8 total caps with CL-14 tolerance."""
    values = frozen_f64(weights, shape=(3,), name="weights")
    return bool(
        np.all(values >= 0) and np.all(values <= 0.4 + WEIGHT_TOLERANCE)
        and float(np.sum(values)) <= 0.8 + WEIGHT_TOLERANCE
    )


def enumerate_actions(valuation: Valuation, *, include_hold: bool = True) -> ActionSet:
    """Return the integer grid plus a distinct exact hold only when feasible.

    A hold that coincides bitwise with a grid point retains its special tie
    preference without adding a duplicate action.
    """
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
