"""Exact small-grid policy used only as a secondary crossed diagnostic."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from frem_s11.errors import NumericalError
from frem_s11.scores import Law


@lru_cache(maxsize=1)
def action_grid() -> NDArray[np.float64]:
    """Cash plus six assets on a 25 percent grid with a 50 percent asset cap."""
    rows: list[tuple[float, ...]] = []
    for cash in range(5):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        for e in range(3):
                            for f in range(3):
                                units = (cash, a, b, c, d, e, f)
                                if sum(units) == 4:
                                    rows.append(tuple(value / 4 for value in units))
    result = np.asarray(rows, dtype=np.float64)
    result.setflags(write=False)
    return result


def choose_action(law: Law, current: NDArray[np.float64], *, cost_bps: float = 10.0) -> NDArray[np.float64]:
    """Maximize a fixed 95/5 nominal-worst expected log utility after turnover cost."""
    state = np.asarray(current, dtype=np.float64)
    if state.shape != (7,) or np.any(state < 0) or not np.isclose(state.sum(), 1.0, atol=1e-12):
        raise NumericalError("current allocation is invalid")
    grid = action_grid()
    gross = grid[:, 0, None] + grid[:, 1:] @ (1.0 + law.atoms).T
    turnover = np.abs(grid[:, 1:] - state[None, 1:]).sum(axis=1)
    investable = 1.0 - turnover[:, None] * cost_bps / 10_000
    net = investable * gross
    if np.any(net <= 0):
        raise NumericalError("candidate law permits nonpositive wealth")
    utility = np.log(net)
    nominal = utility @ law.probabilities
    robust = 0.95 * nominal + 0.05 * utility.min(axis=1)
    best = np.flatnonzero(np.isclose(robust, robust.max(), atol=1e-14, rtol=0))
    if len(best) > 1:
        candidate_turnover = turnover[best]
        best = best[
            np.flatnonzero(np.isclose(candidate_turnover, candidate_turnover.min(), atol=1e-14, rtol=0))
        ]
    result = np.array(grid[int(best[0])], copy=True)
    result.setflags(write=False)
    return result


def realize(
    action: NDArray[np.float64],
    outcome: NDArray[np.float64],
    previous: NDArray[np.float64],
    *,
    cost_bps: float = 10.0,
) -> tuple[float, NDArray[np.float64]]:
    """Return net wealth growth and the post-return drifted allocation."""
    target = np.asarray(action, dtype=np.float64)
    realized = np.asarray(outcome, dtype=np.float64)
    prior = np.asarray(previous, dtype=np.float64)
    if target.shape != (7,) or prior.shape != (7,) or realized.shape != (6,):
        raise NumericalError("policy realization shape is invalid")
    if np.any(target < 0) or np.any(prior < 0):
        raise NumericalError("policy realization allocation is invalid")
    if not np.isclose(target.sum(), 1.0, atol=1e-12) or not np.isclose(prior.sum(), 1.0, atol=1e-12):
        raise NumericalError("policy realization allocation is invalid")
    if not np.isfinite(realized).all() or cost_bps < 0:
        raise NumericalError("policy realization input is invalid")
    turnover = float(np.abs(target[1:] - prior[1:]).sum())
    investable = 1.0 - turnover * cost_bps / 10_000
    holdings = investable * np.concatenate(([target[0]], target[1:] * (1.0 + realized)))
    factor = float(holdings.sum())
    if not np.isfinite(factor) or factor <= 0 or investable <= 0 or np.any(holdings < 0):
        raise NumericalError("realized policy wealth is invalid")
    drifted = np.asarray(holdings / factor, dtype=np.float64)
    drifted.setflags(write=False)
    return factor, drifted
