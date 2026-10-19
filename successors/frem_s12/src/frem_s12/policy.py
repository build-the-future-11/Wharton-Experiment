"""Secondary finite-grid self-financing policy diagnostics."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from frem_s12.errors import NumericalError
from frem_s12.scores import Law


def action_grid() -> NDArray[np.float64]:
    """Return cash plus six risky weights on the frozen 25%/50% grid."""
    actions = []
    for units in itertools.product(range(3), repeat=6):
        risky = np.asarray(units, dtype=np.float64) / 4
        if risky.sum() <= 1 + 1e-12:
            actions.append(np.concatenate([[1 - risky.sum()], risky]))
    return np.asarray(actions, dtype=np.float64)


_ACTIONS = action_grid()


def _choose_action_reference(
    law: Law, current: NDArray[np.float64], *, cost_bps: float = 10.0
) -> NDArray[np.float64]:
    """Retain the scalar definition used to validate the vectorized selector."""
    holdings = np.asarray(current, dtype=np.float64)
    risky_growth = 1 + law.atoms
    utilities = np.empty(len(_ACTIONS), dtype=np.float64)
    fee_rate = cost_bps / 10_000
    for index, action in enumerate(_ACTIONS):
        turnover = float(np.abs(action[1:] - holdings[1:]).sum())
        investable = 1 - fee_rate * turnover
        growth = investable * (action[0] + risky_growth @ action[1:])
        nominal = float(law.probabilities @ np.log(growth))
        worst = float(np.min(np.log(growth)))
        utilities[index] = 0.95 * nominal + 0.05 * worst
    return np.asarray(_ACTIONS[int(np.argmax(utilities))], dtype=np.float64).copy()


def choose_action(law: Law, current: NDArray[np.float64], *, cost_bps: float = 10.0) -> NDArray[np.float64]:
    """Choose maximum frozen 95/5 nominal-worst expected log utility."""
    holdings = np.asarray(current, dtype=np.float64)
    if holdings.shape != (7,) or np.any(holdings < 0) or not np.isclose(holdings.sum(), 1.0):
        raise NumericalError("current allocation is invalid")
    risky_growth = 1 + law.atoms
    if np.any(risky_growth <= 0):
        raise NumericalError("law implies total loss beyond policy domain")
    fee_rate = cost_bps / 10_000
    turnover = np.abs(_ACTIONS[:, 1:] - holdings[None, 1:]).sum(axis=1)
    investable = 1 - fee_rate * turnover
    growth = investable[:, None] * (_ACTIONS[:, :1] + _ACTIONS[:, 1:] @ risky_growth.T)
    log_growth = np.log(growth)
    utilities = 0.95 * (log_growth @ law.probabilities) + 0.05 * log_growth.min(axis=1)
    return np.asarray(_ACTIONS[int(np.argmax(utilities))], dtype=np.float64).copy()


def realize(
    action: NDArray[np.float64],
    outcome: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    cost_bps: float = 10.0,
) -> tuple[float, NDArray[np.float64]]:
    """Apply turnover cost before returns and return drifted next holdings."""
    target = np.asarray(action, dtype=np.float64)
    observed = np.asarray(outcome, dtype=np.float64)
    previous = np.asarray(current, dtype=np.float64)
    if target.shape != (7,) or previous.shape != (7,) or observed.shape != (6,):
        raise NumericalError("policy realization shape is invalid")
    turnover = float(np.abs(target[1:] - previous[1:]).sum())
    investable = 1 - cost_bps / 10_000 * turnover
    post = investable * np.concatenate([[target[0]], target[1:] * (1 + observed)])
    factor = float(post.sum())
    if factor <= 0 or not np.isfinite(factor):
        raise NumericalError("policy wealth factor is invalid")
    holdings = post / factor
    return factor, holdings
