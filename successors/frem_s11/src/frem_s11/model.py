"""Causal finite-scenario specialists and delayed score stacking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frem_s11.errors import NumericalError
from frem_s11.scores import Law, combine_laws

EXPERTS = ("unconditional", "recent", "context", "tail_regime")


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Frozen specialist and stacker settings."""

    minimum_history: int = 120
    library_window: int = 120
    context_neighbors: int = 24
    recency_half_life: float = 36.0
    score_half_life: float = 36.0
    shrink_to_equal: float = 0.25
    maximum_expert_weight: float = 0.60

    def __post_init__(self) -> None:
        """Validate frozen ensemble settings."""
        if not 60 <= self.minimum_history <= self.library_window <= 240:
            raise NumericalError("history/window settings are invalid")
        if not 8 <= self.context_neighbors < self.minimum_history:
            raise NumericalError("context-neighbor setting is invalid")
        if self.recency_half_life <= 0 or self.score_half_life <= 0:
            raise NumericalError("half-lives must be positive")
        if not 0 <= self.shrink_to_equal <= 1 or not 0.25 <= self.maximum_expert_weight <= 1:
            raise NumericalError("stacker shrink/cap settings are invalid")


@dataclass(frozen=True, slots=True)
class ForecastBundle:
    """Issued specialist and stacked laws at one origin."""

    specialists: tuple[Law, ...]
    equal: Law
    stacked: Law
    stack_weights: NDArray[np.float64]
    current_state: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own diagnostic arrays and verify bundle dimensions."""
        weights = np.array(self.stack_weights, dtype=np.float64, copy=True)
        state = np.array(self.current_state, dtype=np.float64, copy=True)
        if len(self.specialists) != len(EXPERTS) or weights.shape != (len(EXPERTS),) or state.shape != (6,):
            raise NumericalError("forecast bundle shape is invalid")
        weights.setflags(write=False)
        state.setflags(write=False)
        object.__setattr__(self, "stack_weights", weights)
        object.__setattr__(self, "current_state", state)


def _states(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute state at t from observations strictly before t."""
    states = np.full((len(returns), 6), np.nan, dtype=np.float64)
    market = returns.mean(axis=1)
    value = returns[:, [2, 5]].mean(axis=1) - returns[:, [0, 3]].mean(axis=1)
    size = returns[:, :3].mean(axis=1) - returns[:, 3:].mean(axis=1)
    for origin in range(12, len(returns)):
        window = slice(origin - 12, origin)
        states[origin] = (
            market[origin - 1],
            market[window].mean(),
            market[window].std(ddof=1),
            returns[window].std(axis=1, ddof=1).mean(),
            value[window].mean(),
            size[window].mean(),
        )
    return states


def _current_state(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the decision state from the latest twelve fully matured returns."""
    if len(returns) < 12:
        raise NumericalError("twelve matured returns are required for a state")
    window = returns[-12:]
    market = window.mean(axis=1)
    value = window[:, [2, 5]].mean(axis=1) - window[:, [0, 3]].mean(axis=1)
    size = window[:, :3].mean(axis=1) - window[:, 3:].mean(axis=1)
    state = np.asarray(
        (
            market[-1],
            market.mean(),
            market.std(ddof=1),
            window.std(axis=1, ddof=1).mean(),
            value.mean(),
            size.mean(),
        ),
        dtype=np.float64,
    )
    state.setflags(write=False)
    return state


def _recency(size: int, half_life: float) -> NDArray[np.float64]:
    age = np.arange(size - 1, -1, -1, dtype=np.float64)
    weights = np.exp2(-age / half_life)
    return np.asarray(weights / float(weights.sum()), dtype=np.float64)


def _context_weights(
    contexts: NDArray[np.float64], current: NDArray[np.float64], neighbors: int
) -> NDArray[np.float64]:
    center = np.median(contexts, axis=0)
    scale = np.median(np.abs(contexts - center), axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    distances = np.linalg.norm((contexts - current) / scale, axis=1)
    order = np.argsort(distances, kind="stable")[:neighbors]
    weights = np.zeros(len(contexts), dtype=np.float64)
    weights[order] = 1.0 / neighbors
    return weights


def _tail_weights(contexts: NDArray[np.float64], current: NDArray[np.float64]) -> NDArray[np.float64]:
    volatility = contexts[:, 2]
    negative = contexts[:, 1] < 0
    current_high_vol = current[2] >= np.median(volatility)
    current_negative = current[1] < 0
    same = (volatility >= np.median(volatility)) == current_high_vol
    same &= negative == current_negative
    if int(same.sum()) < 12:
        same = (
            volatility >= np.quantile(volatility, 0.60)
            if current_high_vol
            else volatility < np.quantile(volatility, 0.60)
        )
    weights = np.full(len(contexts), 0.10 / len(contexts), dtype=np.float64)
    weights[same] += 0.90 / int(same.sum())
    return weights / weights.sum()


def _stack_weights(loss_history: NDArray[np.float64], config: EnsembleConfig) -> NDArray[np.float64]:
    equal = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
    if len(loss_history) < 24:
        return equal
    history = loss_history[-min(len(loss_history), 120) :]
    recency = _recency(len(history), config.score_half_life)
    centered = history - history.mean(axis=1, keepdims=True)
    scale = np.median(np.abs(centered))
    if not np.isfinite(scale) or scale <= 1e-12:
        return equal
    logits = -(recency @ centered) / scale
    logits -= logits.max()
    fitted = np.exp(logits)
    fitted /= fitted.sum()
    fitted = (1 - config.shrink_to_equal) * fitted + config.shrink_to_equal * equal
    if fitted.max() > config.maximum_expert_weight:
        excess = float(fitted.max() - config.maximum_expert_weight)
        index = int(fitted.argmax())
        fitted[index] = config.maximum_expert_weight
        mask = np.arange(len(fitted)) != index
        fitted[mask] += excess * fitted[mask] / fitted[mask].sum()
    return np.asarray(fitted / float(fitted.sum()), dtype=np.float64)


def issue_forecast(
    returns_before_origin: NDArray[np.float64],
    *,
    loss_history: NDArray[np.float64] | None = None,
    config: EnsembleConfig | None = None,
) -> ForecastBundle:
    """Issue laws using only returns and matured score feedback before the origin."""
    cfg = config or EnsembleConfig()
    values = np.asarray(returns_before_origin, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < cfg.minimum_history:
        raise NumericalError("insufficient or malformed return history")
    if not np.isfinite(values).all():
        raise NumericalError("return history is nonfinite")
    states = _states(values)
    current = _current_state(values)
    start = max(12, len(values) - cfg.library_window)
    atoms = values[start:]
    contexts = states[start:]
    valid = np.isfinite(contexts).all(axis=1)
    atoms, contexts = atoms[valid], contexts[valid]
    if len(atoms) < cfg.context_neighbors:
        raise NumericalError("insufficient mature scenario contexts")
    laws = (
        Law(atoms, np.full(len(atoms), 1.0 / len(atoms))),
        Law(atoms, _recency(len(atoms), cfg.recency_half_life)),
        Law(atoms, _context_weights(contexts, current, cfg.context_neighbors)),
        Law(atoms, _tail_weights(contexts, current)),
    )
    equal_weights = np.full(len(laws), 1.0 / len(laws))
    losses = np.empty((0, len(laws)), dtype=np.float64) if loss_history is None else np.asarray(loss_history)
    if losses.ndim != 2 or losses.shape[1] != len(laws) or not np.isfinite(losses).all():
        raise NumericalError("matured score history is invalid")
    weights = _stack_weights(losses, cfg)
    return ForecastBundle(
        laws, combine_laws(laws, equal_weights), combine_laws(laws, weights), weights, current
    )
