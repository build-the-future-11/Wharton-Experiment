"""Causal S11 forecast and crossed-policy development evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frem_s11.data import ReturnPanel
from frem_s11.errors import DataContractError
from frem_s11.model import EXPERTS, EnsembleConfig, issue_forecast
from frem_s11.policy import choose_action, realize
from frem_s11.scores import Law, energy_score, moving_block_interval

VARIANTS = ("stacked", "equal", *EXPERTS)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete retained score, stack, policy, and uncertainty evidence."""

    months: NDArray[np.int64]
    scores: NDArray[np.float64]
    stack_weights: NDArray[np.float64]
    allocations: NDArray[np.float64]
    wealth: NDArray[np.float64]
    benchmark_wealth: NDArray[np.float64]
    intervals: dict[str, tuple[float, float]]
    relative_improvements: dict[str, float]
    primary_supported: bool
    secondary_supported: bool

    def __post_init__(self) -> None:
        """Own retained arrays so a completed result cannot be mutated in place."""
        for name in ("months", "scores", "stack_weights", "allocations", "wealth", "benchmark_wealth"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _laws(bundle: object) -> tuple[Law, ...]:
    from frem_s11.model import ForecastBundle

    if not isinstance(bundle, ForecastBundle):
        raise TypeError("forecast bundle required")
    return (bundle.stacked, bundle.equal, *bundle.specialists)


def run_backtest(
    panel: ReturnPanel,
    *,
    start_month: int,
    end_month: int,
    config: EnsembleConfig | None = None,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Run one declared fold with no outcome entering its own forecast."""
    cfg = config or EnsembleConfig()
    start = int(np.searchsorted(panel.months, start_month, side="left"))
    stop = int(np.searchsorted(panel.months, end_month, side="right"))
    exact_start = start < len(panel.months) and panel.months[start] == start_month
    exact_end = stop > 0 and panel.months[stop - 1] == end_month
    if (
        start < cfg.minimum_history
        or stop - start < 24
        or stop > len(panel.months)
        or not exact_start
        or not exact_end
    ):
        raise DataContractError("declared backtest fold is unavailable or too short")
    score_rows: list[list[float]] = []
    weight_rows: list[NDArray[np.float64]] = []
    losses: list[list[float]] = []
    allocations = [np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64) for _ in VARIANTS]
    wealth = np.ones(len(VARIANTS), dtype=np.float64)
    wealth_rows: list[NDArray[np.float64]] = []
    allocation_rows: list[NDArray[np.float64]] = []
    benchmark = np.ones(2, dtype=np.float64)
    benchmark_rows: list[NDArray[np.float64]] = []
    for origin in range(start, stop):
        history = np.asarray(losses, dtype=np.float64) if losses else None
        bundle = issue_forecast(panel.returns[:origin], loss_history=history, config=cfg)
        laws = _laws(bundle)
        outcome = panel.returns[origin]
        scores = [energy_score(law, outcome) for law in laws]
        specialist_scores = scores[2:]
        losses.append(specialist_scores)
        score_rows.append(scores)
        weight_rows.append(bundle.stack_weights)
        for index, law in enumerate(laws):
            action = choose_action(law, allocations[index], cost_bps=cost_bps)
            factor, allocations[index] = realize(action, outcome, allocations[index], cost_bps=cost_bps)
            wealth[index] *= factor
        allocation_rows.append(np.asarray(allocations, dtype=np.float64))
        wealth_rows.append(wealth.copy())
        benchmark[1] *= 1.0 + float(outcome.mean())
        benchmark_rows.append(benchmark.copy())
    score_array = np.asarray(score_rows, dtype=np.float64)
    intervals = {
        "equal_minus_stacked": moving_block_interval(score_array[:, 1] - score_array[:, 0]),
        "recent_minus_stacked": moving_block_interval(score_array[:, 3] - score_array[:, 0]),
    }
    stacked_mean = float(score_array[:, 0].mean())
    equal_mean = float(score_array[:, 1].mean())
    recent_mean = float(score_array[:, 3].mean())
    if equal_mean <= 0 or recent_mean <= 0:
        raise DataContractError("control energy score must be strictly positive")
    relative = {
        "equal": (equal_mean - stacked_mean) / equal_mean,
        "recent": (recent_mean - stacked_mean) / recent_mean,
    }
    supported = (
        relative["equal"] >= 0.05
        and relative["recent"] >= 0.05
        and intervals["equal_minus_stacked"][0] > 0
        and intervals["recent_minus_stacked"][0] > 0
    )
    terminal = wealth_rows[-1]
    terminal_benchmarks = benchmark_rows[-1]
    secondary_supported = bool(
        terminal[0] > terminal[1]
        and terminal[0] > terminal[3]
        and terminal[0] > terminal_benchmarks[0]
        and terminal[0] > terminal_benchmarks[1]
    )
    return BacktestResult(
        panel.months[start:stop].copy(),
        score_array,
        np.asarray(weight_rows, dtype=np.float64),
        np.asarray(allocation_rows, dtype=np.float64),
        np.asarray(wealth_rows, dtype=np.float64),
        np.asarray(benchmark_rows, dtype=np.float64),
        intervals,
        relative,
        supported,
        secondary_supported,
    )
