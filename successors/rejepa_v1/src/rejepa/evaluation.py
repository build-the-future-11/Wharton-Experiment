"""Leakage-resistant walk-forward splits, metrics and paired uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]


class PurgedWalkForwardSplitter:
    def __init__(self, train_dates: int, validation_dates: int, test_dates: int, embargo_dates: int) -> None:
        if min(train_dates, validation_dates, test_dates) < 1 or embargo_dates < 0:
            raise ValueError("Invalid walk-forward sizes")
        self.train_size = train_dates
        self.validation_size = validation_dates
        self.test_size = test_dates
        self.embargo = embargo_dates

    def split(self, dates: list[object] | np.ndarray) -> list[WalkForwardFold]:
        unique = list(pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).unique().sort_values())
        folds: list[WalkForwardFold] = []
        cursor = self.train_size
        while cursor + self.embargo + self.validation_size + self.embargo + self.test_size <= len(unique):
            train = unique[:cursor]
            validation_start = cursor + self.embargo
            validation = unique[validation_start : validation_start + self.validation_size]
            test_start = validation_start + self.validation_size + self.embargo
            test = unique[test_start : test_start + self.test_size]
            folds.append(WalkForwardFold(tuple(train), tuple(validation), tuple(test)))
            cursor += self.test_size
        if not folds:
            raise ValueError("Not enough unique dates for one fold")
        return folds


def portfolio_metrics(
    gross_returns: np.ndarray,
    *,
    turnover: np.ndarray | None = None,
    cost_rate: float = 0.0,
    periods_per_year: int = 252,
    cvar_alpha: float = 0.95,
) -> dict[str, float]:
    returns = np.asarray(gross_returns, dtype=float)
    if returns.ndim != 1 or returns.size < 2 or not np.isfinite(returns).all():
        raise ValueError("Finite one-dimensional returns are required")
    if turnover is None:
        traded = np.zeros_like(returns)
    else:
        traded = np.asarray(turnover, dtype=float)
        if traded.shape != returns.shape or (traded < 0).any() or not np.isfinite(traded).all():
            raise ValueError("turnover must be finite, nonnegative and aligned")
    net = returns - cost_rate * traded
    mean = float(net.mean())
    std = float(net.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(net, 0) ** 2)))
    wealth = np.cumprod(1 + net)
    peaks = np.maximum.accumulate(wealth)
    drawdown = wealth / peaks - 1
    loss = -net
    var = float(np.quantile(loss, cvar_alpha))
    tail = loss[loss >= var]
    return {
        "net_total_return": float(wealth[-1] - 1),
        "annualized_mean": mean * periods_per_year,
        "sharpe": mean / std * np.sqrt(periods_per_year) if std > 0 else 0.0,
        "sortino": mean / downside * np.sqrt(periods_per_year) if downside > 0 else 0.0,
        "maximum_drawdown": float(drawdown.min()),
        "cvar_loss": float(tail.mean()) if tail.size else var,
        "mean_turnover": float(traded.mean()),
        "transaction_cost": float((cost_rate * traded).sum()),
    }


def distribution_calibration(
    observations: np.ndarray, lower: np.ndarray, upper: np.ndarray, expected_coverage: float
) -> dict[str, float]:
    y, lo, hi = (np.asarray(value, dtype=float) for value in (observations, lower, upper))
    if y.shape != lo.shape or y.shape != hi.shape or y.ndim != 1:
        raise ValueError("Calibration arrays must be aligned vectors")
    if not 0 < expected_coverage < 1 or not np.isfinite(np.concatenate([y, lo, hi])).all():
        raise ValueError("Invalid calibration inputs")
    if (lo > hi).any():
        raise ValueError("Lower interval exceeds upper interval")
    covered = (y >= lo) & (y <= hi)
    return {
        "empirical_coverage": float(covered.mean()),
        "coverage_error": float(abs(covered.mean() - expected_coverage)),
        "mean_interval_width": float((hi - lo).mean()),
    }


def paired_moving_block_interval(
    candidate_loss: np.ndarray,
    control_loss: np.ndarray,
    *,
    block_length: int = 20,
    repetitions: int = 5000,
    confidence: float = 0.95,
    seed: int = 11,
) -> tuple[float, float, float]:
    candidate = np.asarray(candidate_loss, dtype=float)
    control = np.asarray(control_loss, dtype=float)
    if candidate.shape != control.shape or candidate.ndim != 1 or candidate.size < block_length:
        raise ValueError("Paired vectors must align and contain at least one block")
    if not np.isfinite(candidate).all() or not np.isfinite(control).all():
        raise ValueError("Loss vectors must be finite")
    difference = control - candidate
    rng = np.random.default_rng(seed)
    starts = np.arange(candidate.size - block_length + 1)
    draws = np.empty(repetitions)
    block_count = int(np.ceil(candidate.size / block_length))
    for index in range(repetitions):
        selected = rng.choice(starts, size=block_count, replace=True)
        sample = np.concatenate([difference[start : start + block_length] for start in selected])[
            : candidate.size
        ]
        draws[index] = sample.mean()
    tail = (1 - confidence) / 2
    return float(difference.mean()), float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))
