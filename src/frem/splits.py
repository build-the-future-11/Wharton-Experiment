"""Chronological, maturity-purged splits with explicit date-bar embargoes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data import WindowData


@dataclass
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    protected: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.tolist(),
            "validation": self.validation.tolist(),
            "test": self.test.tolist(),
            "protected": self.protected.tolist(),
            "metadata": self.metadata,
        }


def _timestamp(value: str | int | pd.Timestamp | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Split boundaries must be timezone-qualified timestamps")
    return int(timestamp.value)


def _development_pool(
    data: WindowData, protected_start: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    boundary = _timestamp(protected_start)
    if boundary is None:
        eligible = np.ones(len(data), dtype=bool)
        protected = np.array([], dtype=np.int64)
    else:
        # Never use a label from the protected period even for an earlier decision.
        eligible = (data.time < boundary) & (data.label_time < boundary)
        protected = np.flatnonzero(data.time >= boundary)
    pool = np.flatnonzero(eligible)
    dates = np.unique(data.time[pool])
    if len(dates) < 3:
        raise ValueError("At least three safely matured development decision times are required")
    return pool, dates, protected, boundary


def purge_before_boundary(data: WindowData, indices: np.ndarray, next_start: int) -> np.ndarray:
    """An outcome published exactly at the next split start is still excluded."""
    index = np.asarray(indices, dtype=np.int64)
    return index[data.label_time[index] < int(next_start)]


def _select(data: WindowData, pool: np.ndarray, dates: np.ndarray) -> np.ndarray:
    return pool[np.isin(data.time[pool], dates)]


def _finish(
    data: WindowData,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    protected: np.ndarray,
    metadata: dict[str, Any],
) -> SplitIndices:
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("A split is empty after the embargo/maturity purge; collect more history")
    pre_train, pre_validation = len(train), len(validation)
    train = purge_before_boundary(data, train, int(data.time[validation].min()))
    validation = purge_before_boundary(data, validation, int(data.time[test].min()))
    if not len(train) or not len(validation):
        raise ValueError("All train or validation labels cross the next split boundary")
    metadata = {
        **metadata,
        "purged_training_rows": pre_train - len(train),
        "purged_validation_rows": pre_validation - len(validation),
        "ranges": {
            name: {
                "rows": len(index),
                "decision_times": len(np.unique(data.time[index])),
                "decision_start_ns": int(data.time[index].min()),
                "decision_end_ns": int(data.time[index].max()),
                "latest_label_ns": int(data.label_time[index].max()),
            }
            for name, index in (("train", train), ("validation", validation), ("test", test))
        },
    }
    return SplitIndices(train, validation, test, protected, metadata)


def chronological_split(
    data: WindowData,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo_bars: int = 5,
    protected_start: str | int | pd.Timestamp | None = None,
) -> SplitIndices:
    """Split unique decision times, not asset rows.

    Fractions define nominal development boundaries. ``embargo_bars`` unique
    decision times are then skipped at the start of validation and test. Training
    and validation labels must be available strictly before the following split.
    """
    if not (
        0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Positive train/validation fractions must sum to less than one")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be nonnegative")
    pool, dates, protected, boundary = _development_pool(data, protected_start)
    train_end = int(np.floor(len(dates) * train_fraction))
    validation_end = int(np.floor(len(dates) * (train_fraction + validation_fraction)))
    train_dates = dates[:train_end]
    validation_dates = dates[train_end + embargo_bars : validation_end]
    test_dates = dates[validation_end + embargo_bars :]
    metadata = {
        "method": "chronological_unique_decision_times",
        "embargo_bars": embargo_bars,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "protected_start_ns": boundary,
        "excluded_protected_crossing_labels": int(
            np.sum((data.time < boundary) & (data.label_time >= boundary))
        )
        if boundary is not None
        else 0,
        "embargo_rule": "skip beginning of validation/test by unique decision-time bars",
    }
    return _finish(
        data,
        _select(data, pool, train_dates),
        _select(data, pool, validation_dates),
        _select(data, pool, test_dates),
        protected,
        metadata,
    )


def walk_forward_splits(
    data: WindowData,
    min_train_dates: int,
    validation_dates: int,
    test_dates: int,
    step_dates: int | None = None,
    embargo_bars: int = 5,
    protected_start: str | int | pd.Timestamp | None = None,
    max_folds: int | None = None,
) -> list[SplitIndices]:
    """Expanding-window folds; all development test blocks are disjoint.

    Embargo is a count of unique decision timestamps, not rows or calendar days.
    The frozen protected tail is excluded from fold construction altogether.
    """
    if min(min_train_dates, validation_dates, test_dates) < 1 or embargo_bars < 0:
        raise ValueError("Positive window sizes and a nonnegative embargo are required")
    step = test_dates if step_dates is None else step_dates
    if step < test_dates:
        raise ValueError("step_dates < test_dates would reuse development test observations")
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive")
    pool, dates, protected, boundary = _development_pool(data, protected_start)
    folds: list[SplitIndices] = []
    end = min_train_dates
    while True:
        validation_start = end + embargo_bars
        validation_end = validation_start + validation_dates
        test_start = validation_end + embargo_bars
        test_end = test_start + test_dates
        if test_end > len(dates) or (max_folds is not None and len(folds) >= max_folds):
            break
        metadata = {
            "method": "expanding_walk_forward",
            "fold": len(folds),
            "embargo_bars": embargo_bars,
            "step_dates": step,
            "protected_start_ns": boundary,
            "test_blocks_disjoint": True,
        }
        folds.append(
            _finish(
                data,
                _select(data, pool, dates[:end]),
                _select(data, pool, dates[validation_start:validation_end]),
                _select(data, pool, dates[test_start:test_end]),
                protected,
                metadata,
            )
        )
        end += step
    if not folds:
        raise ValueError("Insufficient unprotected, matured history for one complete walk-forward fold")
    return folds


def low_data_training_subset(
    data: WindowData, indices: np.ndarray, fraction: float, mode: str = "most_recent"
) -> np.ndarray:
    """Predeclared low-data selection using complete chronological date groups."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    if mode not in {"most_recent", "earliest"}:
        raise ValueError("mode must be most_recent or earliest")
    index = np.asarray(indices, dtype=np.int64)
    dates = np.unique(data.time[index])
    keep = max(1, int(np.ceil(len(dates) * fraction)))
    selected = dates[-keep:] if mode == "most_recent" else dates[:keep]
    return index[np.isin(data.time[index], selected)]
