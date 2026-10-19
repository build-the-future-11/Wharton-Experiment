"""Expanding walk-forward splits with purge gap and lockbox."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterator, Optional, Sequence


@dataclass(frozen=True)
class WalkForwardSplit:
    train_indices: slice
    test_indices: slice
    fold_id: int
    is_lockbox: bool = False


def _sample_count(value: int, name: str, *, minimum: int) -> int:
    """Reject coercions that can silently change index or loop semantics."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def walk_forward_folds(
    n_samples: int,
    *,
    min_train: int = 64,
    test_size: int = 16,
    step: Optional[int] = None,
    purge_gap: int = 0,
    lockbox_size: int = 0,
) -> list[WalkForwardSplit]:
    """
    Expanding-window walk-forward.

    Train always starts at 0 and grows; test follows train + purge_gap.
    Final lockbox fold (if lockbox_size > 0) is held out and marked is_lockbox.

    Sizes and an explicit step must be positive integers; purge_gap and
    lockbox_size may be zero. A lockbox-only result is valid when at least
    min_train samples and the purge gap precede the lockbox, even when no
    ordinary test_size window fits. Without a lockbox, at least one ordinary
    fold must fit. Smaller steps may deliberately overlap successive test
    windows; training and testing within an individual fold never overlap.
    """
    n_samples = _sample_count(n_samples, "n_samples", minimum=1)
    min_train = _sample_count(min_train, "min_train", minimum=1)
    test_size = _sample_count(test_size, "test_size", minimum=1)
    purge_gap = _sample_count(purge_gap, "purge_gap", minimum=0)
    lockbox_size = _sample_count(lockbox_size, "lockbox_size", minimum=0)
    step = test_size if step is None else _sample_count(step, "step", minimum=1)
    effective_n = n_samples - lockbox_size
    if lockbox_size and effective_n - purge_gap < min_train:
        raise ValueError("Not enough pre-lockbox training samples after the purge gap")
    if not lockbox_size and n_samples < min_train + purge_gap + test_size:
        raise ValueError("Not enough samples for walk-forward configuration")
    folds: list[WalkForwardSplit] = []

    def append_fold(train_stop: int, test_start: int, test_stop: int,
                    *, is_lockbox: bool = False) -> None:
        # Validate numeric boundaries before Python can interpret negative
        # slice stops relative to the end of an array.
        if not (min_train <= train_stop <= test_start - purge_gap
                and 0 <= test_start < test_stop <= n_samples):
            raise ValueError("Invalid walk-forward train/test boundaries")
        folds.append(WalkForwardSplit(
            train_indices=slice(0, train_stop),
            test_indices=slice(test_start, test_stop),
            fold_id=len(folds),
            is_lockbox=is_lockbox,
        ))

    train_end = min_train
    while True:
        test_start = train_end + purge_gap
        test_end = test_start + test_size
        if test_end > effective_n:
            break
        append_fold(train_end, test_start, test_end)
        train_end += step

    if lockbox_size > 0:
        append_fold(effective_n - purge_gap, effective_n, n_samples, is_lockbox=True)
    return folds


def iter_fold_arrays(
    X,
    y,
    folds: Sequence[WalkForwardSplit],
) -> Iterator[tuple[WalkForwardSplit, object, object, object, object]]:
    for fold in folds:
        tr, te = fold.train_indices, fold.test_indices
        yield fold, X[tr], y[tr], X[te], y[te]
