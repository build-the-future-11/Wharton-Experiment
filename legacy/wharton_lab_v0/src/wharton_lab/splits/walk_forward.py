"""Expanding walk-forward splits with purge gap and lockbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence


@dataclass(frozen=True)
class WalkForwardSplit:
    train_indices: slice
    test_indices: slice
    fold_id: int
    is_lockbox: bool = False


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
    """
    if n_samples < min_train + purge_gap + test_size:
        raise ValueError("Not enough samples for walk-forward configuration")
    step = step or test_size
    effective_n = n_samples - lockbox_size if lockbox_size > 0 else n_samples
    folds: list[WalkForwardSplit] = []
    fold_id = 0
    train_end = min_train
    while True:
        test_start = train_end + purge_gap
        test_end = test_start + test_size
        if test_end > effective_n:
            break
        folds.append(
            WalkForwardSplit(
                train_indices=slice(0, train_end),
                test_indices=slice(test_start, test_end),
                fold_id=fold_id,
            )
        )
        fold_id += 1
        train_end += step

    if lockbox_size > 0:
        lb_start = n_samples - lockbox_size
        folds.append(
            WalkForwardSplit(
                train_indices=slice(0, lb_start - purge_gap),
                test_indices=slice(lb_start, n_samples),
                fold_id=fold_id,
                is_lockbox=True,
            )
        )
    return folds


def iter_fold_arrays(
    X,
    y,
    folds: Sequence[WalkForwardSplit],
) -> Iterator[tuple[WalkForwardSplit, object, object, object, object]]:
    for fold in folds:
        tr, te = fold.train_indices, fold.test_indices
        yield fold, X[tr], y[tr], X[te], y[te]
