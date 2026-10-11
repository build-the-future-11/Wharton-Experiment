"""Expanding-window walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def iter_walk_forward(
    n: int,
    *,
    n_folds: int = 5,
    min_train: int = 40,
    test_size: int = 8,
) -> Iterator[WalkForwardFold]:
    if n < min_train + test_size:
        raise ValueError("series too short for walk-forward")
    for fold in range(n_folds):
        test_end = n - fold * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end < min_train or test_start < min_train:
            break
        yield WalkForwardFold(
            fold=fold,
            train_idx=np.arange(0, train_end),
            test_idx=np.arange(test_start, test_end),
        )
