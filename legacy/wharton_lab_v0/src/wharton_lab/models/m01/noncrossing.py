"""Non-crossing quantile repair (post-hoc)."""

from __future__ import annotations

import numpy as np


def sort_quantiles_inplace(q_matrix: np.ndarray) -> np.ndarray:
    """
    Enforce q_1 <= q_2 <= ... <= q_K per row by sorting.

    Simple post-hoc fix; does not re-fit. Documented alternative: isotonic
    regression per row on (tau_k, q_k) with increasing constraint.
    """
    out = np.array(q_matrix, dtype=float, copy=True)
    out.sort(axis=1)
    return out


def isotonic_average_adjacent(q_matrix: np.ndarray) -> np.ndarray:
    """
    Pool adjacent crossing pairs by averaging until monotonic (per row).

    Lightweight isotonic-style pass: if q[k] > q[k+1], replace both with mean.
    Repeat until stable or single pass for speed (one pass used here).
    """
    out = np.array(q_matrix, dtype=float, copy=True)
    n_rows, k = out.shape
    for i in range(n_rows):
        row = out[i]
        changed = True
        while changed:
            changed = False
            for j in range(k - 1):
                if row[j] > row[j + 1]:
                    m = 0.5 * (row[j] + row[j + 1])
                    row[j] = m
                    row[j + 1] = m
                    changed = True
    return out


def enforce_non_crossing(q_matrix: np.ndarray, method: str = "sort") -> np.ndarray:
    if method == "isotonic_adjacent":
        return isotonic_average_adjacent(q_matrix)
    return sort_quantiles_inplace(q_matrix)
