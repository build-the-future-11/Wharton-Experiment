from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def date_level_spearman_ic(
    scores: np.ndarray,
    labels: np.ndarray,
    date_ids: np.ndarray,
) -> float:
    """Mean Spearman correlation between scores and labels within each date."""
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    date_ids = np.asarray(date_ids)
    ics = []
    for d in np.unique(date_ids):
        m = date_ids == d
        if m.sum() < 3:
            continue
        r, _ = spearmanr(scores[m], labels[m])
        if np.isfinite(r):
            ics.append(r)
    if not ics:
        return float("nan")
    return float(np.mean(ics))
