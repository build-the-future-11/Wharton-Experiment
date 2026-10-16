"""Shared-atom unconditional, recency and contextual empirical experts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from frem_s10.arrays import frozen_f64
from frem_s10.domain import ScenarioLibrary, Transform
from frem_s10.errors import ChronologyError, ContractError
from frem_s10.math.distances import extrapolation_threshold, neighbor_order, pairwise_distances
from frem_s10.math.probabilities import recency_weights
from frem_s10.model.config import ForecastConfig
from frem_s10.model.records import ExpertSet, scientific_id


def issue_experts(
    library: ScenarioLibrary,
    context: ArrayLike,
    *,
    decision_us: int,
    transform: Transform,
    config: ForecastConfig,
) -> ExpertSet:
    """Context is raw; fixed burn-in transform is applied here to all candidates."""
    n = len(library.record_ids)
    if n < config.library_minimum or n > config.library_window:
        raise ContractError("scenario library violates configured admission/window")
    if np.any(library.available_us >= decision_us):
        raise ChronologyError("unavailable scenario entered issuance")
    x = frozen_f64(context, shape=(12,), name="raw issue context")
    scaled = (x - transform.feature_mean) / transform.feature_scale
    candidates = (library.contexts - transform.feature_mean) / transform.feature_scale
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    recent = recency_weights(
        library.origin_us, decision_us=decision_us, half_life_weeks=config.half_life_weeks
    )
    distances = pairwise_distances(scaled[None, :], candidates, squared=True)[0]
    order = neighbor_order(distances, library.origin_us, library.record_ids)
    selected = order[: config.neighbors]
    cutoff: float | None = None
    kth: float | None = None
    reason: str | None = None
    contextual = uniform.copy()
    if n <= config.neighbors:
        reason = "INSUFFICIENT_LEAVE_ONE_OUT_CONTEXTS"
    else:
        kth = float(distances[selected[-1]])
        cutoff = extrapolation_threshold(candidates, neighbors=config.neighbors)
        if kth > cutoff:
            reason = "CONTEXT_EXTRAPOLATION"
        else:
            contextual = np.zeros(n, dtype=np.float64)
            contextual[list(selected)] = 1.0 / config.neighbors
    probabilities = np.stack((uniform, recent, contextual))
    identity = scientific_id(
        "experts-v2",
        (library.record_ids, decision_us, config.neighbors, transform.record_id),
        probabilities,
        scaled,
    )
    return ExpertSet(identity, library, probabilities, scaled, selected, kth, cutoff, reason)
