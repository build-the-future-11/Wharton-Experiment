"""Causal mixture-level global/local fits from immutable matured score records."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.errors import ChronologyError, ContractError
from frem_s10.math.distances import extrapolation_threshold, neighbor_order, pairwise_distances
from frem_s10.math.probabilities import recency_weights, softmax
from frem_s10.math.simplex import CertifiedMixture, solve_simplex
from frem_s10.model.config import ForecastConfig
from frem_s10.model.records import MaturedFeedback, MixtureBundle


def _fit(
    records: tuple[MaturedFeedback, ...], weights: ArrayLike, center: ArrayLike, config: ForecastConfig
) -> CertifiedMixture:
    p = frozen_f64(weights, shape=(len(records),), name="feedback weights")
    a = np.stack([record.a for record in records])
    b = np.stack([record.b for record in records])
    return solve_simplex(p @ a, np.einsum("i,ijk->jk", p, b), center, l2_coefficient=config.l2_coefficient)


def _individual(
    records: tuple[MaturedFeedback, ...], decision_us: int, config: ForecastConfig
) -> NDArray[np.float64]:
    if len(records) < config.global_minimum:
        return np.full(3, 1 / 3)
    recent = records[-config.global_minimum :]
    weights = recency_weights(
        np.array([item.origin_us for item in recent], dtype=np.int64),
        decision_us=decision_us,
        half_life_weeks=config.half_life_weeks,
    )
    standalone = np.stack([item.a - 0.5 * np.diag(item.b) for item in recent])
    return softmax(-(weights @ standalone))


def fit_mixtures(
    records: tuple[MaturedFeedback, ...],
    scaled_context: ArrayLike,
    *,
    decision_us: int,
    config: ForecastConfig,
) -> MixtureBundle:
    """Fit declared mixtures; all contexts supplied here use the fixed transform."""
    context = frozen_f64(scaled_context, shape=(12,), name="current scaled context")
    if len(records) > config.feedback_window or len({r.record_id for r in records}) != len(records):
        raise ContractError("feedback window is oversized or has duplicate records")
    order = tuple((r.available_us, r.origin_us, r.record_id) for r in records)
    if order != tuple(sorted(order)) or any(r.available_us >= decision_us for r in records):
        raise ChronologyError("feedback must be ordered and strictly mature")
    uniform = np.full(3, 1 / 3)
    global_fit = None
    global_alpha = uniform
    if len(records) >= config.global_minimum:
        global_fit = _fit(records, np.full(len(records), 1 / len(records)), uniform, config)
        global_alpha = global_fit.alpha
    individual = _individual(records, decision_us, config)
    if len(records) <= config.neighbors:
        return MixtureBundle(
            global_alpha,
            global_alpha,
            global_alpha,
            individual,
            0,
            0,
            "INSUFFICIENT_LOCAL_RECORDS",
            global_fit,
            None,
        )
    contexts = np.stack([record.context for record in records])
    distances = pairwise_distances(context[None, :], contexts, squared=True)[0]
    origins = np.array([record.origin_us for record in records], dtype=np.int64)
    selected = neighbor_order(distances, origins, tuple(r.record_id for r in records))[: config.neighbors]
    local_records = tuple(records[i] for i in selected)
    weights = recency_weights(
        origins[list(selected)], decision_us=decision_us, half_life_weeks=config.half_life_weeks
    )
    effective = 1 / float(weights @ weights)
    reason = None
    if effective < config.concentration_floor:
        reason = "LOCAL_CONCENTRATION"
    elif float(distances[selected[-1]]) > extrapolation_threshold(contexts, neighbors=config.neighbors):
        reason = "LOCAL_EXTRAPOLATION"
    if reason:
        return MixtureBundle(
            global_alpha, global_alpha, global_alpha, individual, 0, effective, reason, global_fit, None
        )
    local_fit = _fit(local_records, weights, global_alpha, config)
    beta = min(config.maximum_blend, effective / (effective + config.shrink_pseudocount))
    contextual = local_fit.alpha if config.no_shrink else (1 - beta) * global_alpha + beta * local_fit.alpha
    return MixtureBundle(
        global_alpha, local_fit.alpha, contextual, individual, beta, effective, None, global_fit, local_fit
    )
