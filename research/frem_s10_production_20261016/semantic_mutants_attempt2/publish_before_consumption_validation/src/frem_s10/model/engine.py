"""Immutable delayed-feedback transitions for the declared weekly S10 algorithm."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from frem_s10.domain import ForecastDistribution, OriginRecord, Transform
from frem_s10.errors import ChronologyError, IntegrityError
from frem_s10.model.config import ForecastConfig
from frem_s10.model.experts import issue_experts
from frem_s10.model.feedback import mature_forecast
from frem_s10.model.library import empty_library, update_library
from frem_s10.model.records import (
    ForecastState,
    ForecastTransition,
    IssuanceBatch,
    IssuedForecast,
    IssueRequest,
    scientific_id,
)
from frem_s10.model.stacking import fit_mixtures


def state_identity(state: ForecastState) -> str:
    """Hash scientific state, including retained numeric atoms and event history."""
    library = state.library
    feedback_meta = tuple(
        (item.record_id, item.issued_id, item.outcome_id, item.origin_us, item.available_us)
        for item in state.feedback
    )
    pending_meta = tuple(
        (
            item.record_id,
            item.arm,
            item.state_before_id,
            item.distribution.record_id,
            item.request.record_id,
            item.request.origin_us,
            item.request.endpoint_us,
            item.request.available_us,
            item.request.feature_available_us,
            item.request.anchor_index,
            item.experts.record_id,
            item.experts.library.record_ids,
            tuple(int(x) for x in item.experts.library.origin_us),
            tuple(int(x) for x in item.experts.library.available_us),
            item.experts.neighbor_indices,
            item.experts.neighbor_distance,
            item.experts.extrapolation_threshold,
            item.experts.fallback_reason,
        )
        for item in state.pending
    )
    metadata = (
        state.transform.record_id,
        state.transform.record_ids,
        asdict(state.config),
        library.record_ids,
        tuple(int(x) for x in library.origin_us),
        tuple(int(x) for x in library.available_us),
        feedback_meta,
        pending_meta,
        state.cursor_us,
        tuple(sorted(state.consumed_event_ids)),
        tuple(sorted(state.issued_origin_ids)),
        state.previous_state_id,
    )
    arrays = [
        library.g,
        library.d,
        library.contexts,
        state.transform.feature_mean,
        state.transform.feature_scale,
        state.transform.score_scale,
    ]
    for item in state.feedback:
        arrays.extend((item.context, item.a, item.b))
    for issued in state.pending:
        arrays.extend(
            (
                issued.request.x,
                issued.g,
                issued.d,
                issued.probabilities,
                issued.alpha,
                issued.experts.context,
                issued.experts.probabilities,
                issued.experts.library.g,
                issued.experts.library.d,
                issued.experts.library.contexts,
            )
        )
    return scientific_id("forecast-state-v2", metadata, *arrays)


def initial_state(transform: Transform, config: ForecastConfig | None = None) -> ForecastState:
    """Construct an empty causal state; market source admission is external."""
    state = ForecastState(
        "pending",
        transform,
        config or ForecastConfig(),
        empty_library(),
        (),
        (),
        -(2**63),
        frozenset(),
        frozenset(),
        None,
    )
    return replace(state, record_id=state_identity(state))


def _issue(state: ForecastState, request: IssueRequest) -> IssuanceBatch:
    experts = issue_experts(
        state.library,
        request.x,
        decision_us=request.decision_us,
        transform=state.transform,
        config=state.config,
    )
    mixture = fit_mixtures(
        state.feedback, experts.context, decision_us=request.decision_us, config=state.config
    )
    weights = (
        ("unconditional", np.array([1.0, 0.0, 0.0])),
        ("recent", np.array([0.0, 1.0, 0.0])),
        ("context", np.array([0.0, 0.0, 1.0])),
        ("equal", np.full(3, 1 / 3)),
        ("global_stack", mixture.global_alpha),
        ("contextual_stack", mixture.contextual_alpha),
        ("individual_score_mix", mixture.individual_alpha),
        ("zero", np.full(3, 1 / 3)),
    )
    forecasts = []
    request_meta = (
        request.record_id,
        request.origin_us,
        request.endpoint_us,
        request.available_us,
        request.feature_available_us,
        request.anchor_index,
    )
    for arm, alpha in weights:
        g, d = state.library.g, state.library.d
        p = alpha @ experts.probabilities
        if arm == "zero":
            g, d, p = np.ones((1, 3)), np.zeros((1, 3)), np.ones(1)
        identity = scientific_id(
            "issued-forecast-v2",
            (arm, request_meta, experts.record_id, state.record_id),
            request.x,
            g,
            d,
            p,
            alpha,
        )
        distribution = ForecastDistribution(identity, g, d, p)
        forecasts.append(
            IssuedForecast(identity, arm, request, distribution, experts, alpha, state.record_id)
        )
    return IssuanceBatch(tuple(forecasts), mixture)


def advance(
    state: ForecastState, request: IssueRequest, matured: tuple[OriginRecord, ...]
) -> ForecastTransition:
    """Apply every newly mature label once, then issue without the current outcome.

    The caller supplies all outcomes newly available since the saved cursor. A
    missing due issued target is an error. Each returned state is immutable, so
    rejected transitions never partially update library or feedback histories.
    """
    if state.record_id != state_identity(state):
        raise IntegrityError("forecast state identity mismatch")
    if request.decision_us <= state.cursor_us or request.record_id in state.issued_origin_ids:
        raise ChronologyError("duplicate issue or chronological cursor regression")
    ordered = tuple(sorted(matured, key=lambda item: (item.available_us, item.origin_us, item.record_id)))
    ids = tuple(item.record_id for item in ordered)
    if len(ids) != len(set(ids)) or any(item in state.consumed_event_ids for item in ids):
        raise ChronologyError("duplicate applied outcome, including evicted history")
    if any(
        item.available_us >= request.decision_us or item.available_us < state.cursor_us for item in ordered
    ):
        raise ChronologyError("new outcome is immature or precedes the prior cursor")
    pending = {item.request.record_id: item for item in state.pending}
    if any(
        item.request.available_us < request.decision_us and item.request.record_id not in ids
        for item in state.pending
    ):
        raise ChronologyError("a due issued outcome was omitted")
    new_feedback = tuple(
        mature_forecast(
            pending[item.record_id], item, decision_us=request.decision_us, transform=state.transform
        )
        for item in ordered
        if item.record_id in pending
    )
    feedback = tuple(
        sorted(
            (*state.feedback, *new_feedback),
            key=lambda item: (item.available_us, item.origin_us, item.record_id),
        )
    )
    feedback = feedback[-state.config.feedback_window :]
    library = update_library(state.library, ordered, decision_us=request.decision_us, config=state.config)
    working = replace(
        state,
        library=library,
        feedback=feedback,
        pending=tuple(item for item in state.pending if item.request.record_id not in ids),
        consumed_event_ids=state.consumed_event_ids | frozenset(ids),
        previous_state_id=state.record_id,
    )
    working = replace(working, record_id=state_identity(working))
    batch = _issue(working, request)
    # All arms share expert coefficients. Retain one exact issued representative.
    result = replace(
        working,
        pending=(*working.pending, batch.get("contextual_stack")),
        cursor_us=request.decision_us,
        issued_origin_ids=state.issued_origin_ids | frozenset((request.record_id,)),
    )
    result = replace(result, record_id=state_identity(result))
    return ForecastTransition(result, batch, new_feedback)
