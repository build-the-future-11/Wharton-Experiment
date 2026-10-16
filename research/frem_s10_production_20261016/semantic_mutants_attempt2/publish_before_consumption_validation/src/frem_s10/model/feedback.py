"""Score the exact previously issued expert laws only after target maturity."""

from __future__ import annotations

import numpy as np

from frem_s10.domain import OriginRecord, Transform
from frem_s10.errors import ChronologyError, IntegrityError
from frem_s10.math.scores import score_coefficients
from frem_s10.model.records import IssuedForecast, MaturedFeedback, scientific_id


def mature_forecast(
    issued: IssuedForecast, outcome: OriginRecord, *, decision_us: int, transform: Transform
) -> MaturedFeedback:
    """Use stored atoms/probabilities; never regenerate an earlier forecast."""
    request = issued.request
    if outcome.available_us >= decision_us:
        raise ChronologyError("outcome is unavailable at feedback processing")
    if (
        request.record_id,
        request.origin_us,
        request.endpoint_us,
        request.available_us,
        request.feature_available_us,
        request.anchor_index,
    ) != (
        outcome.record_id,
        outcome.origin_us,
        outcome.endpoint_us,
        outcome.available_us,
        outcome.feature_available_us,
        outcome.anchor_index,
    ):
        raise IntegrityError("matured outcome does not match issued target metadata")
    if not np.array_equal(request.x, outcome.x):
        raise IntegrityError("issued feature context differs from target context")
    coefficients = score_coefficients(
        issued.experts.library.r / transform.score_scale,
        issued.experts.probabilities,
        outcome.r / transform.score_scale,
    )
    outcome_id = scientific_id("matured-outcome-v2", (outcome.record_id,), outcome.g, outcome.d)
    record_id = scientific_id(
        "matured-feedback-v2",
        (issued.record_id, outcome_id, transform.record_id),
        coefficients.a,
        coefficients.b,
    )
    return MaturedFeedback(
        record_id,
        issued.record_id,
        outcome_id,
        outcome.origin_us,
        outcome.available_us,
        issued.experts.context,
        coefficients.a,
        coefficients.b,
    )
