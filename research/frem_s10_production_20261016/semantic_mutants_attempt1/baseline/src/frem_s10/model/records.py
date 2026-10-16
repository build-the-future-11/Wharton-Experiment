"""Immutable model-owned records and deterministic scientific identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.domain import ForecastDistribution, ScenarioLibrary, Transform
from frem_s10.errors import ChronologyError, ContractError
from frem_s10.math.probabilities import validate_probabilities
from frem_s10.math.simplex import CertifiedMixture
from frem_s10.metadata import strict_int
from frem_s10.model.config import ForecastConfig


def _require_identity(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name}: nonempty string identity required")


def _require_identity_set(value: object, name: str) -> None:
    if not isinstance(value, frozenset) or any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"{name}: immutable nonempty-string identity set required")


def scientific_id(kind: str, metadata: tuple[object, ...], *arrays: NDArray[np.float64]) -> str:
    """Hash scientific metadata and little-endian binary64 arrays, excluding IO state."""
    digest = hashlib.sha256()
    digest.update(json.dumps((kind, metadata), separators=(",", ":"), allow_nan=False).encode())
    for array in arrays:
        canonical = np.asarray(array, dtype="<f8", order="C")
        digest.update(json.dumps(canonical.shape).encode())
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IssueRequest:
    """Forecast metadata and available features, with no target-value field."""

    record_id: str
    origin_us: int
    endpoint_us: int
    available_us: int
    feature_available_us: int
    x: NDArray[np.float64]
    anchor_index: int = 0

    def __post_init__(self) -> None:
        """Validate the event clock and freeze the owned feature vector."""
        for name in ("origin_us", "endpoint_us", "available_us", "feature_available_us"):
            strict_int(getattr(self, name), name=name)
        strict_int(self.anchor_index, name="anchor_index", minimum=0)
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ContractError("issue request needs a nonempty identity")
        if not self.origin_us < self.endpoint_us < self.available_us:
            raise ChronologyError("issued target metadata is not chronologically ordered")
        if self.feature_available_us >= self.decision_us:
            raise ChronologyError("request features are unavailable at decision")
        object.__setattr__(self, "x", frozen_f64(self.x, shape=(12,), name="issue context"))

    @property
    def decision_us(self) -> int:
        """Return the CL-01 instant one microsecond after the opening quote."""
        return self.origin_us + 1


@dataclass(frozen=True, slots=True)
class ExpertSet:
    """Immutable shared scenario atoms and issued U/R/C probabilities."""

    record_id: str
    library: ScenarioLibrary
    probabilities: NDArray[np.float64]
    context: NDArray[np.float64]
    neighbor_indices: tuple[int, ...]
    neighbor_distance: float | None
    extrapolation_threshold: float | None
    fallback_reason: str | None

    def __post_init__(self) -> None:
        """Check each exact issued probability law."""
        if not isinstance(self.library, ScenarioLibrary) or not isinstance(self.neighbor_indices, tuple):
            raise ContractError("expert library and immutable neighbor index tuple required")
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ContractError("expert identity is required")
        if len(set(self.neighbor_indices)) != len(self.neighbor_indices) or any(
            type(i) is not int or not 0 <= i < len(self.library.record_ids) for i in self.neighbor_indices
        ):
            raise ContractError("expert neighbor indices are invalid")
        for value in (self.neighbor_distance, self.extrapolation_threshold):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ContractError("expert distance diagnostic is invalid")
        p = frozen_f64(self.probabilities, shape=(3, len(self.library.record_ids)), name="expert law")
        for row in p:
            validate_probabilities(row)
        object.__setattr__(self, "probabilities", p)
        object.__setattr__(self, "context", frozen_f64(self.context, shape=(12,), name="scaled context"))


@dataclass(frozen=True, slots=True)
class MaturedFeedback:
    """Exact score coefficients referring to one past issued forecast."""

    record_id: str
    issued_id: str
    outcome_id: str
    origin_us: int
    available_us: int
    context: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own finite coefficients and reject impossible maturity order."""
        if any(
            not isinstance(value, str) or not value
            for value in (self.record_id, self.issued_id, self.outcome_id)
        ):
            raise ContractError("maturity identities are required")
        strict_int(self.origin_us, name="feedback origin")
        strict_int(self.available_us, name="feedback availability")
        object.__setattr__(self, "context", frozen_f64(self.context, shape=(12,), name="feedback context"))
        object.__setattr__(self, "a", frozen_f64(self.a, shape=(3,), name="feedback A"))
        object.__setattr__(self, "b", frozen_f64(self.b, shape=(3, 3), name="feedback B"))
        if self.available_us <= self.origin_us:
            raise ChronologyError("feedback matures before its forecast")


@dataclass(frozen=True, slots=True)
class MixtureBundle:
    """Declared mixture arms, fallback reason and numerical certificates."""

    global_alpha: NDArray[np.float64]
    local_alpha: NDArray[np.float64]
    contextual_alpha: NDArray[np.float64]
    individual_alpha: NDArray[np.float64]
    beta: float
    effective_count: float
    fallback_reason: str | None
    global_certificate: CertifiedMixture | None
    local_certificate: CertifiedMixture | None

    def __post_init__(self) -> None:
        """Validate every simplex and the concentration/blend diagnostics."""
        for field in ("global_alpha", "local_alpha", "contextual_alpha", "individual_alpha"):
            object.__setattr__(self, field, validate_probabilities(getattr(self, field), size=3))
        if (
            not np.isfinite(self.beta)
            or not 0 <= self.beta <= 0.5
            or not np.isfinite(self.effective_count)
            or self.effective_count < 0
        ):
            raise ContractError("invalid mixture concentration or blend")


@dataclass(frozen=True, slots=True)
class IssuedForecast:
    """A saved finite law with its exact pre-issue state and expert identities."""

    record_id: str
    arm: str
    request: IssueRequest
    distribution: ForecastDistribution
    experts: ExpertSet
    alpha: NDArray[np.float64]
    state_before_id: str

    def __post_init__(self) -> None:
        """Own the exact expert mixture rather than exposing mutable weights."""
        _require_identity(self.record_id, "issued record")
        _require_identity(self.state_before_id, "issued parent state")
        if (
            not isinstance(self.request, IssueRequest)
            or not isinstance(self.distribution, ForecastDistribution)
            or not isinstance(self.experts, ExpertSet)
        ):
            raise ContractError("issued forecast contains an invalid nested record")
        if self.distribution.record_id != self.record_id:
            raise ContractError("issued distribution identity mismatch")
        if self.arm not in (
            "unconditional",
            "recent",
            "context",
            "equal",
            "global_stack",
            "contextual_stack",
            "individual_score_mix",
            "zero",
        ):
            raise ContractError("unknown issued forecast arm")
        object.__setattr__(self, "alpha", validate_probabilities(self.alpha, size=3))

    @property
    def g(self) -> NDArray[np.float64]:
        """Expose immutable price gross atoms."""
        return self.distribution.g

    @property
    def d(self) -> NDArray[np.float64]:
        """Expose immutable dividend-per-starting-dollar atoms."""
        return self.distribution.d

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Expose the immutable final atom probabilities."""
        return self.distribution.probabilities


@dataclass(frozen=True, slots=True)
class IssuanceBatch:
    """All eight declared forecast arms from one common information state."""

    forecasts: tuple[IssuedForecast, ...]
    mixtures: MixtureBundle

    def get(self, arm: str) -> IssuedForecast:
        """Return one declared arm or reject an unknown identity."""
        for forecast in self.forecasts:
            if forecast.arm == arm:
                return forecast
        raise ContractError(f"unknown issued arm: {arm}")


@dataclass(frozen=True, slots=True)
class ForecastState:
    """Complete persistent causal forecast state, independent of holdings."""

    record_id: str
    transform: Transform
    config: ForecastConfig
    library: ScenarioLibrary
    feedback: tuple[MaturedFeedback, ...]
    pending: tuple[IssuedForecast, ...]
    cursor_us: int
    consumed_event_ids: frozenset[str]
    issued_origin_ids: frozenset[str]
    previous_state_id: str | None

    def __post_init__(self) -> None:
        """Reject malformed nested state before identity calculation or replay."""
        _require_identity(self.record_id, "state")
        if self.previous_state_id is not None:
            _require_identity(self.previous_state_id, "previous state")
        strict_int(self.cursor_us, name="state cursor")
        if (
            not isinstance(self.transform, Transform)
            or not isinstance(self.config, ForecastConfig)
            or not isinstance(self.library, ScenarioLibrary)
        ):
            raise ContractError("state transform, configuration and scenario library must be typed records")
        _require_identity_set(self.consumed_event_ids, "consumed events")
        _require_identity_set(self.issued_origin_ids, "issued origins")
        self._validate_record_queues()
        if not set(self.library.record_ids).issubset(self.consumed_event_ids):
            raise ContractError("library contains an unconsumed outcome identity")
        if len(self.library.record_ids) > self.config.library_window:
            raise ContractError("scenario library exceeds its declared window")

    def _validate_record_queues(self) -> None:
        if not isinstance(self.feedback, tuple) or any(
            not isinstance(item, MaturedFeedback) for item in self.feedback
        ):
            raise ContractError("feedback must be an immutable tuple of matured records")
        if not isinstance(self.pending, tuple) or any(
            not isinstance(item, IssuedForecast) for item in self.pending
        ):
            raise ContractError("pending must be an immutable tuple of issued forecasts")
        order = tuple((item.available_us, item.origin_us, item.record_id) for item in self.feedback)
        if order != tuple(sorted(order)) or len({item.record_id for item in self.feedback}) != len(
            self.feedback
        ):
            raise ContractError("feedback order or identity uniqueness is invalid")
        if len(self.feedback) > self.config.feedback_window:
            raise ContractError("matured feedback exceeds its declared window")
        origins = tuple(item.request.record_id for item in self.pending)
        if len(set(origins)) != len(origins) or not set(origins).issubset(self.issued_origin_ids):
            raise ContractError("pending targets must be distinct previously issued origins")
        if set(origins).intersection(self.consumed_event_ids):
            raise ContractError("consumed outcome remains pending")


@dataclass(frozen=True, slots=True)
class ForecastTransition:
    """Next state, newly issued laws and exactly-once matured feedback."""

    state: ForecastState
    batch: IssuanceBatch
    matured_feedback: tuple[MaturedFeedback, ...]
