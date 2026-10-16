"""Immutable model-owned records and deterministic scientific identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.domain import ForecastDistribution, ScenarioLibrary, Transform
from frem_s10.errors import ChronologyError, ContractError
from frem_s10.math.probabilities import validate_probabilities
from frem_s10.math.simplex import CertifiedMixture
from frem_s10.model.config import ForecastConfig


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
    record_id: str
    origin_us: int
    endpoint_us: int
    available_us: int
    feature_available_us: int
    x: NDArray[np.float64]
    anchor_index: int = 0

    def __post_init__(self) -> None:
        if not self.record_id or not self.origin_us < self.endpoint_us < self.available_us:
            raise ChronologyError("issued target metadata is not chronologically ordered")
        if self.feature_available_us >= self.decision_us:
            raise ChronologyError("request features are unavailable at decision")
        object.__setattr__(self, "x", frozen_f64(self.x, shape=(12,), name="issue context"))

    @property
    def decision_us(self) -> int:
        return self.origin_us + 1


@dataclass(frozen=True, slots=True)
class ExpertSet:
    record_id: str
    library: ScenarioLibrary
    probabilities: NDArray[np.float64]
    context: NDArray[np.float64]
    neighbor_indices: tuple[int, ...]
    neighbor_distance: float | None
    extrapolation_threshold: float | None
    fallback_reason: str | None

    def __post_init__(self) -> None:
        p = frozen_f64(self.probabilities, shape=(3, len(self.library.record_ids)), name="expert law")
        for row in p:
            validate_probabilities(row)
        object.__setattr__(self, "probabilities", p)
        object.__setattr__(self, "context", frozen_f64(self.context, shape=(12,), name="scaled context"))


@dataclass(frozen=True, slots=True)
class MaturedFeedback:
    record_id: str
    issued_id: str
    outcome_id: str
    origin_us: int
    available_us: int
    context: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", frozen_f64(self.context, shape=(12,), name="feedback context"))
        object.__setattr__(self, "a", frozen_f64(self.a, shape=(3,), name="feedback A"))
        object.__setattr__(self, "b", frozen_f64(self.b, shape=(3, 3), name="feedback B"))
        if self.available_us <= self.origin_us:
            raise ChronologyError("feedback matures before its forecast")


@dataclass(frozen=True, slots=True)
class MixtureBundle:
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
        for field in ("global_alpha", "local_alpha", "contextual_alpha", "individual_alpha"):
            object.__setattr__(self, field, validate_probabilities(getattr(self, field), size=3))


@dataclass(frozen=True, slots=True)
class IssuedForecast:
    record_id: str
    arm: str
    request: IssueRequest
    distribution: ForecastDistribution
    experts: ExpertSet
    alpha: NDArray[np.float64]
    state_before_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_probabilities(self.alpha, size=3))

    @property
    def g(self) -> NDArray[np.float64]:
        return self.distribution.g

    @property
    def d(self) -> NDArray[np.float64]:
        return self.distribution.d

    @property
    def probabilities(self) -> NDArray[np.float64]:
        return self.distribution.probabilities


@dataclass(frozen=True, slots=True)
class IssuanceBatch:
    forecasts: tuple[IssuedForecast, ...]
    mixtures: MixtureBundle

    def get(self, arm: str) -> IssuedForecast:
        for forecast in self.forecasts:
            if forecast.arm == arm:
                return forecast
        raise ContractError(f"unknown issued arm: {arm}")


@dataclass(frozen=True, slots=True)
class ForecastState:
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


@dataclass(frozen=True, slots=True)
class ForecastTransition:
    state: ForecastState
    batch: IssuanceBatch
    matured_feedback: tuple[MaturedFeedback, ...]
