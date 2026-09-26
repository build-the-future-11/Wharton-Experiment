"""Typed dataclasses for lab pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class HistoricalBatch:
    """Features and labels available as-of decision time."""

    as_of: datetime
    entity_ids: np.ndarray
    features: np.ndarray
    labels: Optional[np.ndarray] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_ids", np.asarray(self.entity_ids))
        object.__setattr__(self, "features", np.asarray(self.features, dtype=float))
        if self.labels is not None:
            object.__setattr__(self, "labels", np.asarray(self.labels, dtype=float))


@dataclass(frozen=True)
class MaturedLabel:
    """Outcome observed only after horizon + reporting delay."""

    entity_id: str | int
    decision_time: datetime
    maturity_time: datetime
    value: float
    horizon_days: int
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Forecast:
    """Point or distributional forecast for one entity."""

    entity_id: str | int
    as_of: datetime
    point: float
    quantiles: Optional[Mapping[float, float]] = None
    std: Optional[float] = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CovarianceForecast:
    as_of: datetime
    covariance: np.ndarray
    entity_ids: Sequence[str | int]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "covariance", np.asarray(self.covariance, dtype=float))


@dataclass(frozen=True)
class ScenarioPaths:
    as_of: datetime
    paths: np.ndarray
    entity_ids: Sequence[str | int]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", np.asarray(self.paths, dtype=float))


@dataclass(frozen=True)
class ControllerState:
    as_of: datetime
    positions: np.ndarray
    cash: float
    risk_budget: float
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    as_of: datetime
    target_weights: np.ndarray
    entity_ids: Sequence[str | int]
    rationale: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    supports_quantiles: bool = False
    supports_ranking: bool = False
    supports_factors: bool = False
    supports_memory: bool = False
    requires_maturity: bool = False
    max_entities: Optional[int] = None


@dataclass(frozen=True)
class ModelMetadata:
    """Model-card-like metadata."""

    model_id: str
    title: str
    version: str
    description: str
    inputs: Sequence[str]
    outputs: Sequence[str]
    training_constraints: Sequence[str] = field(default_factory=tuple)
    metrics: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
