"""Validated immutable data records shared by S10 model and portfolio layers."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from .arrays import FloatArray, IntArray, frozen_f64, frozen_i64
from .errors import ChronologyError, ContractError
from .metadata import ASSETS, strict_int


def _ids(ids: tuple[str, ...], count: int, name: str) -> None:
    if not isinstance(ids, tuple) or len(ids) != count or any(not isinstance(x, str) or not x for x in ids):
        raise ContractError(f"{name}: nonempty ordered string identities required")
    if len(set(ids)) != len(ids):
        raise ContractError(f"{name}: duplicate identities")


@dataclass(frozen=True, slots=True)
class Transform:
    """A single immutable pre-2008 context and score scaling transform."""

    feature_mean: FloatArray
    feature_scale: FloatArray
    score_scale: FloatArray
    record_ids: tuple[str, ...]
    record_id: str

    def __post_init__(self) -> None:
        for name, length in (("feature_mean", 12), ("feature_scale", 12), ("score_scale", 3)):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(length,), name=name))
        if np.any(self.feature_scale <= 0) or np.any(self.score_scale < 1e-4):
            raise ContractError("transform scales violate fixed positive floors")
        _ids(self.record_ids, len(self.record_ids), "burnin_record_ids")
        if not self.record_id:
            raise ContractError("transform identity required")

    def scale_context(self, x: FloatArray) -> FloatArray:
        """Standardize a single 12-dimensional raw context with frozen values."""
        source = frozen_f64(x, shape=(12,), name="context")
        return frozen_f64((source - self.feature_mean) / self.feature_scale,
                          shape=(12,), name="scaled_context")

    def scale_return(self, r: FloatArray) -> FloatArray:
        """Scale a [3] or [N,3] total simple return without recentering."""
        shape = (3,) if r.ndim == 1 else (None, 3)
        source = frozen_f64(r, shape=shape, name="total_simple_return")
        return frozen_f64(source / self.score_scale, shape=shape, name="standardized_return")


@dataclass(frozen=True, slots=True)
class OriginRecord:
    """Features and separate H4 price/dividend targets with explicit availability."""

    record_id: str
    anchor_index: int
    origin_us: int
    endpoint_us: int
    available_us: int
    feature_available_us: int
    x: FloatArray
    g: FloatArray
    d: FloatArray
    h1_g: FloatArray | None = None
    h1_d: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ContractError("origin record identity required")
        strict_int(self.anchor_index, name="anchor_index", minimum=0)
        for name in ("origin_us", "endpoint_us", "available_us", "feature_available_us"):
            strict_int(getattr(self, name), name=name)
        if not self.feature_available_us < self.origin_us < self.endpoint_us < self.available_us:
            raise ChronologyError("origin feature/endpoint/availability order is invalid", record_id=self.record_id)
        object.__setattr__(self, "x", frozen_f64(self.x, shape=(12,), name="x"))
        for name in ("g", "d"):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(3,), name=name))
        if np.any(self.g <= 0) or np.any(self.d < 0):
            raise ContractError("positive price gross and nonnegative cash dividend required")
        if (self.h1_g is None) != (self.h1_d is None):
            raise ContractError("both one-interval components must be supplied together")
        if self.h1_g is not None and self.h1_d is not None:
            object.__setattr__(self, "h1_g", frozen_f64(self.h1_g, shape=(3,), name="h1_g"))
            object.__setattr__(self, "h1_d", frozen_f64(self.h1_d, shape=(3,), name="h1_d"))
            if np.any(self.h1_g <= 0) or np.any(self.h1_d < 0):
                raise ContractError("invalid one-interval price/dividend components")

    @property
    def r(self) -> FloatArray:
        """Return total simple H4 return while retaining g/d as separate state."""
        return frozen_f64(self.g + self.d - 1.0, shape=(3,), name="r")


@dataclass(frozen=True, slots=True)
class ScenarioLibrary:
    """Ordered eligible raw contexts and joint atoms; N=0 supports initial state."""

    record_ids: tuple[str, ...]
    origin_us: IntArray
    available_us: IntArray
    g: FloatArray
    d: FloatArray
    contexts: FloatArray

    def __post_init__(self) -> None:
        count = len(self.record_ids)
        _ids(self.record_ids, count, "library_record_ids")
        for name in ("origin_us", "available_us"):
            object.__setattr__(self, name, frozen_i64(getattr(self, name), shape=(count,), name=name))
        for name, dimension in (("g", 3), ("d", 3), ("contexts", 12)):
            object.__setattr__(self, name,
                               frozen_f64(getattr(self, name), shape=(count, dimension), name=name))
        if np.any(self.g <= 0) or np.any(self.d < 0):
            raise ContractError("invalid library price/dividend atoms")
        if np.any(self.available_us <= self.origin_us):
            raise ChronologyError("library label must mature after origin")
        ordering = tuple(zip(self.origin_us.tolist(), self.record_ids, strict=True))
        if ordering != tuple(sorted(ordering)):
            raise ChronologyError("library must be sorted by origin then stable identity")

    @property
    def r(self) -> FloatArray:
        """Return total simple outcomes in original asset order."""
        return frozen_f64(self.g + self.d - 1.0, shape=(len(self.record_ids), 3), name="library_r")


@dataclass(frozen=True, slots=True)
class ForecastDistribution:
    """A finite law on jointly paired price gross and cash-dividend components."""

    record_id: str
    g: FloatArray
    d: FloatArray
    probabilities: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ContractError("forecast identity required")
        g = frozen_f64(self.g, shape=(None, 3), name="forecast_g")
        count = len(g)
        if not count:
            raise ContractError("a forecast must have at least one atom")
        d = frozen_f64(self.d, shape=(count, 3), name="forecast_d")
        p = frozen_f64(self.probabilities, shape=(count,), name="probabilities")
        if np.any(g <= 0) or np.any(d < 0) or np.any(p < 0) or abs(float(np.sum(p)) - 1) > 1e-12:
            raise ContractError("invalid finite forecast distribution")
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "probabilities", p)

    @property
    def r(self) -> FloatArray:
        """Return total simple outcomes without replacing price/dividend records."""
        return frozen_f64(self.g + self.d - 1.0, shape=self.g.shape, name="forecast_r")


@dataclass(frozen=True, slots=True)
class SessionPanel:
    """Synchronized positive-price daily quotes and cash dividends in USD/share."""

    dates: tuple[date, ...]
    open: FloatArray
    close: FloatArray
    dividends: FloatArray
    source_ids: tuple[str, str, str]
    event_ids: tuple[tuple[str, ...], ...]
    assets: tuple[str, str, str] = ASSETS

    def __post_init__(self) -> None:
        count = len(self.dates)
        if not count or any(type(day) is not date for day in self.dates):
            raise ContractError("nonempty calendar dates required")
        if self.dates != tuple(sorted(set(self.dates))):
            raise ChronologyError("session dates must be unique and increasing")
        if self.assets != ASSETS:
            raise ContractError("asset order must be SPY, TLT, GLD")
        _ids(self.source_ids, 3, "source_ids")
        for name in ("open", "close", "dividends"):
            object.__setattr__(self, name,
                               frozen_f64(getattr(self, name), shape=(count, 3), name=name))
        if np.any(self.open <= 0) or np.any(self.close <= 0) or np.any(self.dividends < 0):
            raise ContractError("positive daily marks and nonnegative cash dividends required")
        if len(self.event_ids) != count:
            raise ContractError("dividend event IDs must align with sessions")
        flat = tuple(item for group in self.event_ids for item in group)
        _ids(flat, len(flat), "dividend_event_ids")
