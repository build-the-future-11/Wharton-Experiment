"""Bounded S6 sequence features, dated engrams, and delayed quantized updates.

Each reservoir branch obeys h'=(1-a)h+a*tanh(W h+U x+b), with
||W||_2=rho<1. Thus its state stays in [-1,1] and its same-input
contraction factor is at most 1-a+a*rho. Features concatenate final and
average branch states, cross-branch products, bounded input and random
Fourier features. This is a fixed feature map, not a trained JEPA predictor.

Engrams use a Gaussian similarity kernel and dated exponential forgetting.
Only labels with origin, maturity AND availability strictly before the
query may contribute. Retrieval is a convex combination of eligible values.

Ripple updates distribute a clipped gradient over a finite delayed geometric
schedule. Matured gradients are quantized with error feedback, then the
parameter increment is norm bounded. The quantization residual has coordinate
bound q/2; clipping loss is deliberately not recycled into that residual.
These are mathematical/engineering properties, not financial guarantees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from numbers import Real
from typing import Any

import numpy as np


def _finite(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or (result < 0 if zero else result <= 0):
        raise ValueError(f"{name} is out of range")
    return result


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _time(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric time ID")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = _finite(value, name)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result.copy()


def _norm_clip(value: np.ndarray, bound: float) -> np.ndarray:
    """Clip without overflowing even when finite coordinates are enormous."""
    largest = float(np.max(np.abs(value))) if value.size else 0.0
    if largest == 0:
        return value.copy()
    scaled = value / largest
    scaled_norm = float(np.sqrt(np.sum(scaled * scaled)))
    if largest <= bound / scaled_norm:
        return value.copy()
    return scaled * (bound / scaled_norm)


class TrainOnlyScaler:
    """A one-fit standardizer that rejects rows later than its declared cutoff.

    ``row_times`` labels the first array axis; all other axes except the final
    feature axis are pooled. Callers must independently enforce feature release
    times and overlapping-window information boundaries.
    """

    def __init__(self, scale_floor: float = 1e-8, clip: float = 8.0) -> None:
        self.scale_floor = _positive(scale_floor, "scale_floor")
        self.clip = _positive(clip, "clip")
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.train_end_: float | None = None
        self.rows_: int = 0

    def fit(self, x: Any, *, row_times: Any, train_end: Real) -> TrainOnlyScaler:
        if self.mean_ is not None:
            raise ValueError("scaler is frozen after one training fit")
        data = _finite(x, "x")
        if data.ndim < 2 or any(d == 0 for d in data.shape):
            raise ValueError("x needs nonempty row and feature axes")
        times = _array(row_times, (data.shape[0],), "row_times")
        cutoff = _time(train_end, "train_end")
        if np.any(times > cutoff):
            raise ValueError("normalization contains rows after train_end")
        # Rescaling first avoids overflow when fitting large finite inputs.
        flat = data.reshape(-1, data.shape[-1])
        magnitude = np.maximum(np.max(np.abs(flat), axis=0), 1.0)
        reduced = flat / magnitude
        mean = np.mean(reduced, axis=0) * magnitude
        scale = np.std(reduced, axis=0) * magnitude
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("normalization statistics overflowed")
        self.mean_ = mean
        self.scale_ = np.where(scale >= self.scale_floor, scale, 1.0)
        self.train_end_ = cutoff
        self.rows_ = data.shape[0]
        return self

    def transform(self, x: Any) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("scaler has not been fitted")
        data = _finite(x, "x")
        if data.ndim < 1 or data.shape[-1] != self.mean_.size:
            raise ValueError("x feature dimension does not match fitted scaler")
        # Divide first when subtraction could exceed the float64 range.
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (data - self.mean_) / self.scale_
            fallback = data / self.scale_ - self.mean_ / self.scale_
        normalized = np.where(np.isfinite(normalized), normalized, fallback)
        if np.any(np.isnan(normalized)):
            raise ValueError("normalization is numerically undefined")
        return np.clip(normalized, -self.clip, self.clip)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "scale_floor": self.scale_floor, "clip": self.clip,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "train_end": self.train_end_, "rows": self.rows_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TrainOnlyScaler:
        if state.get("version") != 1:
            raise ValueError("unsupported scaler state version")
        obj = cls(state["scale_floor"], state["clip"])
        if state["mean"] is None:
            if state["scale"] is not None or state["train_end"] is not None or state["rows"] != 0:
                raise ValueError("inconsistent unfitted scaler state")
            return obj
        mean = _finite(state["mean"], "mean")
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("mean must be a nonempty feature vector")
        scale = _array(state["scale"], mean.shape, "scale")
        if np.any(scale <= 0):
            raise ValueError("scale must be positive")
        obj.mean_, obj.scale_ = mean.copy(), scale
        obj.train_end_ = _time(state["train_end"], "train_end")
        obj.rows_ = _integer(state["rows"], "rows")
        return obj


@dataclass(frozen=True)
class ReservoirConfig:
    input_dim: int
    state_dim: int = 16
    leaks: tuple[float, ...] = (0.2, 0.6, 1.0)
    recurrent_norm: float = 0.8
    rff_dim: int = 16
    input_clip: float = 8.0
    seed: int = 11

    def __post_init__(self) -> None:
        _integer(self.input_dim, "input_dim")
        _integer(self.state_dim, "state_dim")
        _integer(self.rff_dim, "rff_dim", minimum=0)
        _integer(self.seed, "seed", minimum=0)
        rho = _positive(self.recurrent_norm, "recurrent_norm", zero=True)
        if rho >= 1:
            raise ValueError("recurrent_norm must be below one")
        _positive(self.input_clip, "input_clip")
        if not isinstance(self.leaks, (tuple, list)) or not self.leaks:
            raise ValueError("leaks must be a nonempty sequence")
        leaks = tuple(_positive(a, "leak") for a in self.leaks)
        if any(a > 1 for a in leaks):
            raise ValueError("each leak must be at most one")
        object.__setattr__(self, "leaks", leaks)


class ReservoirEngramKernel:
    """Fixed bounded multi-timescale features; each window starts at zero.

    Independent orthogonal recurrent matrices give a known spectral norm.
    Branches diverge through their matrices, leaks and input projections;
    products reconnect their final states without changing recurrence stability.
    """

    def __init__(self, config: ReservoirConfig) -> None:
        if not isinstance(config, ReservoirConfig):
            raise ValueError("config must be a ReservoirConfig")
        self.config = config
        rng = np.random.default_rng(config.seed)
        branches, width = len(config.leaks), config.state_dim
        matrices = []
        for _ in range(branches):
            q, r = np.linalg.qr(rng.normal(size=(width, width)))
            signs = np.where(np.diag(r) >= 0, 1.0, -1.0)
            matrices.append(q * signs * config.recurrent_norm)
        self.recurrent = np.stack(matrices)
        self.input_weights = rng.normal(size=(branches, width, config.input_dim)) / np.sqrt(config.input_dim)
        self.bias = rng.uniform(-0.1, 0.1, size=(branches, width))
        self.rff_weights = rng.normal(size=(branches * width, config.rff_dim))
        self.rff_phase = rng.uniform(0, 2 * np.pi, size=config.rff_dim)

    @property
    def output_dim(self) -> int:
        branches = len(self.config.leaks)
        return (self.config.input_dim + 2 * branches * self.config.state_dim
                + branches * (branches - 1) // 2 * self.config.state_dim + self.config.rff_dim)

    @property
    def contraction_bounds(self) -> np.ndarray:
        leaks = np.asarray(self.config.leaks)
        return 1 - leaks + leaks * self.config.recurrent_norm

    def step(self, x: Any, state: Any | None = None) -> np.ndarray:
        c = self.config
        data = _array(x, (c.input_dim,), "x")
        shape = (len(c.leaks), c.state_dim)
        previous = np.zeros(shape) if state is None else _array(state, shape, "state")
        if np.any(np.abs(previous) > 1 + 1e-12):
            raise ValueError("reservoir state must be inside [-1,1]")
        return self._step_batch(np.clip(data, -c.input_clip, c.input_clip)[None], previous[None])[0]

    def _step_batch(self, x: np.ndarray, state: np.ndarray) -> np.ndarray:
        leaks = np.asarray(self.config.leaks)[None, :, None]
        drive = np.einsum("bij,nbj->nbi", self.recurrent, state)
        drive += np.einsum("bif,nf->nbi", self.input_weights, x) + self.bias
        return (1 - leaks) * state + leaks * np.tanh(drive)

    def encode(self, windows: Any, *, context: str | int = "full") -> np.ndarray:
        data = _finite(windows, "windows")
        c = self.config
        if data.ndim != 3 or data.shape[-1] != c.input_dim or min(data.shape) == 0:
            raise ValueError("windows must be nonempty [rows,time,input_dim]")
        if isinstance(context, str):
            if context not in {"full", "zero"}:
                raise ValueError("context must be full, zero, or an integer")
            length = data.shape[1] if context == "full" else 0
        else:
            length = _integer(context, "context", minimum=0)
            if length > data.shape[1]:
                raise ValueError("context exceeds available window length")
        if length == 0:
            return np.zeros((data.shape[0], self.output_dim))
        data = np.clip(data[:, -length:], -c.input_clip, c.input_clip)
        states = np.zeros((data.shape[0], len(c.leaks), c.state_dim))
        averages = np.zeros_like(states)
        for t in range(length):
            states = self._step_batch(data[:, t], states)
            averages += (states - averages) / (t + 1)
        flat = states.reshape(data.shape[0], -1)
        pieces = [np.tanh(data[:, -1]), flat, averages.reshape(data.shape[0], -1)]
        pieces.extend(states[:, left] * states[:, right]
                      for left, right in combinations(range(len(c.leaks)), 2))
        if c.rff_dim:
            pieces.append(np.sqrt(2.0 / c.rff_dim) * np.cos(flat @ self.rff_weights + self.rff_phase))
        return np.concatenate(pieces, axis=1)

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "config": asdict(self.config),
                **{name: getattr(self, name).tolist() for name in
                   ("recurrent", "input_weights", "bias", "rff_weights", "rff_phase")}}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ReservoirEngramKernel:
        if state.get("version") != 1:
            raise ValueError("unsupported reservoir state version")
        obj = cls(ReservoirConfig(**state["config"]))
        for name in ("recurrent", "input_weights", "bias", "rff_weights", "rff_phase"):
            setattr(obj, name, _array(state[name], getattr(obj, name).shape, name))
        norms = np.linalg.norm(obj.recurrent, ord=2, axis=(1, 2))
        if np.any(norms > obj.config.recurrent_norm + 1e-12):
            raise ValueError("checkpoint violates the recurrence contraction bound")
        return obj


@dataclass(frozen=True)
class MemoryRead:
    values: np.ndarray
    support: np.ndarray
    effective_count: np.ndarray


class DelayedEngramMemory:
    """Bounded dated storage with strict label release and origin retention.

    Numeric times must use one common unit. ``recency_decay`` is inverse time
    in that unit. Pending episodes cannot evict mature episodes. Queries advance
    monotonically; released episodes join active storage, which keeps the latest
    ``capacity`` origins. Pending storage has a separate explicit hard bound.
    Memory support reports the eligible record count, not a causal-effect claim.
    """

    def __init__(self, key_dim: int, value_dim: int, capacity: int = 256,
                 bandwidth: float = 1.0, recency_decay: float = 0.0,
                 max_pending: int = 4096) -> None:
        self.key_dim = _integer(key_dim, "key_dim")
        self.value_dim = _integer(value_dim, "value_dim")
        self.capacity = _integer(capacity, "capacity")
        self.bandwidth = _positive(bandwidth, "bandwidth")
        self.recency_decay = _positive(recency_decay, "recency_decay", zero=True)
        self.max_pending = _integer(max_pending, "max_pending")
        self._records: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []
        self._query_time: float | None = None
        self._next_order = 0

    def add(self, key: Any, value: Any, *, origin_time: Real,
            maturity_time: Real, available_time: Real) -> None:
        key_arr = _array(key, (self.key_dim,), "key")
        value_arr = _array(value, (self.value_dim,), "value")
        origin = _time(origin_time, "origin_time")
        maturity = _time(maturity_time, "maturity_time")
        available = _time(available_time, "available_time")
        if maturity <= origin or available < maturity:
            raise ValueError("require origin < maturity <= availability")
        if self._query_time is not None and available < self._query_time:
            raise ValueError("cannot insert feedback already available before the query clock")
        if len(self._pending) >= self.max_pending:
            raise ValueError("pending engram capacity exceeded")
        self._pending.append({"key": key_arr, "value": value_arr, "origin_time": origin,
                              "maturity_time": maturity, "available_time": available,
                              "order": self._next_order})
        self._next_order += 1

    def retrieve(self, keys: Any, *, query_time: Real) -> MemoryRead:
        query = _finite(keys, "keys")
        if query.ndim != 2 or query.shape[1] != self.key_dim:
            raise ValueError("keys must have shape [rows,key_dim]")
        time = _time(query_time, "query_time")
        if self._query_time is not None and time < self._query_time:
            raise ValueError("memory query times must be chronological")
        released = [r for r in self._pending if r["origin_time"] < time
                    and r["maturity_time"] < time and r["available_time"] < time]
        eligible = sorted(self._records + released,
                          key=lambda r: (r["origin_time"], r["available_time"], r["order"]))[-self.capacity:]
        if not eligible:
            self._query_time = time
            return MemoryRead(np.zeros((query.shape[0], self.value_dim)),
                              np.zeros(query.shape[0], dtype=np.int64), np.zeros(query.shape[0]))
        keys_arr = np.stack([r["key"] for r in eligible])
        values = np.stack([r["value"] for r in eligible])
        # Row-wise distance calculation avoids a rows*memory*dimension allocation.
        weights = np.empty((query.shape[0], len(eligible)))
        dates = np.asarray([r["available_time"] for r in eligible])
        for index, row in enumerate(query):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                delta = (keys_arr - row) / self.bandwidth
                log_weights = -0.5 * np.sum(delta * delta, axis=1)
                log_weights -= self.recency_decay * (time - dates)
            if np.any(np.isnan(log_weights)) or not np.any(np.isfinite(log_weights)):
                raise ValueError("memory distances exceed numerical range")
            log_weights -= np.max(log_weights)
            weights[index] = np.exp(log_weights)
            weights[index] /= np.sum(weights[index])
        # Convex combinations computed after featurewise rescaling avoid overflow.
        magnitude = np.maximum(np.max(np.abs(values), axis=0), 1.0)
        result = (weights @ (values / magnitude)) * magnitude
        # Commit chronology/storage only after all arithmetic succeeds.
        self._records = eligible
        released_ids = {r["order"] for r in released}
        self._pending = [r for r in self._pending if r["order"] not in released_ids]
        self._query_time = time
        return MemoryRead(result, np.full(query.shape[0], len(eligible), dtype=np.int64),
                          1.0 / np.sum(weights * weights, axis=1))

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "key_dim": self.key_dim, "value_dim": self.value_dim,
                "capacity": self.capacity, "bandwidth": self.bandwidth,
                "recency_decay": self.recency_decay, "max_pending": self.max_pending,
                "query_time": self._query_time, "next_order": self._next_order,
                "records": [{**r, "key": r["key"].tolist(), "value": r["value"].tolist()}
                            for r in self._records],
                "pending": [{**r, "key": r["key"].tolist(), "value": r["value"].tolist()}
                            for r in self._pending]}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> DelayedEngramMemory:
        if state.get("version") != 1:
            raise ValueError("unsupported memory state version")
        obj = cls(**{name: state[name] for name in
                     ("key_dim", "value_dim", "capacity", "bandwidth", "recency_decay", "max_pending")})
        if len(state["records"]) > obj.capacity or len(state["pending"]) > obj.max_pending:
            raise ValueError("memory checkpoint exceeds capacity")
        clock = None if state["query_time"] is None else _time(state["query_time"], "query_time")
        next_order = _integer(state["next_order"], "next_order", minimum=0)
        seen = set()
        for group in ("records", "pending"):
            destination = []
            for record in state[group]:
                origin = _time(record["origin_time"], "origin_time")
                maturity = _time(record["maturity_time"], "maturity_time")
                available = _time(record["available_time"], "available_time")
                order = _integer(record["order"], "order", minimum=0)
                if not origin < maturity <= available or order in seen or order >= next_order:
                    raise ValueError("invalid memory chronology or duplicate record identity")
                if group == "records" and (clock is None or available >= clock):
                    raise ValueError("active memory contains unreleased feedback")
                if group == "pending" and clock is not None and available < clock:
                    raise ValueError("pending memory contains already released feedback")
                destination.append({"key": _array(record["key"], (obj.key_dim,), "key"),
                                    "value": _array(record["value"], (obj.value_dim,), "value"),
                                    "origin_time": origin, "maturity_time": maturity,
                                    "available_time": available, "order": order})
                seen.add(order)
            setattr(obj, "_records" if group == "records" else "_pending", destination)
        obj._query_time, obj._next_order = clock, next_order
        return obj


class QuantizedRippleUpdater:
    """Delayed finite-horizon error-feedback gradient descent.

    ``observe`` clips each gradient and enqueues weights proportional to
    decay**j at step+delay_steps+j. ``apply`` consumes all matured contributions
    once, rounds the sum plus previous residual to multiples of q, and applies
    the negative learning-rate-scaled update within max_update_norm. Skipping
    times catches up matured contributions; it does not synthesize new labels.
    """

    def __init__(self, shape: tuple[int, ...], learning_rate: float = 0.01,
                 quantization_step: float = 1e-3, delay_steps: int = 1,
                 ripple_steps: int = 3, decay: float = 0.5,
                 max_gradient_norm: float = 10.0, max_update_norm: float = 0.1,
                 max_pending: int = 4096) -> None:
        if not isinstance(shape, (tuple, list)) or not shape:
            raise ValueError("shape must be a nonempty sequence")
        self.shape = tuple(_integer(n, "shape dimension") for n in shape)
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.quantization_step = _positive(quantization_step, "quantization_step", zero=True)
        self.delay_steps = _integer(delay_steps, "delay_steps", minimum=0)
        self.ripple_steps = _integer(ripple_steps, "ripple_steps")
        self.decay = _positive(decay, "decay", zero=True)
        if self.decay > 1:
            raise ValueError("decay must be at most one")
        self.max_gradient_norm = _positive(max_gradient_norm, "max_gradient_norm")
        self.max_update_norm = _positive(max_update_norm, "max_update_norm")
        self.max_pending = _integer(max_pending, "max_pending")
        if self.max_pending < self.ripple_steps:
            raise ValueError("max_pending must admit one full ripple schedule")
        self.residual = np.zeros(self.shape)
        self._pending: list[tuple[int, np.ndarray]] = []
        self._clock = -1
        self._last_applied = -1

    def observe(self, gradient: Any, *, step: int) -> None:
        time = _integer(step, "step", minimum=0)
        if time < self._clock or time <= self._last_applied:
            raise ValueError("ripple operations must be chronological")
        grad = _array(gradient, self.shape, "gradient")
        if len(self._pending) + self.ripple_steps > self.max_pending:
            raise ValueError("ripple pending capacity exceeded")
        grad = _norm_clip(grad, self.max_gradient_norm)
        weights = self.decay ** np.arange(self.ripple_steps, dtype=np.float64)
        weights /= np.sum(weights)
        self._pending.extend((time + self.delay_steps + j, grad * w) for j, w in enumerate(weights))
        self._clock = time

    def apply(self, parameters: Any, *, step: int) -> np.ndarray:
        time = _integer(step, "step", minimum=0)
        if time < self._clock or time <= self._last_applied:
            raise ValueError("ripple apply times must advance chronologically")
        current = _array(parameters, self.shape, "parameters")
        matured = [gradient for due, gradient in self._pending if due <= time]
        if matured:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                raw = np.sum(matured, axis=0) + self.residual
                quantized = (np.rint(raw / self.quantization_step) * self.quantization_step
                             if self.quantization_step > 0 else raw.copy())
                residual = raw - quantized
                increment = self.learning_rate * quantized
            if not all(np.all(np.isfinite(x)) for x in (quantized, residual, increment)):
                raise ValueError("ripple arithmetic exceeds numerical range")
            increment = _norm_clip(increment, self.max_update_norm)
            with np.errstate(over="ignore", invalid="ignore"):
                updated = current - increment
            if not np.all(np.isfinite(updated)):
                raise ValueError("parameter update exceeds numerical range")
            self.residual = residual
            self._pending = [(due, gradient) for due, gradient in self._pending if due > time]
        else:
            updated = current
        self._last_applied = self._clock = time
        return updated

    def state_dict(self) -> dict[str, Any]:
        names = ("shape", "learning_rate", "quantization_step", "delay_steps", "ripple_steps",
                 "decay", "max_gradient_norm", "max_update_norm", "max_pending")
        return {"version": 1, "config": {name: getattr(self, name) for name in names},
                "residual": self.residual.tolist(), "clock": self._clock,
                "last_applied": self._last_applied,
                "pending": [{"due": due, "gradient": grad.tolist()} for due, grad in self._pending]}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> QuantizedRippleUpdater:
        if state.get("version") != 1:
            raise ValueError("unsupported ripple state version")
        obj = cls(**state["config"])
        residual = _array(state["residual"], obj.shape, "residual")
        if (np.any(np.abs(residual) > obj.quantization_step / 2 + 1e-12)
                or (obj.quantization_step == 0 and np.any(residual != 0))):
            raise ValueError("quantization residual exceeds q/2")
        clock = _integer(state["clock"], "clock", minimum=-1)
        last = _integer(state["last_applied"], "last_applied", minimum=-1)
        if last > clock or len(state["pending"]) > obj.max_pending:
            raise ValueError("inconsistent ripple clock or pending capacity")
        pending = []
        for record in state["pending"]:
            due = _integer(record["due"], "due", minimum=0)
            grad = _array(record["gradient"], obj.shape, "pending gradient")
            if due <= last or not np.allclose(grad, _norm_clip(grad, obj.max_gradient_norm),
                                             rtol=1e-12, atol=1e-12):
                raise ValueError("invalid pending ripple contribution")
            pending.append((due, grad))
        obj.residual, obj._pending = residual, pending
        obj._clock, obj._last_applied = clock, last
        return obj
