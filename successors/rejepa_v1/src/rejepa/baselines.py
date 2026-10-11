"""Simple forecasting and allocation baselines that every complex model must beat fairly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge


@dataclass
class FittedBaseline:
    name: str
    model: object

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(features, dtype=float)), dtype=float)


@dataclass(frozen=True)
class RetrievalPrediction:
    """Prediction and auditable record identifiers from a causal retrieval control."""

    values: np.ndarray
    retrieved_identifiers: tuple[tuple[str, ...], ...]


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    norms = np.linalg.norm(array, axis=1)
    if (norms <= 0).any():
        raise ValueError(f"{name} rows must have positive norm")
    return array / norms[:, None]


def _integer_times(values: np.ndarray, length: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (length,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain one integer timestamp per row")
    return raw.astype(np.int64, copy=True)


class CausalContinuationRetriever:
    """RAFT-style nearest-history continuation control with maturity gating.

    This is a compact causal control, not a claim of reproducing every detail of
    RAFT. Records are immutable after fitting and a query may retrieve only
    continuations whose outcomes were available by its decision timestamp.
    """

    def __init__(
        self,
        keys: np.ndarray,
        continuations: np.ndarray,
        outcome_available_ns: np.ndarray,
        *,
        identifiers: tuple[str, ...] | None = None,
        neighbors: int = 8,
        temperature: float = 0.1,
    ) -> None:
        self._keys: Final = _matrix(keys, "keys").copy()
        count = len(self._keys)
        targets = np.asarray(continuations, dtype=float)
        if targets.shape != (count,) or not np.isfinite(targets).all():
            raise ValueError("continuations must be one finite value per key")
        self._continuations: Final = targets.copy()
        self._outcome_available_ns: Final = _integer_times(
            outcome_available_ns, count, "outcome_available_ns"
        )
        if identifiers is None:
            identifiers = tuple(f"record-{index}" for index in range(count))
        if (
            len(identifiers) != count
            or len(set(identifiers)) != count
            or any(not item for item in identifiers)
        ):
            raise ValueError("identifiers must be unique, nonempty and aligned")
        if not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self._identifiers: Final = tuple(identifiers)
        self.neighbors: Final = neighbors
        self.temperature: Final = float(temperature)
        for array in (self._keys, self._continuations, self._outcome_available_ns):
            array.flags.writeable = False

    def predict(self, query_keys: np.ndarray, as_of_ns: np.ndarray) -> RetrievalPrediction:
        queries = _matrix(query_keys, "query_keys")
        if queries.shape[1] != self._keys.shape[1]:
            raise ValueError("query key width does not match fitted keys")
        decisions = _integer_times(as_of_ns, len(queries), "as_of_ns")
        predictions = np.zeros(len(queries), dtype=float)
        retrieved: list[tuple[str, ...]] = []
        for row, (query, decision) in enumerate(zip(queries, decisions, strict=True)):
            eligible = np.flatnonzero(self._outcome_available_ns <= decision)
            if not len(eligible):
                retrieved.append(())
                continue
            similarities = self._keys[eligible] @ query
            order = np.argsort(-similarities, kind="stable")[: self.neighbors]
            selected = eligible[order]
            logits = similarities[order] / self.temperature
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum()
            predictions[row] = float(weights @ self._continuations[selected])
            retrieved.append(tuple(self._identifiers[index] for index in selected))
        return RetrievalPrediction(predictions, tuple(retrieved))


@dataclass(frozen=True)
class ResidualAlignment:
    """Frozen scalar learned on development/validation residual corrections."""

    coefficient: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.coefficient):
            raise ValueError("coefficient must be finite")

    @classmethod
    def fit(
        cls,
        base_predictions: np.ndarray,
        raw_corrections: np.ndarray,
        outcomes: np.ndarray,
        *,
        ridge: float = 1e-6,
    ) -> ResidualAlignment:
        base = np.asarray(base_predictions, dtype=float)
        correction = np.asarray(raw_corrections, dtype=float)
        target = np.asarray(outcomes, dtype=float)
        if (
            base.ndim != 1
            or not len(base)
            or correction.shape != base.shape
            or target.shape != base.shape
            or not np.isfinite(np.concatenate([base, correction, target])).all()
        ):
            raise ValueError("Calibration arrays must be finite aligned vectors")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        denominator = float(correction @ correction + ridge)
        coefficient = 0.0 if denominator == 0 else float(correction @ (target - base) / denominator)
        return cls(coefficient)

    def apply(self, base_predictions: np.ndarray, raw_corrections: np.ndarray) -> np.ndarray:
        base = np.asarray(base_predictions, dtype=float)
        correction = np.asarray(raw_corrections, dtype=float)
        if base.shape != correction.shape or not np.isfinite(np.concatenate([base, correction])).all():
            raise ValueError("Prediction arrays must be finite and aligned")
        return base + self.coefficient * correction


class CausalStreamingResidualMemory:
    """FreshCast-style streaming residual control with explicit outcome maturity."""

    def __init__(self, *, neighbors: int = 8, temperature: float = 0.1) -> None:
        if not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.neighbors = neighbors
        self.temperature = float(temperature)
        self._records: list[tuple[str, np.ndarray, float, int, int]] = []
        self._identifiers: set[str] = set()
        self._key_width: int | None = None

    def add(
        self,
        identifier: str,
        key: np.ndarray,
        *,
        base_prediction: float,
        outcome: float,
        outcome_available_ns: int,
        recorded_at_ns: int,
    ) -> None:
        vector = np.asarray(key, dtype=float)
        norm = float(np.linalg.norm(vector))
        scalars = np.asarray([base_prediction, outcome], dtype=float)
        if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all() or norm <= 0:
            raise ValueError("key must be a nonempty finite vector with positive norm")
        if not np.isfinite(scalars).all():
            raise ValueError("prediction and outcome must be finite")
        if not identifier or identifier in self._identifiers:
            raise ValueError("identifier must be unique and nonempty")
        if not isinstance(outcome_available_ns, (int, np.integer)) or not isinstance(
            recorded_at_ns, (int, np.integer)
        ):
            raise TypeError("timestamps must be integers")
        if outcome_available_ns > recorded_at_ns:
            raise ValueError("an outcome cannot enter memory before it is available")
        if self._records and recorded_at_ns < self._records[-1][4]:
            raise ValueError("streaming records must be appended in recorded-time order")
        if self._key_width is not None and len(vector) != self._key_width:
            raise ValueError("all memory keys must have the same width")
        normalized = (vector / norm).copy()
        normalized.flags.writeable = False
        self._records.append(
            (
                identifier,
                normalized,
                float(outcome - base_prediction),
                int(outcome_available_ns),
                int(recorded_at_ns),
            )
        )
        self._identifiers.add(identifier)
        self._key_width = len(vector)

    def raw_correction(self, query_keys: np.ndarray, as_of_ns: np.ndarray) -> RetrievalPrediction:
        queries = _matrix(query_keys, "query_keys")
        decisions = _integer_times(as_of_ns, len(queries), "as_of_ns")
        values = np.zeros(len(queries), dtype=float)
        identifiers: list[tuple[str, ...]] = []
        for row, (query, decision) in enumerate(zip(queries, decisions, strict=True)):
            eligible = [record for record in self._records if record[3] <= decision and record[4] <= decision]
            if not eligible:
                identifiers.append(())
                continue
            if any(len(record[1]) != len(query) for record in eligible):
                raise ValueError("query key width does not match memory keys")
            similarities = np.asarray([record[1] @ query for record in eligible])
            order = np.argsort(-similarities, kind="stable")[: self.neighbors]
            logits = similarities[order] / self.temperature
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum()
            values[row] = float(weights @ np.asarray([eligible[index][2] for index in order]))
            identifiers.append(tuple(eligible[index][0] for index in order))
        return RetrievalPrediction(values, tuple(identifiers))


def fit_linear_baseline(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    kind: str = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> FittedBaseline:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite aligned features and targets are required")
    if kind == "ridge":
        model = Ridge(alpha=alpha)
    elif kind == "elastic_net":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000)
    else:
        raise ValueError("kind must be ridge or elastic_net")
    model.fit(x, y)
    return FittedBaseline(kind, model)


def momentum_signal(prices: np.ndarray, lookback: int) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or lookback < 1 or values.shape[0] <= lookback or (values <= 0).any():
        raise ValueError("Price panel or lookback is invalid")
    return values[lookback:] / values[:-lookback] - 1


def mean_reversion_signal(prices: np.ndarray, lookback: int) -> np.ndarray:
    return -momentum_signal(prices, lookback)


def equal_weight(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    return np.full(count, 1 / count)


def inverse_volatility_weight(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.allclose(cov, cov.T):
        raise ValueError("covariance must be square and symmetric")
    volatility = np.sqrt(np.diag(cov))
    if (volatility <= 0).any() or not np.isfinite(volatility).all():
        raise ValueError("volatility must be positive and finite")
    inverse = 1 / volatility
    return inverse / inverse.sum()
