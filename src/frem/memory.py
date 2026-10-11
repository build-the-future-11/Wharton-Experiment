"""Bounded episodic memory with strict outcome-availability semantics.

Records are immutable-key, matured return/residual episodes. Utility credit is
attached only after an episode's realized outcome becomes available. Retrieval
is read-only so frozen evaluation does not silently adapt memory.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

DAY_NS = 86_400_000_000_000
MEMORY_FEATURES = 7


@dataclass
class MemoryRecord:
    identifier: int
    key: list[float]
    asset_id: int
    decision_time: int
    label_time: int
    outcome: float
    residual: float
    ngram_code: int
    novelty: float = 0.0
    utility: float = 0.0
    credited_uses: int = 0


@dataclass
class PendingOutcome:
    key: list[float]
    asset_id: int
    decision_time: int
    label_time: int
    base_prediction: float
    prediction: float
    memory_prediction: float
    scale: float
    ngram_code: int
    novelty: float
    outcome: float | None
    retrieved_ids: list[int] = field(default_factory=list)
    retrieved_weights: list[float] = field(default_factory=list)


@dataclass
class Retrieval:
    # residual mean, outcome mean, residual std, similarity, log(1+count),
    # credited utility mean, age/(half-life+age).
    features: np.ndarray
    identifiers: list[int]
    weights: list[float]


class EngramMemory:
    def __init__(
        self,
        *,
        key_dim: int,
        capacity: int = 512,
        neighbors: int = 8,
        same_asset: bool = True,
        exact_ngram: bool = False,
        retention: str = "utility",
        utility_rate: float = 0.1,
        half_life_days: float = 180.0,
        temperature: float = 0.2,
        pending_capacity: int = 100_000,
        surprise_threshold: float = 0.0,
        seed: int = 0,
    ) -> None:
        if min(key_dim, capacity, neighbors, pending_capacity) < 1:
            raise ValueError("Memory dimensions and bounds must be positive")
        if retention not in {"utility", "fifo", "random", "uniform"}:
            raise ValueError("Unknown retention policy")
        if not 0 < utility_rate <= 1 or half_life_days <= 0 or temperature <= 0:
            raise ValueError("Memory learning/decay/temperature parameters are invalid")
        if surprise_threshold < 0:
            raise ValueError("surprise_threshold must be nonnegative")
        self.parameters = {
            "key_dim": key_dim,
            "capacity": capacity,
            "neighbors": neighbors,
            "same_asset": same_asset,
            "exact_ngram": exact_ngram,
            "retention": retention,
            "utility_rate": utility_rate,
            "half_life_days": half_life_days,
            "temperature": temperature,
            "pending_capacity": pending_capacity,
            "surprise_threshold": surprise_threshold,
            "seed": seed,
        }
        self.key_dim = key_dim
        self.capacity = capacity
        self.neighbors = neighbors
        self.same_asset = same_asset
        self.exact_ngram = exact_ngram
        self.retention = retention
        self.utility_rate = utility_rate
        self.half_life_days = half_life_days
        self.temperature = temperature
        self.pending_capacity = pending_capacity
        self.surprise_threshold = surprise_threshold
        self.records: dict[int, MemoryRecord] = {}
        self.pending: list[PendingOutcome] = []
        self.next_identifier = 0
        # UTC nanoseconds before 1970 are negative and remain valid history.
        self.last_advance_time = -(1 << 63)
        self.total_matured = 0
        self.total_evicted = 0
        self.total_write_filtered = 0
        self.random = np.random.default_rng(seed)

    def _key(self, key: Any) -> np.ndarray:
        array = np.asarray(key, dtype=np.float64)
        if array.shape != (self.key_dim,) or not np.isfinite(array).all():
            raise ValueError(f"Memory key must be finite with shape ({self.key_dim},)")
        norm = float(np.linalg.norm(array))
        return array / max(norm, 1e-12)

    def retrieve(self, key: Any, *, asset_id: int, decision_time: int, ngram_code: int = 0) -> Retrieval:
        if decision_time < self.last_advance_time:
            raise ValueError(
                "Cannot query a historical time after memory utility has advanced beyond it; restore an earlier checkpoint"
            )
        query = self._key(key)
        eligible = [
            record
            for record in self.records.values()
            if record.label_time < decision_time
            and record.decision_time < decision_time
            and (not self.same_asset or record.asset_id == asset_id)
            and (not self.exact_ngram or record.ngram_code == ngram_code)
        ]
        if not eligible:
            return Retrieval(np.zeros(MEMORY_FEATURES, dtype=np.float32), [], [])
        similarities = np.asarray([np.dot(query, self._key(record.key)) for record in eligible])
        # Stable ordering gives reproducible ties independently of container order.
        ordered = sorted(
            range(len(eligible)), key=lambda index: (-similarities[index], eligible[index].identifier)
        )
        chosen = ordered[: self.neighbors]
        records = [eligible[index] for index in chosen]
        similarity = similarities[chosen]
        ages = np.asarray([(decision_time - record.label_time) / DAY_NS for record in records])
        utility = np.asarray([record.utility for record in records])
        logits = similarity / self.temperature - math.log(2) * ages / self.half_life_days
        # Keep retrieval weighting identical across eviction policies so a
        # utility-vs-FIFO/random comparison isolates the retention decision.
        logits += np.tanh(utility)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        residuals = np.asarray([record.residual for record in records])
        outcomes = np.asarray([record.outcome for record in records])
        mean_residual = float(weights @ residuals)
        residual_std = float(np.sqrt(max(float(weights @ np.square(residuals - mean_residual)), 0)))
        mean_age = float(weights @ ages)
        features = np.asarray(
            [
                mean_residual,
                weights @ outcomes,
                residual_std,
                weights @ similarity,
                math.log1p(len(records)),
                weights @ utility,
                mean_age / (self.half_life_days + mean_age),
            ],
            dtype=np.float32,
        )
        return Retrieval(features, [record.identifier for record in records], weights.tolist())

    def schedule(
        self,
        *,
        key: Any,
        asset_id: int,
        decision_time: int,
        label_time: int,
        base_prediction: float,
        prediction: float,
        memory_prediction: float,
        scale: float,
        ngram_code: int = 0,
        novelty: float = 0.0,
        outcome: float | None = None,
        retrieval: Retrieval | None = None,
    ) -> None:
        if label_time <= decision_time:
            raise ValueError("The outcome must become available after its prediction decision")
        if decision_time < self.last_advance_time:
            raise ValueError("Cannot schedule an event behind the memory clock")
        if len(self.pending) >= self.pending_capacity:
            raise RuntimeError(
                "Pending outcome capacity exhausted; supply matured feedback or increase the explicit bound"
            )
        if any(event.decision_time == decision_time and event.asset_id == asset_id for event in self.pending):
            raise ValueError("Duplicate pending (decision_time, asset_id)")
        values = (base_prediction, prediction, memory_prediction, scale, novelty)
        if not all(math.isfinite(value) for value in values) or scale <= 0:
            raise ValueError("Pending predictions and uncertainty must be finite, with positive scale")
        if outcome is not None and not math.isfinite(outcome):
            raise ValueError("Use outcome=None for an unavailable live outcome")
        if retrieval is None:
            retrieval = Retrieval(np.zeros(MEMORY_FEATURES, dtype=np.float32), [], [])
        if len(retrieval.identifiers) != len(retrieval.weights):
            raise ValueError("Retrieved identifiers and credit weights must align")
        if len(retrieval.identifiers) > self.neighbors or len(set(retrieval.identifiers)) != len(
            retrieval.identifiers
        ):
            raise ValueError(
                "Retrieved credit metadata must respect the neighbor bound and have unique identifiers"
            )
        if retrieval.weights and (
            not all(math.isfinite(weight) and weight >= 0 for weight in retrieval.weights)
            or not math.isclose(sum(retrieval.weights), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("Retrieved credit weights must be a normalized finite distribution")
        self.pending.append(
            PendingOutcome(
                key=self._key(key).tolist(),
                asset_id=int(asset_id),
                decision_time=int(decision_time),
                label_time=int(label_time),
                base_prediction=float(base_prediction),
                prediction=float(prediction),
                memory_prediction=float(memory_prediction),
                scale=float(scale),
                ngram_code=int(ngram_code),
                novelty=float(novelty),
                outcome=None if outcome is None else float(outcome),
                retrieved_ids=list(retrieval.identifiers),
                retrieved_weights=list(retrieval.weights),
            )
        )

    def provide_feedback(
        self, *, decision_time: int, asset_id: int, outcome: float, available_at: int, as_of_time: int
    ) -> None:
        if available_at > as_of_time:
            raise ValueError("Live feedback cannot be supplied before its publication")
        if not math.isfinite(outcome):
            raise ValueError("Observed outcome must be finite")
        for event in self.pending:
            if event.decision_time == decision_time and event.asset_id == asset_id:
                if event.label_time != np.iinfo(np.int64).max and available_at < event.label_time:
                    raise ValueError("Reported availability precedes the declared label maturity")
                if event.outcome is not None and not math.isclose(event.outcome, outcome, abs_tol=1e-12):
                    raise ValueError("Conflicting outcome correction requires a new audited replay")
                event.outcome = float(outcome)
                event.label_time = int(available_at)
                return
        raise KeyError("No matching pending prediction for this feedback")

    def _credit(self, event: PendingOutcome) -> None:
        assert event.outcome is not None
        baseline_error = (event.outcome - event.base_prediction) ** 2
        memory_error = (event.outcome - event.prediction) ** 2
        improvement = (baseline_error - memory_error) / (event.scale**2 + baseline_error + 1e-12)
        improvement = float(np.clip(improvement, -1, 1))
        for identifier, weight in zip(event.retrieved_ids, event.retrieved_weights, strict=True):
            record = self.records.get(identifier)
            if record is None:
                continue  # A finite-capacity record may be evicted before feedback.
            rate = self.utility_rate * float(weight)
            record.utility = (1 - rate) * record.utility + rate * improvement
            record.credited_uses += 1

    def _retention_score(self, record: MemoryRecord, as_of_time: int) -> float:
        age = max(0, as_of_time - record.label_time) / DAY_NS
        return (
            math.tanh(record.utility)
            + 0.1 * math.tanh(record.novelty)
            + 0.03 * math.log1p(record.credited_uses)
            - math.log(2) * age / self.half_life_days
        )

    def _insert(self, event: PendingOutcome, as_of_time: int) -> None:
        assert event.outcome is not None
        if event.novelty < self.surprise_threshold:
            self.total_write_filtered += 1
            return
        identifier = self.next_identifier
        self.next_identifier += 1
        self.records[identifier] = MemoryRecord(
            identifier,
            event.key,
            event.asset_id,
            event.decision_time,
            event.label_time,
            event.outcome,
            event.outcome - event.base_prediction,
            event.ngram_code,
            event.novelty,
        )
        if len(self.records) <= self.capacity:
            return
        if self.retention == "fifo":
            remove = min(self.records, key=lambda item: (self.records[item].label_time, item))
        elif self.retention in {"random", "uniform"}:
            # Uniform eviction among the current capacity+1 candidates. This is
            # not uniform sampling of the complete historical stream.
            remove = int(self.random.choice(sorted(self.records)))
        else:
            remove = min(
                self.records, key=lambda item: (self._retention_score(self.records[item], as_of_time), item)
            )
        del self.records[remove]
        self.total_evicted += 1

    def advance(self, as_of_time: int) -> list[PendingOutcome]:
        if as_of_time < self.last_advance_time:
            raise ValueError("Online memory clock cannot move backwards")
        ready = [
            event for event in self.pending if event.label_time < as_of_time and event.outcome is not None
        ]
        waiting = [
            event
            for event in self.pending
            if not (event.label_time < as_of_time and event.outcome is not None)
        ]
        ready.sort(key=lambda event: (event.label_time, event.decision_time, event.asset_id))
        for event in ready:
            self._credit(event)
            self._insert(event, as_of_time)
        self.pending = waiting
        self.total_matured += len(ready)
        self.last_advance_time = int(as_of_time)
        return ready

    def diagnostics(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "occupancy": len(self.records),
            "pending": len(self.pending),
            "total_matured": self.total_matured,
            "total_evicted": self.total_evicted,
            "total_write_filtered": self.total_write_filtered,
            "credited_uses": sum(record.credited_uses for record in self.records.values()),
            "mean_utility": float(np.mean([record.utility for record in self.records.values()]))
            if self.records
            else 0.0,
            "policy": self.retention,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "records": [asdict(record) for record in self.records.values()],
            "pending": [asdict(event) for event in self.pending],
            "next_identifier": self.next_identifier,
            "last_advance_time": self.last_advance_time,
            "total_matured": self.total_matured,
            "total_evicted": self.total_evicted,
            "total_write_filtered": self.total_write_filtered,
            "rng_state": self.random.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> EngramMemory:
        memory = cls(**state["parameters"])
        memory.records = {row["identifier"]: MemoryRecord(**row) for row in state["records"]}
        memory.pending = [PendingOutcome(**row) for row in state["pending"]]
        if len(memory.records) > memory.capacity or len(memory.pending) > memory.pending_capacity:
            raise ValueError("Checkpoint memory exceeds declared capacity")
        for record in memory.records.values():
            memory._key(record.key)
        for name in (
            "next_identifier",
            "last_advance_time",
            "total_matured",
            "total_evicted",
            "total_write_filtered",
        ):
            setattr(memory, name, int(state[name]))
        memory.random.bit_generator.state = state["rng_state"]
        return memory
