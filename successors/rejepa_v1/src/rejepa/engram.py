"""Immutable, delayed-credit spectral engram memory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EngramRecord:
    identifier: int
    key: torch.Tensor
    value: torch.Tensor
    decision_time_ns: int
    outcome_available_at_ns: int
    recorded_at_ns: int
    utility: float


class SpectralEngramMemory:
    def __init__(self, key_dim: int, value_dim: int, capacity: int = 512, temperature: float = 0.2) -> None:
        if min(key_dim, value_dim, capacity) < 1:
            raise ValueError("Memory dimensions and capacity must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        self.temperature = temperature
        self.records: list[EngramRecord] = []
        self._next_identifier = 0

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        decision_time_ns: int,
        outcome_available_at_ns: int,
        recorded_at_ns: int,
        utility: float,
    ) -> int:
        if key.shape != (self.key_dim,) or value.shape != (self.value_dim,):
            raise ValueError("Engram key or value has the wrong shape")
        if outcome_available_at_ns <= decision_time_ns:
            raise ValueError("An outcome must become available strictly after its decision")
        if recorded_at_ns < outcome_available_at_ns:
            raise ValueError("An engram cannot be recorded before its outcome is available")
        if not math.isfinite(utility):
            raise ValueError("Utility must be finite")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("Engram tensors must be finite")
        identifier = self._next_identifier
        self._next_identifier += 1
        self.records.append(
            EngramRecord(
                identifier=identifier,
                key=key.detach().cpu().clone(),
                value=value.detach().cpu().clone(),
                decision_time_ns=int(decision_time_ns),
                outcome_available_at_ns=int(outcome_available_at_ns),
                recorded_at_ns=int(recorded_at_ns),
                utility=float(utility),
            )
        )
        if len(self.records) > self.capacity:
            self.records.sort(key=lambda row: (row.utility, row.outcome_available_at_ns, row.identifier))
            self.records = self.records[-self.capacity :]
        return identifier

    def retrieve(
        self,
        query: torch.Tensor,
        as_of_ns: int,
        neighbors: int = 8,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if query.ndim != 2 or query.shape[-1] != self.key_dim:
            raise ValueError("Expected query [batch, key_dim]")
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        eligible = [row for row in self.records if row.outcome_available_at_ns <= as_of_ns]
        if not eligible:
            return torch.zeros(query.shape[0], self.value_dim, device=query.device, dtype=query.dtype), ()
        keys = torch.stack([row.key for row in eligible]).to(device=query.device, dtype=query.dtype)
        values = torch.stack([row.value for row in eligible]).to(device=query.device, dtype=query.dtype)
        utility = torch.tensor(
            [max(row.utility, 0.0) for row in eligible], device=query.device, dtype=query.dtype
        )
        similarity = F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).T
        similarity = similarity + torch.log1p(utility).unsqueeze(0)
        count = min(neighbors, len(eligible))
        scores, indices = torch.topk(similarity, k=count, dim=-1)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        selected = values[indices]
        retrieved = (weights.unsqueeze(-1) * selected).sum(dim=1)
        identifiers = tuple(sorted({eligible[index].identifier for index in indices.flatten().tolist()}))
        return retrieved, identifiers

    def export(self) -> dict[str, object]:
        return {
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "capacity": self.capacity,
            "temperature": self.temperature,
            "next_identifier": self._next_identifier,
            "records": [
                {
                    "identifier": row.identifier,
                    "key": row.key.tolist(),
                    "value": row.value.tolist(),
                    "decision_time_ns": row.decision_time_ns,
                    "outcome_available_at_ns": row.outcome_available_at_ns,
                    "recorded_at_ns": row.recorded_at_ns,
                    "utility": row.utility,
                }
                for row in self.records
            ],
        }

    @classmethod
    def from_export(cls, payload: dict[str, object]) -> SpectralEngramMemory:
        memory = cls(
            key_dim=int(payload["key_dim"]),
            value_dim=int(payload["value_dim"]),
            capacity=int(payload["capacity"]),
            temperature=float(payload["temperature"]),
        )
        records = payload.get("records")
        if not isinstance(records, list):
            raise TypeError("Engram export records must be a list")
        seen: set[int] = set()
        for item in records:
            if not isinstance(item, dict):
                raise TypeError("Engram export record must be an object")
            identifier = int(item["identifier"])
            if identifier in seen:
                raise ValueError("Engram identifiers must be unique")
            seen.add(identifier)
            key = torch.as_tensor(item["key"], dtype=torch.float32)
            value = torch.as_tensor(item["value"], dtype=torch.float32)
            if key.shape != (memory.key_dim,) or value.shape != (memory.value_dim,):
                raise ValueError("Engram export tensor has the wrong shape")
            record = EngramRecord(
                identifier=identifier,
                key=key,
                value=value,
                decision_time_ns=int(item["decision_time_ns"]),
                outcome_available_at_ns=int(item["outcome_available_at_ns"]),
                recorded_at_ns=int(item["recorded_at_ns"]),
                utility=float(item["utility"]),
            )
            if record.outcome_available_at_ns <= record.decision_time_ns:
                raise ValueError("Exported outcome maturity is invalid")
            if record.recorded_at_ns < record.outcome_available_at_ns:
                raise ValueError("Exported record predates outcome availability")
            if (
                not math.isfinite(record.utility)
                or not torch.isfinite(key).all()
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Exported engram contains non-finite values")
            memory.records.append(record)
        next_identifier = int(payload.get("next_identifier", 0))
        minimum_next = max(seen, default=-1) + 1
        if next_identifier < minimum_next:
            raise ValueError("next_identifier would overwrite an existing record")
        memory._next_identifier = next_identifier
        if len(memory.records) > memory.capacity:
            raise ValueError("Export contains more records than capacity")
        return memory
