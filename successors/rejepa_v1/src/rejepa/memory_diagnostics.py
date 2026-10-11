"""Diagnostics for temporal engram retrieval without mutating memory."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .engram import SpectralEngramMemory


def memory_diagnostics(
    memory: SpectralEngramMemory,
    query: torch.Tensor,
    *,
    as_of_ns: int,
    rare_identifiers: set[int] | None = None,
) -> dict[str, float]:
    if query.ndim != 2 or query.shape[-1] != memory.key_dim:
        raise ValueError("Query width does not match memory")
    eligible = [record for record in memory.records if record.outcome_available_at_ns <= as_of_ns]
    if not eligible:
        return {
            "eligible_records": 0.0,
            "retrieval_entropy": 0.0,
            "key_redundancy": 0.0,
            "rare_regime_recall": 0.0,
        }
    keys = torch.stack([record.key for record in eligible]).to(query)
    similarity = F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).T
    weights = torch.softmax(similarity / memory.temperature, dim=-1)
    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1).mean()
    normalized = F.normalize(keys, dim=-1)
    gram = normalized @ normalized.T
    mask = torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)
    redundancy = gram[mask].abs().mean() if mask.any() else torch.tensor(0.0, device=query.device)
    recalled = set()
    top = weights.argmax(dim=-1).tolist()
    for index in top:
        recalled.add(eligible[index].identifier)
    rare = rare_identifiers or set()
    recall = len(recalled & rare) / len(rare) if rare else 0.0
    return {
        "eligible_records": float(len(eligible)),
        "retrieval_entropy": float(entropy.item()),
        "key_redundancy": float(redundancy.item()),
        "rare_regime_recall": float(recall),
        "retrieved_unique": float(len(recalled)),
        "capacity_utilization": float(len(memory.records) / memory.capacity),
    }


def utility_forgetting_candidates(memory: SpectralEngramMemory, count: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    ordered = sorted(
        memory.records,
        key=lambda record: (record.utility, record.outcome_available_at_ns, record.identifier),
    )
    return tuple(record.identifier for record in ordered[: min(count, len(ordered))])


def retrieval_precision(retrieved: list[int], relevant: set[int]) -> float:
    if not retrieved:
        return 0.0
    return float(np.mean([identifier in relevant for identifier in retrieved]))
