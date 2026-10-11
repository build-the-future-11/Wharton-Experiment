"""Immutable horizon-matured spectral engram memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EngramRecord:
    asset_id: str
    horizon: int
    branch: int
    origin_ns: int
    available_ns: int
    context: torch.Tensor
    spectrum: torch.Tensor
    value: torch.Tensor
    record_id: str

    def __post_init__(self) -> None:
        if self.branch not in (0, 1, 2):
            raise ValueError("Engram branch must be 0, 1, or 2")
        if self.available_ns < self.origin_ns:
            raise ValueError("Engram outcome cannot be available before its origin")
        if (
            self.context.shape != (168,)
            or self.spectrum.shape != (32,)
            or self.value.shape != (35,)
        ):
            raise ValueError("Engram record violates the manuscript dimension contract")
        if any(tensor.requires_grad for tensor in (self.context, self.spectrum, self.value)):
            raise ValueError("Stored memory tensors must be detached")


class EngramBank:
    def __init__(self, *, max_records: int = 1024, frozen: bool = False) -> None:
        self.max_records = max_records
        self.frozen = frozen
        self._records: dict[tuple[str, int, int], list[EngramRecord]] = {}

    def insert(self, record: EngramRecord) -> None:
        if self.frozen:
            raise RuntimeError("Evaluation memory is frozen")
        key = (record.asset_id, record.horizon, record.branch)
        records = self._records.setdefault(key, [])
        if records and record.origin_ns <= records[-1].origin_ns:
            raise ValueError("Engram origins must be strictly increasing")
        records.append(record)
        if len(records) > self.max_records:
            del records[0 : len(records) - self.max_records]

    def freeze(self) -> None:
        self.frozen = True

    def eligible(
        self, asset_id: str, horizon: int, branch: int, query_ns: int
    ) -> tuple[EngramRecord, ...]:
        return tuple(
            record
            for record in self._records.get((asset_id, horizon, branch), ())
            if record.origin_ns < query_ns and record.available_ns <= query_ns
        )

    def permuted_values(
        self, asset_id: str, horizon: int, branch: int, query_ns: int, seed: int
    ) -> tuple[torch.Tensor, ...]:
        records = self.eligible(asset_id, horizon, branch, query_ns)
        if not records:
            return ()
        order = np.random.default_rng(seed).permutation(len(records))
        return tuple(records[index].value for index in order)

    def __len__(self) -> int:
        return sum(len(records) for records in self._records.values())


@dataclass(frozen=True)
class RetrievalResult:
    value: torch.Tensor
    entropy: torch.Tensor
    effective_records: torch.Tensor
    missing: torch.Tensor
    top_record_ids: tuple[str, ...]


class SpectralEngramRetriever(nn.Module):
    def __init__(self, branch: int, *, spectral_coefficient: float = 0.5) -> None:
        super().__init__()
        if branch not in (0, 1, 2):
            raise ValueError("branch must be 0, 1, or 2")
        self.branch = branch
        self.query_projection = nn.Linear(168, 32)
        self.key_projection = nn.Linear(168, 32)
        self.value_projection = nn.Linear(35, 32)
        self.spectral_logits = nn.Parameter(torch.zeros(32))
        self.spectral_coefficient = float(spectral_coefficient)
        mask = torch.zeros(32, dtype=torch.bool)
        start, stop = ((1, 4), (5, 16), (17, 32))[branch]
        mask[start - 1 : stop] = True
        self.register_buffer("band_mask", mask)

    def _spectral_similarity(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(self.spectral_logits[self.band_mask])
        weights = weights / weights.sum()
        query_band = query[self.band_mask]
        key_band = keys[:, self.band_mask]
        numerator = (weights * (query_band.unsqueeze(0) * key_band.conj()).real).sum(dim=-1)
        query_norm = (weights * query_band.abs().square()).sum()
        key_norm = (weights * key_band.abs().square()).sum(dim=-1)
        return numerator / torch.sqrt(query_norm * key_norm + 1e-8)

    def forward(
        self,
        query_context: torch.Tensor,
        query_spectrum: torch.Tensor,
        records: tuple[EngramRecord, ...],
        *,
        query_ns: int,
        min_records: int = 8,
        override_values: tuple[torch.Tensor, ...] | None = None,
    ) -> RetrievalResult:
        if len(records) < min_records:
            zero = query_context.new_zeros(32)
            scalar = query_context.new_tensor(0.0)
            return RetrievalResult(zero, scalar, scalar, query_context.new_tensor(True), ())
        contexts = torch.stack([record.context.to(query_context) for record in records])
        spectra = torch.stack([record.spectrum.to(query_spectrum) for record in records])
        values = torch.stack(
            [value.to(query_context) for value in override_values]
            if override_values is not None
            else [record.value.to(query_context) for record in records]
        )
        q = F.normalize(self.query_projection(query_context), dim=-1)
        k = F.normalize(self.key_projection(contexts), dim=-1)
        content = 10.0 * torch.einsum("d,md->m", q, k)
        spectral = self._spectral_similarity(query_spectrum, spectra)
        ages = query_context.new_tensor(
            [(query_ns - record.origin_ns) / 86_400_000_000_000 for record in records]
        )
        scores = (
            content + self.spectral_coefficient * spectral - torch.log2(ages.clamp_min(1.0)) / 504.0
        )
        weights = torch.softmax(scores, dim=0)
        retrieved = self.value_projection(torch.einsum("m,mv->v", weights, values))
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
        effective = entropy.exp()
        top = torch.topk(weights, k=min(8, len(records))).indices.tolist()
        return RetrievalResult(
            retrieved,
            entropy,
            effective,
            query_context.new_tensor(False),
            tuple(records[index].record_id for index in top),
        )
