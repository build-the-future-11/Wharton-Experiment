"""Expert create / merge / prune driven by matured losses only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from wharton_lab.models.m05.experts import ExpertKind, ExpertSpec


@dataclass
class MaturedLossRecord:
    expert_id: str
    loss: float
    step: int


@dataclass
class PendingMaturity:
    """Losses waiting until maturity before lifecycle updates."""

    expert_id: str
    loss: float
    birth_step: int
    mature_at_step: int


@dataclass
class LifecycleState:
    pending: list[PendingMaturity] = field(default_factory=list)
    matured: list[MaturedLossRecord] = field(default_factory=list)
    step: int = 0

    def queue_loss(self, expert_id: str, loss: float, maturity_delay: int = 2) -> None:
        self.pending.append(
            PendingMaturity(
                expert_id=expert_id,
                loss=float(loss),
                birth_step=self.step,
                mature_at_step=self.step + maturity_delay,
            )
        )

    def advance_step(self) -> list[MaturedLossRecord]:
        self.step += 1
        ready: list[MaturedLossRecord] = []
        still: list[PendingMaturity] = []
        for p in self.pending:
            if p.mature_at_step <= self.step:
                rec = MaturedLossRecord(expert_id=p.expert_id, loss=p.loss, step=self.step)
                self.matured.append(rec)
                ready.append(rec)
            else:
                still.append(p)
        self.pending = still
        return ready


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    if a.shape != b.shape:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def merge_experts(
    id_a: str,
    id_b: str,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    new_id: str,
) -> tuple[str, np.ndarray]:
    wa = np.asarray(weights_a, dtype=float).ravel()
    wb = np.asarray(weights_b, dtype=float).ravel()
    if wa.shape != wb.shape:
        # Never merge linear (n+1) with gaussian (n+2) — broadcasting/concat fails.
        return id_a, wa
    merged = 0.5 * (wa + wb)
    return new_id, merged


def prune_worst(
    expert_ids: list[str],
    matured: list[MaturedLossRecord],
    fraction: float,
) -> list[str]:
    if not expert_ids or fraction <= 0:
        return []
    by_id: dict[str, list[float]] = {}
    for m in matured:
        by_id.setdefault(m.expert_id, []).append(m.loss)
    scores = []
    for eid in expert_ids:
        losses = by_id.get(eid, [float("inf")])
        scores.append((eid, float(np.mean(losses))))
    scores.sort(key=lambda x: -x[1])
    k = max(1, int(len(expert_ids) * fraction))
    return [eid for eid, _ in scores[:k]]


def maybe_create_expert(
    specs: list[ExpertSpec],
    max_experts: int,
    mean_routing_loss: float,
    threshold: float,
    n_features: int,
    rng: np.random.Generator,
    kind_cycle: Callable[[int], ExpertKind],
) -> ExpertSpec | None:
    if len(specs) >= max_experts:
        return None
    if mean_routing_loss < threshold:
        return None
    idx = len(specs)
    kind = kind_cycle(idx)
    return ExpertSpec(expert_id=f"e{idx}", kind=kind, n_features=n_features)
