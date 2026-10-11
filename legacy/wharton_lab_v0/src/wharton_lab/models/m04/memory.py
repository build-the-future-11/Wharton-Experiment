"""Financial APEN memory: pending queue, similarity retrieval, bounded store."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


@dataclass(order=True)
class _PendingItem:
    maturity_time: datetime
    episode_id: int = field(compare=True)
    decision_time: datetime = field(compare=False)
    context_key: np.ndarray = field(compare=False, repr=False)
    residual: float = field(compare=False)
    meta: dict = field(default_factory=dict, compare=False)


@dataclass
class MemoryEpisode:
    episode_id: int
    decision_time: datetime
    context_key: np.ndarray
    residual: float
    inserted_at: datetime
    meta: dict = field(default_factory=dict)


class PendingOutcomeQueue:
    """Episodes retrievable only after horizon + delay matured."""

    def __init__(self, horizon_steps: int, maturity_delay: int) -> None:
        self.horizon_steps = horizon_steps
        self.maturity_delay = maturity_delay
        self._heap: List[_PendingItem] = []
        self._counter = 0

    def enqueue(
        self,
        decision_time: datetime,
        context_key: np.ndarray,
        residual: float,
        *,
        step: timedelta = timedelta(days=1),
        meta: Optional[dict] = None,
    ) -> int:
        eid = self._counter
        self._counter += 1
        maturity = decision_time + step * (self.horizon_steps + self.maturity_delay)
        item = _PendingItem(
            maturity_time=maturity,
            episode_id=eid,
            decision_time=decision_time,
            context_key=np.asarray(context_key, dtype=float).copy(),
            residual=float(residual),
            meta=meta or {},
        )
        heapq.heappush(self._heap, item)
        return eid

    def pop_matured(self, as_of: datetime) -> list[MemoryEpisode]:
        out: list[MemoryEpisode] = []
        while self._heap and self._heap[0].maturity_time <= as_of:
            item = heapq.heappop(self._heap)
            out.append(
                MemoryEpisode(
                    episode_id=item.episode_id,
                    decision_time=item.decision_time,
                    context_key=item.context_key,
                    residual=item.residual,
                    inserted_at=as_of,
                    meta=item.meta,
                )
            )
        return out

    def pending_count(self) -> int:
        return len(self._heap)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BoundedMemoryStore:
    """FIFO eviction with insertion/retrieval/eviction logs."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.episodes: list[MemoryEpisode] = []
        self.insertion_log: list[tuple[datetime, int]] = []
        self.retrieval_log: list[tuple[datetime, int, float]] = []
        self.eviction_log: list[tuple[datetime, int]] = []

    def insert(self, episode: MemoryEpisode) -> None:
        self.episodes.append(episode)
        self.insertion_log.append((episode.inserted_at, episode.episode_id))
        while len(self.episodes) > self.capacity:
            ev = self.episodes.pop(0)
            self.eviction_log.append((episode.inserted_at, ev.episode_id))

    def retrieve_similar(
        self,
        query_key: np.ndarray,
        as_of: datetime,
        k: int = 8,
        sim_fn: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
    ) -> list[tuple[MemoryEpisode, float]]:
        scored = []
        for ep in self.episodes:
            if ep.decision_time > as_of:
                continue
            s = sim_fn(query_key, ep.context_key)
            scored.append((ep, s))
        scored.sort(key=lambda x: -x[1])
        top = scored[:k]
        for ep, s in top:
            self.retrieval_log.append((as_of, ep.episode_id, s))
        return top


def confidence_from_similarities(sims: list[float], min_confidence: float) -> float:
    if not sims:
        return 0.0
    conf = float(np.mean(np.clip(sims, 0, 1)))
    return conf if conf >= min_confidence else 0.0


def novelty_gate(max_sim: float, threshold: float) -> bool:
    """True if context is novel (not too similar to past)."""
    return max_sim < threshold
