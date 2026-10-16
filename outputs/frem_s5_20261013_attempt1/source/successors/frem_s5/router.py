"""Delayed date-level expert routing; standard library, no neural imports."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def nll(y, mu, scale, df):
    if not all(math.isfinite(v) for v in (y, mu, scale, df)) or scale <= 0 or df <= 2:
        raise ValueError("Invalid Student-t distribution")
    return (
        math.lgamma(df / 2)
        - math.lgamma((df + 1) / 2)
        + 0.5 * math.log(df * math.pi)
        + math.log(scale)
        + (df + 1) / 2 * math.log1p(((y - mu) / scale) ** 2 / df)
    )


def mixture_nll(losses, weights):
    if len(losses) != len(weights) or not losses or any(w < 0 for w in weights):
        raise ValueError("Invalid mixture")
    if not all(math.isfinite(v) for v in [*losses, *weights]) or abs(sum(weights) - 1) > 1e-10:
        raise ValueError("Invalid mixture values")
    pivot = min(losses)
    return pivot - math.log(sum(w * math.exp(pivot - l) for w, l in zip(weights, losses)))


def softmax_loss(losses, temperature):
    if temperature <= 0:
        raise ValueError("Positive temperature required")
    best = min(losses)
    e = [math.exp(max(-700, -(v - best) / temperature)) for v in losses]
    return [v / sum(e) for v in e]


def solve(matrix, vector):
    """Small pivoted linear solve for explicitly regularized regression."""
    n = len(vector)
    a = [list(row) + [v] for row, v in zip(matrix, vector)]
    for j in range(n):
        p = max(range(j, n), key=lambda i: abs(a[i][j]))
        a[j], a[p] = a[p], a[j]
        if abs(a[j][j]) < 1e-12:
            raise ValueError("Singular regularized system")
        d = a[j][j]
        a[j] = [v / d for v in a[j]]
        for i in range(n):
            if i != j:
                d = a[i][j]
                a[i] = [v - d * w for v, w in zip(a[i], a[j])]
    return [row[-1] for row in a]


@dataclass(frozen=True)
class Episode:
    time: int
    maturity: int
    context: tuple
    losses: tuple


class Router:
    MODES = ("equal", "recent", "context", "random", "linear")

    def __init__(self, mode, temperature=1.0, seed=11, experts=4):
        if mode not in self.MODES or temperature not in (0.5, 1.0) or experts < 2:
            raise ValueError("Unknown routing contract")
        self.mode, self.temperature, self.seed, self.experts = mode, temperature, seed, experts
        self.pending, self.memory = [], []
        self.clock = -1
        self.seen = set()

    def observe(self, episode):
        if episode.time != self.clock or episode.maturity <= episode.time or episode.time in self.seen:
            raise ValueError("Feedback must follow one prediction and mature later")
        if len(episode.losses) != self.experts or not episode.context:
            raise ValueError("Episode shape mismatch")
        if not all(math.isfinite(v) for v in (*episode.losses, *episode.context)):
            raise ValueError("Nonfinite episode")
        self.seen.add(episode.time)
        self.pending.append(episode)

    def weights(self, time, context):
        if time <= self.clock or not context or not all(math.isfinite(v) for v in context):
            raise ValueError("Strictly increasing predictions with finite context required")
        self.clock = time
        self.memory.extend(e for e in self.pending if e.maturity < time)
        self.pending = [e for e in self.pending if e.maturity >= time]
        self.memory.sort(key=lambda e: e.time)
        self.memory = self.memory[-60:]
        if any(len(e.context) != len(context) for e in self.memory):
            raise ValueError("Context dimension changed")
        uniform = [1 / self.experts] * self.experts
        if self.mode == "equal" or not self.memory:
            return uniform, {"dates": len(self.memory), "ess": 0, "shrinkage": 0}
        recent = self.memory[-24:]
        if self.mode in ("context", "random"):
            nearest = sorted(
                self.memory, key=lambda e: (sum((a - b) ** 2 for a, b in zip(e.context, context)), e.time)
            )[:12]
            if self.mode == "random":
                # Match the contextual selection's count in fixed six-month age buckets.
                rng = random.Random(self.seed + time)
                buckets = {}
                for e in nearest:
                    b = int((time - e.time) / (182.625 * 86400))
                    buckets[b] = buckets.get(b, 0) + 1
                selected = []
                for b, count in sorted(buckets.items()):
                    pool = [e for e in self.memory if int((time - e.time) / (182.625 * 86400)) == b]
                    selected.extend(rng.sample(pool, count))
            else:
                selected = nearest
        else:
            selected = recent
        raw = [math.exp(-math.log(2) * (time - e.time) / (365.25 * 86400)) for e in selected]
        total = sum(raw)
        if total <= 0:
            return uniform, {"dates": len(selected), "ess": 0, "shrinkage": 0}
        w = [v / total for v in raw]
        ess = 1 / sum(v * v for v in w)
        losses = [sum(a * e.losses[j] for a, e in zip(w, selected)) for j in range(self.experts)]
        if self.mode == "linear":
            x = [[1.0, *e.context] for e in selected]
            d = len(x[0])
            gram = [
                [
                    sum(q * row[i] * row[j] for q, row in zip(w, x)) + (0.001 if i == 0 else 1.0) * (i == j)
                    for j in range(d)
                ]
                for i in range(d)
            ]
            predicted = []
            for expert in range(self.experts):
                rhs = [
                    sum(q * row[i] * e.losses[expert] for q, row, e in zip(w, x, selected)) for i in range(d)
                ]
                beta = solve(gram, rhs)
                predicted.append(sum(a * b for a, b in zip(beta, [1.0, *context])))
            losses = predicted
        adaptive = softmax_loss(losses, self.temperature)
        shrink = ess / (ess + 12)
        result = [(1 - shrink) / self.experts + shrink * v for v in adaptive]
        return result, {
            "dates": len(selected),
            "ess": ess,
            "shrinkage": shrink,
            "latest_maturity": max(e.maturity for e in selected),
            "episode_times": [e.time for e in selected],
        }

    def state(self):
        return {
            "mode": self.mode,
            "temperature": self.temperature,
            "seed": self.seed,
            "experts": self.experts,
            "clock": self.clock,
            "seen": sorted(self.seen),
            "pending": [e.__dict__ for e in self.pending],
            "memory": [e.__dict__ for e in self.memory],
        }

    @classmethod
    def restore(cls, state):
        r = cls(state["mode"], state["temperature"], state["seed"], state["experts"])
        r.clock, r.seen = state["clock"], set(state["seen"])
        for key in ("pending", "memory"):
            setattr(
                r,
                key,
                [
                    Episode(e["time"], e["maturity"], tuple(e["context"]), tuple(e["losses"]))
                    for e in state[key]
                ],
            )
        return r
