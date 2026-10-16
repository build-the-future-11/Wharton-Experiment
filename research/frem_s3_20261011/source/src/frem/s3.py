"""FREM-S3 causal residual replay. Additive; legacy checkpoints are never edited.

All calibration is explicit. Replay updates only from strictly matured outcomes;
the frozen base and key metric never learn from the replay evaluation stream.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

DAY = 86_400_000_000_000
WEIGHTS = (0.0, 0.25, 0.5, 1.0)
POWERS = (0.0, 0.5, 1.0)


def balanced(values, times, assets):
    groups = {}
    for value, time, asset in zip(values, times, assets, strict=True):
        groups.setdefault((int(time) // DAY, int(asset)), []).append(float(value))
    dates = {}
    for (date, _), group in groups.items():
        dates.setdefault(date, []).append(float(np.mean(group)))
    if not dates:
        raise ValueError("Empty score")
    return float(np.mean([np.mean(v) for v in dates.values()]))


def losses(y, mu, scale, df):
    arrays = np.broadcast_arrays(y, mu, scale, df)
    if not all(np.isfinite(a).all() for a in arrays) or np.any(arrays[2] <= 0) or np.any(arrays[3] <= 2):
        raise ValueError("Invalid Student-t distribution")
    return -student_t.logpdf(arrays[0], arrays[3], loc=arrays[1], scale=arrays[2])


def calibrate(frame, *, shrink=True):
    """Common validation-only distribution calibration for every base family."""
    choices = []
    for alpha in (0.0, 0.5, 1.0) if shrink else (1.0,):
        for scale in (0.5, 0.75, 1.0, 1.5, 2.0):
            score = balanced(
                losses(frame.actual, alpha * frame.prediction, scale * frame.scale, frame.student_df),
                frame.time,
                frame.asset_id,
            )
            choices.append({"alpha": alpha, "scale": scale, "nll": score})
    return {
        "selected": min(choices, key=lambda c: (c["nll"], c["alpha"], abs(math.log(c["scale"])))),
        "candidates": choices,
    }


def apply_calibration(frame, calibration):
    result = frame.copy()
    chosen = calibration["selected"]
    result["prediction"] *= chosen["alpha"]
    result["scale"] *= chosen["scale"]
    return result


class KeyMetric:
    """Frozen train-only ridge projection of descriptors to residual moments."""

    def fit(self, x, z, label_times, cutoff):
        if len(x) < 8 or np.max(label_times) >= cutoff:
            raise ValueError("Metric labels must mature before validation")
        self.center = np.mean(x, axis=0)
        self.unit = np.maximum(np.std(x, axis=0), 1e-6)
        design = (x - self.center) / self.unit
        targets = np.column_stack((z, np.square(z)))
        self.coef = np.linalg.solve(design.T @ design + np.eye(design.shape[1]), design.T @ targets)
        self.fitted_through = int(np.max(label_times))
        return self

    def transform(self, x):
        return ((x - self.center) / self.unit) @ self.coef

    def to_dict(self):
        return {
            "center": self.center.tolist(),
            "unit": self.unit.tolist(),
            "coef": self.coef.tolist(),
            "fitted_through": self.fitted_through,
        }


@dataclass(frozen=True)
class ReplayConfig:
    name: str
    online: bool = True
    mode: str = "combined"
    retrieval: str = "nearest"
    retention: str = "fifo"
    capacity: int = 128
    neighbors: int = 4
    same_asset: bool = True
    confidence: bool = True
    half_life_days: float = 180.0
    seed: int = 11

    def __post_init__(self):
        if self.mode not in {"off", "mean", "risk", "combined"}:
            raise ValueError("Unknown replay mode")
        if self.retrieval not in {"nearest", "recent", "random", "learned"}:
            raise ValueError("Unknown retrieval")
        if self.retention not in {"fifo", "balanced", "dual", "utility"}:
            raise ValueError("Unknown retention")
        if min(self.capacity, self.neighbors, self.half_life_days) <= 0:
            raise ValueError("Invalid replay bounds")


def arms(seed=11):
    base = ReplayConfig("combined", seed=seed)
    return [
        replace(base, name="off", mode="off"),
        replace(base, name="frozen", online=False, confidence=False, mode="mean"),
        replace(base, name="recent", retrieval="recent"),
        replace(base, name="random", retrieval="random"),
        replace(base, name="mean", mode="mean"),
        replace(base, name="risk", mode="risk"),
        base,
        replace(base, name="no_confidence", confidence=False),
        replace(base, name="balanced", retention="balanced"),
        replace(base, name="larger", capacity=512),
        replace(base, name="dual", retention="dual"),
        replace(base, name="cross_asset", same_asset=False),
        replace(base, name="learned", retrieval="learned"),
        replace(base, name="utility", retention="utility"),
    ]


class Replay:
    """Stateful event replay; serializable, deterministic and bounded."""

    def __init__(self, config, version):
        self.config = config
        self.version = version
        self.records = []
        self.pending = []
        self.clock = -(1 << 63)
        self.next_id = 0
        self.rng = np.random.default_rng(config.seed)
        self.matured = self.evicted = 0
        self.seen = set()

    def schedule(self, row, key, prediction, scale, ids):
        identity = (int(row["time"]), int(row["asset_id"]))
        if identity in self.seen or identity[0] < self.clock:
            raise ValueError("Duplicate or backward prediction")
        if int(row["label_time"]) <= identity[0]:
            raise ValueError("Outcome is not future")
        if row.get("version", self.version) != self.version:
            raise ValueError("Residual predictor version mismatch")
        if not np.isfinite(key).all() or not all(
            math.isfinite(float(row[k])) for k in ("actual", "prediction", "scale", "student_df")
        ):
            raise ValueError("Nonfinite episode")
        self.seen.add(identity)
        # Actual remains opaque to retrieval until advance observes maturity.
        self.pending.append(
            {
                "id": self.next_id,
                "time": identity[0],
                "asset": identity[1],
                "label_time": int(row["label_time"]),
                "key": np.asarray(key).tolist(),
                "actual": float(row["actual"]),
                "mu": float(row["prediction"]),
                "scale": float(row["scale"]),
                "df": float(row["student_df"]),
                "deployed_mu": float(prediction),
                "deployed_scale": float(scale),
                "ids": list(ids),
                "version": self.version,
            }
        )
        self.next_id += 1

    def advance(self, time):
        if time < self.clock:
            raise ValueError("Memory clock cannot move backwards")
        ready = sorted(
            [p for p in self.pending if p["label_time"] < time],
            key=lambda p: (p["label_time"], p["time"], p["asset"]),
        )
        self.pending = [p for p in self.pending if p["label_time"] >= time]
        for episode in ready:
            # Normalizing constants cancel because the deployed df is unchanged.
            y, nu = episode["actual"], episode["df"]

            def score(mu, scale, y=y, nu=nu):
                return math.log(scale) + (nu + 1) / 2 * math.log1p(((y - mu) / scale) ** 2 / nu)

            gain = score(episode["mu"], episode["scale"]) - score(
                episode["deployed_mu"], episode["deployed_scale"]
            )
            for record in self.records:
                if record["id"] in episode["ids"]:
                    record["utility"] = 0.9 * record["utility"] + 0.1 * float(np.clip(gain, -1, 1))
            record = {**episode, "z": (episode["actual"] - episode["mu"]) / episode["scale"], "utility": 0.0}
            self.records.append(record)
            self.matured += 1
            self._evict()
        self.clock = time

    def _evict(self):
        if len(self.records) <= self.config.capacity:
            return
        c = self.config
        if c.retention == "balanced":
            counts = {}
            for r in self.records:
                counts[r["asset"]] = counts.get(r["asset"], 0) + 1
            candidates = [r for r in self.records if counts[r["asset"]] == max(counts.values())]
            victim = min(candidates, key=lambda r: (r["label_time"], r["id"]))
        elif c.retention == "utility":
            victim = min(self.records, key=lambda r: (r["utility"], r["label_time"], r["id"]))
        elif c.retention == "dual":
            ordered = sorted(self.records, key=lambda r: (r["label_time"], r["id"]))
            old = ordered[: len(ordered) - c.capacity // 2]
            # Remove one of the most temporally redundant archive records.
            victim = min(
                old[1:], key=lambda r: (r["label_time"] - old[old.index(r) - 1]["label_time"], r["id"])
            )
        else:
            victim = min(self.records, key=lambda r: (r["label_time"], r["id"]))
        self.records.remove(victim)
        self.evicted += 1

    def retrieve(self, key, asset, time):
        if time < self.clock:
            raise ValueError("Query precedes memory state")
        c = self.config
        records = [
            r for r in self.records if r["label_time"] < time and (not c.same_asset or r["asset"] == asset)
        ]
        empty = {"z": 0.0, "second": 1.0, "confidence": 0.0, "count": 0, "age": 0.0, "ess": 0.0, "ids": []}
        if not records:
            return empty
        keys = np.asarray([r["key"] for r in records])
        distances = np.mean(np.square(keys - key), axis=1)
        order = np.argsort(distances, kind="stable")[: c.neighbors]
        if c.retrieval == "recent":
            order = np.asarray(
                sorted(range(len(records)), key=lambda i: (-records[i]["label_time"], records[i]["id"]))[
                    : c.neighbors
                ]
            )
        elif c.retrieval == "random":
            chosen = []
            # Match each nearest neighbor's asset and 90-day age bin.
            for i in order:
                bucket = (time - records[i]["label_time"]) // (90 * DAY)
                pool = [
                    j
                    for j, r in enumerate(records)
                    if j not in chosen
                    and r["asset"] == records[i]["asset"]
                    and (time - r["label_time"]) // (90 * DAY) == bucket
                ]
                chosen.append(int(self.rng.choice(pool)))
            order = np.asarray(chosen)
        chosen = [records[i] for i in order]
        ages = np.asarray([(time - r["label_time"]) / DAY for r in chosen])
        logits = -distances[order] - math.log(2) * ages / c.half_life_days
        if c.retrieval in {"recent", "random"}:
            logits = -math.log(2) * ages / c.half_life_days
        weight = np.exp(logits - logits.max())
        weight /= weight.sum()
        z = np.asarray([r["z"] for r in chosen])
        mean = float(weight @ z)
        ess = float(1 / np.sum(weight**2))
        disagreement = float(weight @ (z - mean) ** 2)
        age = float(weight @ ages)
        similarity = math.exp(-min(float(weight @ distances[order]), 50))
        confidence = (
            min(1, ess / c.neighbors) / (1 + disagreement) * 2 ** (-age / c.half_life_days) * similarity
        )
        return {
            "z": mean,
            "second": float(weight @ z**2),
            "confidence": confidence if c.confidence else 1.0,
            "count": len(chosen),
            "age": age,
            "ess": ess,
            "ids": [r["id"] for r in chosen],
        }

    def to_dict(self):
        return {
            "config": asdict(self.config),
            "version": self.version,
            "records": self.records,
            "pending": self.pending,
            "clock": self.clock,
            "next_id": self.next_id,
            "rng": self.rng.bit_generator.state,
            "seen": sorted(self.seen),
            "matured": self.matured,
            "evicted": self.evicted,
        }

    @classmethod
    def from_dict(cls, state):
        result = cls(ReplayConfig(**state["config"]), state["version"])
        for name in ("records", "pending", "clock", "next_id", "matured", "evicted"):
            setattr(result, name, copy.deepcopy(state[name]))
        result.seen = {tuple(x) for x in state["seen"]}
        result.rng.bit_generator.state = state["rng"]
        if len(result.records) > result.config.capacity or any(
            r["version"] != result.version for r in result.records
        ):
            raise ValueError("Invalid serialized replay")
        return result


def run_replay(initial, frame, keys, config, weight, power, *, keep_state=False):
    replay = Replay.from_dict(initial)
    if replay.config != config:
        raise ValueError("Replay policy differs from initial memory")
    rows = frame.to_dict("records")
    if len(keys) != len(rows):
        raise ValueError("Key alignment error")
    output = []
    for row, key in zip(rows, keys, strict=True):
        time = int(row["time"])
        if config.online:
            replay.advance(time)
        retrieved = replay.retrieve(key, int(row["asset_id"]), time)
        mean_enabled = config.mode in {"mean", "combined"}
        risk_enabled = config.mode in {"risk", "combined"}
        strength = retrieved["confidence"]
        correction = weight * strength * row["scale"] * retrieved["z"] if mean_enabled else 0.0
        # Student-t's normalized residual second moment is nu/(nu-2).
        ratio = retrieved["second"] * (row["student_df"] - 2) / row["student_df"]
        factor = math.sqrt((4 + retrieved["ess"] * ratio) / (4 + retrieved["ess"]))
        factor = float(np.clip(factor, 0.5, 2)) ** (power * strength) if risk_enabled else 1.0
        mu, scale = row["prediction"] + correction, row["scale"] * factor
        output.append(
            {
                **row,
                "base_prediction": row["prediction"],
                "base_scale": row["scale"],
                "prediction": mu,
                "scale": scale,
                "memory_gate": weight * strength,
                "risk_power": power * strength,
                "neighbors": retrieved["count"],
                "memory_age_days": retrieved["age"],
                "effective_neighbors": retrieved["ess"],
            }
        )
        if config.online:
            replay.schedule(row, key, mu, scale, retrieved["ids"])
    import pandas as pd

    result = pd.DataFrame(output)
    result["nll"] = losses(result.actual, result.prediction, result.scale, result.student_df)
    return (result, replay.to_dict()) if keep_state else result


def seed_memory(frame, keys, config, version, cutoff):
    if len(frame) != len(keys) or int(frame.label_time.max()) >= cutoff:
        raise ValueError("Memory seeding must precede validation")
    memory = Replay(config, version)
    for row, key in zip(frame.to_dict("records"), keys, strict=True):
        memory.advance(int(row["time"]))
        memory.schedule(row, key, row["prediction"], row["scale"], [])
    memory.advance(cutoff)
    return memory.to_dict()


def select_replay(initial, validation, keys, config):
    candidates = []
    for weight in WEIGHTS if config.mode in {"mean", "combined"} else (0.0,):
        for power in POWERS if config.mode in {"risk", "combined"} else (0.0,):
            predictions = run_replay(initial, validation, keys, config, weight, power)
            score = balanced(predictions.nll, predictions.time, predictions.asset_id)
            candidates.append({"weight": weight, "power": power, "nll": score})
    chosen = min(candidates, key=lambda c: (c["nll"], c["weight"], c["power"]))
    predictions, state = run_replay(
        initial, validation, keys, config, chosen["weight"], chosen["power"], keep_state=True
    )
    return {"selected": chosen, "candidates": candidates}, state


def save_replay(path, state):
    payload = json.dumps(state, sort_keys=True, allow_nan=False).encode()
    with Path(path).open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()
