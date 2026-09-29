"""M04 prequential evaluation: exploratory synthetic evidence, never a lockbox run.

Uses the original M04 implementation with a wrapper that withholds outcomes until
maturity. No changes to the legacy models, loaders, protocols, or result files.
Run: PYTHONPATH=src python -m wharton_lab.evaluation.m04_prequential --out NEW_DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from wharton_lab.backtesting.walk_forward import iter_walk_forward
from wharton_lab.data.causal_tracks import CausalPanel, causal_return_features
from wharton_lab.data.synthetic import SyntheticWorldConfig, WorldKind, generate_world
from wharton_lab.models.m04.config import M04Config
from wharton_lab.models.m04.model import M04FinancialAPEN

ARMS = ("zero", "historical_mean", "online_mean", "ridge_no_memory", "online_bias",
        "M04_online", "M04_zero_residual")
UPSTREAM_REVISION = "8e3d790dd513a93b7d80d5726a8b2708e8615b71"
SOURCE_BLOBS = {
    "models/m04/config.py": "4a9452f9221fec49c7f3e0f0edeccbd0d54c3c4a",
    "models/m04/memory.py": "1dbbf3f21dd8d8cc39e181d85ffc5b50f56b0bc5",
    "models/m04/model.py": "392bc724c4b5f5fcd1836b140436891b5dda0cb9",
    "models/base.py": "5c93ab12ccd8a54a4c7aa464f62f086532abbdc1",
    "contracts/base.py": "99b4bf249ccdcf7734705265f3d92096d14f005b",
    "data/causal_tracks.py": "c0719a7bebfd8f734753d2835699fc6a36ccaef0",
    "data/synthetic.py": "b26ee364901a66a58b8b8c2da12a1d6d3f547c70",
    "backtesting/walk_forward.py": "b530f6b0ce99f0a382767f0892dab0e0ce53c9ed",
}


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Prediction:
    point: float
    backbone: float
    memory_size: int


class MaturedAPEN:
    """Train once; issue predictions; accept only already-matured outcomes.

    Integer row indices form a logical trading-step clock, not calendar dates.
    No target is stored in the model's pending queue before it becomes available.
    Memory begins empty in each fold. The backbone and scaler never refit within it.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, row_t: np.ndarray, *,
                 first_test: int, config: M04Config | None = None,
                 zero_residual: bool = False) -> None:
        cfg = config or M04Config(horizon_steps=1, maturity_delay=2)
        if _integer(cfg.horizon_steps, "horizon_steps", 1) != 1:
            raise ValueError("This evaluator supports only the one-step target")
        self.delay = 1 + _integer(cfg.maturity_delay, "maturity_delay")
        for name in ("memory_capacity", "retrieval_k"):
            _integer(getattr(cfg, name), name, 1)
        if cfg.use_mlp_backbone:
            raise ValueError("Matched-backbone protocol is ridge only")
        first_test = _integer(first_test, "first_test")
        X, y = np.asarray(X, float), np.asarray(y, float)
        raw_t = np.asarray(row_t)
        if raw_t.ndim != 1 or not np.issubdtype(raw_t.dtype, np.integer):
            raise ValueError("row_t must be a one-dimensional integer array")
        t = raw_t.astype(np.int64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(t) != len(y) or len(y) < 2:
            raise ValueError("Training arrays must be aligned and contain at least two rows")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite training input")
        if np.any(t < 0) or np.any(np.diff(t) <= 0) or np.any(t + self.delay > first_test):
            raise ValueError("Training labels are unordered or not mature at first_test")
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd = np.where(self.sd > 0, self.sd, 1.0)
        self.model = M04FinancialAPEN(M04Config.from_dict(cfg.to_dict()))
        self.model.fit((X - self.mu) / self.sd, y)
        self.first_test = first_test
        self.zero_residual = bool(zero_residual)
        self.last_clock = first_test - 1
        self.last_prediction = first_test - 1
        self.issued: dict[int, tuple[np.ndarray, float]] = {}
        self.observed_count = 0

    def _time(self, row: int) -> datetime:
        row = _integer(row, "row")
        if row < self.last_clock:
            raise ValueError("Time may not move backwards")
        self.last_clock = row
        return datetime(2000, 1, 1) + timedelta(days=row)

    def predict(self, x: np.ndarray, row: int) -> Prediction:
        row = _integer(row, "row")
        if row < self.first_test or row <= self.last_prediction:
            raise ValueError("Prediction rows must be new and strictly increasing")
        x = np.asarray(x, float)
        if x.shape != self.mu.shape or not np.isfinite(x).all():
            raise ValueError("Invalid prediction features")
        now = self._time(row)
        z = ((x - self.mu) / self.sd)[None, :]
        base = float(self.model._backbone_predict(z)[0])
        # The legacy store permits historical queries; this wrapper forbids time travel.
        if any(ep.inserted_at > now for ep in self.model._memory.episodes):
            raise RuntimeError("Future memory insertion detected")
        point = float(self.model.predict(z, as_of=now)[0])
        if not np.isfinite([base, point]).all():
            raise RuntimeError("Non-finite prediction")
        self.issued[row] = (z.copy(), base)  # No target is stored here.
        self.last_prediction = row
        return Prediction(point, base, len(self.model._memory.episodes))

    def observe(self, row: int, outcome: float, *, now_row: int) -> float:
        row, now_row = _integer(row, "row"), _integer(now_row, "now_row")
        if row not in self.issued:
            raise ValueError("No outstanding prediction, or outcome already observed")
        if now_row < row + self.delay:
            raise ValueError("Outcome is not mature")
        if not np.isfinite(outcome):
            raise ValueError("Non-finite outcome")
        now = self._time(now_row)
        z, base = self.issued.pop(row)
        effective_y = base if self.zero_residual else float(outcome)
        self.model.log_prediction(z, np.array([effective_y]),
                                  datetime(2000, 1, 1) + timedelta(days=row))
        if self.model.advance_time(now) != 1 or self.model._pending.pending_count() != 0:
            raise RuntimeError("Unexpected maturity-queue state")
        self.observed_count += 1
        return base - float(outcome)


def evaluate_fold(panel: CausalPanel, train_idx: np.ndarray, test_idx: np.ndarray, *,
                  delay: int = 2, bias_gain: float = 0.1) -> tuple[list[dict], dict]:
    """Score a fixed disjoint test block, releasing outcomes before each prediction."""
    delay = _integer(delay, "delay")
    if not 0 < bias_gain <= 1:
        raise ValueError("bias_gain must be in (0, 1]")
    X, y, times = np.asarray(panel.X), np.asarray(panel.y), np.asarray(panel.row_t)
    if (X.ndim != 2 or y.ndim != 1 or times.ndim != 1 or len(X) != len(y)
            or len(times) != len(y) or not np.issubdtype(times.dtype, np.integer)
            or not np.isfinite(X).all() or not np.isfinite(y).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid causal panel")
    for idx in (train_idx, test_idx):
        idx = np.asarray(idx)
        if (idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or not len(idx)
                or np.any(idx < 0) or np.any(idx >= len(y)) or np.any(np.diff(idx) <= 0)):
            raise ValueError("Invalid fold indices")
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    if train_idx[-1] >= test_idx[0]:
        raise ValueError("Training must strictly precede test")
    first = int(times[test_idx[0]])
    mature_train = train_idx[times[train_idx] + 1 + delay <= first]
    cfg = M04Config(horizon_steps=1, maturity_delay=delay)
    live = MaturedAPEN(X[mature_train], y[mature_train], times[mature_train],
                      first_test=first, config=cfg)
    zeroed = MaturedAPEN(X[mature_train], y[mature_train], times[mature_train],
                        first_test=first, config=cfg, zero_residual=True)
    train_sum, train_n = float(y[mature_train].sum()), len(mature_train)
    historical_mean = train_sum / train_n
    bias, released = 0.0, 0
    rows = []
    for j, ix in enumerate(test_idx):
        t = int(times[ix])
        while released < j and int(times[test_idx[released]]) + 1 + delay <= t:
            old = int(test_idx[released])
            # This is the sole point where an evaluation label enters any learner.
            residual = live.observe(int(times[old]), float(y[old]), now_row=t)
            zeroed.observe(int(times[old]), float(y[old]), now_row=t)
            bias = (1 - bias_gain) * bias + bias_gain * residual
            train_sum += float(y[old]); train_n += 1
            released += 1
        pred = live.predict(X[ix], t)
        ablated = zeroed.predict(X[ix], t)
        if abs(ablated.point - pred.backbone) > 1e-12:
            raise RuntimeError("Zero-residual ablation differs from the matched backbone")
        row = {"row_t": t, "target": float(y[ix]), "zero": 0.0,
               "historical_mean": historical_mean, "online_mean": train_sum / train_n,
               "ridge_no_memory": pred.backbone, "online_bias": pred.backbone - bias,
               "M04_online": pred.point, "M04_zero_residual": ablated.point,
               "available_test_labels": released, "memory_size": pred.memory_size}
        rows.append(row)
    metrics = {name: float(np.mean([(r[name] - r["target"])**2 for r in rows])) for name in ARMS}
    return rows, {"n_train": len(mature_train), "n_purged": len(train_idx)-len(mature_train),
                  "train_last_row": int(times[mature_train[-1]]), "test_first_row": first,
                  "n_test": len(rows), "mse": metrics, "observed_labels": released,
                  "correction_count": sum(abs(r["M04_online"]-r["ridge_no_memory"]) > 1e-12
                                          for r in rows)}


def verify_sources() -> dict:
    root = Path(__file__).resolve().parents[1]
    out = {}
    for path, expected in SOURCE_BLOBS.items():
        b = (root / path).read_bytes()
        blob = hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
        if blob != expected:
            raise RuntimeError(f"Upstream source changed: {path}; review before rerunning")
        out[path] = {"git_blob_sha1": blob, "sha256": hashlib.sha256(b).hexdigest()}
    return out


def run_suite(out: Path, protocol: dict) -> dict:
    """Freeze the resolved protocol before running; refuse overwrites and real-data claims."""
    if protocol.get("confirmatory") is not False or protocol.get("data_scope") != "synthetic_only":
        raise ValueError("Only explicitly exploratory synthetic protocols are supported")
    if protocol.get("arms") != list(ARMS):
        raise ValueError("Protocol arm roster does not match this evaluator")
    if _integer(protocol["horizon_steps"], "horizon_steps", 1) != 1:
        raise ValueError("Protocol horizon must match the one-step target")
    seeds, kinds = protocol["seeds"], protocol["worlds"]
    if not seeds or len(set(seeds)) != len(seeds) or not kinds or len(set(kinds)) != len(kinds):
        raise ValueError("Seeds/worlds must be nonempty and unique")
    if set(seeds) - {11, 23, 47, 89, 131}:
        raise ValueError("Only previously used development seeds are allowed; no lockbox seed")
    for name in ("n_steps", "n_assets", "n_folds", "min_train", "test_size"):
        _integer(protocol[name], name, 1)
    _integer(protocol["maturity_delay"], "maturity_delay")
    for kind in kinds:
        WorldKind(kind)
    out = Path(out).resolve()
    if any(p in {"lockbox", "repaired", "factorial", "protocol"} for p in out.parts):
        raise ValueError("Refusing a protected legacy/protocol output path")
    sources = verify_sources()
    out.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(protocol, sort_keys=True, indent=2) + "\n"
    (out / "PROTOCOL.json").write_text(encoded)
    manifest = {"status": "RUNNING", "generation": "M04_prequential_exploratory_20260929",
                "confirmatory": False, "data_scope": "synthetic_only",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "upstream_revision": UPSTREAM_REVISION, "sources": sources,
                "protocol_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "sklearn": sklearn.__version__, "platform": platform.platform()},
                "real_market_validation": "NOT_RUN", "other_core_models": "NOT_RUN"}
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    cases = []
    for kind in kinds:
        for seed in seeds:
            world = generate_world(SyntheticWorldConfig(kind=WorldKind(kind), seed=seed,
                                    n_steps=protocol["n_steps"], n_assets=protocol["n_assets"]))
            panel = causal_return_features(world.returns, source=f"{kind}_s{seed}")
            folds = sorted(iter_walk_forward(len(panel.y), n_folds=protocol["n_folds"],
                           min_train=protocol["min_train"], test_size=protocol["test_size"]),
                           key=lambda f: int(f.test_idx[0]))
            if len(folds) != protocol["n_folds"]:
                raise ValueError("Insufficient data for the full declared fold count")
            seen, all_rows, fold_metrics = set(), [], []
            for f in folds:
                if seen.intersection(f.test_idx.tolist()):
                    raise RuntimeError("Overlapping test folds")
                seen.update(f.test_idx.tolist())
                rows, metric = evaluate_fold(panel, f.train_idx, f.test_idx,
                                            delay=protocol["maturity_delay"],
                                            bias_gain=protocol["bias_gain"])
                all_rows.extend({"fold": f.fold, **row} for row in rows)
                fold_metrics.append({"fold": f.fold, **metric})
            mse = {a: float(np.mean([(r[a]-r["target"])**2 for r in all_rows])) for a in ARMS}
            contrasts = {b: 1-mse["M04_online"]/mse[b] if mse[b] > 0 else None
                         for b in ("ridge_no_memory", "online_bias", "historical_mean", "online_mean")}
            case = {"world": kind, "seed": seed, "n_test": len(all_rows), "mse": mse,
                    "m04_oos_r2_vs": contrasts, "folds": fold_metrics,
                    "correction_count": sum(f["correction_count"] for f in fold_metrics)}
            stem = f"{kind}_s{seed}"
            with (out / f"{stem}_predictions.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
                writer.writeheader(); writer.writerows(all_rows)
            (out / f"{stem}_metrics.json").write_text(json.dumps(case, indent=2, allow_nan=False)+"\n")
            cases.append(case)
            print(f"{stem}: {len(folds)} folds; {len(all_rows)} targets; "
                  f"M04 vs backbone R2={contrasts['ridge_no_memory']:+.6f}", flush=True)
    result = {"confirmatory": False, "data_scope": "synthetic_only", "core_models_run": ["M04"],
              "dataset_seed_cases": len(cases), "fold_evaluations": sum(len(c["folds"]) for c in cases),
              "arms_per_fold": len(ARMS), "target_rows": sum(c["n_test"] for c in cases),
              "cases": cases}
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    manifest.update(status="COMPLETED", finished_at=datetime.now(timezone.utc).isoformat())
    manifest["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(out.iterdir()) if p.name != "RUN_MANIFEST.json"}
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="A new directory; existing paths fail")
    parser.add_argument("--protocol", type=Path, default=Path(
        "protocol/M04_PREQUENTIAL_EXPLORATORY_2026-09-29.json"))
    args = parser.parse_args()
    run_suite(args.out, json.loads(args.protocol.read_text()))


if __name__ == "__main__":
    main()
