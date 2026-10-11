"""Bounded, explicitly non-scientific multi-dataset compatibility runner."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import ExperimentManifest, write_receipt
from .config import ModelConfig
from .data import build_examples, select_fold_universe, synthetic_panel, validate_panel
from .evaluation import representation_diagnostics
from .model import RENGJEPA, stage_a_loss

ENGINEERING_DATASETS = ("calm_trend", "mean_reversion", "correlation_break", "missing_actions")


def source_tree_hash(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tensor_batch(examples, horizon: int, count: int = 2) -> dict[str, object]:
    selection = slice(0, min(count, len(examples.context)))
    return {
        "context": torch.from_numpy(examples.context[selection]),
        "future": torch.from_numpy(examples.future[horizon][selection]),
        "asset_mask": torch.from_numpy(examples.asset_mask[selection]),
        "scale": torch.from_numpy(examples.scales[horizon][selection]).to(torch.float32),
        "log_return": torch.from_numpy(examples.log_returns[horizon][selection]).to(torch.float32),
        "simple_return": torch.from_numpy(np.expm1(examples.log_returns[horizon][selection])).to(
            torch.float32
        ),
        "decision_ns": torch.from_numpy(
            examples.decision_times[selection].astype("datetime64[ns]").astype(np.int64)
        ),
        "asset_ids": examples.asset_ids,
        "horizon": horizon,
    }


def run_engineering_dataset(kind: str, *, seed: int = 11) -> dict[str, object]:
    started = time.monotonic()
    frame, manifest = synthetic_panel(kind, sessions=220, assets=4, seed=seed)
    validation = validate_panel(frame, manifest)
    freeze_time = frame["timestamp"].max()
    universe = select_fold_universe(frame, freeze_time)
    if len(universe) != 4:
        raise AssertionError("Engineering universe selection lost an asset")
    examples = build_examples(frame, universe, context_length=128, horizons=(5, 21))
    batch = _tensor_batch(examples, 21)
    torch.manual_seed(seed)
    model = RENGJEPA(ModelConfig())
    model.eval()
    with torch.no_grad():
        stage_a = model.stage_a(batch["context"], batch["future"], batch["asset_mask"], 21)
        simple_scaled = batch["simple_return"] / batch["scale"]
        losses = stage_a_loss(stage_a, simple_scaled)
        members = model(
            batch["context"],
            batch["future"],
            batch["asset_mask"],
            batch["scale"],
            tuple(batch["asset_ids"]),
            batch["decision_ns"],
            21,
        )
    if not all(torch.isfinite(output.return_mean).all() for output in members):
        raise FloatingPointError("Non-finite compatibility output")
    first_decision = examples.decision_times[0]
    changed = frame.copy()
    future_rows = pd_timestamp_gt(changed["timestamp"], first_decision)
    changed.loc[future_rows, "close"] *= 1.5
    changed_examples = build_examples(
        changed,
        universe,
        context_length=128,
        horizons=(5, 21),
        decision_times=[first_decision],
    )
    if not np.array_equal(examples.context[:1], changed_examples.context[:1]):
        raise AssertionError("Future-row mutation altered an earlier context")
    diagnostics = representation_diagnostics(
        stage_a.features.online_latent.flatten(0, 1).cpu().numpy()
    )
    elapsed = time.monotonic() - started
    return {
        "terminal_status": "ENGINEERING_PASS",
        "dataset": kind,
        "synthetic": True,
        "scientific_claims_allowed": False,
        "validation": validation,
        "manifest_sha256": manifest.source_sha256,
        "universe": list(universe),
        "example_dates": int(len(examples.decision_times)),
        "context_shape": list(batch["context"].shape),
        "future_shape": list(batch["future"].shape),
        "parameter_count": model.trainable_parameter_count(),
        "stage_a_loss": float(losses["total"]),
        "member_count": len(members),
        "return_shape": list(members[0].return_mean.shape),
        "future_mutation_invariant": True,
        "representation_diagnostics": diagnostics,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def pd_timestamp_gt(series, value: np.datetime64):
    """Comparison helper that keeps timezone handling explicit."""
    import pandas as pd

    threshold = pd.Timestamp(value).tz_localize("UTC")
    return pd.to_datetime(series, utc=True) > threshold


def run_engineering_matrix(output: str | Path, *, seed: int = 11) -> dict[str, object]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing matrix: {destination}")
    package_root = Path(__file__).resolve().parents[2]
    protocol = json.loads((package_root / "research" / "protocol.json").read_text(encoding="utf-8"))
    configuration = {"datasets": ENGINEERING_DATASETS, "seed": seed, "smoke_dates": 2, "assets": 4}
    destination.mkdir(parents=True)
    receipts = []
    for kind in ENGINEERING_DATASETS:
        result = run_engineering_dataset(kind, seed=seed)
        manifest = ExperimentManifest.create(
            source_tree_sha256=source_tree_hash(package_root),
            protocol=protocol,
            configuration={**configuration, "dataset": kind},
            dataset_sha256=result["manifest_sha256"],
            dataset_label=f"synthetic_engineering:{kind}",
            dataset_synthetic=True,
            seeds=(seed,),
            scientific_claims_allowed=False,
        )
        run_path = write_receipt(destination / kind, manifest, result)
        receipts.append(str(run_path / "receipt.json"))
    summary = {
        "terminal_status": "ENGINEERING_MATRIX_PASS",
        "scientific_status": "NOT_EVALUATED",
        "datasets": list(ENGINEERING_DATASETS),
        "dataset_count": len(receipts),
        "receipts": receipts,
        "scientific_claims_allowed": False,
    }
    from .artifacts import write_new_json

    write_new_json(destination / "SUMMARY.json", summary)
    return summary
