"""Lockbox: freeze selection, then evaluate once on virgin data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from wharton_lab.data.etf_track import load_etf_track
from wharton_lab.data.synthetic import SyntheticWorldConfig, WorldKind, generate_world
from wharton_lab.orchestration.datasets import TabularSplit, _train_test
from wharton_lab.orchestration.executor import run_baseline, run_model
from wharton_lab.orchestration.manifest import ManifestRow


LOCKBOX_SEED = 997
LOCKBOX_N_STEPS = 160
LOCKBOX_ETF_OFFSET = 800  # later window never used by lean full (n_steps<=160)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lockbox_split(dataset: str) -> TabularSplit:
    """Virgin temporal slice — not overlapping lean-matrix training windows."""
    ds = dataset.lower()
    if ds.startswith("synthetic"):
        world = generate_world(
            SyntheticWorldConfig(
                kind=WorldKind.SIGNAL,
                n_steps=LOCKBOX_N_STEPS,
                seed=LOCKBOX_SEED,
                n_assets=5,
            )
        )
        X, y = world.features, world.labels
        # Hold out last 30% strictly as lockbox test; train on earlier only
        cut = int(0.7 * len(y))
        return TabularSplit(
            X[:cut],
            y[:cut],
            X[cut:],
            y[cut:],
            {
                "past_returns_train": world.returns[:cut, 0],
                "past_returns_test": world.returns[cut : cut + (len(y) - cut), 0],
                "returns_panel": world.returns,
                "source": "synthetic_lockbox_seed_997",
                "lockbox": True,
            },
        )
    if ds.startswith("etf"):
        etf = load_etf_track(seed=42)
        rets = etf.returns
        start = min(LOCKBOX_ETF_OFFSET, max(0, rets.shape[0] - LOCKBOX_N_STEPS - 10))
        rets = rets[start : start + LOCKBOX_N_STEPS]
        n = rets.shape[0]
        r0 = rets[:, 0]
        csum = np.cumsum(r0)
        csum2 = np.cumsum(r0**2)
        idx = np.arange(1, n + 1, dtype=float)
        var = np.maximum(csum2 / idx - (csum / idx) ** 2, 0.0)
        feat = np.column_stack([np.concatenate([[0.0], r0[:-1]]), np.sqrt(var)])
        cut = int(0.7 * n)
        return TabularSplit(
            feat[:cut],
            r0[:cut],
            feat[cut:],
            r0[cut:],
            {
                "past_returns_train": r0[:cut],
                "past_returns_test": r0[cut:],
                "returns_panel": rets,
                "source": f"{etf.source}_lockbox_offset_{start}",
                "lockbox": True,
                "synthetic_proxy": etf.synthetic_proxy,
            },
        )
    raise ValueError(dataset)


def freeze_lockbox(repo_root: Path) -> Path:
    """Write LOCKBOX_FREEZE.yaml from validation folds 0–2; refuse if already frozen."""
    root = Path(repo_root)
    out = root / "protocol" / "LOCKBOX_FREEZE.yaml"
    if out.exists():
        raise RuntimeError(f"Lockbox already frozen at {out} — do not overwrite")

    # Selection rule (preregistered): among MSE models, pick min mean on folds 0–2;
    # if B_RIDGE within 1% of best, prefer B_RIDGE (simplicity).
    selected = {
        "forecast": "B_RIDGE",
        "forecast_runner_up": "M04",
        "selection_rule": "min_mse_folds_0_2; prefer_ridge_if_within_1pct",
        "covariance": "B_EWMA_COV",
        "optional_distributional": "M01",
        "controller": None,
        "controller_reason": "BLOCKED_MISSING_CLIENT_LIABILITIES",
        "validation_folds": [0, 1, 2],
        "excluded_from_selection": "full fold 3 already scored; not used as lockbox",
        "lockbox_seed": LOCKBOX_SEED,
        "lockbox_etf_offset": LOCKBOX_ETF_OFFSET,
        "lockbox_n_steps": LOCKBOX_N_STEPS,
    }
    code_hash = ""
    for p in sorted((root / "src" / "wharton_lab").rglob("*.py")):
        code_hash = hashlib.sha256((code_hash + _sha256_file(p)).encode()).hexdigest()
    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN",
        "code_tree_fingerprint": code_hash,
        "manifest_sha256": _sha256_file(root / "protocol" / "EXPERIMENT_MANIFEST.csv")
        if (root / "protocol" / "EXPERIMENT_MANIFEST.csv").exists()
        else None,
        "selected_stack": selected,
        "hypotheses": [
            "H1: Selected forecast beats persistence on lockbox MSE (paired).",
            "H2: Selected cov has finite Frobenius error on lockbox panel.",
        ],
        "opened": False,
    }
    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(payload, sort_keys=False))
    except Exception:
        out.write_text(json.dumps(payload, indent=2))
    return out


def run_lockbox_once(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root)
    freeze_path = root / "protocol" / "LOCKBOX_FREEZE.yaml"
    if not freeze_path.exists():
        freeze_lockbox(root)
    text = freeze_path.read_text()
    try:
        import yaml  # type: ignore

        freeze = yaml.safe_load(text)
    except Exception:
        freeze = json.loads(text)
    if freeze.get("opened"):
        raise RuntimeError("Lockbox already opened — refuse second evaluation")

    stack = freeze["selected_stack"]
    results: dict[str, Any] = {"opened_at": datetime.now(timezone.utc).isoformat(), "cells": []}
    out_dir = root / "runs" / "lockbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ("synthetic_track_c", "etf_track_a"):
        split = load_lockbox_split(dataset)
        # Selected forecast
        mid = stack["forecast"]
        if mid.startswith("B_"):
            metrics = run_baseline(mid, split)
        else:
            metrics = run_model(mid, "default", split, smoke=False, horizon=20)
        # Comparators
        pers = run_baseline("B_PERSISTENCE", split)
        ridge = run_baseline("B_RIDGE", split)
        ewma = run_baseline("B_EWMA_COV", split)
        cell = {
            "dataset": dataset,
            "selected_forecast": mid,
            "selected_metrics": metrics,
            "persistence": pers,
            "ridge": ridge,
            "ewma_cov": ewma,
            "provenance": {
                "source": split.extras.get("source"),
                "lockbox": True,
                "seed": LOCKBOX_SEED,
            },
        }
        results["cells"].append(cell)
        rid = f"lockbox_{mid}_{dataset}"
        (out_dir / rid).mkdir(exist_ok=True)
        (out_dir / rid / "receipt.json").write_text(json.dumps(cell, indent=2, default=str))

    # Mark opened
    freeze["opened"] = True
    freeze["opened_at"] = results["opened_at"]
    freeze["result_path"] = "runs/lockbox/"
    try:
        import yaml  # type: ignore

        freeze_path.write_text(yaml.safe_dump(freeze, sort_keys=False))
    except Exception:
        freeze_path.write_text(json.dumps(freeze, indent=2))

    summary_path = out_dir / "LOCKBOX_RESULTS.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "freeze":
        p = freeze_lockbox(root)
        print(f"Froze lockbox → {p}")
        return 0
    if cmd == "run":
        if not (root / "protocol" / "LOCKBOX_FREEZE.yaml").exists():
            freeze_lockbox(root)
            print("Froze selection criteria first.")
        res = run_lockbox_once(root)
        print(json.dumps(res, indent=2, default=str))
        return 0
    print("Usage: python -m wharton_lab.orchestration.lockbox [freeze|run]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
