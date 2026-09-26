"""Load protocol/MODEL_REGISTRY.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def registry_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "protocol" / "MODEL_REGISTRY.yaml"


def load_registry(repo_root: Path | str) -> dict[str, Any]:
    path = registry_path(repo_root)
    if not path.exists():
        return {"models": []}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data


def model_entry(repo_root: Path | str, model_id: str) -> dict[str, Any] | None:
    reg = load_registry(repo_root)
    for m in reg.get("models", []):
        if m.get("id") == model_id:
            return m
    return None


def primary_metric_for(repo_root: Path | str, model_id: str) -> str:
    entry = model_entry(repo_root, model_id)
    if entry:
        return str(entry.get("primary_metric", "mse"))
    baseline_metrics = {
        "B_RIDGE": "mse",
        "B_PERSISTENCE": "mse",
        "B_EWMA_COV": "frobenius_cov_error",
    }
    return baseline_metrics.get(model_id, "mse")
