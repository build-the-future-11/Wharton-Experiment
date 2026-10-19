"""Complete, hash-bound S11 development evidence bundles."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frem_s11.admission import MemorySample, execution_pressure_stable
from frem_s11.backtest import VARIANTS, BacktestResult
from frem_s11.data import ASSETS, ReturnPanel
from frem_s11.evidence import atomic_json, atomic_write, package_identity, sha256_file


def _hash(path: Path) -> str:
    return sha256_file(path)


def _write_array(path: Path, values: NDArray[Any]) -> None:
    buffer = BytesIO()
    np.save(buffer, values, allow_pickle=False)
    atomic_write(path, buffer.getvalue())


def _fold_summary(label: str, result: BacktestResult) -> dict[str, Any]:
    return {
        "fold": label,
        "start": int(result.months[0]),
        "end": int(result.months[-1]),
        "observations": len(result.months),
        "mean_energy_scores": dict(zip(VARIANTS, result.scores.mean(axis=0).tolist(), strict=True)),
        "terminal_wealth": dict(zip(VARIANTS, result.wealth[-1].tolist(), strict=True)),
        "benchmark_terminal_wealth": {
            "cash": float(result.benchmark_wealth[-1, 0]),
            "equal_assets_monthly_rebalanced_no_cost": float(result.benchmark_wealth[-1, 1]),
        },
        "intervals": result.intervals,
        "relative_improvements": result.relative_improvements,
        "primary_supported": result.primary_supported,
        "secondary_supported": result.secondary_supported,
    }


def write_development_bundle(
    output: Path,
    panel: ReturnPanel,
    fold_labels: tuple[str, ...],
    results: tuple[BacktestResult, ...],
    *,
    protocol_bytes: bytes,
    admission: dict[str, Any],
    post_sample: MemorySample,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Retain every fold array, decision statistic, resource receipt, and hash."""
    if len(fold_labels) != len(results) or not results:
        raise ValueError("fold labels and results must be nonempty and aligned")
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "INCOMPLETE.json"
    atomic_json(marker, {"schema": "frem-s11-incomplete-v1", "complete": False})
    resource_valid = execution_pressure_stable(admission, post_sample)
    primary_supported = resource_valid and all(item.primary_supported for item in results)
    secondary_supported = resource_valid and all(item.secondary_supported for item in results)
    if not resource_valid:
        disposition = "DEVELOPMENT_INVALID_RESOURCE_PRESSURE_PRESERVED"
    elif primary_supported:
        disposition = "DEVELOPMENT_PRIMARY_SUPPORTED_CONFIRMATION_REQUIRED"
    else:
        disposition = "DEVELOPMENT_PRIMARY_NOT_SUPPORTED_STOP_S11"
    files: list[Path] = []
    for label, result in zip(fold_labels, results, strict=True):
        arrays = {
            "months": result.months,
            "scores": result.scores,
            "stack_weights": result.stack_weights,
            "allocations": result.allocations,
            "wealth": result.wealth,
            "benchmark_wealth": result.benchmark_wealth,
        }
        for name, values in arrays.items():
            path = output / f"fold_{label}_{name}.npy"
            _write_array(path, values)
            files.append(path)
    summary = {
        "schema": "frem-s11-development-v2",
        "scientific_status": disposition,
        "source_sha256": panel.source_sha256,
        "source_release_month": panel.release_month,
        "asset_order": ASSETS,
        "variants": VARIANTS,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_identity": package_identity(),
        "provenance": provenance,
        "resources": {
            "admission": admission,
            "post_execution": {**asdict(post_sample), "available_mib": post_sample.available_mib},
            "pageouts_stable": resource_valid,
        },
        "folds": [_fold_summary(label, result) for label, result in zip(fold_labels, results, strict=True)],
        "overall_primary_supported": primary_supported,
        "overall_secondary_supported": secondary_supported,
        "confirmation_complete": False,
    }
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary)
    files.append(summary_path)
    manifest = {
        "schema": "frem-s11-manifest-v1",
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": _hash(path)} for path in sorted(files)},
    }
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    file_receipts = manifest["files"]
    if not isinstance(file_receipts, dict):
        raise TypeError("artifact manifest file map is invalid")
    for name, receipt in file_receipts.items():
        if not isinstance(name, str) or not isinstance(receipt, dict):
            raise TypeError("artifact manifest row is invalid")
        path = output / name
        if path.stat().st_size != receipt["bytes"] or _hash(path) != receipt["sha256"]:
            raise OSError(f"artifact verification failed: {name}")
    marker.unlink()
    return summary
