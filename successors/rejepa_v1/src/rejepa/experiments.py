"""Immutable experiment manifests and complete per-cell receipts."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    model: str
    seed: int
    train_fraction: float
    configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cell_id or not self.model or not 0 < self.train_fraction <= 1:
            raise ValueError("Invalid experiment cell")


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    phase: str
    dataset_sha256: str
    source_tree_sha256: str
    protected_start: str | None
    cells: tuple[ExperimentCell, ...]
    frozen: bool = False

    def canonical_payload(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()

    def freeze(self) -> ExperimentManifest:
        if not self.dataset_sha256 or not self.source_tree_sha256 or not self.cells:
            raise ValueError("Dataset, source tree and cells are required before freeze")
        if self.phase in {"confirmatory", "protected"} and not self.protected_start:
            raise ValueError("Confirmatory manifests require a protected boundary")
        return ExperimentManifest(
            experiment_id=self.experiment_id,
            phase=self.phase,
            dataset_sha256=self.dataset_sha256,
            source_tree_sha256=self.source_tree_sha256,
            protected_start=self.protected_start,
            cells=self.cells,
            frozen=True,
        )


def save_manifest(path: str | Path, manifest: ExperimentManifest) -> None:
    payload = {"manifest": asdict(manifest), "manifest_sha256": manifest.sha256}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> ExperimentManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload["manifest"]
    raw["cells"] = tuple(ExperimentCell(**cell) for cell in raw["cells"])
    manifest = ExperimentManifest(**raw)
    if payload.get("manifest_sha256") != manifest.sha256:
        raise ValueError("Experiment manifest hash mismatch")
    return manifest


def run_manifest(
    manifest: ExperimentManifest,
    destination: str | Path,
    execute: Callable[[ExperimentCell], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not manifest.frozen:
        raise ValueError("Manifest must be frozen before execution")
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in manifest.cells:
        if cell.cell_id in seen:
            raise ValueError("Duplicate experiment cell identifier")
        seen.add(cell.cell_id)
        started = time.perf_counter()
        receipt: dict[str, Any] = {
            "cell": asdict(cell),
            "manifest_sha256": manifest.sha256,
            "status": "FAILED",
        }
        try:
            result = execute(cell)
            receipt.update({"status": "COMPLETE", "result": result})
        except Exception as error:  # noqa: BLE001 - every cell failure must receive a receipt
            receipt.update({"error_type": type(error).__name__, "error": str(error)})
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt_path = output / f"{cell.cell_id}.receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipts.append(receipt)
    return receipts


def completion_status(manifest: ExperimentManifest, receipts: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {cell.cell_id for cell in manifest.cells}
    observed = {str(receipt.get("cell", {}).get("cell_id")) for receipt in receipts}
    completed = {
        str(receipt["cell"]["cell_id"]) for receipt in receipts if receipt.get("status") == "COMPLETE"
    }
    return {
        "expected_cells": len(expected),
        "observed_cells": len(observed & expected),
        "completed_cells": len(completed & expected),
        "missing_cells": sorted(expected - observed),
        "failed_cells": sorted((expected & observed) - completed),
        "complete": expected == completed,
    }
