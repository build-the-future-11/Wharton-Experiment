"""Immutable manifests and terminal receipts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class ExperimentManifest:
    run_id: str
    created_at: str
    scientific_claims_allowed: bool
    source_tree_sha256: str
    protocol_sha256: str
    configuration_sha256: str
    dataset_sha256: str
    dataset_label: str
    dataset_synthetic: bool
    seeds: tuple[int, ...]
    environment: dict[str, str]

    @classmethod
    def create(
        cls,
        *,
        source_tree_sha256: str,
        protocol: dict[str, Any],
        configuration: dict[str, Any],
        dataset_sha256: str,
        dataset_label: str,
        dataset_synthetic: bool,
        seeds: tuple[int, ...],
        scientific_claims_allowed: bool = False,
    ) -> ExperimentManifest:
        identity = {
            "source": source_tree_sha256,
            "protocol": sha256_payload(protocol),
            "configuration": sha256_payload(configuration),
            "dataset": dataset_sha256,
            "label": dataset_label,
            "seeds": seeds,
        }
        return cls(
            run_id=sha256_payload(identity)[:20],
            created_at=datetime.now(UTC).isoformat(),
            scientific_claims_allowed=scientific_claims_allowed,
            source_tree_sha256=source_tree_sha256,
            protocol_sha256=identity["protocol"],
            configuration_sha256=identity["configuration"],
            dataset_sha256=dataset_sha256,
            dataset_label=dataset_label,
            dataset_synthetic=dataset_synthetic,
            seeds=seeds,
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pid": str(os.getpid()),
            },
        )


def write_new_json(path: str | Path, payload: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {target}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_receipt(
    directory: str | Path, manifest: ExperimentManifest, result: dict[str, Any]
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    manifest_payload = asdict(manifest)
    manifest_hash = write_new_json(root / "manifest.json", manifest_payload)
    receipt = {
        "run_id": manifest.run_id,
        "manifest_sha256": manifest_hash,
        "terminal_status": result.get("terminal_status", "UNKNOWN"),
        "result": result,
    }
    write_new_json(root / "receipt.json", receipt)
    return root
