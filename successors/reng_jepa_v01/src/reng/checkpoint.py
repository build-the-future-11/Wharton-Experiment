"""Atomic, hash-bound model checkpoints and protected freeze receipts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .artifacts import canonical_json, write_new_json


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    *,
    metadata: dict[str, Any],
) -> Path:
    required = {
        "model_version",
        "source_tree_sha256",
        "dataset_sha256",
        "split_sha256",
        "configuration_sha256",
        "root_seed",
        "ensemble_seed",
        "horizon",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Checkpoint metadata missing: {sorted(missing)}")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    temporary = target / ".state.pt.tmp"
    state_path = target / "state.pt"
    with temporary.open("xb") as handle:
        torch.save(model.state_dict(), handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(state_path)
    payload = {
        **metadata,
        "created_at": datetime.now(UTC).isoformat(),
        "state_sha256": _hash_file(state_path),
    }
    write_new_json(target / "metadata.json", payload)
    return target


def load_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    actual_hash = _hash_file(root / "state.pt")
    if actual_hash != metadata.get("state_sha256"):
        raise ValueError("Checkpoint state hash mismatch")
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint metadata mismatch for {key}")
    state = torch.load(root / "state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return metadata


def create_freeze(
    output: str | Path,
    *,
    protocol: dict[str, Any],
    configuration: dict[str, Any],
    source_tree_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
    universe_sha256: str,
    mandate_sha256: str,
    environment_sha256: str,
    protected_boundary: str,
    rationale: str,
) -> str:
    if not all(
        len(value) == 64
        for value in (
            source_tree_sha256,
            dataset_sha256,
            split_sha256,
            universe_sha256,
            mandate_sha256,
            environment_sha256,
        )
    ):
        raise ValueError("Freeze identities must be SHA-256 digests")
    if not protected_boundary or not rationale:
        raise ValueError("Freeze requires an explicit protected boundary and rationale")
    payload = {
        "status": "FROZEN_UNCONSUMED",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": hashlib.sha256(canonical_json(protocol)).hexdigest(),
        "configuration_sha256": hashlib.sha256(canonical_json(configuration)).hexdigest(),
        "source_tree_sha256": source_tree_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "universe_sha256": universe_sha256,
        "mandate_sha256": mandate_sha256,
        "environment_sha256": environment_sha256,
        "protected_boundary": protected_boundary,
        "rationale": rationale,
    }
    freeze_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    payload["freeze_sha256"] = freeze_hash
    write_new_json(output, payload)
    return freeze_hash


def consume_freeze(freeze_path: str | Path, receipt_path: str | Path, *, attempt_id: str) -> str:
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_UNCONSUMED" or not freeze.get("freeze_sha256"):
        raise ValueError("Freeze is invalid or not consumable")
    receipt = {
        "attempt_id": attempt_id,
        "freeze_sha256": freeze["freeze_sha256"],
        "consumed_at": datetime.now(UTC).isoformat(),
        "status": "CONSUMED_ATTEMPT_STARTED",
    }
    return write_new_json(receipt_path, receipt)
