"""Immutable exact-origin checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frem_s12.errors import ContractError
from frem_s12.evidence import atomic_json, package_identity, read_json, sha256_file


def write_checkpoint(root: Path, state: dict[str, Any]) -> Path:
    """Write one source-bound checkpoint named by its next forecast origin."""
    next_origin = int(state["next_origin"])
    value = {**state, "source_identity": package_identity()["sha256"]}
    path = root / f"checkpoint-{next_origin:04d}.json"
    atomic_json(path, value)
    return path


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read one checkpoint and reject stale package identity."""
    state = read_json(path)
    if state.get("schema") != "frem-s12-fold-checkpoint-v3":
        raise ContractError("checkpoint schema is invalid")
    if state.get("source_identity") != package_identity()["sha256"]:
        raise ContractError("checkpoint source identity is stale")
    state["checkpoint_sha256"] = sha256_file(path)
    return state
