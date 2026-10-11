"""Prospective freeze verification and append-only research decisions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .artifacts import (
    canonical_bytes,
    environment_snapshot,
    file_digest,
    now_utc,
    object_digest,
    read_json,
    source_fingerprint,
    write_json,
)


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in {"output_dir", "description", "name"}}


def identity(config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    inputs = {"panel": file_digest(config["data"]["path"])}
    panel_manifest = Path(str(config["data"]["path"]) + ".manifest.json")
    if not panel_manifest.is_file():
        raise FileNotFoundError("Panel provenance manifest is required before a research freeze or run")
    inputs["panel_manifest"] = file_digest(panel_manifest)
    if config["data"].get("macro_path"):
        inputs["macro"] = file_digest(config["data"]["macro_path"])
        macro_manifest = Path(str(config["data"]["macro_path"]) + ".manifest.json")
        if not macro_manifest.is_file():
            raise FileNotFoundError("Macro provenance manifest is required")
        inputs["macro_manifest"] = file_digest(macro_manifest)
    return {
        "code": source_fingerprint(root),
        "config_sha256": object_digest(scientific_config(config)),
        "data": inputs,
        "protected_start": config["split"]["protected_start"],
    }


def freeze_protocol(
    config: dict[str, Any], root: str | Path, destination: str | Path, rationale: str
) -> dict[str, Any]:
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError("Freeze already exists; create a new version, never overwrite")
    if len(rationale.strip()) < 20:
        raise ValueError("Record a concrete freeze rationale and development-selection decision")
    record = {
        "schema_version": 1,
        "status": "FROZEN_NOT_EVALUATED",
        "created": now_utc(),
        "rationale": rationale,
        "identity": identity(config, root),
        "environment": environment_snapshot(),
        "configuration": scientific_config(config),
        "rule": "Do not tune on protected outcomes; failed runs and deviations must be retained.",
    }
    record["freeze_sha256"] = object_digest(record)
    write_json(dest, record)
    return record


def verify_freeze(
    config: dict[str, Any], root: str | Path, freeze_path: str | Path, strict_environment: bool = True
) -> dict[str, Any]:
    freeze = read_json(freeze_path)
    content = {k: v for k, v in freeze.items() if k != "freeze_sha256"}
    if object_digest(content) != freeze.get("freeze_sha256"):
        raise ValueError("Freeze checksum is invalid")
    current = identity(config, root)
    if current != freeze["identity"]:
        raise ValueError("Code, data or scientific configuration changed after freeze")
    if strict_environment and environment_snapshot() != freeze["environment"]:
        raise ValueError("Frozen execution environment changed; record and freeze a new protocol")
    return freeze


def consume_confirmatory_freeze(root: str | Path, freeze: dict[str, Any]) -> Path:
    """Prevent accidentally reopening the same protected protocol repeatedly.

    Consumption survives interrupted/failed attempts. A genuine invalidation
    needs a recorded defect and a new freeze; deleting this record is not the
    supported recovery process. This is an operational guard, not access control
    against a user intentionally editing their own research files.
    """
    directory = Path(root) / "research" / "confirmatory_consumption"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (freeze["freeze_sha256"] + ".json")
    payload = {
        "freeze_sha256": freeze["freeze_sha256"],
        "consumed_at": now_utc(),
        "status": "CONSUMED",
        "note": "Match freeze identity to retained run manifests; failed attempts remain consumed.",
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            "This confirmatory freeze has already been consumed; inspect its retained evidence"
        ) from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    """Hash-chained local journal. Tamper evidence, not cryptographic authorship."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        prior = "0" * 64
        if path.exists():
            import json

            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    existing = json.loads(line)
                    digest = existing.pop("sha256")
                    if existing.get("previous") != prior or object_digest(existing) != digest:
                        raise ValueError("Research journal hash chain is invalid")
                    prior = digest
        record = {"timestamp": now_utc(), "previous": prior, "event": event}
        record["sha256"] = object_digest(record)
        with path.open("ab") as stream:
            stream.write(canonical_bytes(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record
    finally:
        os.close(fd)
        lock.unlink()
