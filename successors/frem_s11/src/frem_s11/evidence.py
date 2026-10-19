"""Fail-closed evidence identities and atomic JSON helpers."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from frem_s11.errors import DataContractError


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 identity of bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one regular, non-symlink file."""
    if path.is_symlink() or not path.is_file():
        raise DataContractError(f"evidence path is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    """Read a bounded JSON object while rejecting duplicate keys."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise DataContractError(f"JSON evidence path is unavailable or oversized: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"JSON evidence cannot be read: {path}") from exc
    if not isinstance(value, dict):
        raise DataContractError("JSON evidence root must be an object")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    """Create a new evidence file atomically and never overwrite."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"evidence output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    """Write canonical human-readable JSON as a new evidence file."""
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def package_identity() -> dict[str, object]:
    """Hash every scientific source and embedded contract file."""
    root = Path(__file__).resolve().parent
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".json", ".typed"}:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return {"schema": "frem-s11-source-identity-v1", "sha256": digest.hexdigest(), "files": rows}


def verify_admission(path: Path, *, maximum_age: timedelta = timedelta(hours=1)) -> dict[str, Any]:
    """Verify a fresh passing receipt from this interpreter and exact source."""
    receipt = read_json(path)
    if receipt.get("schema") != "frem-s11-host-admission-v1" or receipt.get("status") != "PASS":
        raise DataContractError("a passing FREM-S11 host admission receipt is required")
    if receipt.get("interpreter") != str(Path(sys.executable).resolve()):
        raise DataContractError("host admission interpreter does not match the executing interpreter")
    if receipt.get("source_identity") != package_identity()["sha256"]:
        raise DataContractError("host admission source identity is stale")
    try:
        observed = datetime.fromisoformat(str(receipt["observed_utc"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("host admission timestamp is invalid") from exc
    now = datetime.now(UTC)
    if observed.tzinfo is None or observed > now + timedelta(minutes=1) or now - observed > maximum_age:
        raise DataContractError("host admission receipt is stale or future-dated")
    return receipt


def verify_archive(path: Path, receipt_path: Path, *, release_month: int) -> dict[str, Any]:
    """Bind an archive to its no-parse acquisition receipt."""
    receipt = read_json(receipt_path)
    if receipt.get("schema") != "frem-s11-acquisition-v2" or receipt.get("release_month") != release_month:
        raise DataContractError("archive receipt schema or release month is invalid")
    if receipt.get("sha256") != sha256_file(path) or receipt.get("bytes") != path.stat().st_size:
        raise DataContractError("archive bytes do not match the acquisition receipt")
    source_catalog = json.loads(
        importlib.resources.files("frem_s11").joinpath("sources.json").read_text(encoding="utf-8")
    )
    if receipt.get("source_url") != source_catalog.get(str(release_month)):
        raise DataContractError("archive receipt does not match the frozen source URL")
    if receipt.get("source_identity") != package_identity()["sha256"]:
        raise DataContractError("archive receipt source identity is stale")
    if receipt.get("parsed") is not False or receipt.get("scientific_result") is not False:
        raise DataContractError("acquisition receipt boundary is invalid")
    return receipt


def verify_data_admission(
    path: Path,
    *,
    archive_sha256: str,
    release_month: int,
    cutoff: int,
) -> dict[str, Any]:
    """Verify the exact pre-run data-admission receipt."""
    receipt = read_json(path)
    expected = {
        "schema": "frem-s11-data-admission-v1",
        "status": "PASS",
        "archive_sha256": archive_sha256,
        "release_month": release_month,
        "cutoff": cutoff,
        "source_identity": package_identity()["sha256"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise DataContractError(
            "data-admission receipt is stale or does not match the fixed development input"
        )
    if receipt.get("scientific_result") is not False:
        raise DataContractError("data-admission receipt crossed the scientific-result boundary")
    return receipt
