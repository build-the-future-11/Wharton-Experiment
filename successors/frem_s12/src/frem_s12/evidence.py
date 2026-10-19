"""Hash-bound no-overwrite evidence utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from frem_s12.errors import ContractError

_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 hex digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file."""
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"not a regular evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object while rejecting duplicate keys."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError("JSON evidence must be an object")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    """Write one new file atomically without replacing evidence."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    """Write canonical JSON as new immutable evidence."""
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write(path, payload)


def package_identity() -> dict[str, object]:
    """Hash every package source and embedded contract file."""
    root = Path(__file__).resolve().parent
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".typed"}:
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    payload = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return {
        "schema": "frem-s12-package-identity-v1",
        "sha256": sha256_bytes(payload),
        "files": files,
        "python": sys.version.split()[0],
        "interpreter": str(Path(sys.executable).resolve()),
    }


def receipt(kind: str, **fields: object) -> dict[str, object]:
    """Create timestamped receipt content without writing it."""
    return {
        "schema": f"frem-s12-{kind}-v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "source_identity": package_identity()["sha256"],
        **fields,
    }


def validate_cell_provenance(
    value: object,
    *,
    phase: str,
    identity: dict[str, int | str],
) -> dict[str, str]:
    """Validate the immutable gate and consumption chain for one scientific cell."""
    if not isinstance(value, dict):
        raise ContractError("cell provenance is absent or invalid")
    required = {
        "environment_receipt_sha256",
        "data_receipt_sha256",
        "acquisition_receipt_sha256",
        "archive_sha256",
        "source_identity",
        "consumption",
    }
    if phase == "fixed":
        required.add("selection_receipt_sha256")
    if set(value) != required:
        raise ContractError("cell provenance fields differ from the scientific contract")
    hashes: dict[str, str] = {}
    for name in sorted(required - {"consumption"}):
        digest = value.get(name)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ContractError(f"cell provenance {name} is invalid")
        hashes[name] = digest
    if hashes["source_identity"] != package_identity()["sha256"]:
        raise ContractError("cell provenance source identity is stale")
    consumption = value.get("consumption")
    if not isinstance(consumption, dict) or set(consumption) != {"path", "sha256", "resumed"}:
        raise ContractError("cell consumption provenance is invalid")
    digest = consumption.get("sha256")
    path = consumption.get("path")
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or not isinstance(path, str)
        or not isinstance(consumption.get("resumed"), bool)
    ):
        raise ContractError("cell consumption identity is invalid")
    if phase == "search":
        suffix = (
            f"search/trial={identity['trial']}/family={identity['family']}/seed={identity['seed']}/claim.json"
        )
    elif phase == "fixed":
        suffix = (
            f"fixed/family={identity['family']}/fold={identity['fold']}/seed={identity['seed']}/claim.json"
        )
    else:
        raise ContractError("cell provenance phase is invalid")
    if not path.replace("\\", "/").endswith(suffix):
        raise ContractError("cell consumption path differs from its embedded identity")
    hashes["consumption_sha256"] = digest
    return hashes
