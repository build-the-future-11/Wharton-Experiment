"""Atomic artifacts, content fingerprints and failure-preserving run directories."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Unsupported artifact type: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=json_default
    ).encode("utf-8")


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    atomic_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n"
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def source_fingerprint(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    files: dict[str, str] = {}
    for folder in ("src", "configs"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(root))] = file_digest(path)
    for name in ("pyproject.toml", "research/protocol.json"):
        path = root / name
        if path.is_file():
            files[name] = file_digest(path)
    if not any(name.endswith(".py") for name in files):
        raise ValueError("Project root has no Python source files")
    return {"sha256": object_digest(files), "files": files}


def environment_snapshot() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


@contextlib.contextmanager
def run_directory(
    output_root: str | Path, config: dict[str, Any], phase: str, identity: dict[str, Any]
) -> Iterator[Path]:
    """Retain incomplete, interrupted and failed runs without overwriting evidence."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
    directory = Path(output_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = {
        "run_id": run_id,
        "phase": phase,
        "status": "RUNNING",
        "started": now_utc(),
        "configuration": config,
        "identity": identity,
        "environment": environment_snapshot(),
    }
    write_json(directory / "manifest.json", manifest)
    try:
        yield directory
    except BaseException as exc:
        manifest.update(
            status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        atomic_text(directory / "failure.txt", traceback.format_exc())
        raise
    else:
        manifest["status"] = "COMPLETE"
    finally:
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest.update(
            finished=now_utc(),
            wall_seconds=elapsed,
            process_peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
            rss_scope="process lifetime high-water mark; not isolated model memory",
        )
        write_json(directory / "manifest.json", manifest)
