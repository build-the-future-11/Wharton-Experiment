"""Strict finite manifests for portable generated S10 engineering exercises."""

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import stat
import sys
import sysconfig
from pathlib import Path, PurePosixPath

from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject, digest_bytes, digest_json, object_json, parse_json
from frem_s10.metadata import finite_float, strict_int

VARIANTS = ("primary", "repeat_h1", "no_shrink", "window260", "neighbors16", "neighbors64")
KEYS = {
    "schema",
    "scope",
    "device",
    "precision",
    "seed",
    "steps",
    "variants",
    "budget_seconds",
    "dataset",
    "checkpoint_every_origins",
    "threads",
}
_IDENTITY_MAX_BYTES = 256 * 2**20
_IDENTITY_MAX_FILES = 4096
_NUMPY_ROOTS = ("numpy", "numpy.libs")


def read_manifest(path: Path) -> JSONObject:
    """Reject ignored flags, nonfinite settings, unbounded matrices and market data."""
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ContractError("bounded plain smoke manifest required")
    value = object_json(parse_json(path.read_bytes()))
    if set(value) != KEYS or value["schema"] != "s10-generated-smoke-v1":
        raise ContractError("exact generated smoke manifest schema required")
    if value["scope"] not in ("smoke", "development-engineering"):
        raise ContractError("only generated engineering scope is supported")
    if (value["device"], value["precision"], value["dataset"]) != (
        "cpu",
        "float64",
        "s10-generated-smoke-data-v1",
    ):
        raise ContractError("S10 supports CPU float64 generated smoke only; CUDA has no implemented backend")
    if type(value["threads"]) is not int or value["threads"] != 1:
        raise ContractError("one thread is required")
    if type(value["checkpoint_every_origins"]) is not int or value["checkpoint_every_origins"] != 1:
        raise ContractError("every completed origin must be checkpointed")
    seed = strict_int(value["seed"], name="seed", minimum=0)
    steps = strict_int(value["steps"], name="steps", minimum=1)
    budget = finite_float(value["budget_seconds"], name="budget_seconds", strictly_positive=True)
    if seed >= 2**32 or steps > 80 or budget > 300:
        raise ContractError("generated engineering bounds: seed<2**32, steps<=80, wall<=300s")
    if value["scope"] == "smoke" and steps > 12:
        raise ContractError("tiny smoke is limited to twelve origins; use admitted development-engineering")
    variants = value["variants"]
    if (
        not isinstance(variants, list)
        or not variants
        or any(not isinstance(item, str) or item not in VARIANTS for item in variants)
        or len(variants) != len({str(item) for item in variants})
    ):
        raise ContractError("unique declared model variants required")
    return value


def smoke_plan(manifest: Path) -> JSONObject:
    """Enumerate every generated model cell without NumPy import or execution."""
    config = read_manifest(manifest)
    variants = config["variants"]
    assert isinstance(variants, list)
    return {
        "status": "PLAN_ONLY",
        "configuration": config,
        "configuration_sha256": digest_json(config),
        "cells": [{"variant": name, "seed": config["seed"], "origins": config["steps"]} for name in variants],
        "market_execution": False,
        "gpu_execution": False,
        "scientific_status": "NOT_EVALUATED",
        "resume_boundary": "all declared variants committed at one completed origin",
        "budget_semantics": "cumulative active wall time; checked between origins; no runtime promise",
    }


def runtime_identity() -> JSONObject:
    """Bind the installed source bytes and numeric environment, excluding file paths."""
    package = Path(__file__).resolve().parents[1]
    members = [*sorted(package.rglob("*.py")), package / "science.json"]
    if any(path.is_symlink() for path in members):
        raise IntegrityError("source symlinks cannot identify a generated replay")
    interpreter = _interpreter_identity()
    numpy = numpy_artifact_identity(maximum_bytes=_IDENTITY_MAX_BYTES - int(str(interpreter["bytes"])))
    return {
        "source_sha256": digest_json(
            {str(p.relative_to(package)): digest_bytes(p.read_bytes()) for p in members}
        ),
        "python": sys.version,
        "numpy": numpy["version"],
        "numpy_artifacts": numpy,
        "python_runtime": interpreter,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "precision": "float64",
        "device": "cpu",
    }


def _hash_regular(path: Path, maximum: int) -> tuple[str, int]:
    """Stream a bounded immutable regular file, rejecting symlink and read-time changes."""
    if maximum <= 0:
        raise IntegrityError("runtime artifact identity exceeds its total byte bound")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise IntegrityError("bounded regular runtime artifact required")
            digest, total = hashlib.sha256(), 0
            while chunk := stream.read(min(65_536, maximum - total + 1)):
                total += len(chunk)
                if total > maximum:
                    raise IntegrityError("runtime artifact grew beyond the identity byte bound")
                digest.update(chunk)
            after = os.fstat(stream.fileno())
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if total != before.st_size or any(getattr(before, key) != getattr(after, key) for key in stable):
                raise IntegrityError("runtime artifact changed during identity observation")
        linked = path.stat(follow_symlinks=False)
        if any(getattr(after, key) != getattr(linked, key) for key in stable):
            raise IntegrityError("runtime artifact path changed during identity observation")
        return digest.hexdigest(), total
    except OSError as exc:
        raise IntegrityError("runtime artifact unavailable for exact identity") from exc


def _runtime_member(name: str) -> bool:
    path = PurePosixPath(name)
    if not path.parts or path.parts[0] not in _NUMPY_ROOTS:
        return False
    if str(path) != name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise IntegrityError("NumPy inventory contains a noncanonical relative path")
    return "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")


def _installed_numpy_names(base: Path) -> set[str]:
    """Enumerate actual runtime package files so omitted RECORD members cannot pass."""
    pending = [base / name for name in _NUMPY_ROOTS if (base / name).exists()]
    observed: set[str] = set()
    visited = 0
    while pending:
        path = pending.pop()
        visited += 1
        if visited > 2 * _IDENTITY_MAX_FILES or path.is_symlink():
            raise IntegrityError("NumPy runtime tree is oversized or contains symlinks")
        relative = path.relative_to(base).as_posix()
        if "__pycache__" in PurePosixPath(relative).parts:
            continue
        if path.is_dir():
            pending.extend(path.iterdir())
        elif _runtime_member(relative):
            if not path.is_file():
                raise IntegrityError("NumPy runtime inventory requires regular files")
            observed.add(relative)
        if len(observed) > _IDENTITY_MAX_FILES:
            raise IntegrityError("NumPy runtime file inventory exceeds its bound")
    return observed


def _numpy_inventory(distribution: importlib.metadata.Distribution, base: Path) -> tuple[str, ...]:
    files = distribution.files
    if files is None or len(files) > 2 * _IDENTITY_MAX_FILES:
        raise IntegrityError("bounded installed NumPy distribution inventory required")
    names = [str(item) for item in files if _runtime_member(str(item))]
    if not names or len(names) != len(set(names)) or len(names) > _IDENTITY_MAX_FILES:
        raise IntegrityError("complete unique NumPy runtime inventory required")
    if "numpy/__init__.py" not in names or not any(
        Path(name).name.startswith("_multiarray_umath.") and Path(name).suffix in (".so", ".pyd", ".dll")
        for name in names
    ):
        raise IntegrityError("NumPy Python and native core artifacts are required")
    if set(names) != _installed_numpy_names(base):
        raise IntegrityError("NumPy recorded and actual runtime file inventories differ")
    return tuple(sorted(names))


def numpy_artifact_identity(*, maximum_bytes: int = _IDENTITY_MAX_BYTES) -> JSONObject:
    """Hash installed NumPy runtime artifacts and bundled libraries without importing NumPy.

    Generated bytecode is excluded; Python/native source and bundled data/library
    files are included. Names are installation-relative, preserving relocation.
    External system libraries remain identified only by the OS/build observation.
    """
    try:
        distribution = importlib.metadata.distribution("numpy")
        base = Path(str(distribution.locate_file(""))).resolve(strict=True)
        names = _numpy_inventory(distribution, base)
        spec = importlib.util.find_spec("numpy")
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != base / "numpy/__init__.py":
            raise IntegrityError("NumPy import origin differs from its installed artifact inventory")
        manifest: JSONObject = {}
        total = 0
        for name in names:
            path = base / name
            if Path(str(distribution.locate_file(name))).resolve() != path:
                raise IntegrityError("NumPy artifact resolves outside its exact distribution path")
            identity, size = _hash_regular(path, maximum_bytes - total)
            manifest[name] = {"sha256": identity, "bytes": size}
            total += size
        return {
            "schema": "s10-numpy-artifacts-v1",
            "version": distribution.version,
            "manifest_sha256": digest_json(manifest),
            "files": len(names),
            "bytes": total,
            "scope": "installed NumPy runtime files and bundled libraries; generated bytecode excluded",
        }
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        raise IntegrityError("installed NumPy artifact identity unavailable") from exc


def _interpreter_identity() -> JSONObject:
    """Record actual interpreter bytes and ABI/build metadata without absolute paths."""
    try:
        binary = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise IntegrityError("interpreter binary unavailable") from exc
    identity, size = _hash_regular(binary, _IDENTITY_MAX_BYTES)
    return {
        "sha256": identity,
        "bytes": size,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "soabi": str(sysconfig.get_config_var("SOABI")),
        "build": list(platform.python_build()),
        "compiler": platform.python_compiler(),
    }
