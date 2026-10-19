"""Independent verification of downloaded Kaggle run bundles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from frem_s12.admission import validate_kaggle_environment_receipt
from frem_s12.errors import ContractError
from frem_s12.evidence import read_json, sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}")
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_MODES = {
    "qualify-kaggle",
    "generated-smoke",
    "run-search-cell",
    "select-search",
    "run-cell",
    "analyze",
}


def _relative_file(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("run manifest contains an invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != value:
        raise ContractError("run manifest contains an unsafe relative path")
    return value


def _declared_files(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    """Validate and normalize the producer file inventory."""
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ContractError("run manifest inventory is empty or invalid")
    declared: dict[str, dict[str, object]] = {}
    for raw_name, raw_evidence in files.items():
        name = _relative_file(raw_name)
        if not isinstance(raw_evidence, dict) or set(raw_evidence) != {"sha256", "size"}:
            raise ContractError("run manifest file evidence is invalid")
        digest = raw_evidence["sha256"]
        size = raw_evidence["size"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ContractError("run manifest contains an invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError("run manifest contains an invalid file size")
        declared[name] = raw_evidence
    return declared


def _verify_files(root: Path, manifest_path: Path, declared: dict[str, dict[str, object]]) -> int:
    """Verify the downloaded tree is regular, exact, complete and hash-bound."""
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ContractError("run bundle contains a symlink or special file")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(declared) != observed:
        raise ContractError("run manifest inventory differs from downloaded files")
    total_bytes = 0
    for name, evidence in declared.items():
        path = root / name
        if path.stat().st_size != evidence["size"] or sha256_file(path) != evidence["sha256"]:
            raise ContractError(f"run bundle file identity mismatch: {name}")
        total_bytes += int(evidence["size"])
    return total_bytes


def _candidate_identity(
    manifest: dict[str, object],
    expected_wheel_sha256: str | None,
    expected_package_version: str | None,
) -> tuple[str, str]:
    """Validate the package candidate bound into the final manifest."""
    wheel_sha256 = manifest.get("wheel_sha256")
    package_version = manifest.get("package_version")
    if not isinstance(wheel_sha256, str) or _SHA256.fullmatch(wheel_sha256) is None:
        raise ContractError("run manifest wheel identity is invalid")
    if not isinstance(package_version, str) or not package_version:
        raise ContractError("run manifest package version is invalid")
    if expected_wheel_sha256 is not None and wheel_sha256 != expected_wheel_sha256:
        raise ContractError("run manifest wheel differs from the expected candidate")
    if expected_package_version is not None and package_version != expected_package_version:
        raise ContractError("run manifest package version differs from the expected candidate")
    return wheel_sha256, package_version


def _verified_installation_source(root: Path, manifest: dict[str, object], package_version: str) -> str:
    """Validate isolated installation and return its package source identity."""
    installation = read_json(root / "installation.json")
    if (
        installation.get("schema") != "frem-s12-isolated-install-v1"
        or installation.get("returncode") != 0
        or installation.get("verification_returncode") != 0
        or installation.get("thread_environment") != manifest.get("thread_environment")
    ):
        raise ContractError("isolated installation receipt did not pass")
    verification_stdout = installation.get("verification_stdout")
    if not isinstance(verification_stdout, str):
        raise ContractError("isolated installation verification output is absent")
    try:
        installed = json.loads(verification_stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("isolated installation verification output is invalid") from exc
    package = installed.get("package") if isinstance(installed, dict) else None
    installed_source = package.get("sha256") if isinstance(package, dict) else None
    if (
        not isinstance(installed, dict)
        or installed.get("version") != package_version
        or not isinstance(installed_source, str)
        or _SHA256.fullmatch(installed_source) is None
    ):
        raise ContractError("isolated installation identity is inconsistent")
    return installed_source


def _verify_receipts(
    root: Path, manifest: dict[str, object], wheel_sha256: str, package_version: str
) -> tuple[dict[str, object], str, str, bool]:
    """Cross-check command, preflight and optional environment receipts."""
    command = read_json(root / "command.json")
    if (
        command.get("schema") != "frem-s12-command-result-v1"
        or command.get("command") != manifest.get("command")
        or isinstance(command.get("returncode"), bool)
        or not isinstance(command.get("returncode"), int)
    ):
        raise ContractError("run command receipt is inconsistent")
    preflight = read_json(root / "preflight.json")
    if (
        preflight.get("schema_version") != "frem-s12-kaggle-preflight-v3"
        or preflight.get("mode") != manifest.get("mode")
        or preflight.get("execute") != manifest.get("execute")
        or preflight.get("package_version") != package_version
        or preflight.get("wheel_expected_sha256") != wheel_sha256
        or preflight.get("wheel_observed_sha256") != wheel_sha256
        or preflight.get("input_inventory_sha256") != manifest.get("input_inventory_sha256")
    ):
        raise ContractError("run preflight differs from the final manifest")
    input_files = preflight.get("input_files")
    if not isinstance(input_files, dict):
        raise ContractError("run preflight input inventory is invalid")
    input_digest = hashlib.sha256(
        json.dumps(input_files, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if input_digest != manifest.get("input_inventory_sha256"):
        raise ContractError("run preflight input inventory digest does not reconstruct")
    installed_source = _verified_installation_source(root, manifest, package_version)
    environment_path = root / "environment.json"
    admitted = False
    if environment_path.is_file():
        environment = read_json(environment_path)
        if environment.get("scientific_admitted") is True:
            if manifest.get("mode") != "qualify-kaggle" or command.get("returncode") != 0:
                raise ContractError("scientific admission conflicts with mode or command status")
            validate_kaggle_environment_receipt(environment, expected_source_identity=installed_source)
            admitted = True
    elif manifest.get("mode") == "qualify-kaggle" and command.get("returncode") == 0:
        raise ContractError("successful qualification bundle is missing environment.json")
    return command, input_digest, installed_source, admitted


def verify_run_bundle(
    root: Path,
    *,
    expected_wheel_sha256: str | None = None,
    expected_package_version: str | None = None,
    expected_mode: str | None = None,
) -> dict[str, object]:
    """Verify a complete downloaded bundle without trusting producer summaries."""
    if not root.is_dir() or root.is_symlink():
        raise ContractError("run bundle root must be a regular directory")
    manifest_path = root / "run-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "frem-s12-kaggle-run-manifest-v1":
        raise ContractError("run manifest schema is unsupported")
    mode = manifest.get("mode")
    if not isinstance(mode, str) or mode not in _MODES:
        raise ContractError("run manifest mode is unsupported")
    if expected_mode is not None and mode != expected_mode:
        raise ContractError("run manifest mode differs from the expected target")
    execute = manifest.get("execute")
    if not isinstance(execute, bool):
        raise ContractError("run manifest execute flag is invalid")
    if mode in {"run-search-cell", "run-cell"} and execute is not True:
        raise ContractError("scientific cell bundle lacks explicit execution authority")
    declared = _declared_files(manifest)
    total_bytes = _verify_files(root, manifest_path, declared)
    wheel_sha256, package_version = _candidate_identity(
        manifest, expected_wheel_sha256, expected_package_version
    )
    thread_environment = manifest.get("thread_environment")
    if not isinstance(thread_environment, dict) or any(
        thread_environment.get(name) != "1" for name in _THREAD_VARIABLES
    ):
        raise ContractError("run manifest is not bound to the single-thread reference environment")
    command, input_digest, installed_source, scientific_admitted = _verify_receipts(
        root, manifest, wheel_sha256, package_version
    )
    command_succeeded = command["returncode"] == 0
    eligible = bool(mode == "qualify-kaggle" and command_succeeded and scientific_admitted)
    return {
        "schema": "frem-s12-downloaded-run-verification-v1",
        "verified": True,
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(declared),
        "total_bytes": total_bytes,
        "mode": mode,
        "execute": execute,
        "package_version": package_version,
        "wheel_sha256": wheel_sha256,
        "input_inventory_sha256": input_digest,
        "installed_source_identity": installed_source,
        "command_returncode": command["returncode"],
        "command_succeeded": command_succeeded,
        "scientific_admitted": scientific_admitted,
        "eligible_for_next_stage": eligible,
    }
