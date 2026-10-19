from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from frem_s12.errors import ContractError
from frem_s12.evidence import atomic_json, sha256_file
from frem_s12.run_bundle import verify_run_bundle


def _bundle(root: Path) -> tuple[str, str]:
    root.mkdir()
    wheel = "a" * 64
    version = "0.1.4"
    input_files = {"dataset/wheel.whl": {"sha256": wheel, "size": 42}}
    input_digest = hashlib.sha256(
        json.dumps(input_files, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    command = ["python", "-m", "frem_s12.cli", "generated-smoke"]
    atomic_json(
        root / "preflight.json",
        {
            "schema_version": "frem-s12-kaggle-preflight-v3",
            "mode": "generated-smoke",
            "execute": False,
            "package_version": version,
            "wheel_expected_sha256": wheel,
            "wheel_observed_sha256": wheel,
            "input_files": input_files,
            "input_inventory_sha256": input_digest,
        },
    )
    thread_environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    atomic_json(
        root / "installation.json",
        {
            "schema": "frem-s12-isolated-install-v1",
            "returncode": 0,
            "verification_returncode": 0,
            "verification_stdout": json.dumps({"version": version, "package": {"sha256": "b" * 64}}),
            "thread_environment": thread_environment,
        },
    )
    atomic_json(
        root / "command.json",
        {
            "schema": "frem-s12-command-result-v1",
            "command": command,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    files = {
        path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in sorted(root.iterdir())
    }
    atomic_json(
        root / "run-manifest.json",
        {
            "schema": "frem-s12-kaggle-run-manifest-v1",
            "mode": "generated-smoke",
            "execute": False,
            "package_version": version,
            "wheel_sha256": wheel,
            "input_inventory_sha256": input_digest,
            "thread_environment": thread_environment,
            "command": command,
            "files": files,
        },
    )
    return wheel, version


def test_downloaded_run_bundle_verifies_exact_inventory(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    wheel, version = _bundle(root)
    result = verify_run_bundle(
        root,
        expected_wheel_sha256=wheel,
        expected_package_version=version,
    )
    assert result["verified"] is True
    assert result["command_succeeded"] is True
    assert result["scientific_admitted"] is False


def test_downloaded_run_bundle_rejects_tampering_and_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _bundle(root)
    (root / "command.json").write_text("{}")
    with pytest.raises(ContractError, match="identity mismatch"):
        verify_run_bundle(root)

    other = tmp_path / "other"
    _bundle(other)
    (other / "undeclared.txt").write_text("foreign")
    with pytest.raises(ContractError, match="inventory differs"):
        verify_run_bundle(other)


def test_downloaded_run_bundle_rejects_failed_install_and_wrong_mode(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _bundle(root)
    with pytest.raises(ContractError, match="expected target"):
        verify_run_bundle(root, expected_mode="qualify-kaggle")

    installation = json.loads((root / "installation.json").read_text())
    installation["returncode"] = 1
    (root / "installation.json").write_text(json.dumps(installation))
    manifest = json.loads((root / "run-manifest.json").read_text())
    manifest["files"]["installation.json"] = {
        "sha256": sha256_file(root / "installation.json"),
        "size": (root / "installation.json").stat().st_size,
    }
    (root / "run-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="installation receipt did not pass"):
        verify_run_bundle(root)
