"""One-way Kaggle S10 retrospective production orchestration.

The notebook calls ``verify`` while scientific values remain encrypted, then
``execute`` only after supplying the separately retained decryption key.  Every
scientific operation uses the package's existing one-use consumption markers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_once(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def relative(reference: dict[str, str], root: Path) -> dict[str, str]:
    return {"path": str(Path(reference["path"]).relative_to(root)), "sha256": reference["sha256"]}


def read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {path}")
    return value


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"exclusive destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)


def tree_hashes(root: Path) -> dict[str, str]:
    """Hash regular payload files while ignoring generated Python caches."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    }


def stage_source(payload: Path, destination: Path) -> None:
    expected = payload / "successors"
    if destination.exists():
        if tree_hashes(destination) != tree_hashes(expected):
            raise RuntimeError("pre-staged source differs from sealed payload")
        return
    copy_tree(expected, destination)


def stage_metadata(payload: Path, artifact_root: Path) -> None:
    for item in (payload / "metadata").iterdir():
        target = artifact_root / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def prepare_catalog(payload: Path, artifact_root: Path) -> dict[str, str]:
    from frem_s10.runtime.provenance import save_object

    catalog = read(payload / "catalog.json")
    rows = catalog.get("obligations")
    if not isinstance(rows, list):
        raise RuntimeError("catalog obligations missing")
    row = next(item for item in rows if isinstance(item, dict) and item.get("test_id") == "TEST-089")
    row.update(
        {
            "coverage_review": "PARTIAL",
            "execution_state": "REQUIRES_CURRENT_KAGGLE_OBSERVATIONS",
            "remaining_evidence": "Requires unchanged namespace isolation and full ancestor memory evidence; namespace-only probes cannot close this obligation.",
            "test_families": [
                "tests/test_kaggle_production_contract_audit.py::test_contract_is_opt_in_and_rejects_enabled_network",
                "tests/test_kaggle_production_contract_audit.py::test_kaggle_resource_receipts_require_reserve_and_stable_pressure",
                "tests/test_kaggle_production_contract_audit.py::test_verification_requires_bubblewrap_even_with_kaggle_marker",
            ],
        }
    )
    catalog["evidence_scope"] = "RETROSPECTIVE_DEVELOPMENT"
    catalog["authorizes_scientific_execution"] = False
    return dict(save_object(artifact_root, catalog))


def verification(root: Path, payload: Path, sealed: Path) -> dict[str, object]:
    from frem_s10.identities import parse_json
    from frem_s10.runtime.provenance import (
        create_code_manifest,
        create_environment_receipt,
        save_object,
    )
    from frem_s10.runtime.verification import audit_verification, run_verification

    source = root / "source"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    stage_source(payload, source / "successors")
    stage_metadata(payload, artifacts)
    os.environ.update(
        FREM_S10_EXECUTION_CONTRACT="s10-kaggle-production-v1",
        FREM_S10_SEALED_INPUT=str(sealed),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )
    code = dict(create_code_manifest(artifacts, source))
    environment = dict(
        create_environment_receipt(artifacts, source, evidence_scope="RETROSPECTIVE_DEVELOPMENT")
    )
    configuration = dict(
        save_object(
            artifacts, parse_json((source / "successors/frem_s10/src/frem_s10/science.json").read_bytes())
        )
    )
    catalog = prepare_catalog(payload, artifacts)
    fixed = read(payload / "fixed-references.json")
    bindings = {
        "code": code["sha256"],
        "configuration": configuration["sha256"],
        "data": str(dict(fixed["data_pilot"])["sha256"]),
        "calendar": str(dict(fixed["calendar"])["sha256"]),
        "environment": environment["sha256"],
    }
    reference = dict(
        run_verification(
            artifact_root=artifacts,
            source_root=source,
            code_ref=code,
            environment_ref=environment,
            catalog_ref=catalog,
            bindings=bindings,
            operation="prepare-precision",
            evidence_scope="RETROSPECTIVE_DEVELOPMENT",
            output_directory="verification/checks",
        )
    )
    audit_verification(reference, artifact_root=artifacts, source_root=source)
    state = {
        "schema": "s10-kaggle-production-state-v1",
        "stage": "VERIFICATION_PASS",
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(source),
        "artifact_root": str(artifacts),
        "sealed_input": {"path": str(sealed), "sha256": hashlib.sha256(sealed.read_bytes()).hexdigest()},
        "code": code,
        "environment": environment,
        "configuration": configuration,
        "catalog": catalog,
        "verification": reference,
        "fixed": fixed,
        "scientific_values_staged": False,
        "scientific_attempt_consumed": False,
    }
    write_once(root / "state-verification.json", state)
    return state


def decrypt_values(payload: Path, root: Path, key: str) -> None:
    encrypted = payload / "scientific-values.tar.enc"
    archive = root / "scientific-values.tar"
    subprocess.run(
        [
            "openssl",
            "enc",
            "-d",
            "-aes-256-cbc",
            "-pbkdf2",
            "-pass",
            f"pass:{key}",
            "-in",
            str(encrypted),
            "-out",
            str(archive),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(root / "source")], check=True, timeout=30)
    archive.unlink()


def save_manifest(path: Path, value: dict[str, object]) -> None:
    write_once(path, value)


def freeze(
    *,
    root: Path,
    state: dict[str, object],
    operation: str,
    verification_ref: object,
    source_admission: object,
    precision: object,
    data_ref: object,
    name: str,
) -> dict[str, str]:
    from frem_s10.runtime.pipeline import create_declaration

    artifacts = Path(str(state["artifact_root"]))
    fixed = dict(state["fixed"])
    request = {
        "schema": "s10-freeze-request-v1",
        "operation": operation,
        "evidence_scope": "RETROSPECTIVE_DEVELOPMENT",
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": {
            "code": state["code"],
            "configuration": state["configuration"],
            "data": data_ref,
            "calendar": fixed["calendar"],
            "environment": state["environment"],
            "verification": verification_ref,
        },
        "source_admission": source_admission,
        "precision": precision,
    }
    request_path = artifacts / name / "freeze-request.json"
    save_manifest(request_path, request)
    return dict(
        create_declaration(
            request_path,
            artifact_root=artifacts,
            source_root=Path(str(state["source_root"])),
            consumption_root=artifacts / "consumption",
            output_name=f"{name}/freeze",
        )
    )


def launch(
    root: Path, state: dict[str, object], declaration: object, operation: str, attempt: str
) -> dict[str, object]:
    from frem_s10.runtime.pipeline import run_declared

    artifacts = Path(str(state["artifact_root"]))
    manifest = {
        "schema": "s10-launch-manifest-v1",
        "declaration": declaration,
        "artifact_root": str(artifacts),
        "source_root": str(state["source_root"]),
        "consumption_root": str(artifacts / "consumption"),
        "output_root": str(artifacts / "outputs"),
        "attempt_id": attempt,
    }
    path = artifacts / attempt / "launch.json"
    save_manifest(path, manifest)
    return dict(run_declared(path, expected_operation=operation))


def execute(root: Path, payload: Path, key: str) -> dict[str, object]:
    from frem_s10.data.production import create_source_admission
    from frem_s10.runtime.qualification import prepare_qualification, run_qualification
    from frem_s10.runtime.verification import rebind_campaign_verification

    state = read(root / "state-verification.json")
    if state.get("stage") != "VERIFICATION_PASS" or state.get("scientific_values_staged") is not False:
        raise RuntimeError("fresh verified unconsumed state required")
    artifacts, source = Path(str(state["artifact_root"])), Path(str(state["source_root"]))
    sealed_input = dict(state["sealed_input"])
    os.environ.update(
        FREM_S10_EXECUTION_CONTRACT="s10-kaggle-production-v1",
        FREM_S10_SEALED_INPUT=str(sealed_input["path"]),
    )
    decrypt_values(payload, root, key)
    fixed = dict(state["fixed"])
    request_source = artifacts / str(dict(fixed["source_request"])["path"])
    admission = dict(
        create_source_admission(
            request_source,
            artifact_root=artifacts,
            source_root=source,
            output_name="source-admission-current",
        )
    )
    pilot = dict(fixed["data_pilot"])
    campaign = dict(fixed["data_campaign"])
    precision_declaration = freeze(
        root=root,
        state=state,
        operation="prepare-precision",
        verification_ref=state["verification"],
        source_admission=admission,
        precision=None,
        data_ref=pilot,
        name="precision",
    )
    precision = launch(root, state, precision_declaration, "prepare-precision", "precision-attempt-1")
    if precision.get("scientific_disposition") != "PRECISION_PASS":
        state.update(
            stage="SCIENTIFIC_STOP_PRECISION",
            scientific_values_staged=True,
            scientific_attempt_consumed=True,
            precision=precision,
        )
        write_once(root / "state-terminal.json", state)
        return state
    qualification_request = dict(
        prepare_qualification(
            artifact_root=artifacts,
            source_root=source,
            code_ref=state["code"],
            environment_ref=state["environment"],
            calendar_ref=fixed["calendar"],
            clock_ref=fixed["clock"],
        )
    )
    qmanifest = {
        "schema": "s10-qualification-launch-v1",
        "request": qualification_request,
        "artifact_root": str(artifacts),
        "source_root": str(source),
        "consumption_root": str(artifacts / "qualification-consumption"),
        "output_root": str(artifacts / "qualification-outputs"),
        "attempt_id": "qualification-attempt-1",
    }
    qpath = artifacts / "qualification-launch.json"
    save_manifest(qpath, qmanifest)
    qualification = dict(run_qualification(qpath))
    if qualification.get("status") != "PASS":
        state.update(
            stage="FAILED_OR_BLOCKED_QUALIFICATION",
            scientific_values_staged=True,
            scientific_attempt_consumed=True,
            precision=precision,
            qualification=qualification,
        )
        write_once(root / "state-terminal.json", state)
        return state
    campaign_verification = dict(
        rebind_campaign_verification(
            state["verification"],
            artifact_root=artifacts,
            source_root=source,
            bindings={
                "code": dict(state["code"])["sha256"],
                "configuration": dict(state["configuration"])["sha256"],
                "data": campaign["sha256"],
                "calendar": dict(fixed["calendar"])["sha256"],
                "environment": dict(state["environment"])["sha256"],
            },
            qualification=qualification["receipt"],
        )
    )
    campaign_declaration = freeze(
        root=root,
        state=state,
        operation="development-campaign",
        verification_ref=campaign_verification,
        source_admission=admission,
        precision=precision["precision_prerequisite"],
        data_ref=campaign,
        name="campaign",
    )
    result = launch(root, state, campaign_declaration, "development-campaign", "campaign-attempt-1")
    state.update(
        stage="COMPLETE" if result.get("state") == "COMPLETE" else "FAILED_CONSUMED_CAMPAIGN",
        scientific_values_staged=True,
        scientific_attempt_consumed=True,
        precision=precision,
        qualification=qualification,
        campaign=result,
    )
    write_once(root / "state-terminal.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--sealed", type=Path)
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.command == "verify":
        if args.sealed is None:
            raise SystemExit("--sealed required")
        result = verification(args.root, args.payload, args.sealed)
    else:
        if not args.key:
            raise SystemExit("--key required")
        result = execute(args.root, args.payload, args.key)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
