"""Metadata-only operation declarations; structural validity grants no execution.

The prospective schemas below bind explicitly selected local artifacts. They do
not substitute for the as-yet-unimplemented operation-specific receipt producer
and scientific audit. No authentication service or additional approval flow is
required. This module does not launch, consume, import models, or read raw data.
"""

import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from frem_s10.errors import ContractError, ExecutionStateError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, canonical_bytes, digest_bytes, object_json, parse_json
from frem_s10.metadata import parse_utc_timestamp, strict_int
from frem_s10.runtime.artifacts import _plain_directory, read_verified_json

DESIGN = "FREM-S10-DESIGN-4-CL16"
SCOPES = ("GENERATED_ENGINEERING", "RETROSPECTIVE_DEVELOPMENT")
ROLES = ("code", "configuration", "data", "calendar", "environment", "verification", "admission")
BOUND_ROLES = ROLES[:5]


@dataclass(frozen=True, slots=True)
class DeclarationValidation:
    """Exact structural provenance and explicit, non-authorizing remaining gates."""

    declaration_id: str
    operation: str
    evidence_scope: str
    artifact_hashes: tuple[tuple[str, str], ...]
    unconsumed: bool
    blockers: tuple[str, ...]
    authorizes_scientific_execution: bool = False
    status: str = "STRUCTURE_VALID_SCIENTIFIC_GATE_CLOSED"


def _schema(value: JSONValue, fields: set[str], label: str) -> JSONObject:
    result = object_json(value)
    if set(result) != fields:
        raise ContractError(f"{label}: exact schema fields required")
    return result


def _hash(value: JSONValue) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("exact lowercase SHA256 identity required")
    return value


def _reference(value: JSONValue, root: Path) -> tuple[Path, str]:
    reference = _schema(value, {"path", "sha256"}, "artifact reference")
    name = reference["path"]
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ContractError("relative artifact path without traversal required")
    path = root / name
    if path.is_symlink() or any(part.is_symlink() for part in path.parents):
        raise IntegrityError("artifact references cannot traverse symlinks")
    return path, _hash(reference["sha256"])


def _read(value: JSONValue, root: Path) -> tuple[JSONObject, str]:
    path, identity = _reference(value, root)
    return object_json(read_verified_json(path, identity)), identity


def _receipt(
    value: JSONObject,
    *,
    kind: str,
    operation: str,
    scope: str,
    bindings: JSONObject,
    extras: set[str] | frozenset[str] = frozenset(),
    version: int = 1,
) -> None:
    fields = {"schema", "design_id", "operation", "evidence_scope", "bindings", "status"} | extras
    _schema(value, fields, kind)
    if (
        value["schema"] != f"s10-{kind}-v{version}"
        or value["design_id"] != DESIGN
        or value["operation"] != operation
        or value["evidence_scope"] != scope
        or value["status"] != "PASS"
        or value["bindings"] != bindings
    ):
        raise IntegrityError(f"{kind}: scope, operation, current identities or PASS mismatch")


def _code(value: JSONObject, source_root: Path) -> None:
    _schema(value, {"schema", "files"}, "code manifest")
    entries = value["files"]
    if value["schema"] != "s10-code-manifest-v1" or not isinstance(entries, list) or not entries:
        raise ContractError("nonempty code manifest required")
    seen: set[Path] = set()
    for entry in entries:
        path, expected = _reference(entry, source_root)
        if path in seen or not (
            path.suffix == ".py" or path.name in ("science.json", "py.typed", "pyproject.toml", "uv.lock")
        ):
            raise IntegrityError("duplicate or unsupported source-code artifact")
        seen.add(path)
        if not path.is_file() or path.stat().st_size > 1_048_576:
            raise IntegrityError("source-code artifact absent or oversized")
        if digest_bytes(path.read_bytes()) != expected:
            raise IntegrityError("current code differs from verification-bound manifest")


def _budgets(value: JSONValue, operation: str) -> None:
    limits = _schema(
        value, {"wall_seconds", "rss_mib", "workers", "threads", "weekly_origins", "ledgers"}, "budget"
    )
    ceilings = {
        "wall_seconds": 600 if operation == "prepare-precision" else 3600,
        "rss_mib": 256,
        "workers": 1,
        "threads": 1,
        "weekly_origins": 1000,
        "ledgers": 0 if operation == "prepare-precision" else 192,
    }
    for key, maximum in ceilings.items():
        number = strict_int(limits[key], name=key, minimum=0 if key == "ledgers" else 1)
        if number > maximum or (key == "ledgers" and number != maximum):
            raise ContractError(f"{key}: operation-specific frozen budget violated")


def _declaration(value: JSONObject) -> tuple[str, str, str]:
    _schema(
        value,
        {
            "schema",
            "design_id",
            "operation",
            "evidence_scope",
            "created_at",
            "values_before",
            "artifacts",
            "budget",
            "precision",
        },
        "declaration",
    )
    operation, scope = value["operation"], value["evidence_scope"]
    if value["schema"] != "s10-operation-declaration-v1" or value["design_id"] != DESIGN:
        raise ContractError("exact design4 declaration required")
    if operation not in ("prepare-precision", "development-campaign") or scope not in SCOPES:
        raise ContractError("unsupported operation or evidence scope")
    assert isinstance(operation, str) and isinstance(scope, str)
    cutoff = "2010-01-01" if operation == "prepare-precision" else "2017-01-01"
    if value["values_before"] != cutoff:
        raise ContractError("operation-specific fixed source cutoff violated")
    parse_utc_timestamp(value["created_at"])
    _budgets(value["budget"], operation)
    if operation == "prepare-precision" and value["precision"] is not None:
        raise ContractError("precision cannot require its own prior result")
    return operation, scope, cutoff


def _artifacts(
    declaration: JSONObject,
    root: Path,
    source_root: Path,
) -> tuple[dict[str, JSONObject], JSONObject]:
    refs = _schema(declaration["artifacts"], set(ROLES), "artifact registry")
    loaded = {role: _read(refs[role], root) for role in ROLES}
    documents = {role: pair[0] for role, pair in loaded.items()}
    identities: JSONObject = {role: pair[1] for role, pair in loaded.items()}
    _code(documents["code"], source_root)
    expected = files("frem_s10").joinpath("science.json").read_bytes()
    if canonical_bytes(documents["configuration"]) != canonical_bytes(parse_json(expected)):
        raise IntegrityError("configuration differs from exact installed design4 specification")
    return documents, identities


def _validate_prerequisites(
    declaration: JSONObject,
    documents: dict[str, JSONObject],
    identities: JSONObject,
    root: Path,
) -> None:
    operation, scope, cutoff = _declaration(declaration)
    data = _schema(documents["data"], {"schema", "assets", "source_hashes", "values_before"}, "data manifest")
    if (
        data["schema"] != "s10-data-manifest-v1"
        or data["assets"] != ["SPY", "TLT", "GLD"]
        or data["values_before"] != cutoff
    ):
        raise IntegrityError("data manifest assets or source scope mismatch")
    hashes = data["source_hashes"]
    if not isinstance(hashes, list) or len(hashes) != 3 or len({_hash(item) for item in hashes}) != 3:
        raise IntegrityError("three distinct exact raw source identities required")
    bindings = {role: identities[role] for role in BOUND_ROLES}
    for role in ("calendar", "environment"):
        receipt = _schema(
            documents[role], {"schema", "design_id", "evidence_scope", "status", "identity"}, role
        )
        if (
            receipt["schema"] != f"s10-{role}-receipt-v1"
            or receipt["design_id"] != DESIGN
            or receipt["evidence_scope"] != scope
            or receipt["status"] != "PASS"
        ):
            raise IntegrityError(f"{role} receipt scope or schema mismatch")
        _hash(receipt["identity"])
    _receipt(
        documents["verification"],
        kind="verification-receipt",
        operation=operation,
        scope=scope,
        bindings=bindings,
        extras={"logs"},
    )
    logs = documents["verification"]["logs"]
    if not isinstance(logs, list) or not logs:
        raise IntegrityError("verification requires referenced retained result artifacts")
    for log in logs:
        output, _ = _read(log, root)
        version = 2 if output.get("schema") == "s10-test-result-v2" else 1
        fields = {"schema", "status", "code_sha256", "environment_sha256"}
        _schema(output, fields | ({"evidence"} if version == 2 else set()), "test result")
        expected_result = {
            "schema": f"s10-test-result-v{version}",
            "status": "PASS",
            "code_sha256": identities["code"],
            "environment_sha256": identities["environment"],
        }
        if any(output[key] != value for key, value in expected_result.items()):
            raise IntegrityError("test result is not a current passing code/environment result")
    _receipt(
        documents["admission"],
        kind="admission-receipt",
        operation=operation,
        scope=scope,
        bindings=bindings,
        extras={"review", "verification_sha256"},
    )
    if documents["admission"]["verification_sha256"] != identities["verification"]:
        raise IntegrityError("admission does not bind the current verification receipt")
    review, _ = _read(documents["admission"]["review"], root)
    review_version = 2 if review.get("schema") == "s10-operation-review-v2" else 1
    _receipt(
        review,
        kind="operation-review",
        operation=operation,
        scope=scope,
        bindings=bindings | {"verification": identities["verification"]},
        extras={"source_admission", "verification"} if review_version == 2 else frozenset(),
        version=review_version,
    )


def _precision(
    declaration: JSONObject,
    identities: JSONObject,
    documents: dict[str, JSONObject],
    root: Path,
    source_root: Path,
) -> None:
    receipt, _ = _read(declaration["precision"], root)
    if receipt.get("schema") == "s10-precision-prerequisite-v2":
        _precision_v2(receipt, declaration, identities, documents, root, source_root)
        return
    _schema(
        receipt,
        {"schema", "state", "status", "evidence_scope", "declaration", "completion"},
        "precision prerequisite",
    )
    if (
        receipt["schema"] != "s10-precision-prerequisite-v1"
        or receipt["state"] != "COMPLETE"
        or receipt["status"] != "PRECISION_PASS"
        or receipt["evidence_scope"] != declaration["evidence_scope"]
    ):
        raise IntegrityError("campaign requires complete passing precision in the same evidence scope")
    prior, prior_id = _read(receipt["declaration"], root)
    if (
        _declaration(prior)[0] != "prepare-precision"
        or prior["evidence_scope"] != declaration["evidence_scope"]
    ):
        raise IntegrityError("precision prerequisite is not its distinct pilot declaration")
    if parse_utc_timestamp(prior["created_at"]) > parse_utc_timestamp(declaration["created_at"]):
        raise IntegrityError("precision declaration postdates campaign freeze")
    prior_documents, prior_identities = _artifacts(prior, root, source_root)
    _validate_prerequisites(prior, prior_documents, prior_identities, root)
    prior_refs = _schema(prior["artifacts"], set(ROLES), "precision artifact registry")
    for role in ("code", "configuration", "calendar", "environment"):
        if _reference(prior_refs[role], root)[1] != identities[role]:
            raise IntegrityError("precision used different code/config/calendar/environment")
    prior_data, _ = _read(prior_refs["data"], root)
    if prior_data.get("source_hashes") != documents["data"]["source_hashes"]:
        raise IntegrityError("precision and campaign raw source identities differ")
    completion, _ = _read(receipt["completion"], root)
    _schema(
        completion,
        {
            "schema",
            "declaration_sha256",
            "operation",
            "state",
            "precision_status",
            "evidence_scope",
            "consumption",
        },
        "precision completion",
    )
    if completion != completion | {
        "schema": "s10-precision-completion-v1",
        "declaration_sha256": prior_id,
        "operation": "prepare-precision",
        "state": "COMPLETE",
        "precision_status": "PRECISION_PASS",
        "evidence_scope": declaration["evidence_scope"],
    }:
        raise IntegrityError("precision completion is not bound to the exact pilot declaration")
    consumption, _ = _read(completion["consumption"], root)
    _schema(consumption, {"schema", "operation", "declaration_sha256", "state"}, "precision consumption")
    if consumption != {
        "schema": "s10-operation-consumption-v1",
        "operation": "prepare-precision",
        "declaration_sha256": prior_id,
        "state": "CONSUMED",
    }:
        raise IntegrityError("precision completion lacks its distinct consumed identity")


def _precision_v2(
    receipt: JSONObject,
    declaration: JSONObject,
    identities: JSONObject,
    documents: dict[str, JSONObject],
    root: Path,
    source_root: Path,
) -> None:
    """Bind actual supervised pilot evidence without inventing a normalized consumption marker."""
    _schema(
        receipt,
        {"schema", "state", "status", "evidence_scope", "declaration", "completion"},
        "precision prerequisite",
    )
    if (receipt["state"], receipt["status"], receipt["evidence_scope"]) != (
        "COMPLETE",
        "PRECISION_PASS",
        declaration["evidence_scope"],
    ):
        raise IntegrityError("campaign requires complete passing precision in its exact source scope")
    prior, prior_id = _read(receipt["declaration"], root)
    if (
        _declaration(prior)[0] != "prepare-precision"
        or prior["evidence_scope"] != declaration["evidence_scope"]
    ):
        raise IntegrityError("precision prerequisite operation or scope differs")
    if parse_utc_timestamp(prior["created_at"]) > parse_utc_timestamp(declaration["created_at"]):
        raise IntegrityError("precision declaration postdates campaign freeze")
    prior_documents, prior_identities = _artifacts(prior, root, source_root)
    _validate_prerequisites(prior, prior_documents, prior_identities, root)
    for role in ("code", "configuration", "calendar", "environment"):
        if prior_identities[role] != identities[role]:
            raise IntegrityError("precision used different code/config/calendar/environment")
    if prior_documents["data"]["source_hashes"] != documents["data"]["source_hashes"]:
        raise IntegrityError("precision and campaign source identities differ")
    completion, _ = _read(receipt["completion"], root)
    if (
        completion.get("schema"),
        completion.get("state"),
        completion.get("operation"),
        completion.get("declaration_sha256"),
        completion.get("scientific_disposition"),
    ) != ("s10-supervised-result-v1", "COMPLETE", "prepare-precision", prior_id, "PRECISION_PASS"):
        raise IntegrityError("precision result is not bound to its complete supervised operation")
    _verify_precision_execution(completion, prior_id, root)


def _verify_precision_execution(completion: JSONObject, prior_id: str, root: Path) -> None:
    """Verify the real marker, controller completion and retained worker evidence links."""
    marker, _ = _read(completion["consumption"], root)
    supervisor, _ = _read(completion["supervisor"], root)
    worker, _ = _read(completion["worker_evidence"], root)
    if (
        marker.get("operation"),
        marker.get("freeze_id"),
        marker.get("state"),
        marker.get("resume_permitted"),
    ) != ("precision", prior_id, "CONSUMED", False):
        raise IntegrityError("precision persistent consumption identity differs")
    if (
        supervisor.get("state"),
        supervisor.get("operation"),
        supervisor.get("freeze_id"),
        supervisor.get("worker_exit_code"),
        supervisor.get("consumed"),
    ) != ("COMPLETE", "precision", prior_id, 0, True) or marker.get("attempt_id") != supervisor.get(
        "attempt_id"
    ):
        raise IntegrityError("precision supervision did not complete the consumed attempt")
    if strict_int(supervisor.get("resource_observations"), name="resource observations", minimum=1) < 1:
        raise IntegrityError("precision lacks resource observations")
    if (worker.get("operation"), worker.get("declaration_sha256"), worker.get("scientific_disposition")) != (
        "prepare-precision",
        prior_id,
        "PRECISION_PASS",
    ):
        raise IntegrityError("precision worker evidence does not match the declared passing result")
    worker_path, _ = _reference(completion["worker_evidence"], root)
    audit, _ = _read(worker["audit"], worker_path.parent)
    evidence, evidence_id = _read(worker["evidence"], worker_path.parent)
    if (
        audit.get("status") != "PRECISION_PASS"
        or audit.get("evidence_sha256") != evidence_id
        or audit.get("result") != evidence.get("result")
    ):
        raise IntegrityError("precision audit differs from retained numerical evidence")
    archive_ref = _schema(worker["model_archive"], {"path", "sha256"}, "precision archive reference")
    archive, archive_id = _read(archive_ref, worker_path.parent)
    if (
        audit.get("schema") != "s10-precision-archive-audit-v1"
        or audit.get("archive_manifest_sha256") != archive_id
        or audit.get("inputs_id") != evidence.get("inputs_id")
        or audit.get("final_state_id") != evidence.get("final_state_id")
        or archive.get("variants") != ["primary"]
    ):
        raise IntegrityError("precision lacks its verified archive-to-score linkage")


def validate_declaration(
    reference: JSONValue,
    *,
    artifact_root: Path,
    source_root: Path,
    consumption_root: Path,
) -> DeclarationValidation:
    """Read and bind explicit metadata artifacts; leave the scientific gate closed.

    Real source/calendar admission, operation-specific receipt production/audit,
    fresh host admission and exclusive consumption are separate missing stages.
    A valid structure or caller-supplied PASS cannot establish those observations.
    """
    root = _plain_directory(artifact_root)
    sources = _plain_directory(source_root)
    consumption = _plain_directory(consumption_root)
    declaration, identity = _read(reference, root)
    operation, scope, _ = _declaration(declaration)
    stem = "precision" if operation == "prepare-precision" else "campaign"
    marker = consumption / f"{stem}-{identity}.json"
    if marker.exists() or marker.is_symlink():
        raise ExecutionStateError("operation identity already consumed or ambiguous")
    documents, identities = _artifacts(declaration, root, sources)
    _validate_prerequisites(declaration, documents, identities, root)
    if operation == "development-campaign":
        _precision(declaration, identities, documents, root, sources)
    blockers: tuple[str, ...] = (
        "OPERATION_SPECIFIC_RECEIPT_PRODUCER_AND_SCIENTIFIC_AUDIT_NOT_IMPLEMENTED",
        "REAL_SOURCE_AND_CALENDAR_NOT_ADMITTED_BY_THIS_STRUCTURAL_CHECK",
        "FRESH_HOST_ADMISSION_AND_EXCLUSIVE_CONSUMPTION_STILL_REQUIRED",
    )
    if scope == "GENERATED_ENGINEERING":
        blockers = ("GENERATED_ENGINEERING_CANNOT_AUTHORIZE_REAL_WORK", *blockers)
    return DeclarationValidation(
        identity, operation, scope, tuple((role, str(identities[role])) for role in ROLES), True, blockers
    )
