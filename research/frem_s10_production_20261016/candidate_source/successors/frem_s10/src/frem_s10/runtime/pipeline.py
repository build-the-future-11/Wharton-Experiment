"""Fixed local production admission, prospective freeze and supervised dispatch."""

import sys
from datetime import UTC, datetime
from pathlib import Path

from frem_s10.errors import ExecutionStateError, IntegrityError, ResourceLimitError, S10Error
from frem_s10.identities import JSONObject, JSONValue, object_json, parse_json
from frem_s10.metadata import parse_utc_timestamp
from frem_s10.runtime.artifacts import _plain_directory, publish_json, read_verified_json
from frem_s10.runtime.declarations import (
    BOUND_ROLES,
    DESIGN,
    _artifacts,
    _read,
    _reference,
    _schema,
    validate_declaration,
)
from frem_s10.runtime.supervisor import Budget, execute_operation

PILOT_OUTPUT_BYTES = 512 * 2**20
CAMPAIGN_OUTPUT_BYTES = 2 * 2**30


def output_allowance(operation: str) -> int:
    """Return the fixed engineering storage ceiling, independent of scientific values."""
    if operation not in ("prepare-precision", "development-campaign"):
        raise IntegrityError("unsupported operation storage scope")
    return PILOT_OUTPUT_BYTES if operation == "prepare-precision" else CAMPAIGN_OUTPUT_BYTES


def _ref(root: Path, path: Path, identity: str) -> JSONObject:
    return {"path": str(path.relative_to(root)), "sha256": identity}


def _bindings(artifacts: JSONObject, root: Path) -> JSONObject:
    return {role: _reference(artifacts[role], root)[1] for role in BOUND_ROLES}


def audit_operation_review(reference: JSONValue, *, artifact_root: Path, source_root: Path) -> JSONObject:
    """Audit exact local producers; a caller-supplied PASS string is insufficient."""
    from frem_s10.data.production import validate_source_admission
    from frem_s10.runtime.verification import audit_verification

    review, _ = _read(reference, artifact_root)
    if review.get("schema") != "s10-operation-review-v2":
        raise IntegrityError("production requires the operation-specific v2 local audit")
    bindings = object_json(review["bindings"])
    verification = audit_verification(
        review["verification"], artifact_root=artifact_root, source_root=source_root
    )
    if (
        verification.get("bindings") != {role: bindings[role] for role in BOUND_ROLES}
        or verification.get("operation") != review.get("operation")
        or verification.get("evidence_scope") != review.get("evidence_scope")
        or _reference(review["verification"], artifact_root)[1] != bindings.get("verification")
    ):
        raise IntegrityError("verified command evidence differs from operation review bindings")
    source = validate_source_admission(
        dict(object_json(review["source_admission"])), artifact_root=artifact_root, source_root=source_root
    )
    if source.evidence_scope != review["evidence_scope"]:
        raise IntegrityError("source admission and operation scope differ")
    return {
        "bindings": bindings,
        "source_hashes": list(source.source_hashes),
        "calendar_identity": source.calendar_identity,
        "source_admission": review["source_admission"],
    }


def create_declaration(
    request_path: Path, *, artifact_root: Path, source_root: Path, consumption_root: Path, output_name: str
) -> JSONObject:
    """Create a prospective declaration after auditing its actual local prerequisite producers."""
    from frem_s10.data.production import validate_source_admission
    from frem_s10.runtime.verification import audit_verification

    root = _plain_directory(artifact_root)
    request = _schema(
        parse_json(request_path.read_bytes()),
        {"schema", "operation", "evidence_scope", "created_at", "artifacts", "source_admission", "precision"},
        "freeze request",
    )
    if request["schema"] != "s10-freeze-request-v1":
        raise IntegrityError("exact freeze request schema required")
    requested_time = parse_utc_timestamp(request["created_at"])
    if abs(datetime.now(UTC).timestamp() * 1_000_000 - requested_time) > 300_000_000:
        raise IntegrityError(
            "freeze creation time must be prospective and within five minutes of this producer"
        )
    refs = _schema(request["artifacts"], set(BOUND_ROLES) | {"verification"}, "freeze artifacts")
    source = validate_source_admission(
        dict(object_json(request["source_admission"])), artifact_root=root, source_root=source_root
    )
    audit_verification(refs["verification"], artifact_root=root, source_root=source_root)
    data, _ = _read(refs["data"], root)
    calendar, _ = _read(refs["calendar"], root)
    if (
        data.get("source_hashes") != list(source.source_hashes)
        or calendar.get("identity") != source.calendar_identity
        or source.evidence_scope != request["evidence_scope"]
    ):
        raise IntegrityError("source/calendar identities differ from operation bindings")
    if (
        not output_name
        or Path(output_name).is_absolute()
        or ".." in Path(output_name).parts
        or output_name in (".", "..")
    ):
        raise IntegrityError("new relative freeze directory without traversal required")
    output = root / output_name
    _plain_directory(output.parent)
    output.mkdir(exist_ok=False)
    binding = _bindings(refs, root)
    review: JSONObject = {
        "schema": "s10-operation-review-v2",
        "design_id": DESIGN,
        "operation": request["operation"],
        "evidence_scope": request["evidence_scope"],
        "status": "PASS",
        "bindings": binding | {"verification": _reference(refs["verification"], root)[1]},
        "source_admission": request["source_admission"],
        "verification": refs["verification"],
    }
    review_id = publish_json(output, "review.json", review)
    admission: JSONObject = {
        "schema": "s10-admission-receipt-v1",
        "design_id": DESIGN,
        "operation": request["operation"],
        "evidence_scope": request["evidence_scope"],
        "status": "PASS",
        "bindings": binding,
        "review": _ref(root, output / "review.json", review_id),
        "verification_sha256": _reference(refs["verification"], root)[1],
    }
    admission_id = publish_json(output, "admission.json", admission)
    pilot = request["operation"] == "prepare-precision"
    declaration: JSONObject = {
        "schema": "s10-operation-declaration-v1",
        "design_id": DESIGN,
        "operation": request["operation"],
        "evidence_scope": request["evidence_scope"],
        "created_at": request["created_at"],
        "values_before": "2010-01-01" if pilot else "2017-01-01",
        "artifacts": refs | {"admission": _ref(root, output / "admission.json", admission_id)},
        "budget": {
            "wall_seconds": 600 if pilot else 3600,
            "rss_mib": 256,
            "workers": 1,
            "threads": 1,
            "weekly_origins": 1000,
            "ledgers": 0 if pilot else 192,
        },
        "precision": request["precision"],
    }
    identity = publish_json(output, "declaration.json", declaration)
    reference = _ref(root, output / "declaration.json", identity)
    validate_declaration(
        reference, artifact_root=root, source_root=source_root, consumption_root=consumption_root
    )
    return reference


def validate_production_declaration(
    reference: JSONValue, *, artifact_root: Path, source_root: Path, consumption_root: Path
) -> JSONObject:
    """Validate exact source and verification authority before any supervised dispatch."""
    validation = validate_declaration(
        reference, artifact_root=artifact_root, source_root=source_root, consumption_root=consumption_root
    )
    declaration, identity = _read(reference, artifact_root)
    documents, identities = _artifacts(declaration, artifact_root, source_root)
    audit = audit_operation_review(
        documents["admission"]["review"], artifact_root=artifact_root, source_root=source_root
    )
    if audit["bindings"] != {role: identities[role] for role in (*BOUND_ROLES, "verification")}:
        raise IntegrityError("operation review differs from exact frozen artifact bindings")
    if (
        audit["source_hashes"] != documents["data"]["source_hashes"]
        or audit["calendar_identity"] != documents["calendar"]["identity"]
    ):
        raise IntegrityError("operation source admission differs from frozen source/calendar")
    if validation.evidence_scope != "RETROSPECTIVE_DEVELOPMENT":
        raise IntegrityError("production dispatch requires explicit retrospective development scope")
    if validation.operation == "development-campaign":
        audit_precision_prerequisite(declaration, artifact_root=artifact_root, source_root=source_root)
    return {
        "declaration": declaration,
        "declaration_sha256": identity,
        "source_admission": audit["source_admission"],
        "operation": validation.operation,
    }


def audit_precision_prerequisite(declaration: JSONObject, *, artifact_root: Path, source_root: Path) -> None:
    """Reject structural-only historical fixtures as real campaign pilot authority."""
    receipt, _ = _read(declaration["precision"], artifact_root)
    if receipt.get("schema") != "s10-precision-prerequisite-v2":
        raise IntegrityError("production campaign requires its actual supervised v2 pilot evidence")
    prior, _ = _read(receipt["declaration"], artifact_root)
    documents, _ = _artifacts(prior, artifact_root, source_root)
    audit_operation_review(
        documents["admission"]["review"], artifact_root=artifact_root, source_root=source_root
    )
    completion, _ = _read(receipt["completion"], artifact_root)
    from frem_s10.runtime.receipts import audit_execution

    audit_execution(completion, artifact_root=artifact_root)


def run_declared(manifest_path: Path, *, expected_operation: str) -> JSONObject:
    """Bind a fixed worker command to one reviewed declaration and consume only after the host gate."""
    request = _schema(
        parse_json(manifest_path.read_bytes()),
        {
            "schema",
            "declaration",
            "artifact_root",
            "source_root",
            "consumption_root",
            "output_root",
            "attempt_id",
        },
        "launch manifest",
    )
    if request["schema"] != "s10-launch-manifest-v1":
        raise IntegrityError("exact launch manifest required")
    roots = {
        name: _plain_directory(Path(str(request[name])))
        for name in ("artifact_root", "source_root", "consumption_root", "output_root")
    }
    prepared = validate_production_declaration(
        request["declaration"],
        artifact_root=roots["artifact_root"],
        source_root=roots["source_root"],
        consumption_root=roots["consumption_root"],
    )
    if prepared["operation"] != expected_operation:
        raise ExecutionStateError("requested command and declaration operation differ")
    declaration = object_json(prepared["declaration"])
    budget = object_json(declaration["budget"])
    path, identity = _reference(request["declaration"], roots["artifact_root"])
    operation = "precision" if expected_operation == "prepare-precision" else "campaign"
    attempt_id = str(request["attempt_id"])
    output = roots["output_root"] / attempt_id
    if not output.is_relative_to(roots["artifact_root"]) or not roots["consumption_root"].is_relative_to(
        roots["artifact_root"]
    ):
        raise IntegrityError("output and consumption evidence must lie inside the declared artifact root")
    if output.is_relative_to(roots["source_root"] / "data") or output.is_relative_to(
        roots["consumption_root"]
    ):
        raise IntegrityError("operation output cannot overlap source data or consumption storage")
    command = (
        sys.executable,
        "-B",
        "-X",
        f"pycache_prefix={output / 'empty_bytecode'}",
        "-m",
        "frem_s10.runtime.worker",
        "--declaration",
        str(path),
        "--sha256",
        identity,
        "--artifact-root",
        str(roots["artifact_root"]),
        "--source-root",
        str(roots["source_root"]),
        "--consumption-root",
        str(roots["consumption_root"]),
        "--attempt-id",
        attempt_id,
    )
    try:
        result = execute_operation(
            command=command,
            operation=operation,
            freeze_id=identity,
            attempt_id=attempt_id,
            output_root=roots["output_root"],
            consumption_root=roots["consumption_root"],
            input_hashes=((path, identity),),
            estimated_output_bytes=output_allowance(expected_operation),
            budget=Budget(
                wall_seconds=int(str(budget["wall_seconds"])),
                rss_mib=int(str(budget["rss_mib"])),
                output_bytes=output_allowance(expected_operation),
            ),
        )
    except ResourceLimitError:
        from frem_s10.runtime.receipts import read_unconsumed_refusal

        return read_unconsumed_refusal(
            output, consumption_root=roots["consumption_root"], operation=operation, freeze_id=identity
        )
    if result["state"] == "COMPLETE":
        try:
            return finalize_operation(
                output,
                declaration_ref=request["declaration"],
                artifact_root=roots["artifact_root"],
                consumption_root=roots["consumption_root"],
                declaration=declaration,
                declaration_id=identity,
            )
        except (S10Error, OSError) as exc:
            return _finalization_failure(output, operation, identity, exc)
    return result


def _finalization_failure(output: Path, operation: str, identity: str, error: Exception) -> JSONObject:
    result: JSONObject = {
        "state": "FAILED_CONSUMED",
        "consumed": True,
        "operation": operation,
        "freeze_id": identity,
        "error": f"{type(error).__name__}: {error}",
        "automatic_retry_permitted": False,
    }
    try:
        publish_json(output, "finalization-failure.json", result)
    except (S10Error, OSError) as exc:
        result["retention_error"] = f"{type(exc).__name__}: {exc}"
    return result


def finalize_operation(
    output: Path,
    *,
    declaration_ref: JSONValue,
    artifact_root: Path,
    consumption_root: Path,
    declaration: JSONObject,
    declaration_id: str,
) -> JSONObject:
    """Join worker evidence to successful supervision and actual persistent consumption."""
    from frem_s10.identities import digest_bytes

    supervisor_id = digest_bytes((output / "completion.json").read_bytes())
    supervisor = object_json(read_verified_json(output / "completion.json", supervisor_id))
    worker_id = digest_bytes((output / "worker-evidence.json").read_bytes())
    worker = object_json(read_verified_json(output / "worker-evidence.json", worker_id))
    if supervisor.get("state") != "COMPLETE" or worker.get("declaration_sha256") != declaration_id:
        raise IntegrityError("successful supervised exact-declaration evidence required")
    operation = str(declaration["operation"])
    stem = "precision" if operation == "prepare-precision" else "campaign"
    marker = consumption_root / f"{stem}-{declaration_id}.json"
    marker_id = digest_bytes(marker.read_bytes())
    result: JSONObject = {
        "schema": "s10-supervised-result-v1",
        "state": "COMPLETE",
        "operation": operation,
        "declaration_sha256": declaration_id,
        "evidence_scope": declaration["evidence_scope"],
        "worker_evidence": _ref(artifact_root, output / "worker-evidence.json", worker_id),
        "supervisor": _ref(artifact_root, output / "completion.json", supervisor_id),
        "consumption": _ref(artifact_root, marker, marker_id),
        "declaration": declaration_ref,
        "scientific_disposition": worker["scientific_disposition"],
    }
    for name, filename in (
        ("preflight", "preflight.json"),
        ("resources", "resources.jsonl"),
        ("timeline", "timeline.jsonl"),
    ):
        result[name] = _ref(artifact_root, output / filename, digest_bytes((output / filename).read_bytes()))
    from frem_s10.runtime.receipts import audit_execution

    audit_execution(result, artifact_root=artifact_root)
    result_id = publish_json(output, "result.json", result)
    if operation == "prepare-precision":
        prerequisite: JSONObject = {
            "schema": "s10-precision-prerequisite-v2",
            "state": "COMPLETE",
            "status": worker["scientific_disposition"],
            "evidence_scope": declaration["evidence_scope"],
            "declaration": declaration_ref,
            "completion": _ref(artifact_root, output / "result.json", result_id),
        }
        prerequisite_id = publish_json(output, "precision-prerequisite.json", prerequisite)
        result["precision_prerequisite"] = _ref(
            artifact_root, output / "precision-prerequisite.json", prerequisite_id
        )
    return result
