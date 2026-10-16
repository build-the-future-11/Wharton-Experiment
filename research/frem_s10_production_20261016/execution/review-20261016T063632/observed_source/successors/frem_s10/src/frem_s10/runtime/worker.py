"""Fixed consumed-operation worker; no source-selected executable or recovery route."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from frem_s10.errors import ExecutionStateError, IntegrityError, S10Error
from frem_s10.identities import JSONObject, digest_bytes, object_json
from frem_s10.runtime.artifacts import publish_json, read_verified_json
from frem_s10.runtime.declarations import _artifacts, _declaration, _precision, _validate_prerequisites

if TYPE_CHECKING:
    from frem_s10.data.precision import PrecisionInputs


def execute_worker(
    *,
    declaration_path: Path,
    declaration_sha256: str,
    artifact_root: Path,
    source_root: Path,
    consumption_root: Path,
    attempt_id: str,
    output: Path,
) -> JSONObject:
    """Verify the exact persistent consumed attempt before importing numerical work."""
    declaration = object_json(read_verified_json(declaration_path, declaration_sha256))
    operation, scope, _ = _declaration(declaration)
    marker_path = (
        consumption_root
        / f"{'precision' if operation == 'prepare-precision' else 'campaign'}-{declaration_sha256}.json"
    )
    marker = object_json(read_verified_json(marker_path, digest_bytes(marker_path.read_bytes())))
    if (
        marker.get("state"),
        marker.get("freeze_id"),
        marker.get("attempt_id"),
        marker.get("output_path"),
    ) != ("CONSUMED", declaration_sha256, attempt_id, str(output.absolute())) or marker.get(
        "resume_permitted"
    ) is not False:
        raise ExecutionStateError("worker lacks its exact persistent consumed output identity")
    publish_json(
        output,
        "worker-started.json",
        {
            "schema": "s10-worker-start-v1",
            "declaration_sha256": declaration_sha256,
            "attempt_id": attempt_id,
            "operation": operation,
        },
    )
    documents, identities = _artifacts(declaration, artifact_root, source_root)
    _validate_prerequisites(declaration, documents, identities, artifact_root)
    if operation == "development-campaign":
        _precision(declaration, identities, documents, artifact_root, source_root)
    from frem_s10.data.production import validate_source_admission
    from frem_s10.runtime.pipeline import audit_operation_review, audit_precision_prerequisite

    if operation == "development-campaign":
        audit_precision_prerequisite(declaration, artifact_root=artifact_root, source_root=source_root)

    admission = audit_operation_review(
        documents["admission"]["review"], artifact_root=artifact_root, source_root=source_root
    )
    source = validate_source_admission(
        dict(object_json(admission["source_admission"])), artifact_root=artifact_root, source_root=source_root
    )
    publish_json(output, "consumption.json", marker)
    if operation == "development-campaign":
        _reverify_pilot(declaration, artifact_root, source_root, source, output)
        _reverify_qualification(documents["verification"], identities, artifact_root, source_root, output)
    archive_dir = output / "model_archive"
    archive_dir.mkdir(exist_ok=False)
    from frem_s10.experiment.matrix import MODEL_VARIANTS
    from frem_s10.experiment.model_archive import ModelArchiveWriter, verify_model_archive

    writer = ModelArchiveWriter(
        archive_dir,
        experiment_id=attempt_id,
        evidence_scope=scope,
        binding_id=declaration_sha256,
        variants=("primary",) if operation == "prepare-precision" else MODEL_VARIANTS,
    )
    pilot_inputs = None
    if operation == "prepare-precision":
        evidence, pilot_inputs = _run_precision(source, source_root, output, writer)
    else:
        evidence = _run_campaign(source, source_root, output, writer, attempt_id)
    archive_id = writer.finish()
    if pilot_inputs is not None:
        from frem_s10.runtime.results import audit_precision_archive

        evidence_ref = object_json(evidence["evidence"])
        audit = audit_precision_archive(
            output / "precision", str(evidence_ref["sha256"]), archive_dir, archive_id, pilot_inputs
        )
        audit_id = publish_json(output / "precision", "archive-audit.json", audit)
        evidence["audit"] = {"path": "precision/archive-audit.json", "sha256": audit_id}
    else:
        verify_model_archive(archive_dir, archive_id)
    evidence.update(
        {
            "schema": "s10-worker-evidence-v1",
            "operation": operation,
            "evidence_scope": scope,
            "declaration_sha256": declaration_sha256,
            "source_admission_sha256": source.document_identity,
            "model_archive": {"path": "model_archive/manifest.json", "sha256": archive_id},
        }
    )
    publish_json(output, "worker-evidence.json", evidence)
    return evidence


def _run_precision(
    source: object, source_root: Path, output: Path, writer: object
) -> tuple[JSONObject, "PrecisionInputs"]:
    """Run only the fixed primary pilot and audit its retained scores before returning."""
    from frem_s10.data.production import SourceAdmission, load_precision_inputs
    from frem_s10.experiment.model_archive import ModelArchiveWriter
    from frem_s10.experiment.precision import run_precision
    from frem_s10.runtime.progress import ProgressLog, transition_sink
    from frem_s10.runtime.results import audit_precision, publish_precision

    if not isinstance(source, SourceAdmission) or not isinstance(writer, ModelArchiveWriter):
        raise IntegrityError("validated source and durable archive writer required")
    progress = ProgressLog(output / "progress.jsonl", str(writer.header["experiment_id"]))
    progress.emit({"event": "STARTED", "phase": "precision"})
    inputs = load_precision_inputs(source, source_root=source_root)
    try:
        scores = run_precision(inputs, transition_sink=transition_sink(writer, progress))
    except Exception as exc:
        progress.failure("precision", exc)
        raise
    directory = output / "precision"
    directory.mkdir(exist_ok=False)
    evidence_id = publish_precision(
        directory,
        scores,
        projected_a=len(inputs.projected_a.origin_indices),
        projected_b=len(inputs.projected_b.origin_indices),
    )
    audit = audit_precision(directory, evidence_id)
    audit_id = publish_json(directory, "audit.json", audit)
    progress.emit({"event": "PRECISION_AUDITED", "phase": "precision", "status": audit["status"]})
    return {
        "scientific_disposition": audit["status"],
        "evidence": {"path": "precision/precision.json", "sha256": evidence_id},
        "audit": {"path": "precision/audit.json", "sha256": audit_id},
    }, inputs


def _reverify_pilot(
    declaration: JSONObject, root: Path, source_root: Path, source: object, output: Path
) -> None:
    """Recompute the saved pilot inside supervision before reading campaign outcomes."""
    from frem_s10.data.production import SourceAdmission, load_precision_inputs
    from frem_s10.runtime.declarations import _read, _reference
    from frem_s10.runtime.results import audit_precision_archive

    if not isinstance(source, SourceAdmission):
        raise IntegrityError("admitted source required for pilot re-audit")
    prerequisite, _ = _read(declaration["precision"], root)
    completed, _ = _read(prerequisite["completion"], root)
    previous, _ = _read(completed["worker_evidence"], root)
    path, _ = _reference(completed["worker_evidence"], root)
    evidence_path, evidence_id = _reference(previous["evidence"], path.parent)
    archive_path, archive_id = _reference(previous["model_archive"], path.parent)
    observed = audit_precision_archive(
        evidence_path.parent,
        evidence_id,
        archive_path.parent,
        archive_id,
        load_precision_inputs(source, source_root=source_root),
    )
    saved, _ = _read(previous["audit"], path.parent)
    if observed != saved or observed.get("status") != "PRECISION_PASS":
        raise IntegrityError("pilot prerequisite differs from independent saved-law/source regeneration")
    publish_json(output, "prerequisite-audit.json", observed)


def _reverify_qualification(
    verification: JSONObject, identities: JSONObject, root: Path, source_root: Path, output: Path
) -> None:
    """Reconstruct generated qualification evidence under the live campaign supervisor."""
    from frem_s10.runtime.declarations import _read
    from frem_s10.runtime.qualification import audit_qualification_deep

    logs = verification["logs"]
    if not isinstance(logs, list) or len(logs) != 1:
        raise IntegrityError("one exact qualification-bearing verification result required")
    test_result, _ = _read(logs[0], root)
    evidence, _ = _read(test_result["evidence"], root)
    observed = audit_qualification_deep(
        evidence["qualification"],
        artifact_root=root,
        source_root=source_root,
        code_sha256=str(identities["code"]),
        environment_sha256=str(identities["environment"]),
        calendar_sha256=str(identities["calendar"]),
    )
    publish_json(output, "qualification-audit.json", observed)


def _run_campaign(
    source: object, source_root: Path, output: Path, writer: object, attempt_id: str
) -> JSONObject:
    """Publish the entire fixed matrix and regenerate endpoints from saved output."""
    from frem_s10.data.production import SourceAdmission, load_campaign_inputs
    from frem_s10.experiment.model_archive import ModelArchiveWriter

    if not isinstance(source, SourceAdmission) or not isinstance(writer, ModelArchiveWriter):
        raise IntegrityError("validated source and durable archive writer required")
    inputs = load_campaign_inputs(source, source_root=source_root)
    return run_provided_campaign(inputs, output, writer, attempt_id)


def run_provided_campaign(inputs: object, output: Path, writer: object, attempt_id: str) -> JSONObject:
    """Use the exact production persistence and audit path for admitted or generated inputs."""
    from dataclasses import replace

    from frem_s10.experiment.artifacts import publish_campaign
    from frem_s10.experiment.forecasts import build_forecast_tape
    from frem_s10.experiment.model_archive import ModelArchiveWriter
    from frem_s10.experiment.replay import ReplayInputs
    from frem_s10.runtime.progress import ProgressLog, transition_sink
    from frem_s10.runtime.results import audit_campaign

    if not isinstance(inputs, ReplayInputs) or not isinstance(writer, ModelArchiveWriter):
        raise IntegrityError("validated replay inputs and durable archive writer required")
    progress = ProgressLog(output / "progress.jsonl", attempt_id)
    progress.emit({"event": "STARTED", "phase": "forecast"})
    retained = tuple(index for window in inputs.windows for index in window.origin_indices)
    try:
        tape = build_forecast_tape(
            inputs.origins,
            inputs.transform,
            retained_anchor_indices=retained,
            transition_sink=transition_sink(writer, progress),
        )
    except Exception as exc:
        progress.failure("forecast", exc)
        raise
    if inputs.evidence_scope == "GENERATED_ENGINEERING":
        tape = replace(tape, evidence_scope="GENERATED_ENGINEERING")
    directory = output / "campaign"
    directory.mkdir(exist_ok=False)
    progress.emit({"event": "STARTED", "phase": "ledger", "forecast_frames": len(tape.frames)})
    try:
        receipt = publish_campaign(directory, inputs, tape, experiment_id=attempt_id)
    except Exception as exc:
        progress.failure("ledger", exc)
        raise
    if not receipt.complete:
        raise IntegrityError("campaign failed; complete artifacts and partial cell are retained")
    progress.emit(
        {"event": "LEDGERS_ARCHIVED", "phase": "ledger", "completed_cells": receipt.completed_cells}
    )
    del inputs, tape
    audit_dir = output / "analysis"
    audit_dir.mkdir(exist_ok=False)
    try:
        audit_id = audit_campaign(directory, receipt.manifest_sha256, audit_dir)
    except Exception as exc:
        progress.failure("analysis", exc)
        raise
    audit = object_json(read_verified_json(audit_dir / "audit.json", audit_id))
    progress.emit(
        {"event": "CAMPAIGN_AUDITED", "phase": "analysis", "status": audit["numerical_disposition"]}
    )
    return {
        "scientific_disposition": audit["numerical_disposition"],
        "evidence": {"path": "campaign/manifest.json", "sha256": receipt.manifest_sha256},
        "audit": {"path": "analysis/audit.json", "sha256": audit_id},
    }


def main(argv: list[str] | None = None) -> int:
    """Accept only fixed controller arguments and keep failures concise."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("declaration", "artifact-root", "source-root", "consumption-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)
    try:
        execute_worker(
            declaration_path=args.declaration,
            declaration_sha256=args.sha256,
            artifact_root=args.artifact_root,
            source_root=args.source_root,
            consumption_root=args.consumption_root,
            attempt_id=args.attempt_id,
            output=Path.cwd(),
        )
    except (S10Error, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
