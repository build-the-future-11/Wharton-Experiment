"""Regenerate saved campaign statistics in a separately consumed guarded worker."""

import argparse
import hashlib
import sys
from pathlib import Path

from frem_s10.errors import ExecutionStateError, IntegrityError, ResourceLimitError, S10Error
from frem_s10.identities import JSONObject, JSONValue, digest_bytes, object_json, parse_json
from frem_s10.runtime.artifacts import _plain_directory, publish_json, read_verified_json
from frem_s10.runtime.declarations import _read, _reference, _schema
from frem_s10.runtime.pipeline import CAMPAIGN_OUTPUT_BYTES, _finalization_failure
from frem_s10.runtime.provenance import audit_code, audit_environment
from frem_s10.runtime.receipts import _audit_resources, read_unconsumed_refusal
from frem_s10.runtime.supervisor import Budget, execute_operation


def _request(reference: JSONValue, root: Path, source: Path) -> tuple[JSONObject, str]:
    value, identity = _read(reference, root)
    _schema(value, {"schema", "code", "environment", "campaign"}, "saved audit request")
    if value["schema"] != "s10-saved-audit-request-v1":
        raise IntegrityError("exact saved audit request required")
    audit_code(value["code"], artifact_root=root, source_root=source)
    audit_environment(
        value["environment"],
        artifact_root=root,
        source_root=source,
        evidence_scope="RETROSPECTIVE_DEVELOPMENT",
    )
    path, _ = _reference(value["campaign"], root)
    campaign, _ = _read(value["campaign"], root)
    if (
        path.name != "manifest.json"
        or campaign.get("schema") != "s10-campaign-inventory-v1"
        or campaign.get("complete") is not True
        or campaign.get("completed_cells") != 192
    ):
        raise IntegrityError(
            "saved audit requires an exact complete192 inventory; worker independently validates every cell"
        )
    return value, identity


def _file(path: Path, root: Path) -> JSONObject:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_777_216:
        raise IntegrityError("bounded regular saved-audit metadata required")
    with path.open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.relative_to(root)), "sha256": identity}


def run_saved_audit(manifest_path: Path) -> JSONObject:
    """Dispatch only saved-output regeneration, never model fitting or source values."""
    launch = _schema(
        parse_json(manifest_path.read_bytes()),
        {
            "schema",
            "request",
            "artifact_root",
            "source_root",
            "consumption_root",
            "output_root",
            "attempt_id",
        },
        "saved audit launch",
    )
    if launch["schema"] != "s10-saved-audit-launch-v1":
        raise IntegrityError("exact supervised saved audit launch required")
    roots = {
        name: _plain_directory(Path(str(launch[name])))
        for name in ("artifact_root", "source_root", "consumption_root", "output_root")
    }
    root, source = roots["artifact_root"], roots["source_root"]
    output = roots["output_root"] / str(launch["attempt_id"])
    if (
        not roots["consumption_root"].is_relative_to(root)
        or not output.is_relative_to(root)
        or output.is_relative_to(source / "data")
        or output.is_relative_to(roots["consumption_root"])
    ):
        raise IntegrityError("saved audit output or consumption namespace overlaps protected sources")
    _, identity = _request(launch["request"], root, source)
    path, _ = _reference(launch["request"], root)
    command = (
        sys.executable,
        "-B",
        "-X",
        f"pycache_prefix={output / 'empty_bytecode'}",
        "-m",
        "frem_s10.runtime.saved_audit",
        "--request",
        str(path),
        "--sha256",
        identity,
        "--artifact-root",
        str(root),
        "--source-root",
        str(source),
        "--consumption-root",
        str(roots["consumption_root"]),
        "--attempt-id",
        str(launch["attempt_id"]),
    )
    try:
        result = execute_operation(
            command=command,
            operation="saved-audit",
            freeze_id=identity,
            attempt_id=str(launch["attempt_id"]),
            output_root=roots["output_root"],
            consumption_root=roots["consumption_root"],
            input_hashes=((path, identity),),
            estimated_output_bytes=CAMPAIGN_OUTPUT_BYTES,
            budget=Budget(wall_seconds=3600, rss_mib=256, output_bytes=CAMPAIGN_OUTPUT_BYTES),
        )
    except ResourceLimitError:
        return read_unconsumed_refusal(
            output, consumption_root=roots["consumption_root"], operation="saved-audit", freeze_id=identity
        )
    if result["state"] != "COMPLETE":
        return result
    try:
        return _finalize(output, root, roots["consumption_root"], identity)
    except (S10Error, OSError) as exc:
        return _finalization_failure(output, "saved-audit", identity, exc)


def _finalize(output: Path, root: Path, consumption: Path, identity: str) -> JSONObject:
    refs: JSONObject = {
        name: _file(output / filename, root)
        for name, filename in (
            ("completion", "completion.json"),
            ("worker", "audit-evidence.json"),
            ("preflight", "preflight.json"),
            ("resources", "resources.jsonl"),
            ("timeline", "timeline.jsonl"),
        )
    }
    refs["consumption"] = _file(consumption / f"saved-audit-{identity}.json", root)
    completed, _ = _read(refs["completion"], root)
    marker, _ = _read(refs["consumption"], root)
    worker, _ = _read(refs["worker"], root)
    if (
        completed.get("state"),
        completed.get("operation"),
        completed.get("freeze_id"),
        completed.get("worker_exit_code"),
        completed.get("consumed"),
    ) != ("COMPLETE", "saved-audit", identity, 0, True):
        raise IntegrityError("saved audit supervision did not complete")
    if (
        marker.get("schema_version"),
        marker.get("operation"),
        marker.get("state"),
        marker.get("freeze_id"),
        marker.get("output_path"),
        marker.get("attempt_id"),
        marker.get("resume_permitted"),
    ) != (
        1,
        "saved-audit",
        "CONSUMED",
        identity,
        str(output),
        completed.get("attempt_id"),
        False,
    ) or worker.get("request_sha256") != identity:
        raise IntegrityError("saved audit worker differs from its consumed output identity")
    if (
        completed.get("wall_limit_seconds") != 3600
        or completed.get("rss_limit_mib") != 256
        or completed.get("output_limit_bytes") != CAMPAIGN_OUTPUT_BYTES
        or not 0 < float(str(completed.get("elapsed_seconds"))) <= 3600
        or not 0 < float(str(completed.get("maximum_observed_owned_rss_mib"))) <= 256
    ):
        raise IntegrityError("saved audit resource ceilings differ")
    _audit_resources(refs, completed, root)
    audit, _ = _read(worker["audit"], output)
    if audit.get("status") != "COMPLETE" or audit.get("evidence_sha256") != worker.get("campaign_sha256"):
        raise IntegrityError("saved audit numerical evidence differs from worker identity")
    result: JSONObject = {
        "schema": "s10-supervised-saved-audit-v1",
        "status": "COMPLETE",
        "request_sha256": identity,
        "scientific_fits": 0,
        "scientific_support_claimed": False,
        **refs,
    }
    publish_json(output, "result.json", result)
    return result


def execute_saved_audit(
    *,
    request_path: Path,
    request_sha256: str,
    artifact_root: Path,
    source_root: Path,
    consumption_root: Path,
    attempt_id: str,
    output: Path,
) -> JSONObject:
    """Verify exact consumption before importing array decoding or statistical analysis."""
    marker_path = consumption_root / f"saved-audit-{request_sha256}.json"
    marker = object_json(read_verified_json(marker_path, digest_bytes(marker_path.read_bytes())))
    if (
        marker.get("state"),
        marker.get("freeze_id"),
        marker.get("operation"),
        marker.get("attempt_id"),
        marker.get("output_path"),
        marker.get("resume_permitted"),
    ) != ("CONSUMED", request_sha256, "saved-audit", attempt_id, str(output.absolute()), False):
        raise ExecutionStateError("saved audit lacks its exact consumed output identity")
    publish_json(output, "worker-started.json", {"request_sha256": request_sha256, "attempt_id": attempt_id})
    request, _ = _request(
        {"path": str(request_path.relative_to(artifact_root)), "sha256": request_sha256},
        artifact_root,
        source_root,
    )
    path, identity = _reference(request["campaign"], artifact_root)
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    from frem_s10.runtime.results import audit_campaign

    audit_id = audit_campaign(path.parent, identity, analysis)
    result: JSONObject = {
        "schema": "s10-saved-audit-evidence-v1",
        "request_sha256": request_sha256,
        "campaign_sha256": identity,
        "audit": {"path": "analysis/audit.json", "sha256": audit_id},
        "scientific_fits": 0,
        "scientific_support_claimed": False,
    }
    publish_json(output, "audit-evidence.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Accept only the fixed controller arguments for supervised saved evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "artifact-root", "source-root", "consumption-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)
    try:
        execute_saved_audit(
            request_path=args.request,
            request_sha256=args.sha256,
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
