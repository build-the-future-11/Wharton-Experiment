"""Fixed generated calendar-scale qualification, with the real scientific resource guards.

This operation cannot read market sources or support a market-performance claim.
Its fixed inputs are generated after exclusive consumption. A failed consumed
attempt remains retained and cannot automatically resume or rerun.
"""

import math
import sys
from pathlib import Path

from frem_s10.errors import IntegrityError
from frem_s10.identities import JSONObject, JSONValue, digest_bytes, object_json, parse_json
from frem_s10.runtime.artifacts import _plain_directory
from frem_s10.runtime.declarations import DESIGN, _reference, _schema
from frem_s10.runtime.journal import inspect_journal
from frem_s10.runtime.provenance import audit_code, audit_environment, read_object, save_object
from frem_s10.runtime.supervisor import Budget, execute_operation

QUALIFICATION_CONTRACT: JSONObject = {
    "generator": "s10-calendar-panel-sine-v1",
    "scope": "GENERATED_ENGINEERING",
    "model_variants": 6,
    "ledgers": 192,
    "maximum_library": 520,
    "maximum_actions": 610,
    "wall_seconds": 3600,
    "rss_mib": 256,
    "output_bytes": 2_147_483_648,
    "scientific_support_claimed": False,
}


def qualification_calendar(calendar_ref: JSONValue, clock_ref: JSONValue, root: Path) -> JSONObject:
    """Validate the exact admitted clock without loading a single market value."""
    from frem_s10.data.production import _sessions

    receipt, clock = read_object(calendar_ref, root), read_object(clock_ref, root)
    if receipt != {
        "schema": "s10-calendar-receipt-v1",
        "design_id": DESIGN,
        "evidence_scope": "RETROSPECTIVE_DEVELOPMENT",
        "status": "PASS",
        "identity": _reference(clock_ref, root)[1],
    }:
        raise IntegrityError("qualification clock differs from the exact calendar binding")
    sessions = _sessions(dict(clock))
    if sessions[0].day.isoformat() != "2005-01-03" or sessions[-1].day.isoformat() != "2016-12-30":
        raise IntegrityError("qualification requires the complete fixed calendar span")
    return clock


def prepare_qualification(
    *,
    artifact_root: Path,
    source_root: Path,
    code_ref: JSONValue,
    environment_ref: JSONValue,
    calendar_ref: JSONValue,
    clock_ref: JSONValue,
) -> JSONObject:
    """Bind the deterministic engineering experiment before any generated fitting."""
    request: JSONObject = {
        "schema": "s10-qualification-request-v1",
        "contract": QUALIFICATION_CONTRACT,
        "code": code_ref,
        "environment": environment_ref,
        "calendar": calendar_ref,
        "clock": clock_ref,
    }
    validate_request(request, artifact_root=artifact_root, source_root=source_root)
    return save_object(artifact_root, request)


def validate_request(request: JSONObject, *, artifact_root: Path, source_root: Path) -> None:
    """Reject stale source, changed environment and unsupported workload contracts."""
    _schema(
        request, {"schema", "contract", "code", "environment", "calendar", "clock"}, "qualification request"
    )
    if request["schema"] != "s10-qualification-request-v1" or request["contract"] != QUALIFICATION_CONTRACT:
        raise IntegrityError("fixed complete generated qualification contract required")
    audit_code(request["code"], artifact_root=artifact_root, source_root=source_root)
    audit_environment(
        request["environment"],
        artifact_root=artifact_root,
        source_root=source_root,
        evidence_scope="RETROSPECTIVE_DEVELOPMENT",
    )
    qualification_calendar(request["calendar"], request["clock"], artifact_root)


def file_reference(path: Path, root: Path) -> JSONObject:
    """Hash an existing immutable output, retaining its real relative path."""
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise IntegrityError("regular output within artifact root required")
    return {"path": str(path.relative_to(root)), "sha256": digest_bytes(path.read_bytes())}


def run_qualification(manifest_path: Path) -> JSONObject:
    """Run exactly the trusted generated worker after fresh admission and consume once."""
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
        "qualification launch",
    )
    if launch["schema"] != "s10-qualification-launch-v1":
        raise IntegrityError("fixed qualification launch schema required")
    roots = {
        name: _plain_directory(Path(str(launch[name])))
        for name in ("artifact_root", "source_root", "consumption_root", "output_root")
    }
    root, source = roots["artifact_root"], roots["source_root"]
    if any(not roots[name].is_relative_to(root) for name in ("consumption_root", "output_root")):
        raise IntegrityError("qualification outputs and consumption must remain inside artifact root")
    if roots["output_root"].is_relative_to(source / "data") or roots["output_root"].is_relative_to(
        roots["consumption_root"]
    ):
        raise IntegrityError("qualification output overlaps source or consumption storage")
    request = read_object(launch["request"], root)
    validate_request(request, artifact_root=root, source_root=source)
    path, identity = _reference(launch["request"], root)
    attempt = str(launch["attempt_id"])
    command = (
        sys.executable,
        "-m",
        "frem_s10.runtime.qualification_worker",
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
        attempt,
    )
    result = execute_operation(
        command=command,
        operation="campaign",
        freeze_id=identity,
        attempt_id=attempt,
        output_root=roots["output_root"],
        consumption_root=roots["consumption_root"],
        input_hashes=((path, identity),),
        estimated_output_bytes=2_147_483_648,
        budget=Budget(wall_seconds=3600, output_bytes=2_147_483_648),
    )
    if result["state"] != "COMPLETE":
        return {"status": "FAILED_CONSUMED", "completion": result, "scientific_support_claimed": False}
    output = roots["output_root"] / attempt
    receipt = save_object(
        root,
        {
            "schema": "s10-qualification-receipt-v1",
            "request": launch["request"],
            "completion": file_reference(output / "completion.json", root),
            "resources": file_reference(output / "resources.jsonl", root),
            "timeline": file_reference(output / "timeline.jsonl", root),
            "consumption": file_reference(roots["consumption_root"] / f"campaign-{identity}.json", root),
            "worker": file_reference(output / "qualification.json", root),
        },
    )
    audit_qualification(
        receipt,
        artifact_root=root,
        source_root=source,
        code_sha256=_reference(request["code"], root)[1],
        environment_sha256=_reference(request["environment"], root)[1],
        calendar_sha256=_reference(request["calendar"], root)[1],
    )
    return {"status": "PASS", "receipt": receipt, "scientific_support_claimed": False}


def _observed_resources(reference: JSONValue, root: Path, completion: JSONObject) -> None:
    path, expected = _reference(reference, root)
    if digest_bytes(path.read_bytes()) != expected:
        raise IntegrityError("resource journal identity differs")
    count, _ = inspect_journal(path)
    if not count or completion["resource_observations"] != count:
        raise IntegrityError("complete observed resource timeline required")
    peak, previous, pageouts = 0.0, -1.0, None
    with path.open("rb") as stream:
        for line in stream:
            event = object_json(object_json(parse_json(line))["event"])
            values = tuple(
                event[name]
                for name in ("elapsed_seconds", "owned_rss_mib", "output_bytes", "free_plus_speculative_mib")
            )
            if any(
                type(value) not in (float, int) or not math.isfinite(float(str(value))) for value in values
            ):
                raise IntegrityError("finite observed resource measurements required")
            elapsed, rss, size, available = (float(str(value)) for value in values)
            current_pageouts = event["pageouts"]
            if type(current_pageouts) is not int or (pageouts is not None and current_pageouts != pageouts):
                raise IntegrityError("qualification pageouts increased or became unavailable")
            if (
                elapsed < previous
                or elapsed > 3600
                or not 0 < rss <= 256
                or size > 2_147_483_648
                or available < 512
            ):
                raise IntegrityError("qualification exceeded its observed resource bounds")
            peak, previous, pageouts = max(peak, rss), elapsed, current_pageouts
    if peak != completion["maximum_observed_owned_rss_mib"]:
        raise IntegrityError("qualification peak differs from raw observations")


def audit_qualification(
    reference: JSONValue,
    *,
    artifact_root: Path,
    source_root: Path,
    code_sha256: str,
    environment_sha256: str,
    calendar_sha256: str,
) -> JSONObject:
    """Audit observed work/resource identities; a structural PASS cannot qualify a run."""
    root = _plain_directory(artifact_root)
    receipt = read_object(reference, root)
    _schema(
        receipt,
        {"schema", "request", "completion", "resources", "timeline", "consumption", "worker"},
        "qualification receipt",
    )
    if receipt["schema"] != "s10-qualification-receipt-v1":
        raise IntegrityError("actual generated qualification receipt required")
    request = read_object(receipt["request"], root)
    validate_request(request, artifact_root=root, source_root=source_root)
    for role, expected in (
        ("code", code_sha256),
        ("environment", environment_sha256),
        ("calendar", calendar_sha256),
    ):
        if _reference(request[role], root)[1] != expected:
            raise IntegrityError("qualification belongs to a different verified code/environment/calendar")
    identity = _reference(receipt["request"], root)[1]
    completion, marker, worker = (
        read_object(receipt[key], root) for key in ("completion", "consumption", "worker")
    )
    if (
        completion.get("state"),
        completion.get("consumed"),
        completion.get("worker_exit_code"),
        completion.get("freeze_id"),
        completion.get("error"),
    ) != ("COMPLETE", True, 0, identity, None):
        raise IntegrityError("qualification worker did not complete under supervision")
    if completion.get("wall_limit_seconds") != 3600 or completion.get("rss_limit_mib") != 256:
        raise IntegrityError("qualification budget differs from declared ceilings")
    elapsed = completion.get("elapsed_seconds")
    if type(elapsed) not in (float, int) or not 0 < float(str(elapsed)) <= 3600:
        raise IntegrityError("qualification elapsed duration exceeds the fixed ceiling")
    output = _reference(receipt["completion"], root)[0].parent
    if (
        marker.get("operation"),
        marker.get("state"),
        marker.get("freeze_id"),
        marker.get("attempt_id"),
        marker.get("output_path"),
        marker.get("resume_permitted"),
    ) != ("campaign", "CONSUMED", identity, completion.get("attempt_id"), str(output), False):
        raise IntegrityError("qualification consumption does not bind this exact attempt")
    _observed_resources(receipt["resources"], root, completion)
    timeline, timeline_id = _reference(receipt["timeline"], root)
    if digest_bytes(timeline.read_bytes()) != timeline_id or inspect_journal(timeline)[0] < 4:
        raise IntegrityError("complete lifecycle evidence required")
    _schema(
        worker,
        {"schema", "request_sha256", "contract", "counts", "archive", "campaign", "analysis"},
        "qualification worker",
    )
    if (
        worker["schema"] != "s10-qualification-worker-v1"
        or worker["request_sha256"] != identity
        or worker["contract"] != QUALIFICATION_CONTRACT
    ):
        raise IntegrityError("qualification workload identity mismatch")
    counts = object_json(worker["counts"])
    if (
        counts.get("model_variants") != 6
        or counts.get("ledgers") != 192
        or counts.get("maximum_library") != 520
        or counts.get("maximum_actions") != 610
    ):
        raise IntegrityError("qualification did not reach all declared workload bounds")
    for name in ("archive", "campaign", "analysis"):
        document = read_object(worker[name], root)
        if name == "archive" and (
            document.get("evidence_scope") != "GENERATED_ENGINEERING"
            or document.get("scientific_support_claimed") is not False
        ):
            raise IntegrityError("qualification archive must retain generated scope")
    return {
        "status": "PASS",
        "counts": counts,
        "elapsed_seconds": elapsed,
        "maximum_observed_owned_rss_mib": completion["maximum_observed_owned_rss_mib"],
        "evidence_scope": "GENERATED_ENGINEERING",
        "scientific_support_claimed": False,
    }
