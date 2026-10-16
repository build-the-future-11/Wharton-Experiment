"""Fixed generated calendar-scale qualification, with the real scientific resource guards.

This operation cannot read market sources or support a market-performance claim.
Its fixed inputs are generated after exclusive consumption. A failed consumed
attempt remains retained and cannot automatically resume or rerun.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from frem_s10.errors import IntegrityError, ResourceLimitError
from frem_s10.identities import JSONObject, JSONValue, digest_bytes, object_json, parse_json
from frem_s10.runtime.artifacts import _plain_directory
from frem_s10.runtime.declarations import DESIGN, _reference, _schema
from frem_s10.runtime.journal import inspect_journal
from frem_s10.runtime.provenance import audit_code, audit_environment, read_object, save_object
from frem_s10.runtime.supervisor import Budget, execute_operation

if TYPE_CHECKING:
    from frem_s10.experiment.replay import CampaignResult

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
    if path.stat().st_size > 16_777_216:
        raise IntegrityError("qualification metadata exceeds its declared byte bound")
    with path.open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.relative_to(root)), "sha256": identity}


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
    output = roots["output_root"] / attempt
    command = (
        sys.executable,
        "-B",
        "-X",
        f"pycache_prefix={output / 'empty_bytecode'}",
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
    try:
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
    except ResourceLimitError:
        from frem_s10.runtime.receipts import read_unconsumed_refusal

        result = read_unconsumed_refusal(
            output, consumption_root=roots["consumption_root"], operation="campaign", freeze_id=identity
        )
    if result["state"] != "COMPLETE":
        return {
            "status": result["state"],
            "consumed": result.get("consumed"),
            "completion": result,
            "scientific_support_claimed": False,
        }
    receipt = save_object(
        root,
        {
            "schema": "s10-qualification-receipt-v1",
            "request": launch["request"],
            "completion": file_reference(output / "completion.json", root),
            "preflight": file_reference(output / "preflight.json", root),
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
    if path.stat().st_size > 16_777_216:
        raise IntegrityError("qualification resource metadata exceeds byte bound")
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != expected:
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
                or elapsed > float(str(completion["elapsed_seconds"]))
                or elapsed < 0
                or size < 0
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
        {"schema", "request", "completion", "preflight", "resources", "timeline", "consumption", "worker"},
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
    if (
        completion.get("wall_limit_seconds") != 3600
        or completion.get("rss_limit_mib") != 256
        or completion.get("operation") != "campaign"
        or completion.get("output_limit_bytes") != 2_147_483_648
    ):
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
    from frem_s10.runtime.receipts import _audit_resources

    _audit_resources(receipt, completion, root)
    _audit_lifecycle(receipt, root)
    counts = _audit_worker(worker, request, identity, root)
    return {
        "status": "PASS",
        "qualification_sha256": _reference(reference, root)[1],
        "request_sha256": identity,
        "worker_sha256": _reference(receipt["worker"], root)[1],
        "counts": counts,
        "elapsed_seconds": elapsed,
        "maximum_observed_owned_rss_mib": completion["maximum_observed_owned_rss_mib"],
        "evidence_scope": "GENERATED_ENGINEERING",
        "scientific_support_claimed": False,
        "deep_audit_required_inside_consumed_worker": True,
    }


def _audit_lifecycle(receipt: JSONObject, root: Path) -> None:
    timeline, timeline_id = _reference(receipt["timeline"], root)
    if timeline.stat().st_size > 1_048_576:
        raise IntegrityError("qualification lifecycle metadata exceeds byte bound")
    if digest_bytes(timeline.read_bytes()) != timeline_id or inspect_journal(timeline)[0] < 4:
        raise IntegrityError("complete lifecycle evidence required")
    events = [
        object_json(object_json(parse_json(line))["event"]) for line in timeline.read_bytes().splitlines()
    ]
    if [event.get("state") for event in events] != ["PREFLIGHT", "CONSUMED", "RUNNING", "COMPLETE"] or events[
        1
    ].get("marker_sha256") != _reference(receipt["consumption"], root)[1]:
        raise IntegrityError("qualification lifecycle does not show exact consumption before execution")


def _inventory(reference: JSONValue, root: Path) -> tuple[JSONObject, list[JSONObject]]:
    """Check bounded inventory metadata without opening any payload file."""
    path, _ = _reference(reference, root)
    manifest = read_object(reference, root)
    raw = manifest.get("files")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 6000:
        raise IntegrityError("bounded complete artifact inventory required")
    rows = [object_json(row) for row in raw]
    names = [row.get("name") for row in rows]
    if any(not isinstance(name, str) or Path(name).name != name or name in (".", "..") for name in names):
        raise IntegrityError("plain inventory names required")
    if len({str(name) for name in names}) != len(names) or {
        entry.name for entry in path.parent.iterdir()
    } != set(names) | {path.name}:
        raise IntegrityError("inventory contains missing, duplicate or extra files")
    total = 0
    for row in rows:
        child, _ = _reference(
            {"path": str((path.parent / str(row["name"])).relative_to(root)), "sha256": row["sha256"]}, root
        )
        size = child.stat().st_size
        total += size
        if size > 67_108_864 or total > 2_147_483_648:
            raise IntegrityError("saved artifact exceeds its declared file bound")
    return manifest, rows


def _expected_counts(request: JSONObject, root: Path) -> JSONObject:
    from frem_s10.data.calendar import build_calendar_schedule, plan_fold
    from frem_s10.data.production import _sessions

    clocks = _sessions(dict(qualification_calendar(request["calendar"], request["clock"], root)))
    calendar = build_calendar_schedule(tuple(clock.day for clock in clocks))
    a, b = plan_fold(calendar, "A"), plan_fold(calendar, "B")
    weekly = sum(
        anchor.date.year >= 2008 and anchor.index <= b.origin_indices[-1] for anchor in calendar.anchors
    )
    return {
        "sessions": len(clocks),
        "model_variants": 6,
        "transitions": weekly * 6,
        "ledgers": 192,
        "maximum_library": 520,
        "maximum_actions": 610,
        "fold_a_blocks": len(a.origin_indices),
        "fold_b_blocks": len(b.origin_indices),
    }


def _audit_worker(worker: JSONObject, request: JSONObject, identity: str, root: Path) -> JSONObject:
    from frem_s10.experiment.matrix import MODEL_VARIANTS

    _schema(
        worker,
        {"schema", "request_sha256", "contract", "counts", "archive", "campaign", "analysis", "stress"},
        "qualification worker",
    )
    if (
        worker["schema"] != "s10-qualification-worker-v1"
        or worker["request_sha256"] != identity
        or worker["contract"] != QUALIFICATION_CONTRACT
    ):
        raise IntegrityError("qualification workload identity mismatch")
    counts = object_json(worker["counts"])
    if counts != _expected_counts(request, root):
        raise IntegrityError("qualification did not reach all declared workload bounds")
    archive, generations = _inventory(worker["archive"], root)
    shapes = object_json(archive.get("shape_counts"))
    if (
        archive.get("schema"),
        archive.get("evidence_scope"),
        archive.get("scientific_support_claimed"),
        archive.get("binding_id"),
        archive.get("variants"),
    ) != ("s10-weekly-model-archive-v1", "GENERATED_ENGINEERING", False, identity, list(MODEL_VARIANTS)):
        raise IntegrityError("qualification archive scope, binding or variants differ")
    if (
        shapes.get("transitions") != counts["transitions"]
        or shapes.get("max_library") != 520
        or len(generations) != counts["transitions"]
    ):
        raise IntegrityError("qualification counts differ from verified saved archive generations")
    if [row.get("variant") for row in generations] != [
        MODEL_VARIANTS[index % 6] for index in range(len(generations))
    ]:
        raise IntegrityError("qualification archive has incomplete weekly model variants")
    _audit_campaign_inventory(worker, counts, root)
    _audit_stress(read_object(worker["stress"], root))
    return counts


def _audit_stress(stress: JSONObject) -> None:
    if (
        stress.get("schema") != "s10-generated-policy-stress-v1"
        or stress.get("evidence_scope") != "GENERATED_ENGINEERING"
    ):
        raise IntegrityError("generated maximum-action stress evidence required")
    rows = stress.get("rows")
    expected = [
        (n, cost, robust) for n in (64, 128, 260, 520) for cost in (0, 10, 20, 40) for robust in (False, True)
    ]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise IntegrityError("complete fixed support/cost/policy stress ladder required")
    for raw, key in zip(rows, expected, strict=True):
        row = object_json(raw)
        _schema(row, {"atoms", "cost_bps", "robust", "wall_seconds", "certificate"}, "stress observation")
        elapsed = row["wall_seconds"]
        if type(elapsed) not in (int, float) or not 0 <= float(str(elapsed)) <= 3600:
            raise IntegrityError("finite bounded stress elapsed time required")
        if (
            type(row["robust"]) is not bool
            or type(row["atoms"]) is not int
            or type(row["cost_bps"]) is not int
        ):
            raise IntegrityError("exact typed stress axes required")
        certificate = object_json(row.get("certificate"))
        if (row.get("atoms"), row.get("cost_bps"), row.get("robust")) != key or certificate.get(
            "candidate_count"
        ) != 610:
            raise IntegrityError("stress evidence did not evaluate the maximum candidate set")
        gap = certificate.get("optimality_gap")
        if type(gap) not in (float, int) or not 0 <= float(str(gap)) <= 1e-10:
            raise IntegrityError("maximum-action stress decision lacks its finite certified optimum")


def _audit_campaign_inventory(worker: JSONObject, counts: JSONObject, root: Path) -> None:
    from frem_s10.experiment.matrix import campaign_matrix

    campaign, rows = _inventory(worker["campaign"], root)
    frames = 6 * (int(str(counts["fold_a_blocks"])) + int(str(counts["fold_b_blocks"])))
    if (
        campaign.get("schema"),
        campaign.get("evidence_scope"),
        campaign.get("complete"),
        campaign.get("completed_cells"),
        campaign.get("failure_cell_id"),
        campaign.get("scientific_support_claimed"),
        campaign.get("forecast_frames"),
    ) != ("s10-campaign-inventory-v1", "GENERATED_ENGINEERING", True, 192, None, False, frames):
        raise IntegrityError("complete generated192 campaign publication required")
    if len(rows) != 2 + frames + 192 or [row.get("cell_id") for row in rows[-192:]] != [
        cell.cell_id for cell in campaign_matrix()
    ]:
        raise IntegrityError("qualification campaign inventory is incomplete or reordered")
    analysis = read_object(worker["analysis"], root)
    if (
        analysis.get("schema"),
        analysis.get("evidence_sha256"),
        analysis.get("evidence_scope"),
        analysis.get("status"),
        analysis.get("cells"),
        analysis.get("source_authority_approved"),
    ) != (
        "s10-campaign-audit-v1",
        _reference(worker["campaign"], root)[1],
        "GENERATED_ENGINEERING",
        "COMPLETE",
        192,
        False,
    ):
        raise IntegrityError("qualification saved-output analysis identity or scope differs")


def audit_qualification_deep(
    reference: JSONValue,
    *,
    artifact_root: Path,
    source_root: Path,
    code_sha256: str,
    environment_sha256: str,
    calendar_sha256: str,
) -> JSONObject:
    """Reconcile all saved payloads inside a consumed supervised campaign worker.

    The controller uses audit_qualification only. This deliberately separate
    numerical/IO-heavy verifier must finish before campaign market inputs load.
    It never fits a model, executes a portfolio, or reads source market values.
    """
    result = audit_qualification(
        reference,
        artifact_root=artifact_root,
        source_root=source_root,
        code_sha256=code_sha256,
        environment_sha256=environment_sha256,
        calendar_sha256=calendar_sha256,
    )
    receipt = read_object(reference, artifact_root)
    worker = read_object(receipt["worker"], artifact_root)
    request = read_object(receipt["request"], artifact_root)
    identity = _reference(receipt["request"], artifact_root)[1]
    deep_worker_evidence(worker, request=request, identity=identity, artifact_root=artifact_root)
    return result | {"deep_audit": True, "deep_audit_required_inside_consumed_worker": False}


def deep_worker_evidence(
    worker: JSONObject, *, request: JSONObject, identity: str, artifact_root: Path
) -> None:
    """Verify generated payloads under the current worker's resource budget."""
    from frem_s10.experiment.analysis import analyze_campaign
    from frem_s10.experiment.readback import load_campaign
    from frem_s10.runtime.qualification_worker import stress_actions

    _audit_worker(worker, request, identity, artifact_root)
    for name in ("archive", "campaign"):
        _verify_inventory_payloads(worker[name], artifact_root)
    path, digest = _reference(worker["campaign"], artifact_root)
    campaign = load_campaign(path.parent, digest)
    if not campaign.complete or campaign.evidence_scope != "GENERATED_ENGINEERING":
        raise IntegrityError("qualification must reconstruct the complete generated campaign")
    _archive_campaign_links(worker, campaign, request, artifact_root)
    analysis = analyze_campaign(campaign)
    _analysis_payloads(worker["analysis"], analysis, artifact_root)
    retained = read_object(worker["stress"], artifact_root)
    recomputed = stress_actions()
    _compare_stress(retained, recomputed)


def _verify_inventory_payloads(reference: JSONValue, root: Path) -> None:
    """Hash every bounded payload only after supervised consumption."""
    path, _ = _reference(reference, root)
    _, rows = _inventory(reference, root)
    for row in rows:
        child, expected = _reference(
            {"path": str((path.parent / str(row["name"])).relative_to(root)), "sha256": row["sha256"]}, root
        )
        with child.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != expected:
            raise IntegrityError("saved artifact bytes differ from qualification inventory")


def _archive_campaign_links(
    worker: JSONObject, campaign: CampaignResult, request: JSONObject, root: Path
) -> None:
    from frem_s10.data.calendar import build_calendar_schedule, plan_fold
    from frem_s10.data.production import _sessions
    from frem_s10.experiment.matrix import MODEL_VARIANTS
    from frem_s10.experiment.model_archive import iter_verified_transitions

    clocks = _sessions(dict(qualification_calendar(request["calendar"], request["clock"], root)))
    calendar = build_calendar_schedule(tuple(clock.day for clock in clocks))
    last = plan_fold(calendar, "B").origin_indices[-1]
    anchors = tuple(
        anchor for anchor in calendar.anchors if anchor.date.year >= 2008 and anchor.index <= last
    )
    expected = tuple((variant, anchor) for anchor in anchors for variant in MODEL_VARIANTS)
    archive, digest = _reference(worker["archive"], root)
    law_ids: dict[tuple[str, int, str], str] = {}
    origins: dict[int, str] = {}
    count = 0
    for variant, step in iter_verified_transitions(archive.parent, digest):
        if count >= len(expected):
            raise IntegrityError("qualification archive exceeds exact calendar issue schedule")
        expected_variant, anchor = expected[count]
        issue = step.batch.get("contextual_stack").request
        if (variant, issue.anchor_index, issue.decision_us) != (
            expected_variant,
            anchor.index,
            anchor.decision_us,
        ):
            raise IntegrityError("qualification archive differs from exact calendar issue schedule")
        origins[anchor.index] = issue.record_id
        for law in step.batch.forecasts:
            law_ids[(variant, anchor.index, law.arm)] = law.record_id
        count += 1
    if count != len(expected):
        raise IntegrityError("qualification archive omits fixed weekly transitions")
    for cell in campaign.results:
        for decision in cell.decisions:
            selected = (
                cell.decisions[0].anchor_index if cell.spec.arm == "frozen_state" else decision.anchor_index
            )
            key = (cell.spec.variant, selected, cell.spec.forecast_arm)
            if decision.forecast_id != law_ids.get(key) or decision.origin_id != origins.get(
                decision.anchor_index
            ):
                raise IntegrityError("campaign decision differs from its original archived issue")


def _analysis_payloads(reference: JSONValue, analysis: object, root: Path) -> None:
    import numpy as np

    from frem_s10.analysis.report import render_statistics
    from frem_s10.experiment.analysis import AnalysisBundle
    from frem_s10.runtime.artifacts import read_verified_json
    from frem_s10.runtime.results import _array, _statistics

    if not isinstance(analysis, AnalysisBundle):
        raise IntegrityError("typed complete generated analysis required")
    saved = read_object(reference, root)
    path, _ = _reference(reference, root)
    if (
        saved.get("primary") != _statistics(analysis.primary)
        or saved.get("ablations") != _statistics(analysis.ablations)
        or saved.get("numerical_disposition") != analysis.primary.disposition
    ):
        raise IntegrityError("saved qualification analysis differs from regenerated endpoints")
    streams = saved.get("bootstrap")
    if not isinstance(streams, list) or len(streams) != len(analysis.bootstrap):
        raise IntegrityError("all qualification bootstrap arrays required")
    for raw, (fold, block, expected) in zip(streams, analysis.bootstrap, strict=True):
        row = object_json(raw)
        name = f"bootstrap-{fold}-{block}.json"
        if row.get("fold") != fold or row.get("block") != block or row.get("path") != name:
            raise IntegrityError("qualification bootstrap axes differ")
        payload = read_verified_json(path.parent / name, str(row.get("sha256")), maximum_bytes=67_108_864)
        observed = _array(payload, unit="observation_index", axis="draw,date", dtype="<i8")
        if not np.array_equal(observed, expected):
            raise IntegrityError("qualification bootstrap array differs from fixed RNG")
    report = read_object(saved["report"], path.parent)
    expected_report = render_statistics(
        analysis.primary,
        analysis.ablations,
        evidence_sha256=str(saved["evidence_sha256"]),
        scope="GENERATED_ENGINEERING",
    )
    if report != {
        "schema": "s10-rendered-report-v1",
        "markdown": expected_report,
        "evidence_sha256": saved["evidence_sha256"],
        "evidence_scope": "GENERATED_ENGINEERING",
    }:
        raise IntegrityError("qualification report differs from retained numerical evidence")


def _compare_stress(retained: JSONObject, recomputed: JSONObject) -> None:
    _audit_stress(retained)
    expected, observed = recomputed["rows"], retained["rows"]
    if not isinstance(expected, list) or not isinstance(observed, list):
        raise IntegrityError("stress rows required")
    for old, new in zip(observed, expected, strict=True):
        left, right = object_json(old), object_json(new)
        if {key: value for key, value in left.items() if key != "wall_seconds"} != {
            key: value for key, value in right.items() if key != "wall_seconds"
        }:
            raise IntegrityError("saved stress certificates differ from generated recomputation")
