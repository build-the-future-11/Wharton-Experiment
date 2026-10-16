"""Advance one fixed prospective research plan without repeating consumed work.

This controller reads metadata only. Every numerical operation retains its own
fresh admission, process supervision and exclusive consumption boundary.
"""

import fcntl
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from frem_s10.errors import ExecutionStateError, IntegrityError, S10Error
from frem_s10.identities import (
    JSONObject,
    JSONValue,
    canonical_bytes,
    checked_json,
    digest_bytes,
    object_json,
    parse_json,
)
from frem_s10.runtime.artifacts import _plain_directory, publish_json
from frem_s10.runtime.declarations import BOUND_ROLES, _read, _reference, _schema
from frem_s10.runtime.journal import Journal, inspect_journal
from frem_s10.runtime.resources import check_host

STAGES = ("verification", "precision", "qualification", "campaign")
REFS = {
    "code",
    "configuration",
    "environment",
    "calendar",
    "clock",
    "source_admission",
    "data_pilot",
    "data_campaign",
    "catalog",
}
ROOTS = ("artifact_root", "source_root", "consumption_root", "output_root", "work_root")


@dataclass(frozen=True)
class Plan:
    """Verified metadata and directories for a single immutable advancement plan."""

    document: JSONObject
    identity: str
    artifacts: JSONObject
    roots: dict[str, Path]
    prefix: str

    @property
    def root(self) -> Path:
        """Return the immutable artifact namespace root."""
        return self.roots["artifact_root"]

    @property
    def source(self) -> Path:
        """Return the verified source checkout."""
        return self.roots["source_root"]

    @property
    def work(self) -> Path:
        """Return this controller's exclusive journal directory."""
        return self.roots["work_root"]

    def bindings(self, *, campaign: bool) -> JSONObject:
        """Return the exact five scientific verification identities."""
        refs = self.operation_artifacts(campaign=campaign)
        return {key: _reference(refs[key], self.root)[1] for key in BOUND_ROLES}

    def operation_artifacts(self, *, campaign: bool) -> JSONObject:
        """Select only the predefined pilot or campaign data cutoff."""
        return {
            key: self.artifacts["data_campaign" if campaign else "data_pilot"]
            if key == "data"
            else self.artifacts[key]
            for key in BOUND_ROLES
        }


def _roots(value: JSONObject) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for key in ROOTS:
        name = value[key]
        if not isinstance(name, str) or not Path(name).is_absolute():
            raise IntegrityError("continuation roots must be explicit absolute directories")
        roots[key] = _plain_directory(Path(name))
    root, source = roots["artifact_root"], roots["source_root"]
    owned = tuple(roots[key] for key in ("consumption_root", "output_root", "work_root"))
    if any(not item.is_relative_to(root) or item.is_relative_to(source / "data") for item in owned):
        raise IntegrityError("continuation evidence must stay inside artifact root and outside source values")
    if (
        any(left.is_relative_to(right) for left in owned for right in owned if left != right)
        or len(set(owned)) != 3
    ):
        raise IntegrityError("continuation work, output and consumption directories must not overlap")
    return roots


def _plan(path: Path) -> Plan:
    from importlib.resources import files

    from frem_s10.data.production import validate_source_admission
    from frem_s10.runtime.provenance import audit_code, audit_environment
    from frem_s10.runtime.qualification import qualification_calendar

    raw = path.read_bytes()
    value = _schema(parse_json(raw), {"schema", "artifacts", "attempt_prefix", *ROOTS}, "continuation plan")
    if value["schema"] != "s10-continuation-plan-v1":
        raise IntegrityError("exact fixed continuation plan required")
    roots = _roots(value)
    root, source = roots["artifact_root"], roots["source_root"]
    prefix = value["attempt_prefix"]
    if not isinstance(prefix, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", prefix) is None:
        raise IntegrityError("bounded plain continuation attempt prefix required")
    refs = _schema(value["artifacts"], REFS, "continuation artifact registry")
    audit_code(refs["code"], artifact_root=root, source_root=source)
    audit_environment(
        refs["environment"],
        artifact_root=root,
        source_root=source,
        evidence_scope="RETROSPECTIVE_DEVELOPMENT",
    )
    admission = validate_source_admission(
        dict(object_json(refs["source_admission"])), artifact_root=root, source_root=source
    )
    qualification_calendar(refs["calendar"], refs["clock"], root)
    if (
        admission.evidence_scope != "RETROSPECTIVE_DEVELOPMENT"
        or admission.calendar_identity != _reference(refs["clock"], root)[1]
    ):
        raise IntegrityError("continuation requires exact retrospective source/calendar admission")
    configuration, _ = _read(refs["configuration"], root)
    if canonical_bytes(configuration) != canonical_bytes(
        parse_json(files("frem_s10").joinpath("science.json").read_bytes())
    ):
        raise IntegrityError("continuation configuration differs from installed frozen science")
    for key, cutoff in (("data_pilot", "2010-01-01"), ("data_campaign", "2017-01-01")):
        data, _ = _read(refs[key], root)
        if data != {
            "schema": "s10-data-manifest-v1",
            "assets": ["SPY", "TLT", "GLD"],
            "source_hashes": list(admission.source_hashes),
            "values_before": cutoff,
        }:
            raise IntegrityError("continuation source hashes or fixed cutoffs differ")
    return Plan(value, digest_bytes(raw), refs, roots, prefix)


def _append(work: Path, event: JSONObject) -> None:
    """Append under the held process lock; verify every previous hash before writing."""
    path = work / "stages.jsonl"
    if not path.exists():
        Journal(path).append(event)
        return
    count, previous = inspect_journal(path)
    raw = canonical_bytes({"sequence": count, "previous": previous, "event": event}) + b"\n"
    if len(raw) > 1_048_576:
        raise IntegrityError("continuation event exceeds metadata bound")
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _history(plan: Plan) -> tuple[dict[str, JSONObject], int]:
    path = plan.work / "stages.jsonl"
    if not path.exists():
        return {}, 0
    count, _ = inspect_journal(path)
    latest: dict[str, JSONObject] = {}
    pending: JSONObject | None = None
    with path.open("rb") as stream:
        for line in stream:
            event = object_json(object_json(parse_json(line))["event"])
            if event.get("plan_sha256") != plan.identity:
                raise IntegrityError("continuation journal belongs to a different plan")
            kind = event.get("kind")
            if kind == "HOST_BLOCKED":
                if pending is not None:
                    raise IntegrityError("host gate cannot interrupt an unresolved stage")
            elif kind == "STAGE_STARTED":
                if pending is not None or event.get("stage") not in STAGES:
                    raise IntegrityError("continuation has an invalid overlapping stage")
                _check_stage_start(str(event["stage"]), latest)
                pending = event
            elif kind == "STAGE_FINISHED":
                latest[str(event["stage"])] = _finished(plan, pending, event)
                pending = None
            else:
                raise IntegrityError("unknown continuation lifecycle event")
    if pending is not None:
        raise ExecutionStateError(
            "interrupted continuation stage is ambiguous; retained work cannot automatically repeat"
        )
    return latest, count


def _check_stage_start(stage: str, latest: dict[str, JSONObject]) -> None:
    index = STAGES.index(stage)
    if any(latest.get(prior, {}).get("status") != "COMPLETE" for prior in STAGES[:index]):
        raise IntegrityError("continuation stage has an incomplete predecessor")
    prior = latest.get(stage)
    if prior is not None and (prior.get("status") != "BLOCKED" or prior.get("retryable") is not True):
        raise IntegrityError("completed, failed or consumed stages cannot be restarted")
    if any(later in latest for later in STAGES[index + 1 :]):
        raise IntegrityError("continuation stage cannot regress behind later evidence")


def _finished(plan: Plan, pending: JSONObject | None, event: JSONObject) -> JSONObject:
    if pending is None or any(event.get(key) != pending.get(key) for key in ("stage", "attempt")):
        raise IntegrityError("continuation completion lacks its exact start")
    outcome, _ = _read(event["outcome"], plan.root)
    if outcome.get("stage") != event["stage"] or outcome.get("plan_sha256") != plan.identity:
        raise IntegrityError("continuation stage outcome binding differs")
    if outcome.get("status") not in ("COMPLETE", "BLOCKED", "FAILED", "FAILED_CONSUMED", "SCIENTIFIC_STOP"):
        raise IntegrityError("unknown continuation stage disposition")
    if outcome.get("retryable") is True and outcome.get("status") != "BLOCKED":
        raise IntegrityError("only an unconsumed blocked stage can permit automatic retry")
    return outcome


def _save(plan: Plan, directory: Path, name: str, value: JSONObject) -> JSONObject:
    identity = publish_json(directory, name, value)
    return {"path": str((directory / name).relative_to(plan.root)), "sha256": identity}


def _status(result: JSONObject) -> str:
    return str(result.get("state", result.get("status")))


def _verification(plan: Plan, directory: Path) -> JSONObject:
    from frem_s10.runtime.provenance import read_object
    from frem_s10.runtime.verification import audit_verification, run_verification

    reference = run_verification(
        artifact_root=plan.root,
        source_root=plan.source,
        code_ref=plan.artifacts["code"],
        environment_ref=plan.artifacts["environment"],
        catalog_ref=plan.artifacts["catalog"],
        bindings=plan.bindings(campaign=False),
        operation="prepare-precision",
        evidence_scope="RETROSPECTIVE_DEVELOPMENT",
        output_directory=str((directory / "checks").relative_to(plan.root)),
    )
    receipt = read_object(reference, plan.root)
    if receipt["status"] == "PASS":
        audit_verification(reference, artifact_root=plan.root, source_root=plan.source)
        return {"status": "COMPLETE", "verification": reference}
    logs = receipt["logs"]
    if not isinstance(logs, list) or len(logs) != 1:
        raise IntegrityError("verification failure lacks its exact observed evidence")
    evidence = read_object(read_object(logs[0], plan.root)["evidence"], plan.root)
    host = read_object(evidence["host"], plan.root)
    resource_only = evidence["commands"] == [] and host.get("status") != "PASS"
    return {
        "status": "BLOCKED" if resource_only else "FAILED",
        "verification": reference,
        "retryable": resource_only,
    }


def _freeze(
    plan: Plan, directory: Path, *, campaign: bool, verification: JSONValue, precision: JSONValue
) -> JSONObject:
    from frem_s10.runtime.pipeline import create_declaration

    request: JSONObject = {
        "schema": "s10-freeze-request-v1",
        "operation": "development-campaign" if campaign else "prepare-precision",
        "evidence_scope": "RETROSPECTIVE_DEVELOPMENT",
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": plan.operation_artifacts(campaign=campaign) | {"verification": verification},
        "source_admission": plan.artifacts["source_admission"],
        "precision": precision,
    }
    _save(plan, directory, "freeze-request.json", request)
    return create_declaration(
        directory / "freeze-request.json",
        artifact_root=plan.root,
        source_root=plan.source,
        consumption_root=plan.roots["consumption_root"],
        output_name=str((directory / "freeze").relative_to(plan.root)),
    )


def _scientific(
    plan: Plan, stage: str, directory: Path, prior: JSONObject | None, complete: dict[str, JSONObject]
) -> JSONObject:
    from frem_s10.runtime.pipeline import run_declared
    from frem_s10.runtime.verification import rebind_campaign_verification

    campaign = stage == "campaign"
    operation = "development-campaign" if campaign else "prepare-precision"
    if prior is not None:
        declaration = prior["declaration"]
    else:
        verification = complete["verification"]["verification"]
        precision: JSONValue = None
        if campaign:
            verification = rebind_campaign_verification(
                verification,
                artifact_root=plan.root,
                source_root=plan.source,
                bindings=plan.bindings(campaign=True),
                qualification=complete["qualification"]["qualification"],
            )
            precision = complete["precision"]["precision_prerequisite"]
        declaration = _freeze(
            plan, directory, campaign=campaign, verification=verification, precision=precision
        )
    manifest: JSONObject = {
        "schema": "s10-launch-manifest-v1",
        "declaration": declaration,
        **{key: str(plan.roots[key]) for key in ROOTS if key != "work_root"},
        "attempt_id": f"{plan.prefix}-{directory.name}",
    }
    _save(plan, directory, "launch.json", manifest)
    result = run_declared(directory / "launch.json", expected_operation=operation)
    result_ref = _save(plan, directory, "returned-result.json", result)
    status = _status(result)
    retryable = status == "REJECTED_UNCONSUMED" and result.get("consumed") is False
    if status == "COMPLETE":
        if campaign:
            return {
                "status": "COMPLETE",
                "declaration": declaration,
                "result": result_ref,
                "scientific_disposition": result["scientific_disposition"],
            }
        if result["scientific_disposition"] == "PRECISION_PASS":
            return {
                "status": "COMPLETE",
                "declaration": declaration,
                "result": result_ref,
                "precision_prerequisite": result["precision_prerequisite"],
            }
        return {
            "status": "SCIENTIFIC_STOP",
            "declaration": declaration,
            "result": result_ref,
            "scientific_disposition": result["scientific_disposition"],
            "retryable": False,
        }
    return {
        "status": "BLOCKED" if retryable else "FAILED_CONSUMED" if result.get("consumed") else "FAILED",
        "declaration": declaration,
        "result": result_ref,
        "retryable": retryable,
    }


def _qualification(plan: Plan, directory: Path, prior: JSONObject | None) -> JSONObject:
    from frem_s10.runtime.qualification import prepare_qualification, run_qualification

    request = (
        prior["request"]
        if prior is not None
        else prepare_qualification(
            artifact_root=plan.root,
            source_root=plan.source,
            code_ref=plan.artifacts["code"],
            environment_ref=plan.artifacts["environment"],
            calendar_ref=plan.artifacts["calendar"],
            clock_ref=plan.artifacts["clock"],
        )
    )
    launch: JSONObject = {
        "schema": "s10-qualification-launch-v1",
        "request": request,
        **{key: str(plan.roots[key]) for key in ROOTS if key != "work_root"},
        "attempt_id": f"{plan.prefix}-{directory.name}",
    }
    _save(plan, directory, "launch.json", launch)
    result = run_qualification(directory / "launch.json")
    result_ref = _save(plan, directory, "returned-result.json", result)
    if result.get("status") == "PASS":
        return {
            "status": "COMPLETE",
            "request": request,
            "qualification": result["receipt"],
            "result": result_ref,
        }
    completion = object_json(result.get("completion", result))
    retryable = _status(completion) == "REJECTED_UNCONSUMED" and completion.get("consumed") is False
    return {
        "status": "BLOCKED" if retryable else "FAILED_CONSUMED" if completion.get("consumed") else "FAILED",
        "request": request,
        "result": result_ref,
        "retryable": retryable,
    }


def _audit_completed(plan: Plan, stage: str, outcome: JSONObject) -> None:
    from frem_s10.runtime.pipeline import audit_precision_prerequisite
    from frem_s10.runtime.qualification import audit_qualification
    from frem_s10.runtime.receipts import audit_execution
    from frem_s10.runtime.verification import audit_verification

    if stage == "verification":
        audit_verification(outcome["verification"], artifact_root=plan.root, source_root=plan.source)
    elif stage == "qualification":
        bindings = plan.bindings(campaign=True)
        audit_qualification(
            outcome["qualification"],
            artifact_root=plan.root,
            source_root=plan.source,
            code_sha256=str(bindings["code"]),
            environment_sha256=str(bindings["environment"]),
            calendar_sha256=str(bindings["calendar"]),
        )
    else:
        result, _ = _read(outcome["result"], plan.root)
        audit_execution(result, artifact_root=plan.root)
        if stage == "precision":
            from frem_s10.runtime.declarations import _verify_precision_execution

            audit_precision_prerequisite(
                {"precision": outcome["precision_prerequisite"]},
                artifact_root=plan.root,
                source_root=plan.source,
            )
            _verify_precision_execution(result, str(result["declaration_sha256"]), plan.root)


def _bind_plan(plan: Plan) -> None:
    saved = plan.work / "plan.json"
    if saved.is_symlink():
        raise IntegrityError("saved continuation plan cannot be a symlink")
    if saved.exists():
        if parse_json(saved.read_bytes()) != plan.document:
            raise IntegrityError("work directory already belongs to a different immutable continuation plan")
    else:
        publish_json(plan.work, "plan.json", plan.document)


def _execute_stage(
    plan: Plan, stage: str, directory: Path, prior: JSONObject | None, complete: dict[str, JSONObject]
) -> JSONObject:
    try:
        if stage == "verification":
            return _verification(plan, directory)
        if stage == "qualification":
            return _qualification(plan, directory, prior)
        return _scientific(plan, stage, directory, prior, complete)
    except (S10Error, OSError) as exc:
        return {
            "status": "FAILED",
            "retryable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "consumption_requires_inspection": True,
        }


def _advance_locked(plan: Plan) -> JSONObject:
    _bind_plan(plan)
    history, sequence = _history(plan)
    complete: dict[str, JSONObject] = {}
    for stage in STAGES:
        prior = history.get(stage)
        if prior is not None and prior.get("status") == "COMPLETE":
            _audit_completed(plan, stage, prior)
            complete[stage] = prior
            continue
        if prior is not None and (prior.get("status") != "BLOCKED" or prior.get("retryable") is not True):
            return {
                "status": prior["status"],
                "stage": stage,
                "plan_sha256": plan.identity,
                "outcome": prior,
                "automatic_retry_permitted": False,
            }
        host = object_json(checked_json(check_host()))
        sequence += 1
        directory = plan.work / f"{stage}-{sequence:06d}"
        directory.mkdir(exist_ok=False)
        host_ref = _save(plan, directory, "host.json", host)
        if host.get("status") != "PASS":
            _append(
                plan.work,
                {"kind": "HOST_BLOCKED", "plan_sha256": plan.identity, "stage": stage, "host": host_ref},
            )
            return {
                "status": "BLOCKED",
                "stage": stage,
                "plan_sha256": plan.identity,
                "host": host_ref,
                "consumed": False,
                "automatic_retry_permitted": True,
            }
        event: JSONObject = {
            "kind": "STAGE_STARTED",
            "plan_sha256": plan.identity,
            "stage": stage,
            "attempt": directory.name,
            "host": host_ref,
        }
        _append(plan.work, event)
        outcome = _execute_stage(plan, stage, directory, prior, complete)
        outcome.update({"plan_sha256": plan.identity, "stage": stage})
        reference = _save(plan, directory, "outcome.json", outcome)
        _append(plan.work, event | {"kind": "STAGE_FINISHED", "outcome": reference})
        sequence += 1
        if outcome["status"] != "COMPLETE":
            return {
                "status": outcome["status"],
                "stage": stage,
                "plan_sha256": plan.identity,
                "outcome": reference,
                "automatic_retry_permitted": outcome.get("retryable") is True,
            }
        _audit_completed(plan, stage, outcome)
        complete[stage] = outcome
    return {
        "status": "COMPLETE",
        "plan_sha256": plan.identity,
        "scientific_disposition": complete["campaign"]["scientific_disposition"],
        "result": complete["campaign"]["result"],
        "protected_confirmation_executed": False,
    }


def advance(manifest_path: Path) -> JSONObject:
    """Advance until blocked or terminal; consumed or ambiguous stages never rerun."""
    plan = _plan(manifest_path)
    lock = plan.work / "controller.lock"
    if lock.is_symlink():
        raise IntegrityError("continuation lock cannot be a symlink")
    with lock.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExecutionStateError("another controller owns this continuation plan") from exc
        try:
            return _advance_locked(plan)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
