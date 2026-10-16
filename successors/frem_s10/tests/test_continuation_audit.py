"""Controller fault cases use injected operations; no market or numerical fits run."""

import fcntl
import subprocess
import sys
from pathlib import Path

import pytest

from frem_s10.errors import ExecutionStateError, IntegrityError
from frem_s10.identities import JSONObject, canonical_bytes, object_json, parse_json
from frem_s10.runtime import continuation as controller
from frem_s10.runtime.journal import inspect_journal


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> controller.Plan:
    roots = {key: tmp_path / key for key in controller.ROOTS}
    roots["artifact_root"] = tmp_path
    for path in set(roots.values()):
        path.mkdir(exist_ok=True)
    refs: JSONObject = {key: {"path": key + ".json", "sha256": "a" * 64} for key in controller.REFS}
    document: JSONObject = {
        "schema": "s10-continuation-plan-v1",
        "artifacts": refs,
        "attempt_prefix": "bounded",
        **{key: str(value) for key, value in roots.items()},
    }
    plan = controller.Plan(document, "c" * 64, refs, roots, "bounded")
    monkeypatch.setattr(controller, "_plan", lambda _: plan)
    return plan


def calls(
    plan: controller.Plan, monkeypatch: pytest.MonkeyPatch, *, precision: str = "COMPLETE"
) -> list[str]:
    observed: list[str] = []

    def verification(plan: controller.Plan, directory: Path) -> JSONObject:
        observed.append("verification")
        return {"status": "COMPLETE", "verification": {"path": "observed.json", "sha256": "a" * 64}}

    def scientific(
        plan: controller.Plan,
        stage: str,
        directory: Path,
        prior: JSONObject | None,
        complete: dict[str, JSONObject],
    ) -> JSONObject:
        observed.append(stage)
        if stage == "precision":
            return {
                "status": precision,
                "precision_prerequisite": {"path": "pilot.json", "sha256": "b" * 64},
                "retryable": False,
            }
        return {
            "status": "COMPLETE",
            "scientific_disposition": "NEGATIVE",
            "result": {"path": "campaign.json", "sha256": "d" * 64},
        }

    def qualification(plan: controller.Plan, directory: Path, prior: JSONObject | None) -> JSONObject:
        observed.append("qualification")
        return {"status": "COMPLETE", "qualification": {"path": "qualification.json", "sha256": "e" * 64}}

    def audit(plan: controller.Plan, stage: str, outcome: JSONObject) -> None:
        assert outcome["status"] == "COMPLETE"

    monkeypatch.setattr(controller, "_verification", verification)
    monkeypatch.setattr(controller, "_scientific", scientific)
    monkeypatch.setattr(controller, "_qualification", qualification)
    monkeypatch.setattr(controller, "_audit_completed", audit)
    monkeypatch.setattr(controller, "check_host", lambda: {"status": "PASS"})
    return observed


def test_continuation_import_does_not_load_numpy_or_models() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import frem_s10.runtime.continuation; assert 'numpy' not in sys.modules; assert not any(k.startswith('frem_s10.model.') for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_blocked_host_preserves_fresh_receipts_without_any_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    observed = calls(plan, monkeypatch)
    monkeypatch.setattr(
        controller, "check_host", lambda: {"status": "FAIL", "reason": "generated low memory"}
    )
    first = controller.advance(tmp_path / "unused.json")
    before = {path: path.read_bytes() for path in plan.work.rglob("*.json")}
    second = controller.advance(tmp_path / "unused.json")
    assert first["status"] == second["status"] == "BLOCKED"
    assert first["consumed"] is second["consumed"] is False
    assert observed == []
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert inspect_journal(plan.work / "stages.jsonl")[0] == 2
    assert len(tuple(plan.work.glob("verification-*/host.json"))) == 2
    assert not tuple(plan.roots["consumption_root"].iterdir())


def test_fixed_order_completes_once_and_accepts_negative_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    observed = calls(plan, monkeypatch)
    first = controller.advance(tmp_path / "unused.json")
    second = controller.advance(tmp_path / "unused.json")
    assert first == second
    assert first["status"] == "COMPLETE" and first["scientific_disposition"] == "NEGATIVE"
    assert observed == list(controller.STAGES)
    assert inspect_journal(plan.work / "stages.jsonl")[0] == 8


@pytest.mark.parametrize("failure", ("SCIENTIFIC_STOP", "FAILED_CONSUMED", "FAILED"))
def test_terminal_pilot_never_launches_qualification_or_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    observed = calls(plan, monkeypatch, precision=failure)
    assert controller.advance(tmp_path / "unused.json")["status"] == failure
    assert controller.advance(tmp_path / "unused.json")["status"] == failure
    assert observed == ["verification", "precision"]


def test_ambiguous_started_stage_cannot_automatically_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    observed = calls(plan, monkeypatch)
    controller._append(
        plan.work,
        {
            "kind": "STAGE_STARTED",
            "plan_sha256": plan.identity,
            "stage": "verification",
            "attempt": "interrupted",
        },
    )
    with pytest.raises(ExecutionStateError, match="ambiguous"):
        controller.advance(tmp_path / "unused.json")
    assert observed == []


def test_resigned_journal_cannot_skip_predecessor_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    observed = calls(plan, monkeypatch)
    controller._append(
        plan.work,
        {"kind": "STAGE_STARTED", "plan_sha256": plan.identity, "stage": "campaign", "attempt": "skipped"},
    )
    with pytest.raises(IntegrityError, match="incomplete predecessor"):
        controller.advance(tmp_path / "unused.json")
    assert observed == []


def test_journal_corruption_and_concurrent_controller_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    calls(plan, monkeypatch)
    with (plan.work / "controller.lock").open("wb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ExecutionStateError, match="another controller"):
            controller.advance(tmp_path / "unused.json")
    (plan.work / "stages.jsonl").write_bytes(b'{"incomplete":')
    with pytest.raises(IntegrityError, match="partial"):
        controller.advance(tmp_path / "unused.json")


def test_unconsumed_pilot_retry_reuses_prior_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    observed = calls(plan, monkeypatch)
    count = 0
    declaration: JSONObject = {"path": "same-freeze.json", "sha256": "f" * 64}

    def scientific(
        plan: controller.Plan,
        stage: str,
        directory: Path,
        prior: JSONObject | None,
        complete: dict[str, JSONObject],
    ) -> JSONObject:
        nonlocal count
        observed.append(stage)
        count += 1
        if count == 1:
            assert prior is None
            return {"status": "BLOCKED", "retryable": True, "declaration": declaration}
        assert prior is not None and prior["declaration"] == declaration
        return {"status": "SCIENTIFIC_STOP", "retryable": False, "declaration": declaration}

    monkeypatch.setattr(controller, "_scientific", scientific)
    assert controller.advance(tmp_path / "unused.json")["status"] == "BLOCKED"
    assert controller.advance(tmp_path / "unused.json")["status"] == "SCIENTIFIC_STOP"
    assert observed == ["verification", "precision", "precision"]


def test_scientific_dispatch_reuses_exact_freeze_and_rejects_nonpass_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frem_s10.runtime import pipeline

    plan = fixture(tmp_path, monkeypatch)
    declaration: JSONObject = {"path": "frozen.json", "sha256": "a" * 64}
    prior: JSONObject = {"declaration": declaration}
    first, second = plan.work / "first", plan.work / "second"
    first.mkdir()
    second.mkdir()
    observed: list[JSONObject] = []

    def execute(path: Path, *, expected_operation: str) -> JSONObject:
        launch = object_json(parse_json(path.read_bytes()))
        observed.append(launch)
        assert launch["declaration"] == declaration and expected_operation == "prepare-precision"
        if len(observed) == 1:
            return {"state": "REJECTED_UNCONSUMED", "consumed": False}
        return {"state": "COMPLETE", "scientific_disposition": "PRECISION_INADEQUATE"}

    monkeypatch.setattr(pipeline, "run_declared", execute)
    blocked = controller._scientific(plan, "precision", first, prior, {})
    stop = controller._scientific(plan, "precision", second, blocked, {})
    assert blocked["status"] == "BLOCKED" and blocked["retryable"] is True
    assert stop["status"] == "SCIENTIFIC_STOP" and stop["retryable"] is False
    assert observed[0]["attempt_id"] != observed[1]["attempt_id"]


@pytest.mark.parametrize(
    "commands,host,status", (([], "FAIL", "BLOCKED"), ([{"exit_code": 1}], "PASS", "FAILED"))
)
def test_verification_retries_only_zero_command_resource_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: list[JSONObject], host: str, status: str
) -> None:
    from frem_s10.runtime import verification
    from frem_s10.runtime.provenance import save_object

    plan = fixture(tmp_path, monkeypatch)
    host_ref = save_object(tmp_path, {"status": host})
    evidence = save_object(tmp_path, {"commands": list(commands), "host": host_ref})
    result = save_object(tmp_path, {"evidence": evidence})
    receipt = save_object(tmp_path, {"status": "FAIL", "logs": [result]})

    def run(**kwargs: object) -> JSONObject:
        return receipt

    monkeypatch.setattr(verification, "run_verification", run)
    value = controller._verification(plan, plan.work)
    assert value["status"] == status
    assert value["retryable"] is (status == "BLOCKED")


def test_changed_plan_and_overlapping_roots_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    (plan.work / "plan.json").write_bytes(canonical_bytes({"different": "plan"}))
    with pytest.raises(IntegrityError, match="different immutable"):
        controller.advance(tmp_path / "unused.json")
    changed = plan.document | {"work_root": str(plan.roots["output_root"])}
    with pytest.raises(IntegrityError, match="overlap"):
        controller._roots(changed)


def test_saved_completed_stage_is_reaudited_before_continuing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path, monkeypatch)
    observed = calls(plan, monkeypatch)
    assert controller.advance(tmp_path / "unused.json")["status"] == "COMPLETE"

    def refuse(plan: controller.Plan, stage: str, outcome: JSONObject) -> None:
        raise IntegrityError("retained evidence changed")

    monkeypatch.setattr(controller, "_audit_completed", refuse)
    with pytest.raises(IntegrityError, match="retained evidence changed"):
        controller.advance(tmp_path / "unused.json")
    assert observed == list(controller.STAGES)
