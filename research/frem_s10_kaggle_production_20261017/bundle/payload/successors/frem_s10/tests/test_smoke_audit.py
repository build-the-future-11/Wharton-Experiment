"""Bounded real-kernel generated checks; never market or CUDA evidence."""

import json
import signal
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frem_s10.cli import main
from frem_s10.errors import (
    ArtifactIOError,
    ContractError,
    ExecutionStateError,
    IntegrityError,
    ResourceLimitError,
)
from frem_s10.identities import JSONObject
from frem_s10.runtime.smoke import ActiveBudget, _step, evaluate_smoke, run_smoke
from frem_s10.runtime.smoke_checkpoint import commits, run_lock, save_checkpoint
from frem_s10.runtime.smoke_contract import read_manifest, smoke_plan
from frem_s10.runtime.smoke_data import START, generated_inputs
from frem_s10.runtime.smoke_export import export_smoke


def manifest(tmp_path: Path, **changes: object) -> Path:
    value: dict[str, object] = {
        "schema": "s10-generated-smoke-v1",
        "scope": "smoke",
        "device": "cpu",
        "precision": "float64",
        "seed": 11,
        "steps": 3,
        "variants": ["primary"],
        "budget_seconds": 60,
        "dataset": "s10-generated-smoke-data-v1",
        "checkpoint_every_origins": 1,
        "threads": 1,
    }
    value.update(changes)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path


def test_real_interruption_resume_evaluation_export_and_completed_reuse(tmp_path: Path) -> None:
    source = manifest(tmp_path, steps=7)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert run_smoke(source, full)["status"] == "COMPLETE"
    assert run_smoke(source, resumed, stop_after=3)["status"] == "INTERRUPTED"
    assert run_smoke(source, resumed, resume=True)["status"] == "COMPLETE"
    assert evaluate_smoke(full) == evaluate_smoke(resumed)
    left, right = list(commits(full)), list(commits(resumed))
    assert [row["states"] for row in left] == [row["states"] for row in right]
    assert [row["metrics"] for row in left] == [row["metrics"] for row in right]
    assert run_smoke(source, resumed, resume=True)["status"] == "COMPLETE_REUSED"
    assert export_smoke(resumed, tmp_path / "result.zip")["status"] == "COMPLETE"
    with pytest.raises(FileExistsError):
        export_smoke(resumed, tmp_path / "result.zip")
    with pytest.raises(FileExistsError):
        run_smoke(source, full)


def test_sigint_after_checkpoint_waits_for_durable_metrics_then_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frem_s10.runtime import smoke

    source = manifest(tmp_path)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert run_smoke(source, full)["status"] == "COMPLETE"
    save = save_checkpoint

    def interrupt_between_checkpoint_and_log(directory: Path, checkpoint: JSONObject) -> str:
        name = save(directory, checkpoint)
        signal.raise_signal(signal.SIGINT)
        return name

    monkeypatch.setattr(smoke, "save_checkpoint", interrupt_between_checkpoint_and_log)
    result = run_smoke(source, resumed)
    assert result["status"] == "INTERRUPTED"
    assert result["completed"] == 1
    assert evaluate_smoke(resumed)["status"] == "PARTIAL"
    monkeypatch.setattr(smoke, "save_checkpoint", save)
    assert run_smoke(source, resumed, resume=True)["status"] == "COMPLETE"
    assert evaluate_smoke(resumed) == evaluate_smoke(full)


@pytest.mark.parametrize(
    "change",
    [
        {"device": "cuda"},
        {"precision": "float32"},
        {"steps": 0},
        {"steps": 81},
        {"budget_seconds": 0},
        {"budget_seconds": float("nan")},
        {"budget_seconds": 301},
        {"seed": True},
        {"threads": 0},
        {"checkpoint_every_origins": 0},
        {"variants": ["primary", "primary"]},
        {"variants": ["unknown"]},
        {"scope": "confirmatory"},
        {"dataset": "market.csv"},
        {"ignored_flag": True},
    ],
)
def test_invalid_or_ignored_configuration_rejects(tmp_path: Path, change: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        read_manifest(manifest(tmp_path, **change))


def test_changed_config_and_corrupt_checkpoint_refuse_resume(tmp_path: Path) -> None:
    source = manifest(tmp_path)
    output = tmp_path / "run"
    run_smoke(source, output, stop_after=1)
    source = manifest(tmp_path, seed=29)
    with pytest.raises(IntegrityError, match="identical"):
        run_smoke(source, output, resume=True)
    source = manifest(tmp_path)
    checkpoint = next(output.glob("checkpoint-*.json"))
    checkpoint.write_text(checkpoint.read_text() + " ")
    with pytest.raises(IntegrityError, match="hash"):
        run_smoke(source, output, resume=True)


def test_budget_simulation_carries_prior_time_and_rejects_zero() -> None:
    now = [100.0]
    budget = ActiveBudget(10, 7, clock=lambda: now[0])
    budget.check()
    now[0] += 3
    with pytest.raises(ResourceLimitError):
        budget.check()
    with pytest.raises(ContractError):
        ActiveBudget(0, 0)


def test_concurrent_and_symlink_outputs_reject(tmp_path: Path) -> None:
    source = manifest(tmp_path)
    output = tmp_path / "run"
    run_smoke(source, output, stop_after=1)
    with run_lock(output), pytest.raises(ExecutionStateError, match="already active"):
        run_smoke(source, output, resume=True)
    linked = tmp_path / "linked"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(ArtifactIOError):
        run_smoke(source, linked, resume=True)


def test_current_future_target_cannot_change_issuance(tmp_path: Path) -> None:
    from frem_s10.model.engine import initial_state
    from frem_s10.model.records import ForecastState

    inputs = generated_inputs(11, 2)
    changed = replace(inputs.records[START], g=inputs.records[START].g * 1.1)
    alternate = replace(inputs, records=(*inputs.records[:START], changed, *inputs.records[START + 1 :]))
    state = initial_state(inputs.transform)
    first, rows1 = _step({"primary": state}, inputs, 0)
    second, rows2 = _step({"primary": state}, alternate, 0)
    a, b = first["primary"], second["primary"]
    assert isinstance(a, ForecastState) and isinstance(b, ForecastState)
    assert a.record_id == b.record_id
    np.testing.assert_array_equal(a.pending[-1].probabilities, b.pending[-1].probabilities)
    assert rows1 != rows2  # Only the reporting-only score observes the changed target.


def test_cli_dry_run_and_cuda_rejection(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    cells = smoke_plan(path)["cells"]
    assert isinstance(cells, list) and len(cells) == 1
    assert main(["smoke-plan", "--manifest", str(path)]) == 0
    assert main(["preflight", "--device", "cuda"]) != 0


def test_missing_commit_and_source_drift_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import frem_s10.runtime.smoke as smoke

    source = manifest(tmp_path)
    output = tmp_path / "run"
    run_smoke(source, output, stop_after=2)
    monkeypatch.setattr(smoke, "runtime_identity", lambda: {"source_sha256": "changed"})
    with pytest.raises(IntegrityError, match="identical"):
        run_smoke(source, output, resume=True)
    monkeypatch.undo()
    (output / "step-0001.json").unlink()
    with pytest.raises(IntegrityError, match="incomplete"):
        evaluate_smoke(output)


def test_partial_evaluation_and_export_refusal(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    output = tmp_path / "run"
    run_smoke(path, output, stop_after=1)
    assert evaluate_smoke(output)["status"] == "PARTIAL"
    with pytest.raises(IntegrityError, match="complete"):
        export_smoke(output, tmp_path / "result.zip")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 9.0])
def test_budget_rejects_nonfinite_or_regressing_clocks(value: float) -> None:
    now = [10.0]
    budget = ActiveBudget(20, 0, clock=lambda: now[0])
    now[0] = value
    with pytest.raises(ResourceLimitError):
        budget.check()


def test_abandoned_session_and_missing_logs_fail_closed(tmp_path: Path) -> None:
    source = manifest(tmp_path)
    output = tmp_path / "run"
    run_smoke(source, output, stop_after=1)
    (output / "session-start-abandoned.json").write_text("{}")
    with pytest.raises(ResourceLimitError, match="abandoned"):
        run_smoke(source, output, resume=True)
    with pytest.raises(ResourceLimitError, match="abandoned"):
        evaluate_smoke(output)
    with pytest.raises(ResourceLimitError, match="abandoned"):
        export_smoke(output, tmp_path / "abandoned.zip")
    (output / "session-start-abandoned.json").unlink()
    next(output.glob("metrics-*.jsonl")).unlink()
    with pytest.raises(IntegrityError, match="durable log"):
        evaluate_smoke(output)


def test_complete_origin_count_cannot_hide_spent_session_budget(tmp_path: Path) -> None:
    source = manifest(tmp_path, steps=1)
    output = tmp_path / "run"
    assert run_smoke(source, output)["status"] == "COMPLETE"
    # Simulated additional charged session tests the evaluator's aggregate budget,
    # independently of the complete checkpoint count and status convenience view.
    (output / "session-start-charged.json").write_text("{}")
    (output / "session-end-charged.json").write_text('{"active_seconds":60}')
    with pytest.raises(ResourceLimitError, match="budget"):
        evaluate_smoke(output)
    with pytest.raises(ResourceLimitError, match="budget"):
        export_smoke(output, tmp_path / "over-budget.zip")


def test_larger_scope_checks_admission_before_data_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frem_s10.runtime.resources as resources

    monkeypatch.setattr(resources, "check_host", lambda: {"status": "BLOCKED"})
    source = manifest(tmp_path, scope="development-engineering", steps=72)
    with pytest.raises(ResourceLimitError, match="host admission"):
        run_smoke(source, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()
