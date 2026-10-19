"""Bounded progress evidence, optional console failure and archival ordering."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frem_s10.errors import ArtifactIOError, ContractError, IntegrityError
from frem_s10.identities import JSONObject, object_json, parse_json
from frem_s10.runtime.journal import inspect_journal
from frem_s10.runtime.progress import ProgressLog


def events(path: Path) -> list[JSONObject]:
    return [object_json(object_json(parse_json(line))["event"]) for line in path.read_bytes().splitlines()]


def test_log_is_durable_timestamped_ordered_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    syncs: list[int] = []
    original = os.fsync

    def synced(descriptor: int) -> None:
        original(descriptor)
        syncs.append(descriptor)

    monkeypatch.setattr(os, "fsync", synced)
    clock = iter((10.0, 11.0, 13.0))
    printed: list[str] = []
    path = tmp_path / "metrics.jsonl"
    log = ProgressLog(path, "generated-smoke", clock=clock.__next__, console=printed.append)
    log.emit({"event": "STARTED", "phase": "forecast"})
    log.emit({"event": "ORIGIN_ARCHIVED", "phase": "forecast", "global_step": 1})
    rows = events(path)
    assert inspect_journal(path)[0] == 2
    assert [row["elapsed_seconds"] for row in rows] == [1.0, 3.0]
    assert all(str(row["timestamp"]).endswith("+00:00") for row in rows)
    assert all(float(str(row["process_peak_rss_mib"])) > 0 for row in rows)
    assert [json.loads(line) for line in printed] == rows
    assert len(syncs) >= 4
    with pytest.raises(ArtifactIOError):
        ProgressLog(path, "another-attempt")


@pytest.mark.parametrize("parent", (False, True))
def test_log_rejects_symlink_paths(tmp_path: Path, parent: bool) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    if parent:
        link = tmp_path / "linked"
        link.symlink_to(actual, target_is_directory=True)
        path = link / "progress.jsonl"
    else:
        target = actual / "existing.jsonl"
        target.write_text("preserved")
        path = tmp_path / "progress.jsonl"
        path.symlink_to(target)
    with pytest.raises(ArtifactIOError):
        ProgressLog(path, "generated-only")


def test_console_failure_preserves_local_log_and_omits_exception_text(tmp_path: Path) -> None:
    def detached(line: str) -> None:
        raise BrokenPipeError("secret-like exception detail must not be logged")

    path = tmp_path / "metrics.jsonl"
    log = ProgressLog(path, "generated-only", console=detached)
    log.emit({"event": "STARTED"})
    log.failure("forecast", ValueError("never publish this exception message"))
    assert inspect_journal(path)[0] == 2
    assert events(path)[1]["error_type"] == "ValueError"
    assert b"never publish" not in path.read_bytes()
    assert b"secret-like" not in path.read_bytes()


def test_regressing_clock_and_envelope_override_reject_without_append(tmp_path: Path) -> None:
    ticks = iter((10.0, 9.0))
    path = tmp_path / "metrics.jsonl"
    log = ProgressLog(path, "generated-only", clock=ticks.__next__)
    with pytest.raises(ContractError, match="envelope"):
        log.emit({"run_id": "forged"})
    with pytest.raises(ContractError, match="clock"):
        log.emit({"event": "STARTED"})
    assert inspect_journal(path)[0] == 0


def test_disk_error_is_explicit_without_discarding_prior_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metrics.jsonl"
    log = ProgressLog(path, "generated-only", console=lambda line: None)
    log.emit({"event": "STARTED"})
    before = path.read_bytes()

    def unavailable(event: object) -> str:
        raise ArtifactIOError("constructed unavailable disk")

    monkeypatch.setattr(log.journal, "append", unavailable)
    with pytest.raises(ArtifactIOError):
        log.emit({"event": "ORIGIN_ARCHIVED"})
    log.failure("forecast", ValueError("original remains the caller's failure"))
    assert path.read_bytes() == before


def test_progress_import_has_no_numerical_side_effects() -> None:
    code = "import sys; import frem_s10.runtime.progress; assert not ({'numpy', 'torch'} & set(sys.modules))"
    result = subprocess.run((sys.executable, "-B", "-c", code), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_transition_log_follows_real_checkpoint_and_keeps_state_exact(tmp_path: Path) -> None:
    from test_model_audit import first_transition

    from frem_s10.experiment.model_archive import ModelArchiveWriter
    from frem_s10.model.checkpoint import dump_state
    from frem_s10.runtime.progress import transition_sink

    directory = tmp_path / "archive"
    directory.mkdir()
    writer = ModelArchiveWriter(
        directory,
        experiment_id="generated-only",
        evidence_scope="GENERATED_ENGINEERING",
        binding_id="generated-binding",
        variants=("primary",),
    )
    path = tmp_path / "progress.jsonl"
    log = ProgressLog(path, "generated-only", console=lambda line: None)
    step = first_transition()
    original = dump_state(step.state)
    sink = transition_sink(writer, log)
    sink("primary", step)
    row = events(path)[0]
    assert row["state_id"] == step.state.record_id
    assert row["global_step"] == 1
    assert Path(str(row["checkpoint_path"])).is_file()
    assert dump_state(step.state) == original
    writer.finish()
    with pytest.raises(IntegrityError):
        sink("primary", step)
    assert inspect_journal(path)[0] == 1
