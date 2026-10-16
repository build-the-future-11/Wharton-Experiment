"""Generated engineering lifecycle tests; synthetic counters never admit real work."""

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import frem_s10.runtime.supervisor as supervisor
from frem_s10.errors import (
    ArtifactIOError,
    ContractError,
    ExecutionStateError,
    IntegrityError,
    ResourceLimitError,
)
from frem_s10.identities import JSONValue
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.journal import Journal, inspect_journal
from frem_s10.runtime.resources import MemorySample
from frem_s10.runtime.supervisor import (
    Budget,
    MonotonicClock,
    OwnedTree,
    ProcessRecord,
    directory_bytes,
    execute_operation,
    owned_processes,
    parse_processes,
)


def setup_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    output, markers = tmp_path / "output", tmp_path / "markers"
    output.mkdir()
    markers.mkdir()
    identity = publish_json(tmp_path, "fixture.json", {"scope": "generated fixture only"})
    monkeypatch.setattr(supervisor, "check_host", lambda: {"status": "PASS", "samples": [{"pageouts": 10}]})
    monkeypatch.setattr(supervisor, "sample_memory", lambda: MemorySample("fixture", 4096, 262144, 0, 10))
    return output, markers, identity


def run_generated(
    tmp_path: Path,
    output: Path,
    markers: Path,
    identity: str,
    code: str,
    *,
    name: str = "attempt",
    wall: float = 5.0,
    output_bytes: int = 268_435_456,
) -> dict[str, object]:
    return dict(
        execute_operation(
            command=(sys.executable, "-I", "-c", code),
            operation="precision",
            freeze_id="a" * 64,
            attempt_id=name,
            output_root=output,
            consumption_root=markers,
            input_hashes=((tmp_path / "fixture.json", identity),),
            estimated_output_bytes=1024,
            budget=Budget(wall, poll_seconds=0.01, grace_seconds=0.1, output_bytes=output_bytes),
        )
    )


def test_journal_preserves_chain_and_rejects_tamper(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "timeline.jsonl")
    a = journal.append({"state": "CREATED"})
    b = journal.append({"state": "CONSUMED"})
    assert inspect_journal(journal.path) == (2, b)
    rows = journal.path.read_bytes().splitlines(keepends=True)
    assert hashlib.sha256(rows[0]).hexdigest() == a
    assert json.loads(rows[1])["previous"] == a
    journal.path.write_bytes(rows[0].replace(b"CREATED", b"ALTERED") + rows[1])
    with pytest.raises(IntegrityError):
        journal.append({"state": "COMPLETE"})
    with pytest.raises(ArtifactIOError):
        Journal(journal.path)


def test_partial_journal_rejected_without_repair(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    path.write_bytes(b'{"partial":')
    with pytest.raises(IntegrityError):
        inspect_journal(path)
    assert path.read_bytes() == b'{"partial":'


@pytest.mark.parametrize(
    "values",
    [
        {"rss_mib": 257},
        {"wall_seconds": 3601},
        {"poll_seconds": 0.6},
        {"grace_seconds": 3},
        {"output_bytes": 0},
        {"rss_mib": float("nan")},
    ],
)
def test_budget_cannot_relax_frozen_ceiling(values: dict[str, float]) -> None:
    arguments = {"wall_seconds": 5, **values}
    with pytest.raises(ContractError):
        Budget(**arguments)  # type: ignore[arg-type]  # Deliberately invalid dynamic argument domains.


def test_process_accounting_includes_controller_and_descendants() -> None:
    rows = parse_processes(
        "10 1 1 1024 Mon Oct 12 01:02:03 2026\n20 10 20 2048 Mon Oct 12 01:02:04 2026\n21 20 20 4096 Mon Oct 12 01:02:05 2026"
    )
    owned = owned_processes(rows, 10, 20)
    assert sum(p.rss_bytes for p in owned) == 7 * 1024**2
    assert {p.pid for p in owned} == {10, 20, 21}
    escaped = parse_processes("10 1 1 1024 t\n20 10 20 2048 t\n21 20 21 4096 t")
    with pytest.raises(ResourceLimitError, match="escaped"):
        owned_processes(escaped, 10, 20)
    with pytest.raises(ResourceLimitError):
        parse_processes("1 0 1 1024 t\n1 0 1 1024 t")
    with pytest.raises(ResourceLimitError):
        parse_processes("")


def test_output_quota_rejects_symlinks(tmp_path: Path) -> None:
    (tmp_path / "x").write_bytes(b"abc")
    assert directory_bytes(tmp_path) == 3
    (tmp_path / "escape").symlink_to("/tmp")
    with pytest.raises(ResourceLimitError):
        directory_bytes(tmp_path)


def test_complete_generated_attempt_is_consumed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    result = run_generated(
        tmp_path,
        output,
        markers,
        identity,
        "import os,time; assert os.environ['OPENBLAS_NUM_THREADS']=='1'; time.sleep(.05)",
    )
    assert result["state"] == "COMPLETE" and result["consumed"] is True
    assert result["scientific_disposition"] == "NOT_ASSESSED_BY_SUPERVISOR"
    peak = result["maximum_observed_owned_rss_mib"]
    assert isinstance(peak, float) and peak > 0
    assert inspect_journal(output / "attempt/resources.jsonl")[0] > 0
    marker = markers / ("precision-" + "a" * 64 + ".json")
    saved = marker.read_bytes()
    with pytest.raises(ExecutionStateError):
        run_generated(tmp_path, output, markers, identity, "raise AssertionError", name="repeat")
    assert marker.read_bytes() == saved
    refusal = json.loads((output / "repeat/completion.json").read_text())
    assert refusal["state"] == "CONSUMPTION_REFUSED" and refusal["consumed"] is None
    assert refusal["worker_dispatched"] is False
    assert inspect_journal(output / "attempt/timeline.jsonl")[0] == 4


def test_preflight_failure_cannot_consume_or_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "check_host", lambda: {"status": "BLOCKED"})
    with pytest.raises(ResourceLimitError):
        run_generated(tmp_path, output, markers, identity, "raise AssertionError")
    assert list(markers.iterdir()) == []
    result = json.loads((output / "attempt/completion.json").read_text())
    assert result["state"] == "REJECTED_UNCONSUMED"
    assert not (output / "attempt/worker.stdout").exists()


def test_launch_failure_keeps_consumption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)

    def failed(*args: object, **kwargs: object) -> None:
        raise OSError("injected launch failure")

    monkeypatch.setattr(supervisor, "_spawn_worker", failed)
    result = run_generated(tmp_path, output, markers, identity, "pass")
    assert result["state"] == "FAILED" and result["consumed"] is True
    assert len(list(markers.iterdir())) == 1


def test_nonzero_exit_is_retained_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    result = run_generated(
        tmp_path, output, markers, identity, "import time;time.sleep(.05);raise SystemExit(9)"
    )
    assert result["state"] == "FAILED" and result["worker_exit_code"] == 9


def test_timeout_and_cancel_retain_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)

    def expired(*args: object) -> float:
        raise ResourceLimitError("injected time ceiling", details={"observed_rss_mib": 49.0})

    monkeypatch.setattr(supervisor, "_sample_guard", expired)
    result = run_generated(tmp_path, output, markers, identity, "import time;time.sleep(30)")
    assert result["state"] == "BUDGET_EXCEEDED"
    assert result["maximum_observed_owned_rss_mib"] == 49
    elapsed = result["elapsed_seconds"]
    assert isinstance(elapsed, float) and elapsed < 3
    assert len(list(markers.iterdir())) == 1


def test_cancel_terminates_owned_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    result = execute_operation(
        command=(sys.executable, "-I", "-c", "import time;time.sleep(30)"),
        operation="precision",
        freeze_id="a" * 64,
        attempt_id="cancel",
        output_root=output,
        consumption_root=markers,
        input_hashes=((tmp_path / "fixture.json", identity),),
        estimated_output_bytes=1024,
        budget=Budget(5, grace_seconds=0.1),
        cancelled=lambda: True,
    )
    assert result["state"] == "INTERRUPTED"
    events = [
        json.loads(line)["event"] for line in (output / "cancel/timeline.jsonl").read_text().splitlines()
    ]
    pid = next(e["worker_pid"] for e in events if e["state"] == "RUNNING")
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert len(list(markers.iterdir())) == 1


@pytest.mark.parametrize("escape", [False, True])
def test_parent_exit_and_group_escape_cannot_leave_observed_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    escape: bool,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    code = (
        "import os,time,pathlib\n"
        "pid=os.fork()\n"
        "if pid == 0:\n"
        + (" os.setsid()\n" if escape else "")
        + " pathlib.Path('child.pid').write_text(str(os.getpid()))\n"
        " time.sleep(30)\n"
        " os._exit(0)\n"
        "time.sleep(.15)\n"
    )
    result = run_generated(tmp_path, output, markers, identity, code)
    pid = int((output / "attempt/child.pid").read_text())
    current = supervisor.read_processes().get(pid)
    assert current is None or current.state.startswith("Z")
    assert result["state"] == ("BUDGET_EXCEEDED" if escape else "COMPLETE")
    observations = (output / "attempt/resources.jsonl").read_text()
    assert str(pid) in observations


def test_escaped_pid_reuse_never_signals_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = ProcessRecord(10, 1, 10, 1, "controller", "S")
    root = ProcessRecord(20, 10, 20, 1, "worker", "S")
    child = ProcessRecord(21, 20, 21, 1, "owned-child", "S")
    replacement = replace(child, parent=1, started="unrelated-new-start")
    ownership = OwnedTree(controller, 20)
    original = {p.pid: p for p in (controller, root, child)}
    ownership.observe(original)
    snapshots = iter((original, {p.pid: p for p in (controller, root, replacement)}))
    monkeypatch.setattr(supervisor, "read_processes", lambda: next(snapshots))
    groups: list[int] = []
    pids: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: groups.append(pid))
    monkeypatch.setattr(os, "kill", lambda pid, sig: pids.append(pid))

    class Worker:
        pid = 20
        returncode = None

    supervisor._signal_owned(cast("subprocess.Popen[bytes]", Worker()), ownership, 15)
    assert groups == [20] and pids == []


def test_preflight_requires_process_measurement_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)

    def unavailable() -> dict[int, ProcessRecord]:
        raise ResourceLimitError("injected missing process accounting")

    monkeypatch.setattr(supervisor, "read_processes", unavailable)
    with pytest.raises(ResourceLimitError, match="accounting"):
        run_generated(tmp_path, output, markers, identity, "raise AssertionError")
    assert not list(markers.iterdir())
    assert not (output / "attempt/worker.stdout").exists()


def test_no_resource_observations_cannot_report_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)

    def no_journal(
        ownership: OwnedTree,
        output: Path,
        budget: Budget,
        baseline: int,
        elapsed: float,
        journal: Journal,
        clock: MonotonicClock,
    ) -> float:
        ownership.observe(supervisor.read_processes())
        return 0.0

    monkeypatch.setattr(supervisor, "_sample_guard", no_journal)
    result = run_generated(tmp_path, output, markers, identity, "pass")
    assert result["state"] != "COMPLETE" and result["resource_observations"] == 0
    assert "no resource observations" in str(result["error"])


def test_monotonic_regression_is_rejected() -> None:
    with pytest.MonkeyPatch.context() as patch:
        readings = iter((10.0, 10.2, 10.1))
        patch.setattr(time, "monotonic", lambda: next(readings))
        clock = MonotonicClock()
        assert clock.elapsed() == pytest.approx(0.2)
        with pytest.raises(ResourceLimitError, match="regressed"):
            clock.elapsed()


def test_runtime_clock_regression_cleans_worker_and_writes_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    original_time = time.monotonic

    def regressed() -> float:
        return original_time() - (100.0 if (output / "attempt/worker.stdout").exists() else 0.0)

    monkeypatch.setattr(time, "monotonic", regressed)
    with pytest.raises(ResourceLimitError, match="regressed"):
        run_generated(tmp_path, output, markers, identity, "import time;time.sleep(30)")
    failure = json.loads((output / "attempt/failure.json").read_text())
    assert failure["consumed"] is True and failure["completion_valid"] is False
    events = [
        json.loads(row)["event"] for row in (output / "attempt/timeline.jsonl").read_text().splitlines()
    ]
    pid = next(event["worker_pid"] for event in events if event["state"] == "RUNNING")
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancellation_cleans_generated_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    child_path = output / "cancel/child.pid"
    code = (
        "import os,time,pathlib\n"
        "pid=os.fork()\n"
        "if pid == 0:\n"
        " pathlib.Path('child.pid').write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    result = execute_operation(
        command=(sys.executable, "-I", "-c", code),
        operation="precision",
        freeze_id="a" * 64,
        attempt_id="cancel",
        output_root=output,
        consumption_root=markers,
        input_hashes=((tmp_path / "fixture.json", identity),),
        estimated_output_bytes=1024,
        budget=Budget(5.0, poll_seconds=0.01, grace_seconds=0.1),
        cancelled=child_path.exists,
    )
    assert result["state"] == "INTERRUPTED"
    child = supervisor.read_processes().get(int(child_path.read_text()))
    assert child is None or child.state.startswith("Z")


def test_running_journal_failure_still_cleans_child_and_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    original = Journal.append
    workers: list[int] = []

    def append(self: Journal, event: JSONValue) -> str:
        if isinstance(event, dict) and event.get("state") == "RUNNING":
            pid = event["worker_pid"]
            assert isinstance(pid, int)
            workers.append(pid)
            raise ArtifactIOError("injected running journal failure")
        return original(self, event)

    monkeypatch.setattr(Journal, "append", append)
    result = run_generated(tmp_path, output, markers, identity, "import time;time.sleep(30)")
    assert result["state"] == "FAILED" and result["consumed"] is True
    assert "journal failure" in str(result["error"])
    assert len(workers) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(workers[0], 0)


@pytest.mark.parametrize("stage", ["consumed_journal", "resources_creation", "completion"])
def test_post_consumption_artifact_failure_has_independent_durable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    original_append = Journal.append
    original_init = Journal.__init__
    original_publish = publish_json

    def append(self: Journal, event: JSONValue) -> str:
        if stage == "consumed_journal" and isinstance(event, dict) and event.get("state") == "CONSUMED":
            raise ArtifactIOError("injected consumed journal failure")
        return original_append(self, event)

    def initialize(self: Journal, path: Path) -> None:
        if stage == "resources_creation" and path.name == "resources.jsonl":
            raise ArtifactIOError("injected resources creation failure")
        original_init(self, path)

    def publish(directory: Path, name: str, value: JSONValue) -> str:
        if stage == "completion" and name == "completion.json":
            raise ArtifactIOError("injected publication failure")
        return original_publish(directory, name, value)

    monkeypatch.setattr(Journal, "append", append)
    monkeypatch.setattr(Journal, "__init__", initialize)
    monkeypatch.setattr(supervisor, "publish_json", publish)
    with pytest.raises(ArtifactIOError):
        run_generated(tmp_path, output, markers, identity, "pass")
    failure = json.loads((output / "attempt/failure.json").read_text())
    assert failure["state"] == "FAILED" and failure["consumed"] is True
    assert failure["completion_valid"] is False and failure["replay_permitted"] is False
    assert len(list(markers.iterdir())) == 1


@pytest.mark.parametrize("fault", ["rss", "free", "pageouts", "output", "wall"])
def test_resource_faults_terminate_owned_worker_and_preserve_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    original_processes = supervisor.read_processes

    def sample() -> MemorySample:
        running = (output / "attempt/worker.stdout").exists()
        free = 1 if running and fault == "free" else 262144
        pageouts = 11 if running and fault == "pageouts" else 10
        return MemorySample("constructed counter injection", 4096, free, 0, pageouts)

    def processes() -> dict[int, ProcessRecord]:
        rows = original_processes()
        if fault == "rss" and (output / "attempt/worker.stdout").exists():
            rows[os.getpid()] = replace(rows[os.getpid()], rss_bytes=257 * 2**20)
        return rows

    monkeypatch.setattr(supervisor, "sample_memory", sample)
    monkeypatch.setattr(supervisor, "read_processes", processes)
    code = "import pathlib,time;pathlib.Path('generated.bin').write_bytes(b'x'*16384);time.sleep(30)"
    result = run_generated(
        tmp_path,
        output,
        markers,
        identity,
        code,
        wall=0.4 if fault == "wall" else 5.0,
        output_bytes=8192 if fault == "output" else 268_435_456,
    )
    assert result["state"] == "BUDGET_EXCEEDED"
    assert result["resource_observations"] != 0
    assert len(list(markers.iterdir())) == 1
    events = [
        json.loads(row)["event"] for row in (output / "attempt/timeline.jsonl").read_text().splitlines()
    ]
    pid = next(event["worker_pid"] for event in events if event["state"] == "RUNNING")
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
