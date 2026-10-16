"""Both production operation paths use the same fail-closed bounded controller.

Observers and process handles below are constructed fault fixtures. These tests
exercise the real supervisor and durable records without starting a numerical
worker or claiming that synthetic counters admit scientific execution.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pytest
from test_declarations_audit import Fixture

from frem_s10.errors import ContractError, IntegrityError, ResourceLimitError
from frem_s10.identities import JSONObject, object_json, parse_json
from frem_s10.runtime import supervisor
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.resources import MemorySample
from frem_s10.runtime.supervisor import Budget, OwnedTree, ProcessRecord


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    output, markers = tmp_path / "output", tmp_path / "markers"
    output.mkdir()
    markers.mkdir()
    identity = publish_json(tmp_path, "fixture.json", {"scope": "GENERATED_ENGINEERING"})
    controller = ProcessRecord(os.getpid(), 1, os.getpid(), 32 * 2**20, "constructed-controller", "S")
    monkeypatch.setattr(supervisor, "read_processes", lambda: {controller.pid: controller})
    monkeypatch.setattr(supervisor, "check_host", lambda: {"status": "PASS", "samples": [{"pageouts": 10}]})
    monkeypatch.setattr(supervisor, "sample_memory", lambda: MemorySample("fixture", 4096, 262144, 0, 10))
    return output, markers, identity


def _execute(tmp_path: Path, output: Path, markers: Path, identity: str, operation: str) -> JSONObject:
    return supervisor.execute_operation(
        command=("constructed-worker-never-executed",),
        operation=operation,
        freeze_id="a" * 64,
        attempt_id="attempt",
        output_root=output,
        consumption_root=markers,
        input_hashes=((tmp_path / "fixture.json", identity),),
        estimated_output_bytes=1024,
        budget=Budget(5.0, grace_seconds=0.0),
    )


@pytest.mark.parametrize("operation", ["prepare-precision", "development-campaign"])
def test_each_operation_requires_its_own_prerequisites_before_consumption(
    tmp_path: Path, operation: str
) -> None:
    fixture = Fixture(tmp_path)
    if operation == "prepare-precision":
        reference = fixture.operation(operation)
        assert fixture.validate(reference) is False  # Pilot does not require its unproduced result.
        value = fixture.read(reference)
        references = dict(object_json(value["artifacts"]))
        del references["verification"]
        value["artifacts"] = references
    else:
        pilot = fixture.operation("prepare-precision")
        precision = fixture.precision_receipt(pilot)
        reference = fixture.operation(operation, precision)
        assert fixture.validate(reference) is False
        assert reference["sha256"] != pilot["sha256"]
        value = fixture.read(reference)
        value["precision"] = None
    with pytest.raises((ContractError, IntegrityError)):
        fixture.validate(fixture.save("missing-prerequisite", value))
    assert list(fixture.markers.iterdir()) == []


@pytest.mark.parametrize("operation", ["precision", "campaign"])
@pytest.mark.parametrize("fault", ["admission", "observer"])
def test_each_operation_rejects_failed_preflight_without_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, fault: str
) -> None:
    output, markers, identity = _fixture(tmp_path, monkeypatch)

    def unavailable() -> dict[int, ProcessRecord]:
        raise ResourceLimitError("constructed unavailable process observer")

    def unexpected_spawn(command: tuple[str, ...], directory: Path) -> subprocess.Popen[bytes]:
        raise AssertionError("preflight rejection must not dispatch any worker")

    monkeypatch.setattr(supervisor, "_spawn_worker", unexpected_spawn)
    if fault == "admission":
        monkeypatch.setattr(supervisor, "check_host", lambda: {"status": "BLOCKED"})
    else:
        monkeypatch.setattr(supervisor, "read_processes", unavailable)
    with pytest.raises(ResourceLimitError):
        _execute(tmp_path, output, markers, identity, operation)
    value = object_json(parse_json((output / "attempt/completion.json").read_bytes()))
    assert value["state"] == "REJECTED_UNCONSUMED"
    assert value["consumed"] is False and value["worker_dispatched"] is False
    assert list(markers.iterdir()) == []


@pytest.mark.parametrize("operation", ["precision", "campaign"])
def test_each_operation_preserves_immediate_post_consumption_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    output, markers, identity = _fixture(tmp_path, monkeypatch)

    def crash(command: tuple[str, ...], directory: Path) -> subprocess.Popen[bytes]:
        assert (markers / f"{operation}-{'a' * 64}.json").is_file()
        raise OSError("constructed immediate post-consumption crash")

    monkeypatch.setattr(supervisor, "_spawn_worker", crash)
    value = _execute(tmp_path, output, markers, identity, operation)
    assert value["state"] == "FAILED" and value["consumed"] is True
    assert value["worker_exit_code"] is None and value["operation"] == operation
    marker = object_json(parse_json((markers / f"{operation}-{'a' * 64}.json").read_bytes()))
    assert marker["state"] == "CONSUMED" and marker["resume_permitted"] is False
    states = [
        object_json(object_json(parse_json(line))["event"])["state"]
        for line in (output / "attempt/timeline.jsonl").read_bytes().splitlines()
    ]
    assert states == ["PREFLIGHT", "CONSUMED", "FAILED"]


@pytest.mark.parametrize("operation", ["precision", "campaign"])
@pytest.mark.parametrize("fault", ["observer", "rss", "clock"])
def test_each_operation_uses_actual_shared_guard_for_runtime_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, fault: str
) -> None:
    output, markers, identity = _fixture(tmp_path, monkeypatch)
    started = False
    stopped: list[int] = []

    class Worker:
        pid = 987654321  # Synthetic identity; no signal or OS process is created.
        returncode: int | None = None

    worker = Worker()

    def spawn(command: tuple[str, ...], directory: Path) -> subprocess.Popen[bytes]:
        nonlocal started
        started = True
        return cast("subprocess.Popen[bytes]", worker)

    def processes() -> dict[int, ProcessRecord]:
        if started and fault == "observer":
            raise ResourceLimitError("constructed observer disappeared after consumption")
        controller = ProcessRecord(os.getpid(), 1, os.getpid(), 32 * 2**20, "constructed-controller", "S")
        records = {controller.pid: controller}
        if started:
            records[worker.pid] = ProcessRecord(
                worker.pid, controller.pid, worker.pid, 225 * 2**20, "constructed-worker", "S"
            )
        return records

    def stop(handle: subprocess.Popen[bytes], grace: float, ownership: OwnedTree) -> None:
        stopped.append(handle.pid)
        worker.returncode = -15

    monkeypatch.setattr(supervisor, "_spawn_worker", spawn)
    monkeypatch.setattr(supervisor, "_stop_worker", stop)
    monkeypatch.setattr(supervisor, "read_processes", processes)
    monkeypatch.setattr(time, "monotonic", lambda: 9.0 if started and fault == "clock" else 10.0)
    if fault == "clock":
        with pytest.raises(ResourceLimitError, match="regressed"):
            _execute(tmp_path, output, markers, identity, operation)
        value = object_json(parse_json((output / "attempt/failure.json").read_bytes()))
        assert value["completion_valid"] is False and value["replay_permitted"] is False
    else:
        value = _execute(tmp_path, output, markers, identity, operation)
        assert value["state"] == "BUDGET_EXCEEDED"
        if fault == "rss":
            assert value["maximum_observed_owned_rss_mib"] == 257.0
            assert value["resource_observations"] == 1
        else:
            assert value["resource_observations"] == 0
            assert value["maximum_observed_owned_rss_mib"] == 0.0
    assert value["consumed"] is True and value["operation"] == operation
    assert stopped == [worker.pid]
    assert (markers / f"{operation}-{'a' * 64}.json").is_file()


def test_all_controllers_import_with_only_import_machinery_reads(tmp_path: Path) -> None:
    script = r"""
import _thread, importlib, os, sys, sysconfig, threading
# Initialize the interpreter's macOS platform configuration before the
# package boundary. No frem_s10 module or user artifact has been imported.
sysconfig.get_config_vars()
def reject_thread(*args, **kwargs):
    raise RuntimeError('unexpected thread start')
_thread.start_new_thread = reject_thread
threading.Thread.start = reject_thread
def guard(event, args):
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise RuntimeError('unexpected write open')
        frame = sys._getframe(1)
        permitted = False
        while frame:
            if (frame.f_globals.get('__name__') == 'importlib._bootstrap_external'
                    and frame.f_code.co_name == 'get_data'):
                permitted = True
                break
            frame = frame.f_back
        if not permitted or not isinstance(path, str) or not path.endswith(('.py', '.pyc')):
            raise RuntimeError('non-import file read')
    elif event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system',
                   'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.unlink'):
        raise RuntimeError('unexpected import side effect: ' + event)
sys.addaudithook(guard)
for probe in (lambda: open('forbidden-market-values.json'),
              lambda: _thread.start_new_thread(lambda: None, ()),
              lambda: threading.Thread(target=lambda: None).start()):
    try:
        probe()
    except RuntimeError:
        pass
    else:
        raise AssertionError('restriction probe did not reject')
for suffix in ('artifacts', 'catalog', 'claims', 'consumption', 'continuation', 'declarations',
               'journal', 'mutation_check', 'packaging', 'pipeline', 'preservation', 'provenance', 'qualification',
               'qualification_worker', 'receipts', 'resources', 'saved_audit', 'supervisor',
               'verification', 'worker'):
    importlib.import_module('frem_s10.runtime.' + suffix)
importlib.import_module('frem_s10.data.production')
importlib.import_module('frem_s10.cli')
assert 'numpy' not in sys.modules
assert not any(name.startswith(('frem_s10.model.', 'frem_s10.experiment.', 'frem_s10.analysis.',
                                'torch', 'pandas', 'sklearn', 'scipy', 'frem.')) for name in sys.modules)
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-B", "-X", f"pycache_prefix={tmp_path / 'never-written'}", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
