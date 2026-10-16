"""Bounded one-use local process supervision; scientific admission is a caller obligation.

This module never imports the model, reads market values or accepts commands from
source data. The future reviewed launcher must bind its fixed command and all
scientific prerequisites to the declaration before calling ``execute_operation``.
Polling is conservative instrumentation, not a hard kernel-enforced memory limit.
"""

import json
import math
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from frem_s10.errors import ContractError, ResourceLimitError, S10Error
from frem_s10.identities import JSONValue, parse_json
from frem_s10.runtime.artifacts import _plain_directory, publish_json, read_verified_json
from frem_s10.runtime.consumption import consume
from frem_s10.runtime.journal import Journal, inspect_journal, write_failure_receipt
from frem_s10.runtime.resources import check_host, sample_memory


@dataclass(frozen=True)
class Budget:
    """Fixed upper bounds; smaller budgets are permitted for generated checks."""

    wall_seconds: float
    rss_mib: float = 256.0
    poll_seconds: float = 0.25
    grace_seconds: float = 2.0
    output_bytes: int = 268_435_456

    def __post_init__(self) -> None:
        """Reject nonfinite limits and any relaxation of the declared ceilings."""
        for name, maximum in (("wall_seconds", 3600), ("rss_mib", 256), ("poll_seconds", 0.5)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ContractError(f"invalid bounded resource: {name}")
        if type(self.grace_seconds) not in (int, float) or not 0 <= self.grace_seconds <= 2:
            raise ContractError("termination grace must be between zero and two seconds")
        if type(self.output_bytes) is not int or self.output_bytes <= 0:
            raise ContractError("positive integer output byte ceiling required")


@dataclass(frozen=True)
class ProcessRecord:
    """OS process identity and resident bytes; no command or environment captured."""

    pid: int
    parent: int
    group: int
    rss_bytes: int
    started: str
    state: str = ""


def parse_processes(text: str, *, includes_state: bool = False) -> dict[int, ProcessRecord]:
    """Parse macOS/POSIX ps fields, rejecting malformed or ambiguous identities."""
    records = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5 if includes_state else 4)
        try:
            pid, parent, group, kib = (int(x) for x in fields[:4])
        except ValueError as exc:
            raise ResourceLimitError("invalid process accounting fields") from exc
        if (
            len(fields) != (6 if includes_state else 5)
            or pid <= 0
            or min(parent, group, kib) < 0
            or pid in records
        ):
            raise ResourceLimitError("missing or duplicate process accounting identity")
        records[pid] = ProcessRecord(
            pid, parent, group, kib * 1024, fields[-1], fields[4] if includes_state else ""
        )
    if not records:
        raise ResourceLimitError("empty process accounting")
    return records


def read_processes() -> dict[int, ProcessRecord]:
    """Read only PID/parent/group/RSS/start time; unavailable measurement fails."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,stat=,lstart="],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceLimitError("owned process measurement unavailable") from exc
    return parse_processes(result.stdout, includes_state=True)


def owned_processes(
    records: dict[int, ProcessRecord],
    controller_pid: int,
    worker_pid: int,
) -> tuple[ProcessRecord, ...]:
    """Count the controller and spawned group, and reject escaped descendants."""
    if controller_pid not in records or worker_pid not in records:
        raise ResourceLimitError("controller or worker process identity unavailable")
    worker = records[worker_pid]
    if worker.parent != controller_pid or worker.group != worker_pid:
        raise ResourceLimitError("worker is not the newly owned process group")
    descendants = {worker_pid}
    while True:
        expanded = descendants | {p.pid for p in records.values() if p.parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    if any(records[pid].group != worker_pid for pid in descendants):
        raise ResourceLimitError("owned descendant escaped the worker group")
    members = {p.pid for p in records.values() if p.group == worker_pid}
    if members != descendants:
        raise ResourceLimitError("unknown process ownership in worker group")
    return tuple(records[pid] for pid in sorted(members | {controller_pid}))


def directory_bytes(directory: Path) -> int:
    """Measure the owned output tree and reject symlink-based quota escapes."""
    total = 0
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ResourceLimitError("symlink in owned output tree")
        total += sum((Path(parent) / name).stat().st_size for name in files)
    return total


class MonotonicClock:
    """Reject nonfinite or regressing timestamps instead of extending budgets."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        if not math.isfinite(self.started):
            raise ResourceLimitError("monotonic measurement unavailable")
        self.last = self.started

    def read(self) -> float:
        """Return a checked monotonic reading."""
        current = time.monotonic()
        if not math.isfinite(current) or current < self.last:
            raise ResourceLimitError("monotonic clock regressed or became unavailable")
        self.last = current
        return current

    def elapsed(self) -> float:
        """Return elapsed time only after verifying monotonicity."""
        return self.read() - self.started


class OwnedTree:
    """Track observed descendants by PID/start identity for trusted fixed workers.

    The direct child remains unreaped until cleanup, reserving its process-group
    identity even after exit. Observed escaped descendants remain tracked after
    reparenting. This is not a sandbox against unobserved malicious daemonization.
    """

    def __init__(self, controller: ProcessRecord, worker_pid: int) -> None:
        self.controller = controller
        self.worker_pid = worker_pid
        self.known: dict[int, str] = {}
        self.worker_exited = False

    def observe(self, records: dict[int, ProcessRecord]) -> tuple[ProcessRecord, ...]:
        """Retain identities before reporting escaped lineage to the guard."""
        controller = records.get(self.controller.pid)
        if controller is None or controller.started != self.controller.started:
            raise ResourceLimitError("controller start identity unavailable or changed")
        root = records.get(self.worker_pid)
        if root is None or root.parent != controller.pid or root.group != self.worker_pid:
            raise ResourceLimitError("unreaped owned worker identity unavailable")
        if self.worker_pid in self.known and root.started != self.known[self.worker_pid]:
            raise ResourceLimitError("worker start identity changed")
        self.known[self.worker_pid] = root.started
        current = {p.pid for p in records.values() if self.known.get(p.pid) == p.started}
        # The unreaped child exclusively reserves this newly created group.
        current |= {p.pid for p in records.values() if p.group == self.worker_pid}
        while True:
            expanded = current | {p.pid for p in records.values() if p.parent in current}
            if expanded == current:
                break
            current = expanded
        for pid in current:
            record = records[pid]
            if pid in self.known and self.known[pid] != record.started:
                raise ResourceLimitError("observed descendant PID was reused")
            self.known[pid] = record.started
        if not root.state:
            raise ResourceLimitError("worker process state unavailable")
        self.worker_exited = root.state.startswith("Z")
        return (controller, *(records[pid] for pid in sorted(current)))

    def live_known(self, records: dict[int, ProcessRecord]) -> tuple[ProcessRecord, ...]:
        """Return only still-matching non-zombie owned identities."""
        return tuple(
            p for p in records.values() if self.known.get(p.pid) == p.started and not p.state.startswith("Z")
        )


def _signal_owned(
    worker: subprocess.Popen[bytes],
    ownership: OwnedTree,
    signum: int,
) -> bool:
    """Signal the reserved group and only verified start identities outside it."""
    records = read_processes()
    if worker.returncode is None:
        ownership.observe(records)
        if any(p.group == worker.pid for p in ownership.live_known(records)):
            with suppress(ProcessLookupError):
                os.killpg(worker.pid, signum)
    for process in ownership.live_known(records):
        if process.group != worker.pid:
            # Recheck immediately before signaling an escaped PID. Never adopt
            # a reused PID merely because its earlier identity was owned.
            current = read_processes().get(process.pid)
            if current is not None and current.started == process.started:
                with suppress(ProcessLookupError):
                    os.kill(process.pid, signum)
    return bool(ownership.live_known(records))


def _stop_worker(worker: subprocess.Popen[bytes], grace: float, ownership: OwnedTree) -> None:
    """Clean observed descendants even after direct-worker exit, then reap it."""
    cleanup_error: OSError | ResourceLimitError | None = None
    try:
        active = _signal_owned(worker, ownership, signal.SIGTERM)
        if active and grace:
            time.sleep(grace)
        _signal_owned(worker, ownership, signal.SIGKILL)
    except (OSError, ResourceLimitError) as exc:
        cleanup_error = exc
        # A still-unreaped direct child reserves its group, even if ps failed.
        if worker.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGKILL)
    finally:
        worker.wait(timeout=2)
    if cleanup_error is not None:
        raise ResourceLimitError("owned cleanup could not verify all descendants") from cleanup_error
    for _ in range(20):
        if not ownership.live_known(read_processes()):
            return
        time.sleep(0.01)
    raise ResourceLimitError("verified owned descendants survived cleanup")


def _preflight_disk(output_root: Path, estimate: int, budget: Budget) -> None:
    if type(estimate) is not int or estimate < 0 or estimate > budget.output_bytes:
        raise ContractError("output estimate must fit the declared byte budget")
    required = estimate + max(268_435_456, math.ceil(0.2 * estimate))
    if shutil.disk_usage(output_root).free < required:
        raise ResourceLimitError("insufficient output volume reserve")


def _sample_guard(
    ownership: OwnedTree,
    output: Path,
    budget: Budget,
    baseline_pageouts: int,
    elapsed: float,
    journal: Journal,
    clock: MonotonicClock,
) -> float:
    measurement_started = clock.read()
    records = ownership.observe(read_processes())
    rss = sum(p.rss_bytes for p in records) / 2**20
    memory = sample_memory()
    size = directory_bytes(output)
    journal.append(
        {
            "elapsed_seconds": elapsed,
            "owned_rss_mib": rss,
            "output_bytes": size,
            "free_plus_speculative_mib": memory.available_mib,
            "pageouts": memory.pageouts,
            "owned_processes": [{"pid": p.pid, "start_identity": p.started} for p in records],
        }
    )
    if any(p.group != ownership.worker_pid for p in records if p.pid != ownership.controller.pid):
        raise ResourceLimitError(
            "owned descendant escaped the worker group", details={"observed_rss_mib": rss}
        )
    if elapsed > budget.wall_seconds or rss > budget.rss_mib or size > budget.output_bytes:
        raise ResourceLimitError("elapsed/RSS/output budget exceeded", details={"observed_rss_mib": rss})
    if memory.available_mib < 512 or memory.pageouts != baseline_pageouts:
        raise ResourceLimitError("host resource admission no longer holds", details={"observed_rss_mib": rss})
    _preflight_disk(output, 0, budget)
    if clock.read() - measurement_started > 0.5:
        raise ResourceLimitError(
            "resource observation exceeded polling interval", details={"observed_rss_mib": rss}
        )
    return rss


def _poll(
    worker: subprocess.Popen[bytes],
    output: Path,
    budget: Budget,
    baseline_pageouts: int,
    clock: MonotonicClock,
    journal: Journal,
    cancelled: Callable[[], bool],
    ownership: OwnedTree,
) -> tuple[str, float, str | None]:
    maximum_rss = 0.0
    try:
        while True:
            poll_started = clock.read()
            if cancelled():
                raise KeyboardInterrupt
            elapsed = clock.elapsed()
            maximum_rss = max(
                maximum_rss,
                _sample_guard(
                    ownership,
                    output,
                    budget,
                    baseline_pageouts,
                    elapsed,
                    journal,
                    clock,
                ),
            )
            if ownership.worker_exited:
                break
            time.sleep(max(0.0, budget.poll_seconds - (clock.read() - poll_started)))
        if inspect_journal(journal.path)[0] == 0:
            raise ResourceLimitError("no resource observations; completion cannot be certified")
        if clock.elapsed() > budget.wall_seconds:
            raise ResourceLimitError("elapsed budget exceeded before finalization")
        if directory_bytes(output) > budget.output_bytes:
            raise ResourceLimitError("output budget exceeded before finalization")
        return ("FINISHED", maximum_rss, None)
    except KeyboardInterrupt:
        return "INTERRUPTED", maximum_rss, "user cancellation"
    except (OSError, ResourceLimitError) as exc:
        return "BUDGET_EXCEEDED", _observed_peak(maximum_rss, exc), str(exc)


def _observed_peak(previous: float, error: OSError | ResourceLimitError) -> float:
    if isinstance(error, ResourceLimitError):
        observed = error.details.get("observed_rss_mib")
        if isinstance(observed, (int, float)):
            return max(previous, observed)
    return previous


def _preflight(
    input_hashes: tuple[tuple[Path, str], ...],
    output: Path,
    output_root: Path,
    estimated_output_bytes: int,
    budget: Budget,
    clock: MonotonicClock,
) -> tuple[int, ProcessRecord]:
    """Verify bounded local prerequisites and fresh host counters without dispatch."""
    for path, expected in input_hashes:
        read_verified_json(path, expected)
    processes = read_processes()
    controller = processes.get(os.getpid())
    if controller is None or not controller.started or not controller.state:
        raise ResourceLimitError("controller process measurement unavailable before consumption")
    if controller.rss_bytes / 2**20 > budget.rss_mib:
        raise ResourceLimitError("controller RSS already exceeds operation budget")
    _preflight_disk(output_root, estimated_output_bytes, budget)
    host = check_host()
    publish_json(output, "preflight.json", parse_json(json.dumps(host, allow_nan=False)))
    if host["status"] != "PASS":
        raise ResourceLimitError("host admission blocked before consumption")
    if clock.elapsed() >= budget.wall_seconds:
        raise ResourceLimitError("budget expired during preflight")
    host_samples = host["samples"]
    if not isinstance(host_samples, list) or not host_samples or not isinstance(host_samples[-1], dict):
        raise ResourceLimitError("preflight sample provenance unavailable")
    baseline = sample_memory()
    if baseline.pageouts != host_samples[-1]["pageouts"]:
        raise ResourceLimitError("pageouts changed immediately before consumption")
    if baseline.available_mib < 512:
        raise ResourceLimitError("resource floor failed immediately before consumption")
    if clock.elapsed() >= budget.wall_seconds:
        raise ResourceLimitError("budget expired immediately before consumption")
    return baseline.pageouts, controller


def execute_operation(
    *,
    command: tuple[str, ...],
    operation: str,
    freeze_id: str,
    attempt_id: str,
    output_root: Path,
    consumption_root: Path,
    input_hashes: tuple[tuple[Path, str], ...],
    estimated_output_bytes: int,
    budget: Budget,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, JSONValue]:
    """Check local guards, consume once, and retain a supervised operation outcome.

    The API establishes engineering lifecycle/resource integrity only. Input
    identities must be the separately reviewed scientific prerequisite receipts;
    this generic primitive does not judge their scientific adequacy. No CLI market
    route may call it before that operation-specific admission layer is verified.
    """
    if not command or any(not isinstance(x, str) or not x for x in command):
        raise ContractError("nonempty fixed argv required")
    if operation not in ("precision", "campaign", "saved-audit") or (
        operation == "precision" and budget.wall_seconds > 600
    ):
        raise ContractError("operation or precision budget invalid")
    if not input_hashes:
        raise ContractError("reviewed prerequisite receipt identities required")
    output_root = _plain_directory(output_root)
    _plain_directory(consumption_root)
    if not attempt_id or Path(attempt_id).name != attempt_id or attempt_id in (".", ".."):
        raise ContractError("plain attempt identity required")
    output = output_root / attempt_id
    output.mkdir(exist_ok=False)
    journal = Journal(output / "timeline.jsonl")
    clock = MonotonicClock()
    journal.append(
        {
            "state": "PREFLIGHT",
            "time": datetime.now(UTC).isoformat(),
            "estimated_output_bytes": estimated_output_bytes,
            "output_limit_bytes": budget.output_bytes,
        }
    )
    try:
        baseline_pageouts, controller = _preflight(
            input_hashes, output, output_root, estimated_output_bytes, budget, clock
        )
    except Exception as exc:
        _retain_refusal(output, journal, "REJECTED_UNCONSUMED", exc, consumed=False)
        raise
    try:
        marker_hash = consume(
            consumption_root,
            operation=operation,
            freeze_id=freeze_id,
            attempt_id=attempt_id,
            output_path=str(output),
        )
    except Exception as exc:
        # An fsync/publication error can be ambiguous even if bytes appeared.
        _retain_refusal(output, journal, "CONSUMPTION_REFUSED", exc, consumed=None)
        raise
    try:
        journal.append({"state": "CONSUMED", "marker_sha256": marker_hash})
        return _run_consumed(
            command,
            output,
            operation,
            freeze_id,
            attempt_id,
            budget,
            clock,
            baseline_pageouts,
            controller,
            journal,
            cancelled,
        )
    except (Exception, KeyboardInterrupt) as exc:
        write_failure_receipt(
            output,
            {
                "state": "FAILED",
                "consumed": True,
                "operation": operation,
                "freeze_id": freeze_id,
                "attempt_id": attempt_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completion_valid": False,
                "replay_permitted": False,
            },
        )
        raise


def _retain_refusal(
    output: Path,
    journal: Journal,
    state: str,
    error: Exception,
    *,
    consumed: bool | None,
) -> None:
    """Retain rejected attempts, including ambiguous consumption publication."""
    event: dict[str, JSONValue] = {
        "state": state,
        "consumed": consumed,
        "error_type": type(error).__name__,
        "worker_dispatched": False,
        "replay_permitted": False,
    }
    try:
        journal.append(event)
        publish_json(output, "completion.json", event)
    except (OSError, S10Error):
        write_failure_receipt(output, event | {"completion_valid": False})


def _worker_environment() -> dict[str, str]:
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("PYTHONPATH", None)
    return env


def _spawn_worker(command: tuple[str, ...], output: Path) -> subprocess.Popen[bytes]:
    with (output / "worker.stdout").open("xb") as stdout, (output / "worker.stderr").open("xb") as stderr:
        return subprocess.Popen(
            command,
            cwd=output,
            env=_worker_environment(),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


def _run_consumed(
    command: tuple[str, ...],
    output: Path,
    operation: str,
    freeze_id: str,
    attempt_id: str,
    budget: Budget,
    clock: MonotonicClock,
    baseline_pageouts: int,
    controller: ProcessRecord,
    journal: Journal,
    cancelled: Callable[[], bool],
) -> dict[str, JSONValue]:
    resources = Journal(output / "resources.jsonl")
    state, error, peak = "FAILED", None, 0.0
    worker: subprocess.Popen[bytes] | None = None
    ownership: OwnedTree | None = None
    try:
        worker = _spawn_worker(command, output)
        ownership = OwnedTree(controller, worker.pid)
        journal.append({"state": "RUNNING", "worker_pid": worker.pid})
        state, peak, error = _poll(
            worker,
            output,
            budget,
            baseline_pageouts,
            clock,
            resources,
            cancelled,
            ownership,
        )
    except (OSError, S10Error, KeyboardInterrupt) as exc:
        state = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        if worker is not None and ownership is not None:
            try:
                _stop_worker(worker, budget.grace_seconds, ownership)
            except (OSError, ResourceLimitError, subprocess.SubprocessError) as exc:
                state, error = "FAILED", "cleanup: " + str(exc)
    returncode = None if worker is None else worker.returncode
    if state == "FINISHED":
        state = "COMPLETE" if returncode == 0 else "FAILED"
    observations = inspect_journal(resources.path)[0]
    if state == "COMPLETE" and observations == 0:
        state, error = "FAILED", "no resource observations; completion cannot be certified"
    elapsed = clock.elapsed()
    if state == "COMPLETE" and elapsed > budget.wall_seconds:
        state, error = "BUDGET_EXCEEDED", "elapsed budget exceeded during cleanup"
    result: dict[str, JSONValue] = {
        "state": state,
        "consumed": True,
        "operation": operation,
        "freeze_id": freeze_id,
        "attempt_id": attempt_id,
        "worker_exit_code": returncode,
        "error": error,
        "elapsed_seconds": elapsed,
        "maximum_observed_owned_rss_mib": peak,
        "resource_observations": observations,
        "wall_limit_seconds": budget.wall_seconds,
        "rss_limit_mib": budget.rss_mib,
        "output_limit_bytes": budget.output_bytes,
        "sampling_interval_seconds": budget.poll_seconds,
        "termination_grace_seconds": budget.grace_seconds,
        "sampling_can_overshoot": True,
        "scientific_disposition": "NOT_ASSESSED_BY_SUPERVISOR",
        "ownership_scope": "fixed trusted worker and observed descendants; not a security sandbox",
    }
    journal.append({"state": state, "worker_exit_code": returncode})
    inspect_journal(journal.path)
    publish_json(output, "completion.json", result)
    if state == "COMPLETE" and directory_bytes(output) > budget.output_bytes:
        raise ResourceLimitError("output budget exceeded during artifact finalization")
    return result
