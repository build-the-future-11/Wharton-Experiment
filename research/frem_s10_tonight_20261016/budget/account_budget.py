"""Isolated, stdlib-only quota guard. This module never launches or stops compute.

All writers for an account must use the same ledger and external coordination.
Evidence identifiers are caller attestations, not provider authentication. Quota
is provider quota-hours, never silently equated with wall or device-hours.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

DEADLINE = datetime.fromisoformat("2026-10-16T16:30:00+00:00").timestamp()
MAX_AGE_SECONDS = 60.0
MAX_LEDGER_BYTES = 4 * 1024 * 1024
PHASES = ("setup", "compute", "export", "shutdown")


class Blocked(RuntimeError):
    """No permission to allocate or continue GPU work."""


class StopRequired(Blocked):
    """Caller must stop work and obtain a separate platform-stop observation."""


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise Blocked(f"{name}: finite nonnegative number required")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Blocked(f"{name}: finite nonnegative number required") from exc
    if not math.isfinite(result) or result < 0:
        raise Blocked(f"{name}: finite nonnegative number required")
    return result


def dec(value: Any) -> Decimal:
    return Decimal(str(number(value, "counter")))


def text_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise Blocked(f"{name}: bounded nonempty identifier required")
    return value


def caps(initial_quota_hours: float | None) -> dict[str, float]:
    """No assumed initial quota; reserve is 10% of the selected cap."""
    if initial_quota_hours is None:
        return {"preferred": 0.0, "expanded": 0.0}
    q0 = dec(initial_quota_hours)
    return {
        "preferred": float(max(Decimal(0), min(Decimal(6), q0 - 24))),
        "expanded": float(max(Decimal(0), min(Decimal(10), q0 - 20))),
    }


@dataclass(frozen=True)
class PhaseCost:
    wall_seconds_upper: float
    device_hours_upper: float
    estimated_quota_hours_upper: float


@dataclass(frozen=True)
class Plan:
    setup: PhaseCost
    compute: PhaseCost
    export: PhaseCost
    shutdown: PhaseCost
    device_count: int
    quota_per_device_hour_upper: float
    conversion_evidence_id: str

    def totals(self) -> dict[str, float]:
        if type(self.device_count) is not int or self.device_count < 1:
            raise Blocked("explicit positive device count required")
        conversion = dec(self.quota_per_device_hour_upper)
        if conversion <= 0:
            raise Blocked("positive evidenced quota conversion upper bound required")
        text_id(self.conversion_evidence_id, "quota conversion evidence")
        walls = devices = quotas = Decimal(0)
        for name in PHASES:
            phase = getattr(self, name)
            if not isinstance(phase, PhaseCost):
                raise Blocked("explicit bounds for every phase required")
            wall, device, quota = (
                dec(phase.wall_seconds_upper),
                dec(phase.device_hours_upper),
                dec(phase.estimated_quota_hours_upper),
            )
            if device < wall * self.device_count / 3600 or quota < device * conversion:
                raise Blocked("phase bound understates devices or quota conversion")
            walls += wall
            devices += device
            quotas += quota
        if walls <= 0 or quotas <= 0:
            raise Blocked("positive bounded all-in run cost required")
        return {
            "wall_seconds_upper": number(float(walls), "total wall"),
            "device_hours_upper": number(float(devices), "total device"),
            "estimated_quota_hours_upper": number(float(quotas), "total quota"),
        }


@dataclass(frozen=True)
class Observation:
    account_id: str
    quota_period_id: str
    remaining_quota_hours: float
    quota_observed_at: float
    active_session_ids: tuple[str, ...]
    sessions_observed_at: float
    quota_evidence_id: str
    sessions_evidence_id: str
    coordination_evidence_id: str
    coordinated_at: float
    coordination_ledger_id: str


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_path(path: Path) -> Path:
    result = path.absolute()
    if any(part.is_symlink() for part in (result, *result.parents)):
        raise Blocked("ledger path and parents must not be symlinks")
    return result


class AccountLedger:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = safe_path(path)
        self.clock = clock

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        account_id: str,
        quota_period_id: str,
        initial_quota_hours: float | None = None,
        expanded: bool = False,
        expanded_approval_id: str | None = None,
        allocation_enabled: bool = False,
        account_ready: bool = False,
        allocation_approval_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> AccountLedger:
        ledger = cls(path, clock=clock)
        for flag in (expanded, allocation_enabled, account_ready):
            if type(flag) is not bool:
                raise Blocked("explicit boolean policy flags required")
        if expanded:
            text_id(expanded_approval_id, "explicit expanded approval")
        if allocation_enabled:
            text_id(allocation_approval_id, "explicit GPU allocation approval")
        cap = caps(initial_quota_hours)["expanded" if expanded else "preferred"]
        policy = {
            "ledger_id": str(uuid.uuid4()),
            "account_id": text_id(account_id, "account"),
            "quota_period_id": text_id(quota_period_id, "quota period"),
            "initial_quota_hours": initial_quota_hours,
            "selected_cap_hours": cap,
            "spendable_hours": float(dec(cap) * Decimal("0.9")),
            "reserve_fraction": 0.1,
            "account_floor_hours": 20.0,
            "deadline": DEADLINE,
            "expanded": expanded,
            "expanded_approval_id": expanded_approval_id,
            "allocation_enabled": allocation_enabled,
            "account_ready": account_ready,
            "allocation_approval_id": allocation_approval_id,
        }
        fd = os.open(ledger.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            ledger._append(fd, [], {"kind": "created", "policy": policy})
        finally:
            os.close(fd)
        directory = os.open(ledger.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return ledger

    def _now(self) -> float:
        return number(self.clock(), "current UTC clock")

    def _append(self, fd: int, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
        now = self._now()
        if rows and now < rows[-1]["timestamp"]:
            raise Blocked("UTC clock regressed; allocation requires a trusted clock")
        row = {
            "sequence": len(rows), "event_id": str(uuid.uuid4()), "timestamp": now,
            "previous": rows[-1]["sha256"] if rows else None, "payload": payload,
        }
        row["sha256"] = hashlib.sha256(canonical(row)).hexdigest()
        data = canonical(row) + b"\n"
        if os.fstat(fd).st_size + len(data) > MAX_LEDGER_BYTES:
            raise Blocked("bounded ledger is full")
        while data:
            written = os.write(fd, data)
            if written <= 0:
                raise Blocked("ledger append failed")
            data = data[written:]
        os.fsync(fd)

    def _transaction(self, action: Callable[[list[dict[str, Any]]], tuple[Any, dict[str, Any] | None]]) -> Any:
        safe_path(self.path)
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_LEDGER_BYTES:
                raise Blocked("bounded regular ledger required")
            linked = self.path.stat(follow_symlinks=False)
            if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
                raise Blocked("ledger was replaced")
            raw = os.read(fd, MAX_LEDGER_BYTES + 1)
            if not raw.endswith(b"\n") or not raw:
                raise Blocked("incomplete ledger; history requires review")
            rows = []
            previous = None
            for sequence, line in enumerate(raw.splitlines()):
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(Blocked("nonfinite ledger")))
                supplied = row.pop("sha256")
                if row["sequence"] != sequence or row["previous"] != previous:
                    raise Blocked("broken ledger sequence")
                if supplied != hashlib.sha256(canonical(row)).hexdigest():
                    raise Blocked("broken ledger hash")
                row["sha256"] = supplied
                timestamp = number(row["timestamp"], "event timestamp")
                if rows and timestamp < rows[-1]["timestamp"]:
                    raise Blocked("nonmonotonic ledger timestamps")
                rows.append(row)
                previous = supplied
            if len(rows) > 4096 or rows[0]["payload"]["kind"] != "created":
                raise Blocked("unsupported ledger history")
            if self._now() < rows[-1]["timestamp"]:
                raise Blocked("UTC clock regressed")
            result, payload = action(rows)
            if payload is not None:
                self._append(fd, rows, payload)
            return result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Blocked("invalid ledger; history requires review") from exc
        finally:
            os.close(fd)

    def _state(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        policy = rows[0]["payload"]["policy"]
        runs: dict[str, dict[str, Any]] = {}
        for row in rows[1:]:
            event = row["payload"]
            run_id = event["run_id"]
            if event["kind"] == "reserved" and run_id not in runs:
                runs[run_id] = {**event, "launch": None, "outcome": None, "platform_stopped": False}
            elif event["kind"] == "launch_authorized" and run_id in runs and runs[run_id]["launch"] is None:
                runs[run_id]["launch"] = event
            elif event["kind"] == "outcome" and run_id in runs and runs[run_id]["outcome"] is None:
                runs[run_id]["outcome"] = event
            elif event["kind"] == "platform_stopped" and run_id in runs and not runs[run_id]["platform_stopped"]:
                runs[run_id]["platform_stopped"] = True
            else:
                raise Blocked("invalid ledger transition")
        return policy, runs

    def summary(self) -> dict[str, Any]:
        def read(rows: list[dict[str, Any]]) -> tuple[Any, None]:
            policy, runs = self._state(rows)
            estimate = sum((dec(run["totals"]["estimated_quota_hours_upper"]) for run in runs.values()), Decimal(0))
            return {**policy, "reserved_estimated_quota_hours": float(estimate), "runs": len(runs),
                    "unresolved_platform_runs": sum(not run["platform_stopped"] for run in runs.values())}, None
        return self._transaction(read)

    def _observation(self, observation: Observation, policy: dict[str, Any]) -> None:
        now = self._now()
        if (observation.account_id, observation.quota_period_id) != (policy["account_id"], policy["quota_period_id"]):
            raise Blocked("account or quota period differs")
        if observation.coordination_ledger_id != policy["ledger_id"]:
            raise Blocked("coordination must cover this shared account ledger")
        for name in ("quota_evidence_id", "sessions_evidence_id", "coordination_evidence_id"):
            text_id(getattr(observation, name), name)
        times = [number(getattr(observation, name), name) for name in
                 ("quota_observed_at", "sessions_observed_at", "coordinated_at")]
        if any(not 0 <= now - timestamp <= MAX_AGE_SECONDS for timestamp in times):
            raise Blocked("fresh nonfuture quota, active-session and coordination observations required")
        if times[2] < max(times[:2]):
            raise Blocked("coordination must follow both account observations")
        if type(observation.active_session_ids) is not tuple or observation.active_session_ids:
            raise Blocked("a fresh empty account-wide active-session observation is required")
        if policy["initial_quota_hours"] is None:
            raise Blocked("unknown initial quota blocks GPU allocation")
        if dec(observation.remaining_quota_hours) > dec(policy["initial_quota_hours"]):
            raise Blocked("quota increase/reset requires a new reviewed policy; no automatic refill")

    def reserve(self, run_id: str, plan: Plan, observation: Observation) -> dict[str, Any]:
        """Reserve all-in upper cost; this is not permission to launch.

        authorize_launch consumes the reservation with fresh observations. A
        denied/unused reservation remains charged. This API never starts compute.
        """
        text_id(run_id, "run")
        totals = plan.totals()
        def admit(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
            policy, runs = self._state(rows)
            if not policy["allocation_enabled"] or not policy["account_ready"]:
                raise Blocked("GPU allocation disabled or account readiness unverified")
            self._observation(observation, policy)
            now = self._now()
            timeout = totals["wall_seconds_upper"]
            if now >= DEADLINE or now + timeout > DEADLINE:
                raise Blocked("all-in run timeout must fit the absolute deadline")
            if run_id in runs or any(not item["platform_stopped"] for item in runs.values()):
                raise Blocked("duplicate run or unresolved platform reservation")
            if any(item["outcome"] is not None and (
                dec(item["outcome"]["observed_wall_seconds"]) > dec(item["totals"]["wall_seconds_upper"])
                or dec(item["outcome"]["observed_device_hours"]) > dec(item["totals"]["device_hours_upper"])
            ) for item in runs.values()):
                raise Blocked("a prior run exceeded its upper cost; new review required")
            reserved = sum((dec(item["totals"]["estimated_quota_hours_upper"]) for item in runs.values()), Decimal(0))
            observed_use = dec(policy["initial_quota_hours"]) - dec(observation.remaining_quota_hours)
            upper = dec(totals["estimated_quota_hours_upper"])
            # Estimates stay charged permanently for this campaign, including failures.
            if max(reserved, observed_use) + upper > dec(policy["spendable_hours"]):
                raise Blocked("all-in upper quota cost exceeds remaining reserved budget")
            # Subtract prior estimates again here conservatively: provider charges may lag.
            if dec(observation.remaining_quota_hours) - reserved - upper < 20:
                raise Blocked("account quota floor of twenty hours would be breached")
            permit = {"run_id": run_id, "authorized_at": now, "stop_by": now + timeout,
                      "absolute_deadline": DEADLINE, "timeout_seconds": timeout, "totals": totals,
                      "ledger_id": policy["ledger_id"], "status": "RESERVATION_ONLY_NOT_EXECUTED"}
            event = {**permit, "kind": "reserved", "plan": asdict(plan), "observation": asdict(observation)}
            return permit, event
        return self._transaction(admit)

    def authorize_launch(self, run_id: str, observation: Observation) -> dict[str, Any]:
        """Consume once immediately before launch; caller must enforce stop_by.

        The utility cannot control provider scheduling, other clients, clocks or
        process termination. Delayed launches require new review, not token reuse.
        """
        def authorize(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
            policy, runs = self._state(rows)
            self._observation(observation, policy)
            if run_id not in runs:
                raise Blocked("unknown reservation")
            run = runs[run_id]
            if run["launch"] is not None or run["outcome"] is not None or run["platform_stopped"]:
                raise Blocked("launch authorization already consumed or run completed")
            now = self._now()
            if min(observation.quota_observed_at, observation.sessions_observed_at) < run["authorized_at"]:
                raise Blocked("launch needs observations made after reservation")
            if now + run["timeout_seconds"] > DEADLINE or now >= DEADLINE:
                raise Blocked("launch timeout no longer fits the absolute deadline")
            reserved = sum((dec(item["totals"]["estimated_quota_hours_upper"]) for item in runs.values()), Decimal(0))
            observed_use = dec(policy["initial_quota_hours"]) - dec(observation.remaining_quota_hours)
            if max(reserved, observed_use + dec(run["totals"]["estimated_quota_hours_upper"])) > dec(policy["spendable_hours"]):
                raise Blocked("launch quota observations no longer fit the budget")
            if dec(observation.remaining_quota_hours) - reserved < 20:
                raise Blocked("launch would breach account floor")
            permit = {"run_id": run_id, "authorized_at": now,
                      "stop_by": now + run["timeout_seconds"], "timeout_seconds": run["timeout_seconds"],
                      "status": "LAUNCH_AUTHORIZED_NOT_EXECUTED", "ledger_id": policy["ledger_id"]}
            return permit, {**permit, "kind": "launch_authorized", "observation": asdict(observation)}
        return self._transaction(authorize)

    def assert_within_timeout(self, run_id: str) -> None:
        def check(rows: list[dict[str, Any]]) -> tuple[None, None]:
            _, runs = self._state(rows)
            if run_id not in runs:
                raise Blocked("unknown reservation")
            run = runs[run_id]
            if run["launch"] is None:
                raise Blocked("reservation is not a launch authorization")
            if run["outcome"] is not None or run["platform_stopped"] or self._now() >= min(run["launch"]["stop_by"], DEADLINE):
                raise StopRequired("stop work and separately confirm the platform session has stopped")
            return None, None
        self._transaction(check)

    def record_outcome(self, run_id: str, *, outcome: str, observed_wall_seconds: float,
                       observed_device_hours: float, evidence_id: str) -> None:
        if outcome not in ("success", "failed", "interrupted", "timeout", "not_started"):
            raise Blocked("explicit terminal outcome required")
        payload = {"kind": "outcome", "run_id": run_id, "outcome": outcome,
                   "observed_wall_seconds": number(observed_wall_seconds, "observed wall"),
                   "observed_device_hours": number(observed_device_hours, "observed device"),
                   "evidence_id": text_id(evidence_id, "outcome evidence")}
        def finish(rows: list[dict[str, Any]]) -> tuple[None, dict[str, Any]]:
            _, runs = self._state(rows)
            if run_id not in runs or runs[run_id]["outcome"] is not None:
                raise Blocked("unknown or already completed reservation")
            if runs[run_id]["launch"] is None and outcome != "not_started":
                raise Blocked("launch authorization required for an executed outcome")
            return None, payload
        self._transaction(finish)

    def confirm_platform_stopped(self, run_id: str, observation: Observation) -> None:
        """Account-wide provider observation, never inferred from local process exit."""
        def confirm(rows: list[dict[str, Any]]) -> tuple[None, dict[str, Any]]:
            policy, runs = self._state(rows)
            self._observation(observation, policy)
            if run_id not in runs or runs[run_id]["platform_stopped"] or runs[run_id]["outcome"] is None:
                raise Blocked("completed unresolved reservation required")
            run = runs[run_id]
            stop_by = run["launch"]["stop_by"] if run["launch"] is not None else run["stop_by"]
            if min(observation.quota_observed_at, observation.sessions_observed_at) < stop_by:
                raise Blocked("platform-stop observations must follow the entire reserved interval")
            return None, {"kind": "platform_stopped", "run_id": run_id, "observation": asdict(observation)}
        self._transaction(confirm)
