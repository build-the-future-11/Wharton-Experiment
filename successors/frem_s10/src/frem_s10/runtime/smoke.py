"""Portable bounded engineering execution using the real S10 forecasting kernel.

This is a generated-data exercise of the maintained implementation. It cannot
open market sources, consume a scientific declaration, or resume a market run.
"""

import math
import os
import signal
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from frem_s10.errors import ContractError, IntegrityError, ResourceLimitError
from frem_s10.identities import JSONObject, checked_json, digest_json, object_json, parse_json
from frem_s10.metadata import finite_float, strict_int
from frem_s10.runtime.artifacts import _plain_directory, publish_json
from frem_s10.runtime.journal import inspect_journal
from frem_s10.runtime.provenance import THREAD_ENV
from frem_s10.runtime.smoke_checkpoint import (
    commits,
    identity_document,
    publish_status,
    read_plain,
    run_lock,
    save_checkpoint,
    seal_run,
    verify_binding,
)
from frem_s10.runtime.smoke_contract import read_manifest, runtime_identity


class ActiveBudget:
    """Cumulative active wall time across sessions, with an injectable test clock."""

    def __init__(self, limit: float, prior: float, clock: Callable[[], float] = time.monotonic) -> None:
        """Retain prior elapsed work so restarting cannot replenish a run budget."""
        self.limit = finite_float(limit, name="budget", strictly_positive=True)
        self.prior = finite_float(prior, name="prior elapsed", minimum=0)
        self.clock, self.started = clock, clock()
        if not math.isfinite(self.started):
            raise ResourceLimitError("budget clock is unavailable")
        self.last = self.started

    @property
    def elapsed(self) -> float:
        """Return active execution time including this session's preparation."""
        now = self.clock()
        if not math.isfinite(now) or now < self.last:
            raise ResourceLimitError("budget clock regressed or became unavailable")
        self.last = now
        return self.prior + now - self.started

    def check(self) -> None:
        """Stop cooperatively between atomic origins without resetting the budget."""
        if self.elapsed >= self.limit:
            raise ResourceLimitError("generated execution cumulative time budget exhausted")


def _threads() -> None:
    for key, expected in THREAD_ENV.items():
        if key in os.environ and os.environ[key] != expected:
            raise ContractError(f"{key} must equal 1 before importing NumPy")
        if "numpy" in sys.modules and key not in os.environ:
            raise ContractError("thread controls must be set before numerical imports")
        os.environ[key] = expected


@contextmanager
def _origin_transaction() -> Iterator[None]:
    """Deliver Ctrl-C only after the committed origin and metrics are both durable.

    POSIX signal masking does not hide write failures or make SIGKILL resumable.
    It closes the ordinary interrupt window between two durable publications.
    """
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _declaration(config: JSONObject, data_identity: str) -> JSONObject:
    return {
        "schema": "s10-generated-run-v1",
        "configuration": config,
        "runtime": runtime_identity(),
        "data_identity": data_identity,
        "scientific_status": "NOT_EVALUATED",
        "market_resume_allowed": False,
        "randomness": "private PCG64 seed; data regenerated and hash-checked; deterministic online kernel",
        "checkpoint_boundary": "every completed origin across all declared variants",
        "selection": "no epochs or validation checkpoint selection; latest causal online state only",
    }


def _latest(directory: Path, declaration: JSONObject) -> JSONObject | None:
    latest = None
    logged = _logged_metrics(directory)
    seen: set[int] = set()
    for checkpoint in commits(directory):
        verify_binding(checkpoint, declaration)
        index = strict_int(checkpoint["completed"], name="completed origin", minimum=1)
        if logged.get(index) != checkpoint["metrics"]:
            raise IntegrityError("resume requires matching durable metrics for every committed origin")
        seen.add(index)
        latest = checkpoint
    if set(logged) != seen:
        raise IntegrityError("resume found uncommitted logged origins")
    return latest


def _spent(directory: Path, declaration: JSONObject, checkpoint_elapsed: float) -> float:
    """Include all retained session charges, including work after the last commit."""
    prior = max(checkpoint_elapsed, _session_time(directory))
    for path in directory.glob("status-*.json"):
        status = read_plain(path)
        if status.get("binding") != digest_json(declaration):
            raise IntegrityError("status history does not match the generated run")
        prior = max(prior, finite_float(status["active_seconds"], name="session charge", minimum=0))
    return prior


def _session_time(directory: Path) -> float:
    total = 0.0
    for path in directory.glob("session-end-*.json"):
        row = read_plain(path)
        total += finite_float(row["active_seconds"], name="session duration", minimum=0)
    return total


def _check_sessions(directory: Path) -> None:
    for path in directory.glob("session-start-*.json"):
        end = directory / path.name.replace("session-start-", "session-end-", 1)
        if not end.is_file() or end.is_symlink():
            raise ResourceLimitError("abandoned session has unknown active time; budget review required")


def _restore(checkpoint: JSONObject, config: JSONObject, inputs: object) -> dict[str, object]:
    from frem_s10.model.checkpoint import load_state
    from frem_s10.model.config import ForecastConfig
    from frem_s10.runtime.smoke_data import GeneratedInputs

    assert isinstance(inputs, GeneratedInputs)
    variants = config["variants"]
    assert isinstance(variants, list)
    states, ids = object_json(checkpoint["states"]), object_json(checkpoint["state_ids"])
    if set(states) != set(variants) or set(ids) != set(variants):
        raise IntegrityError("checkpoint variant registry differs from the declared matrix")
    restored: dict[str, object] = {}
    for name in states:
        payload, identity = states[name], ids[name]
        if not isinstance(payload, str) or not isinstance(identity, str):
            raise IntegrityError("checkpoint state payload and identity required")
        restored[name] = load_state(
            payload,
            expected_state_id=identity,
            expected_transform_id=inputs.transform.record_id,
            expected_config=ForecastConfig().variant(name),
        )
    return restored


def _step(
    states: dict[str, object],
    inputs: object,
    completed: int,
) -> tuple[dict[str, object], JSONObject]:
    from frem_s10.math.scores import energy_score
    from frem_s10.model.engine import advance
    from frem_s10.model.records import ForecastState
    from frem_s10.runtime.smoke_data import START, GeneratedInputs, issue_request

    assert isinstance(inputs, GeneratedInputs)
    index = START + completed
    target = inputs.records[index]
    matured = inputs.records[:66] if completed == 0 else (inputs.records[index - 5],)
    next_states: dict[str, object] = {}
    rows: JSONObject = {}
    for name, state in states.items():
        assert isinstance(state, ForecastState)
        transition = advance(state, issue_request(target), matured)
        law = transition.batch.get("contextual_stack")
        # This score is reporting-only. It never enters advance or checkpoint selection.
        score = energy_score(
            law.distribution.r / inputs.transform.score_scale,
            law.probabilities,
            target.r / inputs.transform.score_scale,
        )
        next_states[name] = transition.state
        rows[name] = {
            "contextual_energy_score": score,
            "issued_id": law.record_id,
            "state_id": transition.state.record_id,
            "support": len(transition.state.library.record_ids),
            "matured_feedback": len(transition.matured_feedback),
            "global_fit": transition.batch.mixtures.global_certificate is not None,
            "local_fit": transition.batch.mixtures.local_certificate is not None,
        }
    return next_states, rows


def _checkpoint(
    states: dict[str, object],
    rows: JSONObject,
    declaration: JSONObject,
    completed: int,
    elapsed: float,
) -> JSONObject:
    from frem_s10.model.checkpoint import dump_state
    from frem_s10.model.records import ForecastState

    payloads: JSONObject = {}
    identities: JSONObject = {}
    for name, state in states.items():
        assert isinstance(state, ForecastState)
        payloads[name], identities[name] = dump_state(state), state.record_id
    return {
        "schema": "s10-generated-commit-v1",
        "binding": digest_json(declaration),
        "completed": completed,
        "elapsed_seconds": elapsed,
        "states": payloads,
        "state_ids": identities,
        "metrics": rows,
    }


def run_smoke(
    manifest: Path,
    output: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> JSONObject:
    """Run/resume the exact generated matrix with a new per-session log and safe commits."""
    config = read_manifest(manifest)
    if stop_after is not None:
        strict_int(stop_after, name="stop_after", minimum=1)
    if config["scope"] == "development-engineering":
        from frem_s10.runtime.resources import check_host

        if check_host()["status"] != "PASS":
            raise ResourceLimitError("larger generated development exercise requires host admission")
    _threads()
    _plain_directory(output.absolute().parent)
    if not resume:
        output.mkdir(exist_ok=False)
    with run_lock(output):
        _check_sessions(output)
        session = uuid.uuid4().hex
        publish_json(output, f"session-start-{session}.json", {"configuration_sha256": digest_json(config)})
        started = time.monotonic()
        try:
            return _run_owned(config, output, resume=resume, stop_after=stop_after)
        finally:
            publish_json(
                output,
                f"session-end-{session}.json",
                {
                    "configuration_sha256": digest_json(config),
                    "active_seconds": time.monotonic() - started,
                },
            )


def _run_owned(config: JSONObject, output: Path, *, resume: bool, stop_after: int | None) -> JSONObject:
    from frem_s10.model.config import ForecastConfig
    from frem_s10.model.engine import initial_state
    from frem_s10.runtime.progress import ProgressLog
    from frem_s10.runtime.smoke_data import generated_inputs

    session_started = time.monotonic()
    seed, steps = int(str(config["seed"])), int(str(config["steps"]))
    inputs = generated_inputs(seed, steps)
    declaration = _declaration(config, inputs.identity)
    if resume:
        if identity_document(output) != declaration:
            raise IntegrityError("resume requires identical source, environment, data and configuration")
    else:
        seal_run(output, declaration)
    checkpoint = _latest(output, declaration)
    completed = int(str(checkpoint["completed"])) if checkpoint else 0
    if completed > steps:
        raise IntegrityError("checkpoint exceeds declared run length")
    prior = (
        finite_float(checkpoint["elapsed_seconds"], name="saved elapsed", minimum=0) if checkpoint else 0.0
    )
    prior = _spent(output, declaration, prior)
    budget = ActiveBudget(float(str(config["budget_seconds"])), prior + time.monotonic() - session_started)
    budget.check()
    if completed == steps:
        evaluated = _evaluate_owned(output)
        return {**evaluated, "status": "COMPLETE_REUSED", "output": str(output), "validated": True}
    variants = config["variants"]
    assert isinstance(variants, list)
    states: dict[str, object] = (
        _restore(checkpoint, config, inputs)
        if checkpoint
        else {
            str(name): initial_state(inputs.transform, ForecastConfig().variant(str(name)))
            for name in variants
        }
    )
    log = ProgressLog(
        output / f"metrics-{uuid.uuid4().hex}.jsonl", "generated-" + digest_json(declaration)[:16]
    )
    start = completed
    status = "COMPLETE"
    try:
        while completed < steps:
            budget.check()
            states, rows = _step(states, inputs, completed)
            with _origin_transaction():
                name = save_checkpoint(
                    output, _checkpoint(states, rows, declaration, completed + 1, budget.elapsed)
                )
                completed += 1
                log.emit(
                    {
                        "phase": "generated-forecast",
                        "global_step": completed,
                        "seed": seed,
                        "variants": rows,
                        "checkpoint": name,
                        "active_seconds": budget.elapsed,
                        "origins_per_second": (completed - start)
                        / max(time.monotonic() - session_started, 1e-9),
                        "evidence_scope": "GENERATED_ENGINEERING",
                    }
                )
            publish_status(
                output,
                {
                    "status": "RUNNING",
                    "completed": completed,
                    "checkpoint": name,
                    "binding": digest_json(declaration),
                    "active_seconds": budget.elapsed,
                },
            )
            budget.check()
            if stop_after is not None and completed - start >= stop_after and completed < steps:
                status = "INTERRUPTED"
                break
    except KeyboardInterrupt:
        status = "INTERRUPTED"
    except ResourceLimitError:
        status = "BUDGET_EXCEEDED"
    except Exception as exc:
        publish_status(
            output,
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "completed": completed,
                "active_seconds": budget.elapsed,
                "binding": digest_json(declaration),
            },
        )
        raise
    result: JSONObject = {
        "status": status,
        "completed": completed,
        "planned": steps,
        "active_seconds": budget.elapsed,
        "output": str(output),
        "binding": digest_json(declaration),
        "scientific_status": "NOT_EVALUATED",
        "market_execution": False,
        "gpu_execution": False,
    }
    publish_status(output, result)
    log.emit(result)
    return result


def evaluate_smoke(directory: Path) -> JSONObject:
    """Recompute every saved contextual score from frozen states without fitting."""
    _threads()
    with run_lock(directory):
        _check_sessions(directory)
        return _evaluate_owned(directory)


def _evaluate_owned(directory: Path) -> JSONObject:
    from frem_s10.math.scores import energy_score
    from frem_s10.model.records import ForecastState
    from frem_s10.runtime.smoke_data import START, generated_inputs

    declaration = identity_document(directory)
    config = object_json(declaration["configuration"])
    inputs = generated_inputs(int(str(config["seed"])), int(str(config["steps"])))
    if declaration != _declaration(config, inputs.identity):
        raise IntegrityError("evaluation requires exact generated source/environment/data binding")
    totals: dict[str, float] = {}
    count = 0
    final_ids: JSONObject = {}
    logged = _logged_metrics(directory)
    for checkpoint in commits(directory):
        verify_binding(checkpoint, declaration)
        states = _restore(checkpoint, config, inputs)
        count += 1
        if logged.get(count) != checkpoint["metrics"]:
            raise IntegrityError("committed generated metrics have no matching durable log")
        target = inputs.records[START + count - 1]
        for name, state in states.items():
            assert isinstance(state, ForecastState)
            issued = state.pending[-1]
            if issued.request.record_id != target.record_id:
                raise IntegrityError("saved state does not match the completed origin")
            score = energy_score(
                issued.distribution.r / inputs.transform.score_scale,
                issued.probabilities,
                target.r / inputs.transform.score_scale,
            )
            row = object_json(object_json(checkpoint["metrics"])[name])
            if score != row["contextual_energy_score"] or row["state_id"] != state.record_id:
                raise IntegrityError("saved generated score or state does not reconstruct exactly")
            totals[name] = totals.get(name, 0.0) + score
            final_ids[name] = state.record_id
    if count == 0:
        raise IntegrityError("no committed generated origin to evaluate")
    if set(logged) != set(range(1, count + 1)):
        raise IntegrityError("durable generated log contains an uncommitted origin")
    charged = _spent(
        directory, declaration, finite_float(checkpoint["elapsed_seconds"], name="saved elapsed", minimum=0)
    )
    if charged >= float(str(config["budget_seconds"])):
        raise ResourceLimitError("generated exercise exceeded cumulative budget; no completion claim")
    return {
        "status": "COMPLETE" if count == config["steps"] else "PARTIAL",
        "completed": count,
        "planned": config["steps"],
        "state_ids": final_ids,
        "mean_contextual_energy_score": checked_json({name: total / count for name, total in totals.items()}),
        "evidence_scope": "GENERATED_ENGINEERING",
        "scientific_status": "NOT_EVALUATED",
        "score_reconstruction": "exact binary64 equality in the bound environment",
    }


def _logged_metrics(directory: Path) -> dict[int, JSONObject]:
    rows: dict[int, JSONObject] = {}
    for path in directory.glob("metrics-*.jsonl"):
        inspect_journal(path)
        with path.open() as stream:
            for line in stream:
                event = object_json(object_json(parse_json(line))["event"])
                if event.get("phase") != "generated-forecast":
                    continue
                index = strict_int(event["global_step"], name="logged step", minimum=1)
                if index in rows:
                    raise IntegrityError("duplicate generated origin in durable logs")
                rows[index] = object_json(event["variants"])
    return rows
