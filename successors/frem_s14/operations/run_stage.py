"""Run one POSIX stage with streaming evidence and a hard process-group deadline.

This standalone stdlib tool is deliberately outside the frozen 0.1.3 wheel and
notebook. It is suitable for a prospectively qualified launcher; it never uploads
or starts Kaggle jobs and does not change any research budget or acceptance gate.
"""

import argparse
import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _publish(path, value):
    """Publish complete JSON without replacing existing evidence."""
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def _kill_group(process):
    """Only terminate the new session created by this invocation."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _positive_seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def run_stage(
    command, output, max_seconds, *, deadline=None, cwd=None, echo=True, heartbeat_seconds=10.0
):
    """Execute argv (never a shell), preserving every byte and a terminal receipt.

    ``deadline`` is an optional absolute time.monotonic() deadline from the same
    host/boot. It can only shorten max_seconds. No grace period extends the work
    budget. Cleanup/evidence hashing after termination is reported separately.
    A stage needs a fresh output directory, including after failure or timeout.
    """
    if os.name != "posix":
        raise RuntimeError("POSIX process groups are required")
    if (
        not isinstance(command, (list, tuple))
        or not command
        or not command[0]
        or any(not isinstance(arg, str) or "\x00" in arg for arg in command)
    ):
        raise ValueError("command must be a nonempty argv sequence of strings")
    _positive_seconds(max_seconds, "max_seconds")
    _positive_seconds(heartbeat_seconds, "heartbeat_seconds")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    if type(echo) is not bool:
        raise ValueError("echo must be boolean")
    started = time.monotonic()
    expires = (
        min(started + max_seconds, deadline) if deadline is not None else started + max_seconds
    )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    start_record = {
        "schema": 1,
        "command": list(command),
        "cwd": str(Path(cwd or os.getcwd()).resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": started,
        "deadline_monotonic": expires,
        "requested_seconds": max_seconds,
        "effective_seconds": max(0.0, expires - started),
        "output": str(root.resolve()),
    }
    _publish(root / "started.json", start_record)
    process = None
    selector = selectors.DefaultSelector()
    status, error, code = "LAUNCH_FAILED", None, None
    peak_bytes, observed_exit, console_error = 0, None, None
    log_path = root / "stage.log"
    with log_path.open("xb") as log, (root / "events.jsonl").open("x", encoding="utf-8") as events:

        def display(text):
            nonlocal console_error
            if echo and console_error is None:
                try:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                except (OSError, ValueError) as exc:
                    console_error = repr(exc)

        def event(kind, **fields):
            record = {"event": kind, "elapsed_seconds": time.monotonic() - started, **fields}
            line = json.dumps(record, sort_keys=True, allow_nan=False)
            events.write(line + "\n")
            events.flush()
            os.fsync(events.fileno())
            display(line + "\n")

        def consume(chunk):
            nonlocal peak_bytes
            log.write(chunk)
            log.flush()
            peak_bytes += len(chunk)
            # Display replacement characters only; the retained log is exact bytes.
            display(chunk.decode("utf-8", errors="replace"))

        try:
            event("STARTED", effective_seconds=start_record["effective_seconds"])
            if time.monotonic() >= expires:
                status = "NOT_STARTED_BUDGET_EXHAUSTED"
            else:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                event("PROCESS_STARTED", pid=process.pid)
                os.set_blocking(process.stdout.fileno(), False)
                selector.register(process.stdout, selectors.EVENT_READ)
                heartbeat = time.monotonic() + heartbeat_seconds
                while True:
                    now = time.monotonic()
                    code = process.poll()
                    if now >= expires:
                        status = "TIMED_OUT"
                        event("DEADLINE_EXHAUSTED")
                        _kill_group(process)
                        break
                    if code is not None:
                        observed_exit = now
                        status = "COMPLETE" if code == 0 else "FAILED"
                        # A successful leader must not leave background workers running.
                        _kill_group(process)
                        break
                    ready = selector.select(min(0.1, expires - now, max(0, heartbeat - now)))
                    for key, _ in ready:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if chunk:
                            consume(chunk)
                        else:
                            selector.unregister(key.fileobj)
                    if time.monotonic() >= heartbeat:
                        event("RUNNING", log_bytes=peak_bytes)
                        heartbeat = time.monotonic() + heartbeat_seconds
        except (KeyboardInterrupt, SystemExit) as exc:
            status, error = "INTERRUPTED", type(exc).__name__
        except Exception as exc:
            status, error = "LAUNCH_FAILED" if process is None else "RUNNER_FAILED", repr(exc)
        finally:
            if process is not None:
                _kill_group(process)
                code = process.wait()
                # Drain only currently buffered bytes; a descendant that escaped the
                # group cannot hold this read open indefinitely.
                while True:
                    try:
                        chunk = os.read(process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    consume(chunk)
                process.stdout.close()
            selector.close()
            log.flush()
            os.fsync(log.fileno())
            finished = time.monotonic()
            event("FINISHED", status=status, returncode=code, log_bytes=peak_bytes)
    digest = hashlib.sha256()
    with log_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(512 * 1024), b""):
            digest.update(chunk)
    receipt = {
        **start_record,
        "status": status,
        "returncode": code,
        "error": error,
        "process_exit_observed_monotonic": observed_exit,
        "termination_finished_monotonic": finished,
        "termination_wall_seconds": finished - started,
        "receipt_wall_seconds": time.monotonic() - started,
        "log_bytes": peak_bytes,
        "log_sha256": digest.hexdigest(),
        "console_error": console_error,
        "scientific_qualification": False,
    }
    _publish(root / "finished.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--deadline-monotonic", type=float)
    parser.add_argument("--cwd")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run_stage(
        command,
        args.output,
        args.max_seconds,
        deadline=args.deadline_monotonic,
        cwd=args.cwd,
        echo=not args.quiet,
    )
    return (
        0
        if result["status"] == "COMPLETE"
        else 124
        if "BUDGET" in result["status"] or result["status"] == "TIMED_OUT"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
