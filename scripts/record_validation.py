#!/usr/bin/env python3
"""Run one bounded child argv command; retain logs and never overwrite receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or not args.name.replace("_", "").isalnum():
        parser.error("command, positive timeout and alphanumeric receipt name required")
    out = args.output_dir / args.name
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(args.source_dir.resolve()), OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONHASHSEED="0")
    started_utc = datetime.now(timezone.utc).isoformat()
    started, timed_out = time.perf_counter(), False
    with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=args.cwd, env=env, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            status = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            status = process.wait()
        except OSError as exc:
            status = 127
            stderr.write(f"Child launch failed: {type(exc).__name__}: {exc}\n".encode())
    elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = dict(command=command, cwd=str(args.cwd.resolve()),
                   source_directory=str(args.source_dir.resolve()),
                   started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
                   exit_status=status, timed_out=timed_out, wall_seconds=elapsed,
                   timeout_seconds=args.timeout, child_user_seconds=usage.ru_utime,
                   child_system_seconds=usage.ru_stime,
                   max_rss_kib=usage.ru_maxrss / 1024 if sys.platform == "darwin" else usage.ru_maxrss,
                   platform=platform.platform(), python=sys.version,
                   recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   threads={key: env[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                   logs_sha256={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                for name in ("stderr.log", "stdout.log")})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print((out / "stdout.log").read_text(errors="replace")[-3500:])
    if status:
        print((out / "stderr.log").read_text(errors="replace")[-2500:])
    return 124 if timed_out else int(status != 0)


if __name__ == "__main__":
    raise SystemExit(main())
