"""Single-process bounded S6 numerical integrity checks, not market training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def source_identities():
    paths = [*Path("successors/frem_s6").glob("*.py"), *Path("tests/s6").glob("*.py"),
             Path("successors/frem_portfolio_v1/engine.py"), Path("scripts/public_equity_backtests.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        start = time.monotonic()

        def peak():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                2**20 if platform.system() == "Darwin" else 1024)

        def guard(*_):
            if peak() > 128 or time.monotonic() - start > 60:
                raise RuntimeError("S6 integrity resource ceiling exceeded")

        signal.signal(signal.SIGALRM, guard)
        signal.setitimer(signal.ITIMER_REAL, .05, .05)
        try:
            import pytest
            result = pytest.main(["tests/s6", "-q", "-W", "error", "-p", "no:cacheprovider"])
            guard()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            write(args.output / "resources.json", {"peak_rss_mib": peak(), "seconds": time.monotonic() - start,
                                                    "limit_rss_mib": 128, "limit_seconds": 60})
        raise SystemExit(result)
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_identities()
    write(args.output / "source_hashes.json", before)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    start = time.monotonic()
    with (args.output / "tests.log").open("x") as stream:
        try:
            result = subprocess.run([sys.executable, "-m", "successors.frem_s6.check",
                                     "--worker", "--output", str(args.output)], env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=60, check=False)
            status = {"returncode": result.returncode, "seconds": time.monotonic() - start,
                      "scope": "numerical integrity only; no market fits or performance campaign"}
        except subprocess.TimeoutExpired:
            status = {"returncode": None, "status": "TIMEOUT", "seconds": time.monotonic() - start}
    status["sources_unchanged"] = source_identities() == before
    if not status["sources_unchanged"]:
        status["returncode"] = 1
    write(args.output / "receipt.json", status)
    print(json.dumps(status))
    raise SystemExit(status["returncode"] if status["returncode"] is not None else 1)


if __name__ == "__main__":
    main()
