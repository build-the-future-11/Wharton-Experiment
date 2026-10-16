"""Bounded standard-library engineering tests; never imports an ML framework."""

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
import unittest
from pathlib import Path


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def identities():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("successors/laura_planning_v1").rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        start = time.monotonic()

        def peak():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                2**20 if platform.system() == "Darwin" else 1024)

        def guard(*_):
            if peak() > 128 or time.monotonic() - start > 60:
                raise RuntimeError("Planning integrity resource ceiling exceeded")

        signal.signal(signal.SIGALRM, guard)
        signal.setitimer(signal.ITIMER_REAL, .05, .05)
        try:
            suite = unittest.defaultTestLoader.discover("successors/laura_planning_v1/tests")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            guard()
            write(args.output / "test_counts.json", {"run": result.testsRun,
                                                     "failures": len(result.failures), "errors": len(result.errors),
                                                     "skipped": len(result.skipped)})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            write(args.output / "resources.json", {"peak_rss_mib": peak(), "seconds": time.monotonic() - start,
                                                   "rss_limit_mib": 128, "seconds_limit": 60,
                                                   "guard": "50ms sampled peak RSS and elapsed guard; 60s parent subprocess timeout"})
        raise SystemExit(0 if result.wasSuccessful() else 1)
    args.output.mkdir(parents=True, exist_ok=False)
    before = identities()
    write(args.output / "source_hashes.json", before)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    with (args.output / "tests.log").open("x", encoding="utf-8") as stream:
        try:
            completed = subprocess.run([sys.executable, "-m", "successors.laura_planning_v1.check",
                                        "--worker", "--output", str(args.output)],
                                       stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=60, check=False)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = {"returncode": code, "sources_unchanged": identities() == before,
               "seconds": time.monotonic() - started,
               "scope": "Exact arithmetic and validation engineering only; no financial performance claim"}
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt))
    raise SystemExit(code if receipt["sources_unchanged"] else 1)


if __name__ == "__main__":
    main()
