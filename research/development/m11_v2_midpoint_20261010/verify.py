"""Executed bounded regression driver; see the immutable local contract."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

STATE = 'EXECUTION_STATE.json'
TESTS = ['tests/unit/test_m11_v2_midpoint.py', 'tests/unit/test_m11_v2_objective_numerics.py', 'tests/unit/test_m11_v2.py', 'tests/unit/test_competition_stress.py']
EXCLUDE = 'not test_solver_cash_feasibility_and_liability_result and not test_solver_uses_declared_scenarios_to_reduce_shortfall and not test_actual_bounded_optimizer_still_improves_constructed_funding_gap'
PYTHONPATH = 'src'

RECORD = Path(__file__).resolve().parent
ROOT = next(p for p in RECORD.parents if (p / STATE).is_file())
phase = sys.argv[1]
if phase not in ("baseline", "final"):
    raise SystemExit("expected baseline or final")
receipt = RECORD / (phase + "_receipt.json")
if receipt.exists():
    raise SystemExit("refusing to overwrite a retained command receipt")
contract = json.loads((RECORD / "contract.json").read_text())
for path, digest in contract["frozen_sources_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
if phase == "baseline":
    for path, record in contract["incoming"].items():
        if path.startswith("src/"):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == record["sha256"], path
paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / p for p in TESTS]
paths += [Path(__file__).resolve(), RECORD / "contract.json"]
def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before = hashes()
args = [sys.executable, "-m", "pytest", "-q", "-W", "error", "-p", "no:cacheprovider", "-o", "addopts=", "--basetemp", str(RECORD / (phase + "_pytest_tmp"))]
args += [TESTS[0]] if phase == "baseline" else TESTS
if phase == "final" and EXCLUDE:
    args += ["-k", EXCLUDE]
env = dict(os.environ, PYTHONPATH=PYTHONPATH, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
started = time.perf_counter()
try:
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, timeout=30)
    code, stdout, stderr, timed_out = result.returncode, result.stdout, result.stderr, False
except subprocess.TimeoutExpired as error:
    code, stdout, stderr, timed_out = None, error.stdout or b"", error.stderr or b"", True
seconds = time.perf_counter() - started
(RECORD / (phase + ".stdout.log")).write_bytes(stdout)
(RECORD / (phase + ".stderr.log")).write_bytes(stderr)
archive = RECORD / (phase + "_pytest_artifacts.tar.gz")
with tarfile.open(archive, "w:gz") as handle:
    temporary = RECORD / (phase + "_pytest_tmp")
    if temporary.exists():
        handle.add(temporary, arcname=temporary.name)
record = {"phase": phase, "argv": args, "cwd": str(ROOT), "returncode": code, "timed_out": timed_out, "child_command_wall_seconds": seconds, "hard_timeout_seconds": 30, "source_sha256_before": before, "source_sha256_after": hashes(), "sources_unchanged_during_command": before == hashes(), "artifact_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "numeric_threads": 1, "optimizer_calls": 0, "training_steps": 0, "scope": "generated CPU engineering regressions; no scientific result"}
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"phase": phase, "returncode": code, "timed_out": timed_out, "seconds": seconds, "receipt": str(receipt)}))
print(stdout.decode(errors="replace")[-1800:])
print(stderr.decode(errors="replace")[-800:])
