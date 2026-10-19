#!/usr/bin/env python3
"""Replay a pinned integrity checkout on CPU; this is not independent evidence.

Usage: python scripts/run_integrity_kaggle.py --source-sha FULL_SHA --output NEW_DIR [--matrix]
Dependencies must already be installed. This driver neither downloads nor submits
anything and does not infer a Kaggle account/job identity from its environment.
The optional matrix reuses the declared DEVELOPMENT seeds, never protected data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOTAL_SECONDS, COMMAND_SECONDS = 900, 180
MEMORY_RESERVE_BYTES, COMMAND_ALLOWANCE_BYTES = 1024**3, 512 * 1024**2
WORLDS = ("null", "linear_signal", "regime_shift", "heavy_tails")
SEEDS = (2026101911, 2026101912, 2026101913)
ABLATIONS = dict(zip((f"M{i:02d}" for i in range(1, 13)), (
    "no_regimes", "raw_return_labels", "legacy_single_snapshot", "no_episodic_memory",
    "linear_experts_only", "no_predictive_loss", "no_context", "no_edges",
    "fixed_group_weights", "no_context", "fully_invested", "no_memory")))
INTEGRITY_TESTS = (
    "test_split_boundaries.py", "test_mpc_policy_contract.py", "test_point_in_time.py",
    "test_development_demo.py", "test_optional_model_imports.py",
    "test_verified_accounting.py", "test_causal_portfolio.py")
MODEL_TESTS = tuple(f"test_m{i:02d}.py" for i in range(1, 13)) + (
    "test_research_regressions.py", "test_numerical_gradients.py")
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}

BOOTSTRAP = """import runpy, sys, torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
mode, target, *args = sys.argv[1:]
sys.argv = [target, *args]
if mode == 'module':
    runpy.run_module(target, run_name='__main__')
else:
    runpy.run_path(target, run_name='__main__')
"""
DEPENDENCIES = """import importlib, importlib.metadata, json, platform, sys
from pathlib import Path
versions, errors = {}, {}
for module, package in [('numpy','numpy'), ('scipy','scipy'), ('sklearn','scikit-learn'),
                        ('pandas','pandas'), ('torch','torch'), ('pytest','pytest')]:
    try:
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)
    except Exception as exc:
        errors[package] = f'{type(exc).__name__}: {exc}'
threads = None
if 'torch' in versions:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    threads = {'intraop': torch.get_num_threads(), 'interop': torch.get_num_interop_threads()}
result = {'python': sys.version, 'platform': platform.platform(), 'packages': versions,
          'import_errors': errors, 'device': 'cpu', 'torch_threads': threads}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))
"""
COMPARE_DEMO = """import hashlib, json, sys
from pathlib import Path
a, b, output = map(Path, sys.argv[1:])
ha, hb = [json.loads((p / 'hashes.json').read_text()) for p in (a, b)]
errors = []
if ha != hb:
    errors.append('Declared deterministic/source hashes differ between primary and replay')
for root, manifest in ((a, ha), (b, hb)):
    for name, expected in manifest['deterministic_artifacts_sha256'].items():
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            errors.append(f'Unsafe or missing declared artifact: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f'Artifact hash mismatch: {root.name}/{name}')
result = {'evidence_class': 'same_seed_execution_replay', 'verified': not errors,
          'deterministic_files': len(ha['deterministic_artifacts_sha256']), 'errors': errors}
output.write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))
"""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def python_command(target, *args, module=False):
    return [sys.executable, "-c", BOOTSTRAP, "module" if module else "script", str(target), *map(str, args)]


def memory_admission():
    """Read current host/cgroup-v2 headroom; never modify limits or other processes."""
    result = dict(required_bytes=MEMORY_RESERVE_BYTES + COMMAND_ALLOWANCE_BYTES,
                  reserve_bytes=MEMORY_RESERVE_BYTES, command_allowance_bytes=COMMAND_ALLOWANCE_BYTES,
                  host_available_bytes=None, available_bytes=None, cgroups=[], errors=[], warnings=[])
    try:
        host = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        result["host_available_bytes"] = int(host["MemAvailable"].split()[0]) * 1024
        if result["host_available_bytes"] < 0:
            raise ValueError("negative MemAvailable")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        result["errors"].append(f"Host memory unavailable: {type(exc).__name__}: {exc}")

    base = Path("/sys/fs/cgroup")
    directories = [base]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                current = (base / line.split(":", 2)[2].lstrip("/")).resolve()
                if current.is_relative_to(base) and (current / "memory.max").is_file():
                    directories = []
                    while current.is_relative_to(base):
                        directories.append(current)
                        if current == base:
                            break
                        current = current.parent
                break
    except (OSError, ValueError, RuntimeError) as exc:
        result["warnings"].append(f"Using cgroup mount root: {type(exc).__name__}: {exc}")

    for directory in directories:
        maximum = directory / "memory.max"
        if not maximum.is_file():
            continue  # No cgroup-v2 limit: host MemAvailable remains the admission bound.
        try:
            limit = maximum.read_text().strip()
            if limit == "max":
                continue
            limit, current = int(limit), int((directory / "memory.current").read_text())
            if min(limit, current) < 0:
                raise ValueError("negative cgroup limit/current")
            fields = ("inactive_file", "file_dirty", "file_writeback", "shmem")
            stats = {}
            try:
                stats = {key: int(value) for key, value in
                         (line.split() for line in (directory / "memory.stat").read_text().splitlines())}
            except (OSError, ValueError) as exc:
                result["warnings"].append(f"No reclaim allowance at {directory}: {exc}")
            reclaimable = max(0, stats["inactive_file"] - stats["file_dirty"]
                             - stats["file_writeback"] - stats["shmem"]) if all(k in stats for k in fields) else 0
            result["cgroups"].append(dict(path=str(directory), max_bytes=limit, current_bytes=current,
                reclaimable_bytes=reclaimable, available_bytes=limit - current + reclaimable,
                stat_bytes={key: stats.get(key) for key in fields}))
        except (OSError, ValueError) as exc:
            result["errors"].append(f"Cgroup memory unavailable at {directory}: {type(exc).__name__}: {exc}")
    if result["host_available_bytes"] is not None:
        result["available_bytes"] = min([result["host_available_bytes"]]
                                        + [group["available_bytes"] for group in result["cgroups"]])
    result["admitted"] = (not result["errors"] and result["available_bytes"] is not None
                          and result["available_bytes"] >= result["required_bytes"])
    return result


def run_command(name, command, output, env, deadline):
    """One process group at a time, including launch errors and timeout receipts."""
    directory = output / "commands" / name
    directory.mkdir(parents=True, exist_ok=False)
    memory = memory_admission()
    limit = min(COMMAND_SECONDS, math.floor(deadline - time.perf_counter()))
    receipt = dict(name=name, command=command, cwd=str(ROOT), timeout_seconds=max(0, limit),
                   started_utc=datetime.now(timezone.utc).isoformat(), status="not_started_budget",
                   exit_status=None, timed_out=False, wall_seconds=0.0, memory_admission=memory)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started, process, interrupted = time.perf_counter(), None, False
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        if limit > 0 and not memory["admitted"]:
            receipt["status"] = "not_started_memory"
            stderr.write(b"Memory admission denied; current bounds and required reserve are retained in receipt.json.\n")
        elif limit > 0:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                receipt["exit_status"] = process.wait(timeout=limit)
                receipt["status"] = "completed" if receipt["exit_status"] == 0 else "failed"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                interrupted = isinstance(exc, KeyboardInterrupt)
                receipt.update(status="interrupted" if interrupted else "timed_out", timed_out=not interrupted)
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    receipt["exit_status"] = process.wait()
            except OSError as exc:
                receipt.update(status="launch_failed", exit_status=127)
                stderr.write(f"{type(exc).__name__}: {exc}\n".encode())
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt.update(wall_seconds=time.perf_counter() - started,
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   child_user_seconds=after.ru_utime - before.ru_utime,
                   child_system_seconds=after.ru_stime - before.ru_stime,
                   cumulative_child_peak_rss_kib=after.ru_maxrss / 1024 if sys.platform == "darwin" else after.ru_maxrss,
                   logs_sha256={name: digest(directory / name) for name in ("stdout.log", "stderr.log")})
    write_json(directory / "receipt.json", receipt)
    print(f"{name}: {receipt['status']} ({receipt['wall_seconds']:.3f}s)", flush=True)
    return receipt


def build_plan(output, matrix):
    plan = [dict(name="driver_contract_tests", command=[sys.executable, "-B",
                 str(ROOT / "tests/test_kaggle_driver_contract.py"), "-v"])]
    for name, tests in (("integrity_tests", INTEGRITY_TESTS), ("model_tests", MODEL_TESTS)):
        argv = ["-q", "-p", "no:cacheprovider", "--disable-warnings",
                "--basetemp", str(output / "temporary" / name),
                "--junitxml", str(output / "commands" / name / "pytest.xml")]
        plan.append(dict(name=name, command=python_command("pytest", *argv,
                         *(f"tests/unit/{test}" for test in tests), module=True)))
    for name in ("demo_primary", "demo_replay"):
        plan.append(dict(name=name, command=python_command(ROOT / "scripts/run_development_demo.py",
                    "--output-dir", output / name)))
    plan.append(dict(name="demo_comparison", command=[sys.executable, "-c", COMPARE_DEMO,
                str(output / "demo_primary"), str(output / "demo_replay"), str(output / "demo_comparison.json")]))
    if matrix:
        for world in WORLDS:
            for seed in SEEDS:
                name = f"matrix_{world}_{seed}"
                artifacts = output / "matrix" / f"{world}_{seed}"
                cells = [dict(cell_id=f"{mid}__{world}__{seed}__{variant}", model=mid,
                              world=world, seed=seed, variant=variant)
                         for mid, ablation in ABLATIONS.items() for variant in ("full", ablation)]
                plan.append(dict(name=name, artifact_directory=str(artifacts), cells=cells,
                    command=python_command(ROOT / "scripts/run_portfolio_research.py", "--output", artifacts,
                        "--worlds", world, "--seeds", seed, "--models", *ABLATIONS, "--n-test", 20)))
                plan.append(dict(name=f"verify_{world}_{seed}",
                    command=python_command(ROOT / "scripts/verify_portfolio_artifacts.py", artifacts,
                        "--output", output / f"verified_{world}_{seed}.json")))
    return plan


def matrix_receipts(plan, receipts):
    cells, errors = [], []
    attempted = {r["name"]: r for r in receipts if not r["status"].startswith("not_started")}
    for step in plan:
        if "cells" not in step:
            continue
        terminal = {}
        path = Path(step["artifact_directory"]) / "cells.jsonl"
        if path.is_file() and not path.is_symlink():
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    row = json.loads(line)
                    if row["cell_id"] in terminal:
                        raise ValueError("duplicate cell receipt")
                    terminal[row["cell_id"]] = row
                except (ValueError, KeyError) as exc:
                    errors.append(f"{step['name']} line {line_number}: {exc}")
        expected = {cell["cell_id"] for cell in step["cells"]}
        if set(terminal) - expected:
            errors.append(f"{step['name']}: unexpected cell IDs")
        for cell in step["cells"]:
            status = terminal.get(cell["cell_id"], {}).get("status")
            cells.append(dict(cell, status=status or ("no_terminal_receipt" if step["name"] in attempted else "not_started"),
                              command=step["name"], receipt=terminal.get(cell["cell_id"])))
    return dict(cells=cells, parsing_errors=errors,
                completed=sum(c["status"] == "completed" for c in cells), expected=len(cells))


def bundle(output):
    """Hash regular files from this new output only; never follow symlinks or unpickle."""
    archive, checksum = output / "replay_bundle.zip", output / "replay_bundle.sha256"
    archive_partial = output / "replay_bundle.zip.partial"
    checksum_partial = output / "replay_bundle.sha256.partial"
    if any(path.exists() or path.is_symlink() for path in (archive, checksum, archive_partial, checksum_partial)):
        raise FileExistsError("Refusing to overwrite a published or partial replay bundle")
    excluded = {"artifact_manifest.json", archive.name, checksum.name,
                archive_partial.name, checksum_partial.name}
    files, skipped = [], []
    for path in sorted(output.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            skipped.append(str(path.relative_to(output)))
        elif path.is_file() and path.name not in excluded:
            files.append(path)
    manifest = dict(files={str(p.relative_to(output)): {"sha256": digest(p), "bytes": p.stat().st_size}
                           for p in files}, excluded_symlinks_or_external_paths=skipped,
                    pickle_policy="Only synthetic checkpoints created by this pinned replay; trusted-source only. Never load untrusted pickle files.")
    write_json(output / "artifact_manifest.json", manifest)
    with zipfile.ZipFile(archive_partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in files + [output / "artifact_manifest.json"]:
            zipped.write(path, arcname=str(path.relative_to(output)))
    checksum_partial.write_text(f"{digest(archive_partial)}  {archive.name}\n")
    # The normal ZIP filename is the completion marker; publish it only after its checksum.
    checksum_partial.rename(checksum)
    archive_partial.rename(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True, help="Exact 40-character published Git commit")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing paths are rejected")
    parser.add_argument("--matrix", action="store_true", help="Replay all 288 predeclared DEVELOPMENT cells if budget permits")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_sha):
        parser.error("--source-sha must be a full 40-character hexadecimal commit")
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("--output must not already exist")
    output = output.resolve()
    if any(output.is_relative_to(ROOT / name) for name in ("src", "scripts", "tests", "protocol", ".git")):
        parser.error("--output must be outside executable source and Git directories")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + TOTAL_SECONDS
    (output / "temporary").mkdir()
    env = os.environ.copy()
    env.update(THREAD_ENV, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
               CUDA_VISIBLE_DEVICES="", TMPDIR=str(output / "temporary"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline",
               GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    plan = build_plan(output, args.matrix)
    write_json(output / "plan.json", dict(evidence_class="execution_replay_not_independent_evidence",
        required_source_sha=args.source_sha.lower(), phase="DEVELOPMENT", matrix_requested=args.matrix,
        seeds=list(SEEDS) if args.matrix else [], n_test=20, maximum_matrix_cells=288 if args.matrix else 0,
        budget_seconds=TOTAL_SECONDS, command_limit_seconds=COMMAND_SECONDS, lanes=1, device="cpu",
        memory_reserve_bytes=MEMORY_RESERVE_BYTES, command_allowance_bytes=COMMAND_ALLOWANCE_BYTES,
        threads=THREAD_ENV, commands=plan, network_commands=False, protected_inputs=False,
        primary_demo="Fixed source-declared synthetic seed; replay is not a new replication"))
    write_json(output / "driver_environment.json", dict(python=sys.version, platform=platform.platform(),
        executable=sys.executable, driver_sha256=digest(Path(__file__)), required_source_sha=args.source_sha.lower()))
    (output / "README.txt").write_text(
        "Pinned-source CPU execution replay, not new independent or confirmatory evidence.\n"
        "No real-market data or protected evaluation inputs are used. No remote identity is asserted.\n"
        "Each command has logs and a receipt. On failure/timeout, later commands remain unstarted.\n"
        "All .pkl files here are own synthetic checkpoints from trusted pinned source only.\n"
        "Never load a pickle from an untrusted source. The bundle operation only hashes its bytes.\n")
    receipts, failure, observed_source_sha, source_verified = [], None, None, False

    def required(name, command):
        receipt = run_command(name, command, output, env, deadline)
        receipts.append(receipt)
        if receipt["status"] != "completed":
            raise RuntimeError(f"{name}: {receipt['status']}; remaining commands stopped")
        return (output / "commands" / name / "stdout.log").read_text().strip()

    try:
        observed_source_sha = required("git_head", ["git", "rev-parse", "--verify", "HEAD"])
        if observed_source_sha != args.source_sha.lower():
            raise RuntimeError(f"Source mismatch: required {args.source_sha.lower()}, observed {observed_source_sha}")
        top = required("git_root", ["git", "rev-parse", "--show-toplevel"])
        if Path(top).resolve() != ROOT:
            raise RuntimeError("Driver is not at the root of the pinned checkout")
        dirty = required("git_source_status", ["git", "status", "--porcelain=v1", "--untracked-files=all",
            "--", "src", "scripts", "tests", "protocol", "pyproject.toml", "requirements-verified.txt"])
        if dirty:
            raise RuntimeError("Executable source/configuration differs from pinned commit; see git_source_status log")
        source_paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts" / name for name in
            ("run_integrity_kaggle.py", "run_development_demo.py", "run_portfolio_research.py", "verify_portfolio_artifacts.py")]
        source_paths += [ROOT / "tests/unit" / name for name in INTEGRITY_TESTS + MODEL_TESTS]
        source_paths += [ROOT / "tests/test_kaggle_driver_contract.py"]
        source_paths += [ROOT / "protocol/PORTFOLIO_RESEARCH_V1.md", ROOT / "pyproject.toml"]
        if any(p.is_symlink() or not p.resolve().is_relative_to(ROOT) for p in source_paths):
            raise RuntimeError("Source snapshot contains a symlink or external path")
        write_json(output / "source_hashes.json", {str(p.relative_to(ROOT)): digest(p) for p in source_paths})
        source_verified = True
        required("dependencies", [sys.executable, "-c", DEPENDENCIES, str(output / "environment.json")])
        for step in plan:
            required(step["name"], step["command"])
    except (Exception, KeyboardInterrupt) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        (output / "driver_failure.txt").write_text(traceback.format_exc())
    finally:
        matrix = matrix_receipts(plan, receipts)
        write_json(output / "matrix_receipts.json", matrix)
        if matrix["parsing_errors"] or matrix["completed"] != matrix["expected"]:
            failure = failure or "Matrix has failed, unstarted, or nonterminal cells"
        finished = {receipt["name"] for receipt in receipts if receipt["status"] == "completed"}
        summary = dict(
            status="replay_completed" if failure is None else "replay_failed_or_partial", error=failure,
            evidence_class="execution_replay_not_independent_evidence", required_source_sha=args.source_sha.lower(),
            observed_source_sha=observed_source_sha, source_verified=source_verified,
            wall_seconds=time.perf_counter() - started, child_wall_seconds=sum(r["wall_seconds"] for r in receipts),
            command_receipts=receipts, commands_without_success=[s["name"] for s in plan if s["name"] not in finished],
            matrix_expected=matrix["expected"], matrix_completed=matrix["completed"],
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "execution_summary.json", summary)
        try:
            bundle(output)
        except Exception as exc:
            bundle_error = f"Bundle failure: {type(exc).__name__}: {exc}"
            failure = f"{failure}; {bundle_error}" if failure else bundle_error
            summary.update(status="replay_failed_or_partial", error=failure, bundle_error=bundle_error,
                           wall_seconds=time.perf_counter() - started,
                           finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "execution_summary.json", summary)
            (output / "bundle_failure.txt").write_text(traceback.format_exc())
    print(f"Replay artifacts: {output}", flush=True)
    if failure:
        print(failure, file=sys.stderr)
    return int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())
