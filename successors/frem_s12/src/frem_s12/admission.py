"""Fail-closed source, data, and environment admission receipts."""

from __future__ import annotations

import math
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from frem_s12.acquisition import declared_sources
from frem_s12.data import ReturnPanel, parse_archive
from frem_s12.errors import ContractError
from frem_s12.evidence import package_identity, read_json, receipt, sha256_file

SUPPORTED_RUNTIMES: dict[tuple[int, int], str] = {
    (3, 12): "2.5.4",
    (3, 13): "2.1.3",
}
QUALIFICATION_REPEATS = 3


def _runtime_supported(python_version: tuple[int, int], numpy_version: str) -> bool:
    """Return whether the exact interpreter/NumPy pair is prospectively admitted."""
    return SUPPORTED_RUNTIMES.get(python_version) == numpy_version


def _unescape_mount_path(value: str) -> str:
    """Decode the octal escapes used in Linux mountinfo paths."""
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value


def _read_only_mount(path: Path) -> tuple[bool, str]:
    """Inspect the filesystem mount contract without attempting a write."""
    if not path.is_dir():
        return False, "path_missing"
    try:
        if os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1):
            return True, "statvfs_read_only"
    except OSError as exc:
        return False, f"statvfs_failed:{type(exc).__name__}"
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return False, "no_read_only_flag_and_mountinfo_unavailable"
    resolved = path.resolve()
    matches: list[tuple[int, set[str], str]] = []
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return False, f"mountinfo_failed:{type(exc).__name__}"
    for line in lines:
        before, separator, _ = line.partition(" - ")
        fields = before.split()
        if not separator or len(fields) < 6:
            continue
        mount_path = Path(_unescape_mount_path(fields[4])).resolve()
        if resolved == mount_path or resolved.is_relative_to(mount_path):
            matches.append((len(mount_path.parts), set(fields[5].split(",")), str(mount_path)))
    if not matches:
        return False, "mount_not_found"
    _, options, matched_mount = max(matches, key=lambda value: value[0])
    return "ro" in options, f"mountinfo:{matched_mount}:{','.join(sorted(options))}"


def admit_archive(archive: Path, acquisition_receipt: Path) -> tuple[ReturnPanel, dict[str, object]]:
    """Validate an acquired archive against its immutable adjacent receipt."""
    evidence = read_json(acquisition_receipt)
    required = {"family", "release", "url", "final_url", "bytes", "sha256"}
    if not required.issubset(evidence):
        raise ContractError("acquisition receipt is incomplete")
    family = str(evidence["family"])
    release = int(evidence["release"])
    expected_url = declared_sources().get((family, release))
    if expected_url is None or evidence["url"] != expected_url or evidence["final_url"] != expected_url:
        raise ContractError("acquisition receipt violates the source allowlist")
    if sha256_file(archive) != evidence["sha256"] or archive.stat().st_size != int(evidence["bytes"]):
        raise ContractError("archive identity differs from the acquisition receipt")
    cutoff = 201512
    panel = parse_archive(archive.read_bytes(), family=family, cutoff=cutoff)
    result = receipt(
        "data-admission",
        admitted=True,
        family=family,
        release=release,
        cutoff=cutoff,
        first_month=int(panel.months[0]),
        last_month=int(panel.months[-1]),
        rows=len(panel.months),
        dimensions=panel.returns.shape[1],
        archive_sha256=evidence["sha256"],
        acquisition_receipt_sha256=sha256_file(acquisition_receipt),
    )
    return panel, result


def _gpu_observation() -> dict[str, object]:
    """Observe NVIDIA availability without importing an optional GPU stack."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "devices": []}
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"available": completed.returncode == 0 and bool(devices), "devices": devices}


def qualify_environment(*, benchmark_size: int = 384) -> dict[str, object]:
    """Run a bounded local engineering probe that cannot admit science."""
    if benchmark_size < 64 or benchmark_size > 2048:
        raise ContractError("qualification benchmark size is outside bounds")
    rng = np.random.default_rng(12012)
    left = rng.normal(size=(benchmark_size, benchmark_size))
    right = rng.normal(size=(benchmark_size, benchmark_size))
    started = time.perf_counter()
    product = left @ right
    seconds = time.perf_counter() - started
    checksum = float(product[:8, :8].sum())
    repeated = float((left @ right)[:8, :8].sum())
    deterministic = bool(checksum == repeated and np.isfinite(product).all())
    identity = package_identity()
    python_version = sys.version_info[:2]
    expected_numpy = SUPPORTED_RUNTIMES.get(python_version)
    python_ok = expected_numpy is not None
    numpy_ok = expected_numpy == np.__version__
    runtime_ok = _runtime_supported(python_version, np.__version__)
    result = receipt(
        "environment-qualification",
        qualified=runtime_ok and deterministic,
        scope="local_engineering",
        scientific_admitted=False,
        python_ok=python_ok,
        numpy_ok=numpy_ok,
        deterministic=deterministic,
        benchmark={
            "matrix_size": benchmark_size,
            "seconds": seconds,
            "checksum": checksum,
            "repeat_checksum": repeated,
        },
        gpu=_gpu_observation(),
        runtime={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor_count": os.cpu_count(),
        },
        package=identity,
    )
    return result


def _egress_blocked() -> tuple[bool, str]:
    """Require independent IP, DNS and hostname TCP probes all to fail."""
    observations: list[str] = []
    blocked = True
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53), ("example.com", 443)):
        try:
            with socket.create_connection((host, port), timeout=2):
                observations.append(f"{host}:{port}:succeeded")
                blocked = False
        except OSError as exc:
            observations.append(f"{host}:{port}:failed:{type(exc).__name__}")
    return blocked, ";".join(observations)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def qualify_kaggle_environment(*, kaggle_root: Path = Path("/kaggle")) -> dict[str, object]:
    """Measure repeated representative cells under the Kaggle-native contract.

    Admission uses the slowest of three identical generated measurements.  This
    deliberately trades a small qualification cost for protection against an
    unusually fast warmup or transient scheduler interval admitting a campaign
    whose actual shards exceed the frozen wall-time ceiling.
    """
    working = kaggle_root / "working"
    input_root = kaggle_root / "input"
    detected = working.is_dir() and input_root.is_dir()
    python_version = sys.version_info[:2]
    expected_numpy = SUPPORTED_RUNTIMES.get(python_version)
    python_ok = expected_numpy is not None
    numpy_ok = expected_numpy == np.__version__
    runtime_ok = _runtime_supported(python_version, np.__version__)
    input_read_only, input_mount_observation = _read_only_mount(input_root)
    egress_blocked, egress_observation = _egress_blocked() if detected else (False, "not_observed")
    free_bytes = shutil.disk_usage(working).free if working.exists() else 0
    benchmark: dict[str, object] = {"executed": False}
    projected_cpu_hours = float("inf")
    projected_shard_hours = float("inf")
    projected_shard_bytes = 2**63 - 1
    if detected and runtime_ok and input_read_only and egress_blocked:
        from frem_s12.artifacts import write_fold
        from frem_s12.campaign import FOLDS
        from frem_s12.experiment import run_fold
        from frem_s12.generated import generated_panel
        from frem_s12.search import config_for_trial, run_search_cell, write_search_result

        rss_before = _peak_rss_bytes()
        fixed_seconds_samples: list[float] = []
        search_seconds_samples: list[float] = []
        fixed_artifact_bytes_samples: list[int] = []
        search_artifact_bytes_samples: list[int] = []
        with tempfile.TemporaryDirectory(prefix="frem-s12-probe-", dir=working) as temporary:
            panel = generated_panel(months=432, seed=811)
            search_panel = ReturnPanel("size_bm", panel.months, panel.returns)
            for repeat in range(QUALIFICATION_REPEATS):
                fixed_started = time.perf_counter()
                fixed_result = run_fold(
                    panel,
                    FOLDS[-1],
                    seed=11,
                    config=config_for_trial(23),
                )
                fixed_output = Path(temporary) / f"fixed-{repeat}"
                write_fold(
                    fixed_output,
                    fixed_result,
                    provenance={"qualification_probe": True, "repeat": repeat},
                )
                fixed_seconds_samples.append(time.perf_counter() - fixed_started)
                fixed_artifact_bytes_samples.append(
                    sum(path.stat().st_size for path in fixed_output.rglob("*") if path.is_file())
                )

                search_started = time.perf_counter()
                search_result = run_search_cell(search_panel, trial=23, seed=11)
                search_output = Path(temporary) / f"search-{repeat}"
                write_search_result(
                    search_output,
                    search_result,
                    provenance={"qualification_probe": True, "repeat": repeat},
                )
                search_seconds_samples.append(time.perf_counter() - search_started)
                search_artifact_bytes_samples.append(
                    sum(path.stat().st_size for path in search_output.rglob("*") if path.is_file())
                )
        peak_rss = _peak_rss_bytes()
        fixed_seconds = max(fixed_seconds_samples)
        search_seconds = max(search_seconds_samples)
        fixed_artifact_bytes = max(fixed_artifact_bytes_samples)
        search_artifact_bytes = max(search_artifact_bytes_samples)
        artifact_bytes_stable = bool(
            len(set(fixed_artifact_bytes_samples)) == 1 and len(set(search_artifact_bytes_samples)) == 1
        )
        search_cells = 144
        fixed_cells = 120
        shard_cells = int(np.ceil((search_cells + fixed_cells) / 8))
        projected_cpu_hours = (search_seconds * search_cells + fixed_seconds * fixed_cells) / 3600
        projected_shard_hours = max(search_seconds, fixed_seconds) * shard_cells / 3600
        projected_shard_bytes = max(search_artifact_bytes, fixed_artifact_bytes) * shard_cells
        benchmark = {
            "executed": True,
            "worst_declared_trial": 23,
            "latest_fixed_fold": "H",
            "repeats": QUALIFICATION_REPEATS,
            "timing_statistic": "maximum",
            "generated_search_seconds_samples": search_seconds_samples,
            "generated_fixed_seconds_samples": fixed_seconds_samples,
            "generated_search_seconds": search_seconds,
            "generated_fixed_seconds": fixed_seconds,
            "search_artifact_bytes_samples": search_artifact_bytes_samples,
            "fixed_artifact_bytes_samples": fixed_artifact_bytes_samples,
            "artifact_bytes_stable": artifact_bytes_stable,
            "search_artifact_bytes": search_artifact_bytes,
            "fixed_artifact_bytes": fixed_artifact_bytes,
            "peak_rss_bytes": peak_rss,
            "incremental_peak_rss_bytes": max(0, peak_rss - rss_before),
            "search_cells": search_cells,
            "fixed_cells": fixed_cells,
            "projected_cpu_hours": projected_cpu_hours,
            "projected_eight_shard_hours": projected_shard_hours,
            "projected_largest_shard_bytes": projected_shard_bytes,
        }
    projection_ok = bool(
        projected_cpu_hours <= 120
        and projected_shard_hours <= 8.5
        and projected_shard_bytes <= 15 * 2**30
        and free_bytes >= int(projected_shard_bytes * 1.2)
        and benchmark.get("artifact_bytes_stable") is True
    )
    admitted = bool(
        detected
        and runtime_ok
        and input_read_only
        and egress_blocked
        and benchmark["executed"] is True
        and projection_ok
    )
    return receipt(
        "environment-qualification",
        qualified=admitted,
        scope="kaggle_scientific",
        scientific_admitted=admitted,
        kaggle_detected=detected,
        python_ok=python_ok,
        numpy_ok=numpy_ok,
        runtime_ok=runtime_ok,
        expected_numpy=expected_numpy,
        input_read_only=input_read_only,
        input_mount_observation=input_mount_observation,
        network_disabled_observed=egress_blocked,
        network_observation=egress_observation,
        projection_ok=projection_ok,
        benchmark=benchmark,
        limits={
            "total_cpu_hours": 120,
            "active_shard_hours": 8.5,
            "retained_shard_bytes": 15 * 2**30,
            "working_free_bytes": free_bytes,
        },
        gpu={"admitted": False, "reason": "cpu_reference_only_no_equivalence_or_speedup_receipt"},
        runtime={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor_count": os.cpu_count(),
        },
        package=package_identity(),
    )


def _validate_kaggle_header(environment: dict[str, object], expected_source_identity: str) -> None:
    """Validate admission flags and the exact runtime pair."""
    if (
        environment.get("schema") != "frem-s12-environment-qualification-v1"
        or environment.get("qualified") is not True
        or environment.get("scientific_admitted") is not True
        or environment.get("scope") != "kaggle_scientific"
        or environment.get("source_identity") != expected_source_identity
        or environment.get("kaggle_detected") is not True
        or environment.get("runtime_ok") is not True
        or environment.get("python_ok") is not True
        or environment.get("numpy_ok") is not True
        or environment.get("input_read_only") is not True
        or environment.get("network_disabled_observed") is not True
        or environment.get("projection_ok") is not True
    ):
        raise ContractError("Kaggle environment receipt does not satisfy admission booleans")
    runtime = environment.get("runtime")
    if not isinstance(runtime, dict):
        raise ContractError("Kaggle environment runtime receipt is invalid")
    python_text = runtime.get("python")
    numpy_text = runtime.get("numpy")
    if not isinstance(python_text, str) or not isinstance(numpy_text, str):
        raise ContractError("Kaggle environment runtime versions are invalid")
    try:
        python_pair = tuple(int(value) for value in python_text.split(".")[:2])
    except ValueError as exc:
        raise ContractError("Kaggle environment Python version is invalid") from exc
    if len(python_pair) != 2 or not _runtime_supported(python_pair, numpy_text):
        raise ContractError("Kaggle environment runtime pair is outside the allowlist")
    if environment.get("expected_numpy") != numpy_text:
        raise ContractError("Kaggle environment expected NumPy version is inconsistent")


def _validated_timing_maxima(benchmark: dict[str, object]) -> dict[str, float]:
    """Reconstruct maximum timings from the three retained samples."""
    if (
        benchmark.get("executed") is not True
        or benchmark.get("worst_declared_trial") != 23
        or benchmark.get("latest_fixed_fold") != "H"
        or benchmark.get("repeats") != QUALIFICATION_REPEATS
        or benchmark.get("timing_statistic") != "maximum"
        or benchmark.get("artifact_bytes_stable") is not True
        or benchmark.get("search_cells") != 144
        or benchmark.get("fixed_cells") != 120
    ):
        raise ContractError("Kaggle environment benchmark contract is inconsistent")
    timing_pairs = (
        ("generated_search_seconds_samples", "generated_search_seconds"),
        ("generated_fixed_seconds_samples", "generated_fixed_seconds"),
    )
    timings: dict[str, float] = {}
    for samples_name, maximum_name in timing_pairs:
        samples = benchmark.get(samples_name)
        maximum = benchmark.get(maximum_name)
        if (
            not isinstance(samples, list)
            or len(samples) != QUALIFICATION_REPEATS
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in samples)
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in samples)
            or isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or float(maximum) != max(float(value) for value in samples)
        ):
            raise ContractError("Kaggle environment timing samples do not reconstruct")
        timings[maximum_name] = float(maximum)
    return timings


def _validated_artifact_maxima(benchmark: dict[str, object]) -> dict[str, int]:
    """Reconstruct stable artifact byte maxima from retained samples."""
    byte_pairs = (
        ("search_artifact_bytes_samples", "search_artifact_bytes"),
        ("fixed_artifact_bytes_samples", "fixed_artifact_bytes"),
    )
    byte_maxima: dict[str, int] = {}
    for samples_name, maximum_name in byte_pairs:
        samples = benchmark.get(samples_name)
        maximum = benchmark.get(maximum_name)
        if (
            not isinstance(samples, list)
            or len(samples) != QUALIFICATION_REPEATS
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in samples)
            or len(set(samples)) != 1
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum != max(samples)
        ):
            raise ContractError("Kaggle environment artifact sizes do not reconstruct")
        byte_maxima[maximum_name] = maximum
    return byte_maxima


def _validate_kaggle_projections(
    benchmark: dict[str, object],
    limits: dict[str, object],
    timings: dict[str, float],
    byte_maxima: dict[str, int],
) -> None:
    """Recompute campaign projections and enforce every frozen ceiling."""
    working_free_bytes = limits.get("working_free_bytes")
    projected_cpu = (
        timings["generated_search_seconds"] * 144 + timings["generated_fixed_seconds"] * 120
    ) / 3600
    projected_shard = max(timings.values()) * 33 / 3600
    projected_bytes = max(byte_maxima.values()) * 33
    projections = {
        "projected_cpu_hours": projected_cpu,
        "projected_eight_shard_hours": projected_shard,
        "projected_largest_shard_bytes": projected_bytes,
    }
    for name, expected in projections.items():
        observed = benchmark.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ContractError(f"Kaggle environment {name} does not reconstruct")
    if (
        limits.get("total_cpu_hours") != 120
        or limits.get("active_shard_hours") != 8.5
        or limits.get("retained_shard_bytes") != 15 * 2**30
        or isinstance(working_free_bytes, bool)
        or not isinstance(working_free_bytes, int)
        or working_free_bytes < int(projected_bytes * 1.2)
        or projected_cpu > 120
        or projected_shard > 8.5
        or projected_bytes > 15 * 2**30
    ):
        raise ContractError("Kaggle environment projections exceed or misstate frozen limits")


def validate_kaggle_environment_receipt(
    environment: dict[str, object], *, expected_source_identity: str
) -> None:
    """Recompute the internal consistency of a claimed Kaggle admission receipt."""
    _validate_kaggle_header(environment, expected_source_identity)
    benchmark = environment.get("benchmark")
    limits = environment.get("limits")
    if not isinstance(benchmark, dict) or not isinstance(limits, dict):
        raise ContractError("Kaggle environment benchmark or limits are invalid")
    if (
        benchmark.get("executed") is not True
        or benchmark.get("worst_declared_trial") != 23
        or benchmark.get("latest_fixed_fold") != "H"
        or benchmark.get("repeats") != QUALIFICATION_REPEATS
        or benchmark.get("timing_statistic") != "maximum"
        or benchmark.get("artifact_bytes_stable") is not True
        or benchmark.get("search_cells") != 144
        or benchmark.get("fixed_cells") != 120
    ):
        raise ContractError("Kaggle environment benchmark contract is inconsistent")
    timings = _validated_timing_maxima(benchmark)
    byte_maxima = _validated_artifact_maxima(benchmark)
    _validate_kaggle_projections(benchmark, limits, timings, byte_maxima)
    gpu = environment.get("gpu")
    if not isinstance(gpu, dict) or gpu.get("admitted") is not False:
        raise ContractError("Kaggle environment receipt does not preserve the CPU reference contract")


def require_execution_receipts(
    *,
    environment_receipt: Path,
    data_receipt: Path,
    acquisition_receipt: Path,
    archive: Path,
    family: str,
) -> dict[str, object]:
    """Bind a scientific cell to matching source, environment, and data evidence."""
    environment = read_json(environment_receipt)
    data = read_json(data_receipt)
    identity = package_identity()["sha256"]
    if not isinstance(identity, str):
        raise ContractError("package source identity is invalid")
    validate_kaggle_environment_receipt(environment, expected_source_identity=identity)
    acquisition = read_json(acquisition_receipt)
    required_acquisition = {"family", "release", "url", "final_url", "bytes", "sha256"}
    if not required_acquisition.issubset(acquisition):
        raise ContractError("acquisition receipt is incomplete")
    release = int(acquisition["release"])
    expected_url = declared_sources().get((family, release))
    if (
        acquisition.get("family") != family
        or expected_url is None
        or acquisition.get("url") != expected_url
        or acquisition.get("final_url") != expected_url
        or acquisition.get("sha256") != sha256_file(archive)
        or int(acquisition.get("bytes", -1)) != archive.stat().st_size
    ):
        raise ContractError("acquisition receipt does not bind the requested archive")
    if data.get("admitted") is not True or data.get("source_identity") != identity:
        raise ContractError("data receipt is unadmitted or source-stale")
    if data.get("family") != family or data.get("archive_sha256") != sha256_file(archive):
        raise ContractError("data receipt does not bind the requested archive and family")
    if data.get("acquisition_receipt_sha256") != sha256_file(acquisition_receipt):
        raise ContractError("data receipt does not bind the acquisition receipt")
    return {
        "environment_receipt_sha256": sha256_file(environment_receipt),
        "data_receipt_sha256": sha256_file(data_receipt),
        "acquisition_receipt_sha256": sha256_file(acquisition_receipt),
        "archive_sha256": sha256_file(archive),
        "source_identity": identity,
    }
