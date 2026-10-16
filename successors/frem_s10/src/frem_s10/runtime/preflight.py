"""Portable metadata observations, separate from the fixed production admission.

This module never imports numerical libraries, executes a model, installs a
dependency, or authorizes scientific work. Linux kernel observations are not an
implementation of the existing macOS production resource or sandbox contract.
"""

import importlib.metadata
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from frem_s10.errors import ResourceLimitError
from frem_s10.identities import JSONObject, JSONValue
from frem_s10.metadata import parse_utc_timestamp

_MAX_PROC_BYTES = 65_536


@dataclass(frozen=True)
class LinuxResourceSample:
    """Keep Linux counters in their actual units without macOS relabeling."""

    timestamp: str
    mem_available_kib: int
    pswpout_pages: int

    def observation(self) -> JSONObject:
        """Return descriptive evidence, never a passing resource admission."""
        return {
            "status": "OBSERVED_NOT_ADMISSION",
            "timestamp": self.timestamp,
            "mem_available_kib": self.mem_available_kib,
            "mem_available_mib": self.mem_available_kib / 1024,
            "pswpout_pages": self.pswpout_pages,
            "memory_semantics": "Linux MemAvailable estimate; includes reclaimable memory",
            "swap_semantics": "Linux pswpout cumulative pages swapped out; not macOS Pageouts",
            "container_limit": "NOT_OBSERVED; host /proc values may exceed cgroup limits",
            "production_admission": False,
            "sampling_limit": "Single observation; no swap-counter stability or execution monitoring",
        }


def _counter(text: str, field: str, expression: str) -> int:
    if len(text.encode("utf-8")) > _MAX_PROC_BYTES:
        raise ResourceLimitError("Linux resource input exceeds the observation bound")
    rows = [line for line in text.splitlines() if line.split(":", 1)[0].split()[:1] == [field]]
    if len(rows) != 1:
        raise ResourceLimitError(f"Linux resource counter missing or duplicated: {field}")
    match = re.fullmatch(expression, rows[0])
    if match is None:
        raise ResourceLimitError(f"Linux resource counter malformed: {field}")
    return int(match.group(1))


def parse_linux_resources(meminfo: str, vmstat: str, timestamp: str) -> LinuxResourceSample:
    """Parse one strict MemAvailable/pswpout observation with an explicit clock."""
    parse_utc_timestamp(timestamp)
    available = _counter(meminfo, "MemAvailable", r"MemAvailable:\s+([0-9]+)\s+kB\s*")
    swapped = _counter(vmstat, "pswpout", r"pswpout\s+([0-9]+)\s*")
    return LinuxResourceSample(timestamp, available, swapped)


def _read_proc(path: Path) -> str:
    with path.open("rb") as stream:
        raw = stream.read(_MAX_PROC_BYTES + 1)
    if len(raw) > _MAX_PROC_BYTES:
        raise ResourceLimitError("Linux resource file exceeds the observation bound")
    return raw.decode("ascii")


def sample_linux_resources() -> LinuxResourceSample:
    """Read bounded kernel metadata on Linux without changing production guards."""
    if platform.system() != "Linux":
        raise ResourceLimitError("Linux resource observation requires Linux")
    try:
        return parse_linux_resources(
            _read_proc(Path("/proc/meminfo")),
            _read_proc(Path("/proc/vmstat")),
            datetime.now(UTC).isoformat(),
        )
    except (OSError, UnicodeError) as exc:
        raise ResourceLimitError("Linux resource observation unavailable") from exc


def _distribution(name: str) -> JSONObject:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {"status": "NOT_INSTALLED", "version": None}
    except (OSError, ValueError):
        return {"status": "METADATA_UNAVAILABLE", "version": None}
    return {"status": "METADATA_OBSERVED", "version": version, "runtime_import_tested": False}


def _production(system: str, machine: str) -> JSONObject:
    reasons: list[JSONValue] = [
        "EXACT_RESOLVED_ENVIRONMENT_LOCK_NOT_VERIFIED",
        "FRESH_PRODUCTION_RESOURCE_ADMISSION_NOT_RUN",
        "CURRENT_SOURCE_VERIFICATION_AND_OPERATION_DECLARATION_REQUIRED",
    ]
    supported = system == "Darwin" and machine == "arm64" and sys.version_info[:2] == (3, 12)
    if system == "Linux":
        reasons.extend(
            (
                "LINUX_ADAPTERS_REQUIRE_ACTUAL_PLATFORM_VERIFICATION",
                "LINUX_PRODUCTION_MEMORY_SUPERVISOR_NOT_IMPLEMENTED",
                "EXACT_LINUX_PRODUCTION_ENVIRONMENT_NOT_IMPLEMENTED",
            )
        )
    elif not supported:
        reasons.append("PRODUCTION_REQUIRES_DARWIN_ARM64_PYTHON_3_12")
    return {
        "status": "BLOCKED",
        "platform_matches_supported_contract": supported,
        "required_environment": "Darwin arm64, Python 3.12, complete exact resolved macOS lock",
        "reasons": reasons,
        "production_admission": False,
    }


def _resources(system: str) -> JSONObject:
    if system != "Linux":
        return {
            "status": "NOT_SAMPLED",
            "production_admission": False,
            "instruction": "Use the existing guarded production host-admission command",
        }
    try:
        return sample_linux_resources().observation()
    except ResourceLimitError:
        return {"status": "UNAVAILABLE", "production_admission": False}


def preflight(device: str = "cpu") -> JSONObject:
    """Observe CPU-smoke prerequisites and report unsupported devices explicitly.

    A metadata-compatible result still requires the actual bounded smoke;
    larger development also requires resource admission. S10 uses NumPy float64 on CPU;
    installed PyTorch or a visible NVIDIA tool cannot establish a CUDA backend.
    """
    system, machine = platform.system(), platform.machine()
    numpy = _distribution("numpy")
    reasons: list[JSONValue] = []
    if device != "cpu":
        reasons.append("UNSUPPORTED_DEVICE_BACKEND: S10 implements CPU NumPy only; no fallback")
    if sys.version_info[:2] != (3, 12):
        reasons.append("PYTHON_3_12_REQUIRED")
    if numpy.get("version") != "2.5.4":
        reasons.append("NUMPY_2_5_4_METADATA_REQUIRED")
    return {
        "schema": "s10-portable-preflight-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "BLOCKED" if reasons else "READY_FOR_CPU_SMOKE_METADATA_ONLY",
        "reasons": reasons,
        "requested_device": device,
        "effective_device": "cpu" if device == "cpu" else None,
        "supported_devices": ["cpu"],
        "precision": "float64",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": system,
        "machine": machine,
        "kernel_release": platform.release(),
        "numpy": numpy,
        "pytorch": _distribution("torch") | {"required_for_s10": False},
        "cuda": {
            "backend_implemented": False,
            "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
            "nvidia_smi_executed": False,
            "cuda_runtime_version": None,
            "runtime_version_status": "NOT_OBSERVED_WITHOUT_NUMERICAL_IMPORT",
            "gpu_test": "NOT_RUN",
            "gpu_pass": False,
        },
        "cpu_smoke": {
            "metadata_compatible": not reasons,
            "runtime_test": "NOT_RUN",
            "resource_admission": "NOT_PERFORMED",
        },
        "resources": _resources(system),
        "production": _production(system, machine),
        "data": {
            "status": "NOT_INSPECTED",
            "instruction": "Use admit and the exact operation declaration for data admission",
        },
        "scientific_execution": False,
        "dependencies_changed": False,
    }
