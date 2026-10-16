"""Metadata-only portability and honest device/resource-boundary checks."""

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from frem_s10.errors import ResourceLimitError
from frem_s10.identities import JSONObject, object_json
from frem_s10.runtime import preflight as module

CLOCK = "2026-10-16T00:00:00+00:00"


@pytest.fixture
def observed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sys, "version_info", (3, 12, 14))
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def metadata(name: str) -> JSONObject:
        return {"status": "METADATA_OBSERVED", "version": "2.5.4" if name == "numpy" else "2.10.0"}

    monkeypatch.setattr(module, "_distribution", metadata)


def test_cpu_metadata_compatibility_does_not_admit_production(observed: None) -> None:
    result = module.preflight()
    assert result["status"] == "READY_FOR_CPU_SMOKE_METADATA_ONLY"
    assert result["precision"] == "float64"
    assert result["effective_device"] == "cpu"
    assert object_json(result["cpu_smoke"])["runtime_test"] == "NOT_RUN"
    assert object_json(result["production"])["status"] == "BLOCKED"
    assert object_json(result["production"])["production_admission"] is False
    assert object_json(result["cuda"])["gpu_pass"] is False
    assert result["scientific_execution"] is False
    assert result["dependencies_changed"] is False


@pytest.mark.parametrize("device", ("cuda", "cuda:0", "mps", "auto", "CPU", ""))
def test_unsupported_device_never_falls_back(observed: None, device: str) -> None:
    result = module.preflight(device)
    assert result["status"] == "BLOCKED"
    assert result["effective_device"] is None
    assert "UNSUPPORTED_DEVICE_BACKEND" in str(result["reasons"])


def test_visible_nvidia_tool_cannot_claim_gpu_execution(
    observed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    result = module.preflight("cuda")
    cuda = object_json(result["cuda"])
    assert cuda["nvidia_smi_available"] is True
    assert cuda["nvidia_smi_executed"] is False
    assert cuda["gpu_test"] == "NOT_RUN" and cuda["gpu_pass"] is False
    assert result["status"] == "BLOCKED"


@pytest.mark.parametrize("version", (None, "2.5.3", "2.6.0"))
def test_missing_or_changed_numpy_metadata_blocks_smoke(
    observed: None, monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    monkeypatch.setattr(module, "_distribution", lambda name: {"version": version})
    result = module.preflight()
    assert result["status"] == "BLOCKED"
    assert "NUMPY_2_5_4_METADATA_REQUIRED" in str(result["reasons"])


def test_python_mismatch_cannot_pass_metadata_check(observed: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 13, 0))
    result = module.preflight()
    assert result["status"] == "BLOCKED"
    assert "PYTHON_3_12_REQUIRED" in str(result["reasons"])


def test_linux_observation_retains_actual_counter_meaning() -> None:
    sample = module.parse_linux_resources("MemAvailable: 1048576 kB\n", "pswpout 15\n", CLOCK)
    observed = sample.observation()
    assert observed["mem_available_mib"] == 1024
    assert observed["pswpout_pages"] == 15
    assert observed["production_admission"] is False
    assert "cgroup" in str(observed["container_limit"])
    assert "not macOS" in str(observed["swap_semantics"])
    assert "pageouts" not in observed


@pytest.mark.parametrize(
    ("meminfo", "vmstat"),
    (
        ("MemFree: 1000000 kB\n", "pswpout 0\n"),
        ("MemAvailable: 10 kB\nMemAvailable: 20 kB\n", "pswpout 0\n"),
        ("MemAvailable: 10 kB\nMemAvailable: invalid kB\n", "pswpout 0\n"),
        ("MemAvailable: -1 kB\n", "pswpout 0\n"),
        ("MemAvailable: 10 MB\n", "pswpout 0\n"),
        ("MemAvailable: 10 kB\n", "pswpin 0\n"),
        ("MemAvailable: 10 kB\n", "pswpout -1\n"),
        ("MemAvailable: 10 kB\n", "pswpout 1\npswpout invalid\n"),
        ("MemAvailable: 10 kB\n", "pswpout 1.0\n"),
        ("MemAvailable: 10 kB\n" + " " * 65536, "pswpout 0\n"),
    ),
)
def test_missing_duplicate_malformed_or_large_linux_counter_fails(meminfo: str, vmstat: str) -> None:
    with pytest.raises(ResourceLimitError):
        module.parse_linux_resources(meminfo, vmstat, CLOCK)


def test_linux_observation_never_implements_production_gate(
    observed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        module, "sample_linux_resources", lambda: module.LinuxResourceSample(CLOCK, 16 * 1024**2, 0)
    )
    result = module.preflight()
    assert result["status"] == "READY_FOR_CPU_SMOKE_METADATA_ONLY"
    production = object_json(result["production"])
    assert production["status"] == "BLOCKED"
    assert "LINUX_PRODUCTION_MEMORY_SUPERVISOR_NOT_IMPLEMENTED" in str(production["reasons"])
    assert "LINUX_ADAPTERS_REQUIRE_ACTUAL_PLATFORM_VERIFICATION" in str(production["reasons"])
    assert object_json(result["resources"])["production_admission"] is False


def test_linux_resource_read_failure_is_unavailable(observed: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    def unavailable() -> module.LinuxResourceSample:
        raise ResourceLimitError("unavailable")

    monkeypatch.setattr(module, "sample_linux_resources", unavailable)
    assert object_json(module.preflight()["resources"])["status"] == "UNAVAILABLE"


def test_preflight_cannot_import_numerical_packages_or_spawn_a_process() -> None:
    source = Path(module.__file__).resolve().parents[2]
    code = """import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class DenyNumerical(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','torch','pandas','scipy','sklearn'}:
            raise AssertionError('numerical import in metadata preflight')
sys.meta_path.insert(0, DenyNumerical())
def audit(event, args):
    if event in {'subprocess.Popen','os.system','os.posix_spawn'}:
        raise AssertionError('process launch in metadata preflight')
sys.addaudithook(audit)
from frem_s10.runtime.preflight import preflight
result = preflight('cuda')
assert result['status'] == 'BLOCKED'
assert result['effective_device'] is None
assert not ({'numpy','torch','pandas','scipy','sklearn'} & sys.modules.keys())
print(json.dumps({'status': 'PASS', 'scope': 'metadata_import_boundary'}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(source)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["scope"] == "metadata_import_boundary"
