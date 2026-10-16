"""Simulated Linux kernel/cgroup fixtures; no Linux execution evidence is claimed."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from frem_s10.errors import ResourceLimitError
from frem_s10.identities import JSONValue, object_json
from frem_s10.runtime import resources
from frem_s10.runtime.resources import (
    CgroupMemoryLimit,
    LinuxSample,
    MemorySample,
    assess,
    assess_linux,
    audit_memory_rows,
    memory_observation,
    pressure_counter,
    sample_linux_memory,
)

MIB = 2**20


def fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    proc, mount = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (mount / "parent/worker").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 4194304 kB\nMemAvailable: 2097152 kB\n")
    (proc / "vmstat").write_text("pswpin 0\npswpout 17\n")
    (proc / "self/cgroup").write_text("0::/parent/worker\n")
    (proc / "self/mountinfo").write_text(f"30 22 0:28 / {mount} rw - cgroup2 cgroup rw\n")
    (mount / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    for relative, maximum, current in (("parent", 1024, 256), ("parent/worker", 768, 128)):
        (mount / relative / "memory.max").write_text(f"{maximum * MIB}\n")
        (mount / relative / "memory.current").write_text(f"{current * MIB}\n")
    return proc, mount


def samples() -> tuple[LinuxSample, ...]:
    return tuple(
        LinuxSample(
            f"2026-10-16T12:00:0{i}+00:00",
            1024 * 1024,
            17,
            "/parent/worker",
            "/sys/fs/cgroup",
            (
                CgroupMemoryLimit("/parent/worker", 1024 * MIB, 512 * MIB),
                CgroupMemoryLimit("/parent", 2048 * MIB, 100 * MIB),
            ),
        )
        for i in range(3)
    )


def test_fixture_reads_host_swap_and_every_ancestor_limit(tmp_path: Path) -> None:
    proc, mount = fixture_tree(tmp_path)
    observed = sample_linux_memory(proc)
    assert observed.mem_available_kib == 2097152
    assert observed.pswpout_pages == 17
    assert observed.available_mib == 640
    assert tuple(item.path for item in observed.cgroup_limits) == ("/parent/worker", "/parent")
    (mount / "parent/memory.current").write_text(f"{900 * MIB}\n")
    assert sample_linux_memory(proc).available_mib == 124
    (mount / "parent/memory.current").write_text(f"{1200 * MIB}\n")
    assert sample_linux_memory(proc).available_mib == 0


def test_unlimited_constraints_and_real_root_use_host_available(tmp_path: Path) -> None:
    proc, mount = fixture_tree(tmp_path)
    for relative in ("parent", "parent/worker"):
        (mount / relative / "memory.max").write_text("max\n")
    observed = sample_linux_memory(proc)
    assert observed.available_mib == 2048
    assert all(item.memory_max_bytes is None for item in observed.cgroup_limits)
    (proc / "self/cgroup").write_text("0::/\n")
    root = sample_linux_memory(proc)
    assert root.cgroup_limits == ()
    assert root.available_mib == 2048


@pytest.mark.parametrize(
    "defect",
    (
        "missing",
        "malformed",
        "negative",
        "overflow",
        "symlink",
        "v1",
        "hybrid",
        "partial_mount",
        "duplicate_mount",
        "hidden_ancestor",
        "missing_controller",
        "traversal",
        "duplicate_mem",
        "missing_swap",
    ),
)
def test_fixture_fails_closed_on_missing_or_ambiguous_resources(tmp_path: Path, defect: str) -> None:
    proc, mount = fixture_tree(tmp_path)
    current = mount / "parent/memory.current"
    if defect in ("missing", "symlink"):
        current.unlink()
        if defect == "symlink":
            current.symlink_to(mount / "parent/worker/memory.current")
    elif defect in ("malformed", "negative", "overflow"):
        current.write_text({"malformed": "512 MiB", "negative": "-1", "overflow": str(2**64)}[defect])
    elif defect in ("v1", "hybrid", "traversal"):
        (proc / "self/cgroup").write_text(
            {
                "v1": "2:memory:/worker\n",
                "hybrid": "0::/worker\n2:memory:/worker\n",
                "traversal": "0::/../worker\n",
            }[defect]
        )
    elif defect == "partial_mount":
        (proc / "self/mountinfo").write_text(f"30 22 0:28 /parent {mount} rw - cgroup2 cgroup rw\n")
    elif defect == "duplicate_mount":
        path = proc / "self/mountinfo"
        path.write_text(path.read_text() * 2)
    elif defect == "hidden_ancestor":
        (mount / "memory.max").write_text("max\n")
        (mount / "memory.current").write_text("0\n")
    elif defect == "missing_controller":
        (mount / "cgroup.controllers").write_text("cpu io pids\n")
    elif defect == "duplicate_mem":
        (proc / "meminfo").write_text("MemAvailable: 1048576 kB\nMemAvailable: 1048576 kB\n")
    else:
        (proc / "vmstat").write_text("pswpin 17\n")
    with pytest.raises(ResourceLimitError):
        sample_linux_memory(proc)


def test_linux_boundary_assessment_keeps_true_counter_names() -> None:
    original = samples()
    assert assess_linux(original)["status"] == "PASS"
    row = memory_observation(original[0])
    assert row["platform"] == "linux"
    assert row["available_mib"] == 512
    assert not {"pageouts", "free_pages", "speculative_pages", "page_size_bytes"}.intersection(row)
    assert pressure_counter(original[0]) == 17
    low = replace(original[1], mem_available_kib=512 * 1024 - 1)
    assert assess_linux((original[0], low, original[2]))["status"] == "BLOCKED"


@pytest.mark.parametrize("defect", ("swap_growth", "swap_reset", "time", "maximum", "mount"))
def test_linux_stability_is_required(defect: str) -> None:
    first, second, third = samples()
    if defect.startswith("swap"):
        second = replace(second, pswpout_pages=18 if defect == "swap_growth" else 16)
    elif defect == "time":
        second = replace(second, timestamp=first.timestamp)
    elif defect == "maximum":
        second = replace(
            second,
            cgroup_limits=(
                replace(second.cgroup_limits[0], memory_max_bytes=2048 * MIB),
                second.cgroup_limits[1],
            ),
        )
    else:
        second = replace(second, cgroup_mount="/another/mount")
    assert assess_linux((first, second, third))["status"] == "BLOCKED"


def test_receipt_audit_reconstructs_both_platforms_and_preserves_mac_assessment() -> None:
    linux: list[JSONValue] = [memory_observation(sample) for sample in samples()]
    assert audit_memory_rows(linux) == 17
    mac = tuple(MemorySample(f"2026-10-16T12:00:0{i}+00:00", 16384, 32768, 0, 9) for i in range(3))
    assessment = assess(mac)
    assert assessment["status"] == "PASS"
    assert "platform" not in assessment
    assert assessment["samples"] == [memory_observation(sample) for sample in mac]
    assert audit_memory_rows([memory_observation(sample) for sample in mac]) == 9


@pytest.mark.parametrize("defect", ("headroom", "missing_ancestor", "boolean_counter", "clock", "mixed"))
def test_receipt_rejects_tampered_raw_proof(defect: str) -> None:
    rows: list[JSONValue] = deepcopy([memory_observation(sample) for sample in samples()])
    second = object_json(rows[1])
    if defect == "headroom":
        second["available_mib"] = 1024.0
    elif defect == "missing_ancestor":
        limits = second["cgroup_limits"]
        assert isinstance(limits, list)
        second["cgroup_limits"] = limits[:1]
    elif defect == "boolean_counter":
        second["pswpout_pages"] = True
    elif defect == "clock":
        second["timestamp"] = "2026-10-16T12:00:00.5+00:00"
    else:
        second = memory_observation(MemorySample("2026-10-16T12:00:01+00:00", 16384, 32768, 0, 17))
    rows[1] = second
    with pytest.raises(ResourceLimitError):
        audit_memory_rows(rows)


def test_check_host_linux_dispatch_is_three_spaced_fixture_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    iterator = iter(samples())
    sleeps: list[float] = []
    monkeypatch.setattr("frem_s10.runtime.resources.sys.platform", "linux")
    monkeypatch.setattr(resources, "sample_linux_memory", lambda: next(iterator))
    monkeypatch.setattr("frem_s10.runtime.resources.time.sleep", sleeps.append)
    assert resources.check_host()["status"] == "PASS"
    assert sleeps == [1, 1]
    with pytest.raises(ResourceLimitError, match="unverified"):
        resources.sample_memory()
