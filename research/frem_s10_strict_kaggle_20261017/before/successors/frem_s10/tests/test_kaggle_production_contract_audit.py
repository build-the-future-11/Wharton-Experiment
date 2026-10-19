"""Generated fixtures for the explicit Kaggle production successor contract."""

import errno
import os
import platform
import socket
from pathlib import Path

import pytest

from frem_s10.errors import IntegrityError, ResourceLimitError
from frem_s10.runtime import kaggle_contract, packaging, resources, verification


class _Socket:
    def __init__(self, family: int, success: bool = False) -> None:
        self.family = family
        self.success = success

    def __enter__(self) -> "_Socket":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def settimeout(self, timeout: float) -> None:
        assert timeout == 1.0

    def connect(self, address: object) -> None:
        if self.success:
            return
        raise OSError(errno.ENETUNREACH, "generated offline fixture")


def _platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(kaggle_contract.ENVIRONMENT_KEY, kaggle_contract.CONTRACT)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    original_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: True if str(path) in {"/kaggle/input", "/kaggle/working"} else original_is_dir(path),
    )
    monkeypatch.setattr(
        Path, "exists", lambda path: False if str(path) == "/dev/nvidia0" else os.path.exists(path)
    )
    monkeypatch.setattr(Path, "glob", lambda path, pattern: iter(()))
    sealed = tmp_path / "sealed.zip"
    sealed.write_bytes(b"sealed")
    return sealed


def test_contract_is_opt_in_and_rejects_enabled_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(IntegrityError, match="marker"):
        kaggle_contract.require_platform()
    _platform(monkeypatch, tmp_path)
    monkeypatch.setattr(socket, "socket", lambda family, kind: _Socket(family, success=True))
    with pytest.raises(IntegrityError, match="enabled"):
        kaggle_contract.network_unreachable()


def test_kaggle_resource_receipts_require_reserve_and_stable_pressure() -> None:
    rows = tuple(
        resources.KaggleSample(
            f"2026-10-17T00:00:0{index}+00:00",
            8 * 1024**2,
            7,
            4 * 1024**3,
            4 * 1024**3,
            1024**3,
            (("high", 0), ("max", 0), ("oom", 0), ("oom_kill", 0)),
        )
        for index in range(3)
    )
    receipt = resources.assess_kaggle(rows)
    assert receipt["status"] == "PASS" and receipt["floor_mib"] == 1536
    assert resources.audit_memory_rows(receipt["samples"]) == 7
    damaged = list(receipt["samples"])
    damaged[2] = dict(damaged[2], memory_events=[["high", 1], ["max", 0], ["oom", 0], ["oom_kill", 0]])
    with pytest.raises(ResourceLimitError, match="constraint identity"):
        resources.audit_memory_rows(damaged)


def test_verification_uses_fixed_kaggle_boundary_without_bubblewrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = _platform(monkeypatch, tmp_path)
    monkeypatch.setenv("FREM_S10_SEALED_INPUT", str(sealed))
    monkeypatch.setattr(kaggle_contract, "require_sealed_input", lambda path: path)
    work = tmp_path / "work"
    work.mkdir()
    wheel = tmp_path / "research/frem_s10_wheelhouse/numpy.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"generated wheel fixture")
    monkeypatch.setattr(packaging, "dependency_wheel", lambda source: (wheel, "a" * 64))
    commands = dict(verification.verification_commands(tmp_path, work))
    assert tuple(commands) == verification.CHECK_NAMES
    assert commands["boundary_probe"][0] == str(Path(os.sys.executable).absolute())
    assert kaggle_contract.BOUNDARY_PROBE in commands["boundary_probe"]
    assert not any("bwrap" in item for command in commands.values() for item in command)
    assert verification._probe_message() == kaggle_contract.BOUNDARY_MESSAGE
