"""Simulated Linux boundary contracts; these tests do not certify a Linux host."""

import errno
import json
import os
import platform
import resource
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from frem_s10.errors import ArtifactIOError, IntegrityError
from frem_s10.runtime import isolation, packaging, verification


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")
    for name in ("data", "outputs", "research"):
        (tmp_path / name).mkdir()
    work = tmp_path / "research/checks"
    work.mkdir()
    (work / "protected_empty").mkdir()
    isolation.prepare_linux_fixture(work)
    wheel = tmp_path / "research/numpy.whl"
    wheel.write_bytes(b"inert dependency fixture")
    return tmp_path, work, wheel


def test_linux_commands_mask_evidence_before_exact_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, wheel = _fixture(tmp_path, monkeypatch)
    argv = isolation.linux_prefix(source, work, wheel)
    assert argv[:2] == ("/usr/bin/bwrap", "--unshare-user")
    assert {"--unshare-net", "--unshare-pid", "--disable-userns", "--die-with-parent"} <= set(argv)
    assert not any(value.endswith("-try") for value in argv)
    root = argv.index("--ro-bind")
    assert argv[root + 1 : root + 3] == ("/", "/")
    bind = argv.index("--bind")
    assert argv[bind + 1 : bind + 3] == (str(work), str(work))
    assert argv.count("--bind") == 1
    for name in ("data", "outputs", "research"):
        index = argv.index(str(source / name))
        assert argv[index - 1] == "--tmpfs" and index < bind
    assert argv[bind + 3 : bind + 6] == ("--ro-bind", str(wheel), str(wheel))
    assert argv.count("--remount-ro") == 4
    assert argv[-1] == "--"


def test_missing_bubblewrap_never_returns_unsandboxed_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, wheel = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(IntegrityError, match="no unsandboxed fallback"):
        isolation.linux_prefix(source, work, wheel)


@pytest.mark.parametrize("damage", ("source", "research", "symlink", "sentinel", "wheel", "traversal"))
def test_writable_or_aliased_evidence_roots_cannot_escape_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    source, work, wheel = _fixture(tmp_path, monkeypatch)
    if damage == "source":
        work = source
    elif damage == "research":
        work = source / "research"
    elif damage == "symlink":
        alias = source / "alias"
        alias.symlink_to(work, target_is_directory=True)
        work = alias
    elif damage == "sentinel":
        (work / isolation.DENIED_FIXTURE / "marker.txt").write_text("not the generated sentinel")
    elif damage == "wheel":
        wheel = work / "wheel.whl"
        wheel.write_bytes(b"mutable")
    else:
        work = source / "research/checks/.."
    with pytest.raises((IntegrityError, ArtifactIOError)):
        isolation.linux_prefix(source, work, wheel)


def test_linux_verification_uses_namespace_probe_and_selected_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, wheel = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(packaging, "dependency_wheel", lambda root: (wheel, "a" * 64))
    monkeypatch.setattr(isolation, "_parent_network_namespace", lambda: "net:[12345]")
    commands = dict(verification.verification_commands(source, work))
    assert tuple(commands) == verification.CHECK_NAMES
    for argv in commands.values():
        assert argv[0] == "/usr/bin/bwrap"
        assert "--unshare-net" in argv
    assert isolation.LINUX_BOUNDARY_PROBE in commands["boundary_probe"]
    assert "net:[12345]" in commands["boundary_probe"]
    assert verification._probe_message() == isolation.LINUX_PROBE_MESSAGE


def test_darwin_sandbox_argv_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    work = tmp_path / "checks"
    denied = "".join(
        f"(subpath {json.dumps(str(tmp_path / item))})" for item in ("data", "outputs", "research")
    )
    policy = (
        f"(version 1)(allow default)(deny network*)(deny file-read* {denied})"
        f"(allow file-read* (subpath {json.dumps(str(work))})"
        f"(literal {json.dumps(str(tmp_path / packaging.NUMPY_WHEEL))}))"
    )
    commands = dict(verification.verification_commands(tmp_path, work))
    for argv in commands.values():
        assert argv[:3] == ("/usr/bin/sandbox-exec", "-p", policy)
    assert verification.BOUNDARY_PROBE in commands["boundary_probe"]
    assert verification._probe_message() == (
        "NETWORK_DENIED DATA_OUTPUTS_RESEARCH_DENIED PROTECTED_FIXTURE_EMPTY"
    )


def test_other_platform_has_no_isolation_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with pytest.raises(IntegrityError, match="unsupported"):
        verification.verification_commands(tmp_path, tmp_path / "checks")


@pytest.mark.parametrize("system,multiplier", (("Linux", 1024), ("Darwin", 1)))
def test_observed_rss_units_are_not_relabelled(
    monkeypatch: pytest.MonkeyPatch, system: str, multiplier: int
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(resource, "getrusage", lambda kind: SimpleNamespace(ru_maxrss=1234))
    key, value = verification._rss_observation()
    assert key.endswith("linux" if system == "Linux" else "macos")
    assert value == 1234 * multiplier


class _NoRouteSocket:
    def __init__(self, error: int) -> None:
        self.error = error

    def __enter__(self) -> "_NoRouteSocket":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def settimeout(self, timeout: int) -> None:
        pass

    def connect(self, address: tuple[str, int]) -> None:
        assert address == ("192.0.2.1", 9)
        raise OSError(self.error, "simulated namespace observation")


@pytest.mark.parametrize(
    "defect",
    (None, "same_namespace", "external_interface", "generic_refusal", "mask", "writable_root", "leak"),
)
def test_probe_requires_namespace_mount_and_specific_network_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    fixture = work / isolation.DENIED_FIXTURE
    fixture.mkdir()
    (work / "protected_empty").mkdir()
    wheel = tmp_path / "numpy.whl"
    denied = [tmp_path / name for name in ("data", "outputs", "research")]
    rows = [
        ("/", "rootfs", "rw" if defect == "writable_root" else "ro"),
        (str(work), "ext4", "rw"),
        (str(wheel), "ext4", "ro"),
    ]
    rows.extend((str(path), "tmpfs", "ro") for path in [*denied, fixture])
    if defect == "mask":
        rows.pop()
    mountinfo = "\n".join(
        f"{i} 0 0:1 / {path} {flags} - {kind} none {flags}" for i, (path, kind, flags) in enumerate(rows, 1)
    )
    original_read = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/self/mountinfo":
            return mountinfo
        return original_read(path)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(os, "readlink", lambda path: "net:[1]" if defect == "same_namespace" else "net:[2]")
    monkeypatch.setattr(
        socket, "if_nameindex", lambda: [(1, "eth0" if defect == "external_interface" else "lo")]
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda: _NoRouteSocket(errno.ECONNREFUSED if defect == "generic_refusal" else errno.ENETUNREACH),
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "net:[1]", str(work), str(wheel), *(str(path) for path in denied)]
    )
    if defect == "leak":
        (fixture / "marker.txt").write_text("visible marker")
    if defect is None:
        exec(compile(isolation.LINUX_BOUNDARY_PROBE, "generated-linux-probe", "exec"), {})
        assert capsys.readouterr().out.strip() == isolation.LINUX_PROBE_MESSAGE
    else:
        with pytest.raises((AssertionError, KeyError)):
            exec(compile(isolation.LINUX_BOUNDARY_PROBE, "generated-linux-probe", "exec"), {})
