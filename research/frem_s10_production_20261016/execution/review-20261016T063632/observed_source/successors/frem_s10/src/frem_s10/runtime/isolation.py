"""Fixed verification isolation adapters; availability is never execution proof.

Linux uses mandatory bubblewrap namespaces and mounts, without a weaker fallback.
The retained probe must demonstrate those boundaries on the executing host.
Reference: https://github.com/containers/bubblewrap/blob/main/bwrap.xml
"""

import os
import platform
import re
import shutil
from pathlib import Path

from frem_s10.errors import IntegrityError
from frem_s10.runtime.artifacts import _plain_directory

LINUX_PROBE_MESSAGE = (
    "NETWORK_NAMESPACE_ISOLATED DATA_OUTPUTS_RESEARCH_MASKED DENIED_FIXTURE_HIDDEN PROTECTED_FIXTURE_EMPTY"
)
DENIED_FIXTURE = "isolation_denied"
LINUX_BOUNDARY_PROBE = r"""import errno, os, pathlib, re, socket, sys
parent_namespace, work, wheel, *denied = sys.argv[1:]
assert len(denied) == 3
assert re.fullmatch(r'net:\[[0-9]+\]', parent_namespace)
assert os.readlink('/proc/self/ns/net') != parent_namespace, 'network namespace unchanged'
assert {name for _, name in socket.if_nameindex()} == {'lo'}, 'external interface visible'
with socket.socket() as connection:
    connection.settimeout(1)
    try:
        connection.connect(('192.0.2.1', 9))
    except OSError as error:
        assert error.errno == errno.ENETUNREACH, repr(error)
    else:
        raise AssertionError('network route escaped isolated namespace')
def unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)
mounts = {}
for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
    left, right = line.split(' - ', 1)
    fields = left.split()
    mounts[unescape(fields[4])] = (right.split()[0], set(fields[5].split(',')))
assert 'ro' in mounts['/'][1], 'root filesystem is not read-only'
assert 'rw' in mounts[work][1], 'explicit work mount unavailable'
assert 'ro' in mounts[wheel][1], 'exact dependency wheel is not read-only'
fixture = str(pathlib.Path(work) / 'isolation_denied')
for path in [*denied, fixture]:
    kind, flags = mounts[path]
    assert kind == 'tmpfs' and 'ro' in flags, 'denied path lacks read-only tmpfs: ' + path
assert list(pathlib.Path(fixture).iterdir()) == [], 'denied fixture was exposed'
assert not (pathlib.Path(fixture) / 'marker.txt').exists()
assert list((pathlib.Path(work) / 'protected_empty').iterdir()) == []
print('NETWORK_NAMESPACE_ISOLATED DATA_OUTPUTS_RESEARCH_MASKED DENIED_FIXTURE_HIDDEN PROTECTED_FIXTURE_EMPTY')
"""


def _parent_network_namespace() -> str:
    try:
        identity = os.readlink("/proc/self/ns/net")
    except OSError as exc:
        raise IntegrityError("Linux parent network namespace is unavailable") from exc
    if re.fullmatch(r"net:\[[0-9]+\]", identity) is None:
        raise IntegrityError("Linux parent network namespace identity is malformed")
    return identity


def prepare_linux_fixture(work: Path) -> None:
    """Create an exclusive harmless sentinel whose invisibility must be observed."""
    if platform.system() != "Linux":
        return
    fixture = _plain_directory(work) / DENIED_FIXTURE
    fixture.mkdir(exist_ok=False)
    with (fixture / "marker.txt").open("xb") as handle:
        handle.write(b"generated verification isolation sentinel\n")


def _paths(source_root: Path, work: Path, wheel: Path) -> tuple[Path, Path, Path]:
    if any(".." in path.parts for path in (source_root, work, wheel)):
        raise IntegrityError("canonical isolation paths without traversal required")
    source, output = _plain_directory(source_root), _plain_directory(work)
    if source == output or source.is_relative_to(output):
        raise IntegrityError("isolation work cannot expose the source or its parent")
    denied = tuple(source / name for name in ("data", "outputs", "research"))
    for directory in denied:
        _plain_directory(directory)
        if output == directory or directory.is_relative_to(output):
            raise IntegrityError("isolation work cannot expose a denied evidence root")
    dependency = wheel.absolute()
    if (
        not dependency.is_file()
        or dependency.is_symlink()
        or any(parent.is_symlink() for parent in dependency.parents)
        or dependency.is_relative_to(output)
    ):
        raise IntegrityError("exact regular dependency wheel outside writable work required")
    fixture = _plain_directory(output / DENIED_FIXTURE)
    marker = fixture / "marker.txt"
    if (
        marker.is_symlink()
        or not marker.is_file()
        or marker.stat().st_size != 42
        or marker.read_bytes() != b"generated verification isolation sentinel\n"
    ):
        raise IntegrityError("exact generated isolation sentinel required")
    return source, output, dependency


def linux_prefix(source_root: Path, work: Path, wheel: Path) -> tuple[str, ...]:
    """Construct fixed fail-closed mounts; unsupported bwrap flags fail at launch."""
    if platform.system() != "Linux":
        raise IntegrityError("Linux isolation requires Linux")
    executable = shutil.which("bwrap")
    if executable is None:
        raise IntegrityError("Linux verification requires bubblewrap; no unsandboxed fallback")
    source, output, dependency = _paths(source_root, work, wheel)
    masks = tuple(source / name for name in ("data", "outputs", "research"))
    args = [
        executable,
        "--unshare-user",
        "--disable-userns",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-net",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for path in masks:
        args.extend(("--tmpfs", str(path)))
    args.extend(("--bind", str(output), str(output), "--ro-bind", str(dependency), str(dependency)))
    fixture = output / DENIED_FIXTURE
    args.extend(("--tmpfs", str(fixture)))
    for path in (*masks, fixture):
        args.extend(("--remount-ro", str(path)))
    args.extend(
        ("--dir", str(output / "tmp"), "--setenv", "TMPDIR", str(output / "tmp"), "--chdir", str(source))
    )
    return (*args, "--")


def linux_probe_arguments(source_root: Path, work: Path, wheel: Path) -> tuple[str, ...]:
    """Bind actual parent namespace identity and exact mounted paths into the probe."""
    return (
        "-c",
        LINUX_BOUNDARY_PROBE,
        _parent_network_namespace(),
        str(work),
        str(wheel),
        *(str(source_root / name) for name in ("data", "outputs", "research")),
    )
