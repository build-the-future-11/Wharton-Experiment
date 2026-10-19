"""Kaggle production boundary observations for the S10 retrospective study.

This contract is deliberately opt-in and platform-specific.  It does not claim
that Kaggle exposes nested namespaces.  Instead it binds execution to a CPU-only
Kaggle session whose Internet setting is disabled, uses an immutable uploaded
input as the source of sealed bytes, stages no scientific values until the
engineering verification has completed, and retains the existing one-use
consumption, process-tree, RSS, wall-clock, and output-volume supervisors.
"""

from __future__ import annotations

import errno
import os
import platform
import socket
from pathlib import Path

from frem_s10.errors import IntegrityError

CONTRACT = "s10-kaggle-production-v1"
ENVIRONMENT_KEY = "FREM_S10_EXECUTION_CONTRACT"
BOUNDARY_MESSAGE = (
    "KAGGLE_CPU_ONLY NETWORK_UNREACHABLE SEALED_INPUT_READ_ONLY "
    "SCIENTIFIC_VALUES_NOT_STAGED PROTECTED_FIXTURE_EMPTY"
)


def active() -> bool:
    """Return whether the caller explicitly selected the Kaggle contract."""
    return os.environ.get(ENVIRONMENT_KEY) == CONTRACT


def require_platform() -> None:
    """Reject lookalike Linux hosts and accelerator-backed sessions."""
    if not active():
        raise IntegrityError("explicit Kaggle production contract marker required")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise IntegrityError("Kaggle production requires Linux x86_64")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise IntegrityError("Kaggle input and working mounts are required")
    if Path("/dev/nvidia0").exists() or any(Path("/proc/driver/nvidia/gpus").glob("*")):
        raise IntegrityError("Kaggle production requires accelerator None")


def require_sealed_input(path: Path) -> Path:
    """Require an existing regular file below Kaggle's immutable input mount."""
    require_platform()
    item = path.absolute()
    root = Path("/kaggle/input")
    if (
        not item.is_relative_to(root)
        or item.is_symlink()
        or not item.is_file()
        or any(parent.is_symlink() for parent in item.parents if parent != root.parent)
    ):
        raise IntegrityError("sealed Kaggle input must be a regular file below /kaggle/input")
    return item


def network_unreachable() -> None:
    """Observe platform network denial using numeric addresses and bounded calls."""
    require_platform()
    errors: list[int | None] = []
    for family, address in (
        (socket.AF_INET, ("1.1.1.1", 443)),
        (socket.AF_INET6, ("2606:4700:4700::1111", 443, 0, 0)),
    ):
        with socket.socket(family, socket.SOCK_STREAM) as connection:
            connection.settimeout(1.0)
            try:
                connection.connect(address)
            except OSError as exc:
                errors.append(exc.errno)
            else:
                raise IntegrityError("Kaggle Internet setting is enabled")
    # A timeout (errno=None) is ambiguous reachability, not enforced denial.
    allowed = {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPERM, errno.EACCES}
    if not errors or any(value not in allowed for value in errors):
        raise IntegrityError("network refusal did not establish the Kaggle offline boundary")


BOUNDARY_PROBE = r"""import pathlib, sys
from frem_s10.runtime.kaggle_contract import BOUNDARY_MESSAGE, network_unreachable, require_sealed_input
sealed, data, outputs, research, wheel, protected = map(pathlib.Path, sys.argv[1:])
require_sealed_input(sealed)
network_unreachable()
for path in (data, outputs):
    assert not path.exists(), 'scientific values were staged during engineering verification: ' + str(path)
assert wheel.is_file() and wheel.is_relative_to(research / 'frem_s10_wheelhouse')
assert {p for p in research.rglob('*') if p.is_file()} == {wheel}, 'unexpected research evidence exposed'
assert list(protected.iterdir()) == [], 'protected fixture is not empty'
try:
    with (sealed.parent / '.frem-write-probe').open('xb') as stream:
        stream.write(b'x')
except OSError:
    pass
else:
    raise AssertionError('sealed Kaggle input mount is writable')
print(BOUNDARY_MESSAGE)
"""
