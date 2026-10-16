"""Publish finite JSON evidence once, with exclusive creation and durable bytes."""

import os
import tempfile
from pathlib import Path

from frem_s10.errors import ArtifactIOError, IntegrityError
from frem_s10.identities import JSONValue, canonical_bytes, digest_bytes, parse_json


def _plain_directory(directory: Path) -> Path:
    """Require an existing directory without symlink components."""
    absolute = directory.absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ArtifactIOError("artifact directory contains a symlink")
    if not absolute.is_dir():
        raise ArtifactIOError("artifact directory must already exist")
    return absolute


def publish_json(directory: Path, name: str, value: JSONValue) -> str:
    """Durably publish a new basename without replacing prior evidence.

    Hard-link publication is atomic and fails if the destination exists. A
    failure after publication leaves the result in place for inspection.
    """
    directory = _plain_directory(directory)
    if not name or name in (".", "..") or Path(name).name != name:
        raise ArtifactIOError("artifact name must be a single basename")
    payload = canonical_bytes(value) + b"\n"
    temporary: Path | None = None
    try:
        fd, path = tempfile.mkstemp(prefix=".pending-", dir=directory)
        temporary = Path(path)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise ArtifactIOError("temporary artifact verification failed")
        os.link(temporary, directory / name)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ArtifactIOError("artifact publication failed; existing bytes retained") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest_bytes(payload)


def read_verified_json(path: Path, expected_sha256: str, *, maximum_bytes: int = 1_048_576) -> JSONValue:
    """Read a bounded plain file only after checking its exact content hash."""
    _plain_directory(path.parent)
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("artifact must be a regular nonsymlink file")
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise IntegrityError("positive integer artifact byte ceiling required")
    with path.open("rb") as stream:
        payload = stream.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes or digest_bytes(payload) != expected_sha256:
        raise IntegrityError("artifact size or hash mismatch")
    return parse_json(payload)
