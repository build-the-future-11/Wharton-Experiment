"""Single-writer, durable append-only lifecycle journals with verified hash links."""

import os
from pathlib import Path

from frem_s10.errors import ArtifactIOError, IntegrityError
from frem_s10.identities import JSONValue, canonical_bytes, digest_bytes, parse_json
from frem_s10.runtime.artifacts import _plain_directory


def write_failure_receipt(directory: Path, event: JSONValue) -> None:
    """Durably retain a separate failure even when normal journal publication fails.

    This exclusive fallback never repairs or replaces prior evidence. If the
    filesystem itself cannot accept durable bytes, the caller must report that
    failure; no software can promise a receipt on an unavailable volume.
    """
    directory = _plain_directory(directory)
    payload = canonical_bytes(event) + b"\n"
    try:
        descriptor = os.open(directory / "failure.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError as exc:
        raise ArtifactIOError(
            "durable fallback failure receipt unavailable; retain all existing evidence"
        ) from exc


def inspect_journal(path: Path) -> tuple[int, str]:
    """Check complete JSON lines, exact sequence and every prior-event digest."""
    _plain_directory(path.parent)
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("journal must be a plain file")
    sequence, previous = 0, "0" * 64
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n") or len(line) > 1_048_576:
                raise IntegrityError("partial or oversized journal event")
            value = parse_json(line)
            if not isinstance(value, dict) or set(value) != {"sequence", "previous", "event"}:
                raise IntegrityError("invalid journal event schema")
            if type(value["sequence"]) is not int or value["sequence"] != sequence:
                raise IntegrityError("journal sequence mismatch")
            if value["previous"] != previous:
                raise IntegrityError("journal hash chain mismatch")
            previous = digest_bytes(line)
            sequence += 1
    return sequence, previous


class Journal:
    """Own a new journal; check inode/size/timestamps in O(1) before each append.

    Full hash verification remains mandatory at final audit. This is a single
    trusted writer contract, not protection against an adversarial OS owner.
    """

    def __init__(self, path: Path) -> None:
        _plain_directory(path.parent)
        if path.is_symlink():
            raise ArtifactIOError("journal cannot be a symlink")
        try:
            with path.open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise ArtifactIOError("journal creation failed; retain any created bytes") from exc
        self.path = path
        self.sequence = 0
        self.previous = "0" * 64
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> tuple[int, int, int, int, int]:
        info = self.path.stat(follow_symlinks=False)
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def append(self, event: JSONValue) -> str:
        """Commit one finite bounded event after checking the existing chain."""
        if self.path.is_symlink() or self._fingerprint() != self.fingerprint:
            raise IntegrityError("journal changed outside its owner")
        payload = (
            canonical_bytes({"sequence": self.sequence, "previous": self.previous, "event": event}) + b"\n"
        )
        if len(payload) > 1_048_576:
            raise ArtifactIOError("journal event exceeds byte ceiling")
        try:
            with self.path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise ArtifactIOError("journal append failed; retain incomplete evidence") from exc
        self.previous = digest_bytes(payload)
        self.sequence += 1
        self.fingerprint = self._fingerprint()
        return self.previous
