"""Atomic generated-run commits and exact identity-bound checkpoint recovery."""

import fcntl
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from frem_s10.errors import ExecutionStateError, IntegrityError
from frem_s10.identities import JSONObject, canonical_bytes, digest_bytes, object_json, parse_json
from frem_s10.runtime.artifacts import _plain_directory, publish_json, read_verified_json


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    """Own one POSIX run lock; process exit releases it without deleting evidence."""
    root = _plain_directory(directory)
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    fd = os.open(root / ".run.lock", flags, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExecutionStateError("generated run is already active") from exc
        yield
    finally:
        os.close(fd)


def read_plain(path: Path, maximum: int = 65536) -> JSONObject:
    """Read bounded non-symlink metadata; reject unexpected files before allocation."""
    _plain_directory(path.parent)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise IntegrityError("bounded regular generated artifact required")
    return object_json(parse_json(path.read_bytes()))


def commits(directory: Path) -> Iterator[JSONObject]:
    """Read contiguous atomic commit records and verify every referenced checkpoint."""
    for index, path in enumerate(sorted(directory.glob("step-*.json")), 1):
        if path.name != f"step-{index:04}.json":
            raise IntegrityError("generated checkpoint sequence is incomplete")
        commit = read_plain(path)
        if set(commit) != {"step", "path", "sha256"} or commit["step"] != index:
            raise IntegrityError("generated commit schema mismatch")
        name, identity = commit["path"], commit["sha256"]
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("checkpoint-"):
            raise IntegrityError("checkpoint must be a local generated basename")
        if not isinstance(identity, str):
            raise IntegrityError("checkpoint hash missing")
        checkpoint = object_json(read_verified_json(directory / name, identity, maximum_bytes=32 * 2**20))
        if checkpoint.get("completed") != index:
            raise IntegrityError("checkpoint cursor differs from its commit")
        yield checkpoint


def save_checkpoint(directory: Path, checkpoint: JSONObject) -> str:
    """Publish a unique payload then an exclusive atomic commit at the safe boundary."""
    step = checkpoint["completed"]
    if type(step) is not int or step <= 0:
        raise IntegrityError("positive checkpoint cursor required")
    name = f"checkpoint-{step:04}-{uuid.uuid4().hex}.json"
    identity = publish_json(directory, name, checkpoint)
    publish_json(directory, f"step-{step:04}.json", {"step": step, "path": name, "sha256": identity})
    return name


def publish_status(directory: Path, value: JSONObject) -> None:
    """Keep immutable status history and atomically refresh only its convenience view."""
    name = f"status-{uuid.uuid4().hex}.json"
    publish_json(directory, name, value)
    target = directory / "status.json"
    if target.is_symlink():
        raise IntegrityError("status pointer is a symlink")
    # The unique history entry remains even if updating the convenience view fails.
    temporary = directory / f".status-{uuid.uuid4().hex}"
    payload = canonical_bytes(value) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity_document(directory: Path) -> JSONObject:
    """Read the immutable start declaration and its separate exact-byte seal."""
    seal = read_plain(directory / "identity.json")
    if set(seal) != {"sha256"} or not isinstance(seal["sha256"], str):
        raise IntegrityError("generated identity seal malformed")
    return object_json(read_verified_json(directory / "run.json", seal["sha256"]))


def seal_run(directory: Path, declaration: JSONObject) -> None:
    """Bind configuration, code, data, environment and execution semantics once."""
    identity = publish_json(directory, "run.json", declaration)
    publish_json(directory, "identity.json", {"sha256": identity})


def verify_binding(checkpoint: JSONObject, declaration: JSONObject) -> None:
    """Reject stale configurations, data and runtime even when a file exists."""
    expected = digest_bytes(canonical_bytes(declaration))
    if checkpoint.get("schema") != "s10-generated-commit-v1" or checkpoint.get("binding") != expected:
        raise IntegrityError("checkpoint source/configuration/data/environment binding mismatch")
