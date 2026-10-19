"""Exclusive operation-specific evidence markers; no deletion or resume facility."""

import re
from datetime import UTC, datetime
from pathlib import Path

from frem_s10.errors import ArtifactIOError, ContractError, ExecutionStateError
from frem_s10.runtime.artifacts import publish_json


def consume(
    directory: Path, *, operation: str, freeze_id: str, attempt_id: str, output_path: str | None = None
) -> str:
    """Publish a durable one-use identity before any future scientific dispatch.

    This primitive does not establish preflight or launch a worker. Callers must
    supply verified prerequisites; a marker survives every later error.
    """
    if operation not in ("precision", "campaign", "saved-audit"):
        raise ContractError("unknown scientific operation")
    if not re.fullmatch(r"[0-9a-f]{64}", freeze_id):
        raise ContractError("exact lowercase SHA256 freeze identity required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attempt_id):
        raise ContractError("bounded plain attempt identity required")
    name = f"{operation}-{freeze_id}.json"
    if (directory / name).exists() or (directory / name).is_symlink():
        raise ExecutionStateError("existing consumption marker; attempt requires review")
    try:
        return publish_json(
            directory,
            name,
            {
                "schema_version": 1,
                "operation": operation,
                "freeze_id": freeze_id,
                "attempt_id": attempt_id,
                "consumed_at": datetime.now(UTC).isoformat(),
                "state": "CONSUMED",
                "resume_permitted": False,
                "output_path": output_path,
            },
        )
    except ArtifactIOError as exc:
        raise ExecutionStateError("consumption publication failed or ambiguous; no worker may start") from exc
