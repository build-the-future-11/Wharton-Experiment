"""Source-bound one-use scientific cell consumption ledger."""

from __future__ import annotations

from pathlib import Path

from frem_s12.errors import ConsumptionError, ContractError
from frem_s12.evidence import atomic_json, package_identity, read_json, receipt, sha256_file


def _claim_path(root: Path, phase: str, identity: dict[str, int | str]) -> Path:
    if phase == "search":
        required = ("trial", "family", "seed")
    elif phase == "fixed":
        required = ("family", "fold", "seed")
    else:
        raise ContractError("consumption phase is invalid")
    if set(identity) != set(required):
        raise ContractError("consumption identity fields are invalid")
    parts = [f"{name}={identity[name]}" for name in required]
    if any("/" in value or ".." in value for value in parts):
        raise ContractError("consumption identity contains an unsafe path")
    return root / phase / "/".join(parts) / "claim.json"


def claim_consumption(
    root: Path,
    *,
    phase: str,
    identity: dict[str, int | str],
    gate_hashes: dict[str, object],
    resume: bool = False,
) -> dict[str, object]:
    """Create one irreversible claim, or verify it for an exact resume."""
    path = _claim_path(root, phase, identity)
    source_identity = package_identity()["sha256"]
    if resume:
        value = read_json(path)
        if (
            value.get("phase") != phase
            or value.get("identity") != identity
            or value.get("source_identity") != source_identity
            or value.get("gate_hashes") != gate_hashes
        ):
            raise ConsumptionError("resume claim differs from the current cell or gates")
        return {"path": str(path), "sha256": sha256_file(path), "resumed": True}
    if path.exists() or path.is_symlink():
        raise ConsumptionError("scientific cell was already consumed; only exact resume is allowed")
    value = receipt(
        "consumption",
        consumed=True,
        phase=phase,
        identity=identity,
        gate_hashes=gate_hashes,
        automatic_retries=0,
    )
    atomic_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path), "resumed": False}
