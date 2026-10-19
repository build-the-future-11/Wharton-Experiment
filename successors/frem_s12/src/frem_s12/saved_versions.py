"""Fail-closed audit of Kaggle saved-version consumption evidence."""

from __future__ import annotations

from pathlib import Path

from frem_s12.errors import ContractError
from frem_s12.evidence import read_json, sha256_file

_ATTEMPT_FIELDS = {"claims", "saved_version_id", "status"}
_VALID_STATUSES = {"ENGINEERING_ONLY", "REJECTED", "SCIENTIFIC"}
_VALID_PHASES = {"search", "fixed"}


def _validated_attempts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, dict) or set(value) != {"attempts", "complete", "schema", "scope"}:
        raise ContractError("saved-version ledger fields differ from the contract")
    if value["schema"] != "frem-s12-saved-version-ledger-v1" or value["scope"] != "kaggle":
        raise ContractError("saved-version ledger schema or scope is invalid")
    if value["complete"] is not True:
        raise ContractError("saved-version inventory is not externally complete")
    attempts = value["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise ContractError("saved-version ledger has no attempts")
    return attempts


def _validate_attempt(attempt: object, versions: set[int], claims: set[tuple[str, str]]) -> None:
    if not isinstance(attempt, dict) or set(attempt) != _ATTEMPT_FIELDS:
        raise ContractError("saved-version attempt fields differ from the contract")
    version = attempt["saved_version_id"]
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0 or version in versions:
        raise ContractError("saved-version identity is invalid or duplicated")
    versions.add(version)
    if attempt["status"] not in _VALID_STATUSES:
        raise ContractError("saved-version status is invalid")
    raw_claims = attempt["claims"]
    if not isinstance(raw_claims, list):
        raise ContractError("saved-version claims are invalid")
    for claim in raw_claims:
        if not isinstance(claim, dict) or set(claim) != {"identity", "phase"}:
            raise ContractError("saved-version claim fields differ from the contract")
        phase, identity = claim["phase"], claim["identity"]
        if phase not in _VALID_PHASES or not isinstance(identity, str) or not identity:
            raise ContractError("saved-version claim is invalid")
        key = (phase, identity)
        if key in claims:
            raise ContractError("saved-version ledger contains a duplicate scientific claim")
        claims.add(key)


def audit_saved_versions(path: Path) -> dict[str, object]:
    """Validate a complete external saved-version inventory before scientific work."""
    attempts = _validated_attempts(read_json(path))
    versions: set[int] = set()
    claims: set[tuple[str, str]] = set()
    for attempt in attempts:
        _validate_attempt(attempt, versions, claims)
    return {
        "schema": "frem-s12-saved-version-audit-v1",
        "verified": True,
        "ledger_sha256": sha256_file(path),
        "saved_versions": len(versions),
        "scientific_claims": len(claims),
    }
