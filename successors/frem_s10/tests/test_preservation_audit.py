"""Tiny identity-only preservation fixtures never decode numerical payload files."""

import hashlib
import json
from pathlib import Path

import pytest

from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue
from frem_s10.runtime import preservation


def _write(root: Path, name: str, payload: bytes) -> JSONObject:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}


def _object(root: Path, name: str, value: JSONObject) -> JSONObject:
    return _write(root, name, json.dumps(value, sort_keys=True).encode())


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[JSONValue]:
    payload = _write(root, "history/undecoded.bin", b"\xff\x00not-json-numerical-payload")
    manifest = _object(root, "history/manifest.json", {"files": {"history/undecoded.bin": payload["sha256"]}})
    historical = _object(
        root,
        "session/historical.json",
        {
            "scope": "STREAMING_IDENTITY_CHECK_ONLY_NO_NUMERICAL_OUTCOME_INSPECTION",
            "manifest_sha256": {str(manifest["path"]): manifest["sha256"]},
            "checked_references": 1,
            "unique_files": 1,
            "mismatches": [],
            "mutable_history_records": [],
            "references": [
                {
                    "manifest": manifest["path"],
                    "path": payload["path"],
                    "expected": payload["sha256"],
                    "actual": payload["sha256"],
                    "matches": True,
                }
            ],
        },
    )
    code = _write(root, "session/before/source.py", b"# prior source\n")
    journal = _write(root, "session/before/research/events.jsonl", b"old-journal-bytes\n")
    _write(root, "research/events.jsonl", b"old-journal-bytes\nnew-journal-bytes\n")
    old = _object(
        root,
        "history/receipt.json",
        {
            "source_sha256": {"source.py": code["sha256"]},
            "evidence_sha256": {str(payload["path"]): payload["sha256"]},
        },
    )
    monkeypatch.setattr(preservation, "OLD_REPAIR_RECEIPT", old)
    before = _object(
        root,
        "session/before_manifest.json",
        {
            "timestamp": "generated",
            "files": [
                {"path": "source.py", "sha256": code["sha256"]},
                {"path": "research/events.jsonl", "sha256": journal["sha256"]},
            ],
        },
    )
    prior: JSONObject = {
        "scope": "EXACT_PRIOR_SOURCE_AND_EVIDENCE_PRESERVATION",
        "before_files": 2,
        "old_evidence_files": 1,
        "old_source_files": 1,
        "old_receipt_sha256": old["sha256"],
        "old_receipt_matches": True,
        "old_journal_prefix_preserved": True,
        "mismatches": [],
    }
    prior_ref = _object(root, "session/prior.json", prior)
    current = _object(
        root,
        "session/preservation.json",
        prior
        | {
            "schema": "s10-preservation-audit-v2",
            "before_manifest": before,
            "predecessor_receipt": old,
            "prior_audit": prior_ref,
        },
    )
    return [historical, current]


def test_complete_preservation_streams_non_json_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = _fixture(tmp_path, monkeypatch)
    preservation.audit_preservation(references, tmp_path)


@pytest.mark.parametrize("damage", ["payload", "snapshot", "journal", "manifest", "symlink"])
def test_preserved_identity_corruption_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    references = _fixture(tmp_path, monkeypatch)
    names = {
        "payload": "history/undecoded.bin",
        "snapshot": "session/before/source.py",
        "journal": "research/events.jsonl",
        "manifest": "history/manifest.json",
        "symlink": "history/undecoded.bin",
    }
    path = tmp_path / names[damage]
    if damage == "symlink":
        outside = tmp_path / "external.bin"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path.write_bytes(b"corrupt")
    with pytest.raises((ContractError, IntegrityError)):
        preservation.audit_preservation(references, tmp_path)


@pytest.mark.parametrize("damage", ["empty", "count", "duplicate", "escape"])
def test_historical_inventory_cannot_omit_duplicate_or_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    references = _fixture(tmp_path, monkeypatch)
    value = json.loads((tmp_path / "session/historical.json").read_text())
    if damage == "empty":
        value["references"] = []
        value["checked_references"] = value["unique_files"] = 0
    elif damage == "count":
        value["unique_files"] = 2
    elif damage == "duplicate":
        value["references"] *= 2
    else:
        value["manifest_sha256"] = {"../manifest.json": "0" * 64}
    references[0] = _object(tmp_path, "session/historical.json", value)
    with pytest.raises((ContractError, IntegrityError)):
        preservation.audit_preservation(references, tmp_path)


def test_missing_before_source_requires_exact_archived_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    original = tmp_path / "session/before/source.py"
    fallback = _write(tmp_path, "history/integrated_verified_snapshot/source.py", original.read_bytes())
    source: JSONObject = {"source.py": fallback["sha256"]}
    preservation._source_preserved(source, {}, {"source.py": fallback}, tmp_path)
    with pytest.raises(IntegrityError, match="explicit archived"):
        preservation._source_preserved(source, {}, {}, tmp_path)
    (tmp_path / "history/integrated_verified_snapshot/source.py").write_bytes(b"changed archived source")
    with pytest.raises(IntegrityError, match="identity changed"):
        preservation._source_preserved(source, {}, {"source.py": fallback}, tmp_path)
