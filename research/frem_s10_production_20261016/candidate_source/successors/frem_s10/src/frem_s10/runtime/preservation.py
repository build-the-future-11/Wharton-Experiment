"""Recheck preserved historical bytes without decoding numerical outcome payloads."""

import hashlib
from pathlib import Path, PurePosixPath

from frem_s10.errors import IntegrityError
from frem_s10.identities import JSONObject, JSONValue, object_json
from frem_s10.metadata import strict_int
from frem_s10.runtime.artifacts import _plain_directory
from frem_s10.runtime.declarations import _reference, _schema
from frem_s10.runtime.provenance import read_object

OLD_REPAIR_RECEIPT: JSONObject = {
    "path": "research/frem_s10_repair_20261015/RECEIPT.json",
    "sha256": "8a650b4588f76a51badf82cd6eb2eadbdbd0b5ed059b82ba0848578e0338d001",
}
MUTABLE_HISTORY = {
    "RESEARCH_STATE.json",
    "docs/CODEX_HANDOFF.md",
    "docs/RESEARCH_COMPLETION_STATUS.md",
    "docs/STATIC_VERIFICATION.md",
    "research/events.jsonl",
}


def _hash(root: Path, name: str, expected: JSONValue) -> None:
    path, identity = _reference({"path": name, "sha256": expected}, root)
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != identity:
            raise IntegrityError("preserved file identity changed")


def _mapping(value: JSONValue) -> JSONObject:
    mapping = object_json(value)
    if not mapping or len(mapping) > 4096:
        raise IntegrityError("nonempty bounded preservation file inventory required")
    return mapping


def _relative(value: JSONValue) -> str:
    if not isinstance(value, str):
        raise IntegrityError("relative snapshot identity required")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise IntegrityError("normalized relative snapshot identity required")
    return value


def _list(value: JSONValue) -> list[JSONValue]:
    if not isinstance(value, list) or not value or len(value) > 4096:
        raise IntegrityError("nonempty bounded preservation reference list required")
    return value


def _historical(value: JSONObject, root: Path) -> None:
    _schema(
        value,
        {
            "scope",
            "manifest_sha256",
            "checked_references",
            "unique_files",
            "mismatches",
            "mutable_history_records",
            "references",
        },
        "historical preservation audit",
    )
    if (
        value["scope"] != "STREAMING_IDENTITY_CHECK_ONLY_NO_NUMERICAL_OUTCOME_INSPECTION"
        or value["mismatches"] != []
    ):
        raise IntegrityError("clean hash-only historical audit required")
    expected: dict[tuple[str, str], JSONValue] = {}
    for name, identity in _mapping(value["manifest_sha256"]).items():
        manifest = read_object({"path": name, "sha256": identity}, root)
        fields = [key for key in ("files", "identities", "artifact_sha256") if key in manifest]
        if len(fields) != 1:
            raise IntegrityError("supported exact historical manifest inventory required")
        expected.update({(name, path): digest for path, digest in _mapping(manifest[fields[0]]).items()})
    mutable = _mutable(value["mutable_history_records"], expected)
    observed = _historical_files(value["references"], expected, mutable, root)
    if observed != set(expected) - mutable:
        raise IntegrityError("historical audit omits or invents manifest references")
    if strict_int(value["checked_references"], name="reference count", minimum=1) != len(
        observed
    ) or strict_int(value["unique_files"], name="unique count", minimum=1) != len(
        {path for _, path in observed}
    ):
        raise IntegrityError("historical preservation counts differ from full inventory")


def _mutable(value: JSONValue, expected: dict[tuple[str, str], JSONValue]) -> set[tuple[str, str]]:
    if not isinstance(value, list) or len(value) > 256:
        raise IntegrityError("bounded mutable-history registry required")
    result: set[tuple[str, str]] = set()
    for raw in value:
        row = _schema(
            raw,
            {"manifest", "path", "old_sha256", "classification", "old_expected_unchanged"},
            "mutable history",
        )
        pair = (str(row["manifest"]), str(row["path"]))
        if (
            pair in result
            or pair[1] not in MUTABLE_HISTORY
            or expected.get(pair) != row["old_sha256"]
            or row["old_expected_unchanged"] is not True
            or row["classification"] != "INTENTIONALLY_MUTABLE_CANONICAL_HISTORY_NOT_A_RESEARCH_RESULT"
        ):
            raise IntegrityError("mutable-history exception differs from frozen manifest")
        result.add(pair)
    return result


def _historical_files(
    value: JSONValue, expected: dict[tuple[str, str], JSONValue], mutable: set[tuple[str, str]], root: Path
) -> set[tuple[str, str]]:
    observed: set[tuple[str, str]] = set()
    for raw in _list(value):
        row = _schema(raw, {"manifest", "path", "expected", "actual", "matches"}, "historical reference")
        pair = (str(row["manifest"]), str(row["path"]))
        if (
            pair in observed
            or pair in mutable
            or pair not in expected
            or row["expected"] != expected[pair]
            or row["actual"] != row["expected"]
            or row["matches"] is not True
        ):
            raise IntegrityError("historical reference differs from manifest or is duplicated")
        _hash(root, pair[1], row["expected"])
        observed.add(pair)
    return observed


def _snapshot_files(value: JSONObject, root: Path, session: Path) -> JSONObject:
    reference = value["before_manifest"]
    path, _ = _reference(reference, root)
    if path != session / "before_manifest.json":
        raise IntegrityError("before manifest must belong to the same preservation session")
    manifest = read_object(reference, root)
    _schema(manifest, {"timestamp", "files"}, "before manifest")
    result: JSONObject = {}
    for raw in _list(manifest["files"]):
        row = _schema(raw, {"path", "sha256"}, "before snapshot file")
        name = _relative(row["path"])
        if name in result:
            raise IntegrityError("duplicate snapshot identity")
        _hash(root, str((session / "before" / name).relative_to(root)), row["sha256"])
        result[name] = row["sha256"]
    if len(result) != strict_int(value["before_files"], name="before files", minimum=1):
        raise IntegrityError("before snapshot count differs")
    base = _plain_directory(session / "before")
    observed = {
        item.relative_to(base).as_posix() for item in base.rglob("*") if item.is_file() or item.is_symlink()
    }
    if observed != set(result):
        raise IntegrityError("snapshot file inventory differs from before manifest")
    return result


def _journal_prefix(root: Path, session: Path, snapshots: JSONObject) -> None:
    name = "research/events.jsonl"
    if name not in snapshots:
        raise IntegrityError("preserved journal snapshot is mandatory")
    before, _ = _reference(
        {"path": str((session / "before" / name).relative_to(root)), "sha256": snapshots[name]}, root
    )
    current, _ = _reference({"path": name, "sha256": snapshots[name]}, root)
    if before.stat().st_size == 0:
        raise IntegrityError("preserved journal prefix cannot be empty")
    with before.open("rb") as old, current.open("rb") as new:
        while chunk := old.read(65536):
            if new.read(len(chunk)) != chunk:
                raise IntegrityError("current journal rewrote its preserved prefix")


def _before(value: JSONObject, root: Path, path: Path) -> None:
    fallback_fields = (
        {"source_snapshot_fallbacks"} if value.get("schema") == "s10-preservation-audit-v3" else set()
    )
    _schema(
        value,
        {
            "schema",
            "scope",
            "before_files",
            "old_evidence_files",
            "old_source_files",
            "old_receipt_sha256",
            "old_receipt_matches",
            "old_journal_prefix_preserved",
            "mismatches",
            "before_manifest",
            "predecessor_receipt",
            "prior_audit",
        }
        | fallback_fields,
        "preservation audit v2",
    )
    if (
        value["schema"] not in ("s10-preservation-audit-v2", "s10-preservation-audit-v3")
        or value["scope"] != "EXACT_PRIOR_SOURCE_AND_EVIDENCE_PRESERVATION"
        or value["mismatches"] != []
        or value["old_receipt_matches"] is not True
        or value["old_journal_prefix_preserved"] is not True
    ):
        raise IntegrityError("clean exact source/evidence preservation required")
    if (
        value["predecessor_receipt"] != OLD_REPAIR_RECEIPT
        or value["old_receipt_sha256"] != OLD_REPAIR_RECEIPT["sha256"]
    ):
        raise IntegrityError("fixed predecessor receipt identity differs")
    prior_path, _ = _reference(value["prior_audit"], root)
    if prior_path.parent != path.parent:
        raise IntegrityError("prior preservation audit must belong to this session")
    prior = read_object(value["prior_audit"], root)
    if prior != {
        key: item
        for key, item in value.items()
        if key
        not in {
            "schema",
            "before_manifest",
            "predecessor_receipt",
            "prior_audit",
            "source_snapshot_fallbacks",
        }
    }:
        raise IntegrityError("v2 audit rewrites prior preservation observations")
    old = read_object(value["predecessor_receipt"], root)
    snapshots = _snapshot_files(value, root, path.parent)
    source, evidence = _mapping(old["source_sha256"]), _mapping(old["evidence_sha256"])
    if len(source) != strict_int(value["old_source_files"], name="old source files", minimum=1) or len(
        evidence
    ) != strict_int(value["old_evidence_files"], name="old evidence files", minimum=1):
        raise IntegrityError("predecessor source/evidence counts differ")
    _source_preserved(source, snapshots, value.get("source_snapshot_fallbacks", {}), root)
    for name, identity in evidence.items():
        _hash(root, name, identity)
    _journal_prefix(root, path.parent, snapshots)


def _source_preserved(
    source: JSONObject, snapshots: JSONObject, fallback_value: JSONValue, root: Path
) -> None:
    fallbacks = object_json(fallback_value)
    missing = set(source) - set(snapshots)
    if set(fallbacks) != missing:
        raise IntegrityError("explicit archived fallbacks required for missing source snapshots")
    if any(snapshots[name] != source[name] for name in set(source) & set(snapshots)):
        raise IntegrityError("before snapshot does not preserve exact predecessor source")
    predecessor, _ = _reference(OLD_REPAIR_RECEIPT, root)
    for name, reference in fallbacks.items():
        relative = _relative(name)
        path, identity = _reference(reference, root)
        expected = predecessor.parent / "integrated_verified_snapshot" / relative
        if path != expected or identity != source[name]:
            raise IntegrityError("source fallback differs from exact verified predecessor snapshot")
        _hash(root, str(path.relative_to(root)), identity)


def audit_preservation(references: JSONValue, artifact_root: Path) -> None:
    """Revalidate both exact audits and every historical byte link before engineering PASS."""
    root = _plain_directory(artifact_root)
    refs = _list(references)
    if len(refs) != 2 or refs[0] == refs[1]:
        raise IntegrityError("exact historical and before preservation references required")
    seen: set[str] = set()
    for reference in refs:
        value = read_object(reference, root)
        path, _ = _reference(reference, root)
        scope = str(value.get("scope"))
        if scope in seen:
            raise IntegrityError("duplicate preservation audit scope")
        seen.add(scope)
        if scope == "STREAMING_IDENTITY_CHECK_ONLY_NO_NUMERICAL_OUTCOME_INSPECTION":
            _historical(value, root)
        elif scope == "EXACT_PRIOR_SOURCE_AND_EVIDENCE_PRESERVATION":
            _before(value, root, path)
        else:
            raise IntegrityError("unsupported preservation audit scope")
