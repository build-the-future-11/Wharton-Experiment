"""Retain a source comparison without executing the changing candidate."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path.cwd()
DEST = Path(__file__).resolve().parent
BASE = ROOT / "research/frem_s10_production_20261016"
PACKAGE = ROOT / "successors/frem_s10"
EXPECTED = "11a0e935a9e0c8ba208b520704f3c6b75180273a32ca069b0f9d817a9b960696"
raw_manifest = (ROOT / f"research/frem_s10_objects/{EXPECTED}.json").read_bytes()
assert hashlib.sha256(raw_manifest).hexdigest() == EXPECTED
frozen = {row["path"]: row["sha256"] for row in json.loads(raw_manifest)["files"]}
paths = sorted([*(PACKAGE / "src/frem_s10").rglob("*.py"), *(PACKAGE / "tests").rglob("*.py"),
    *(PACKAGE / name for name in ("pyproject.toml", "uv.lock", "README.md", "MANIFEST.in", "src/frem_s10/science.json", "src/frem_s10/py.typed"))])
observed = {}
timestamps = {}
unstable_reads = []
for path in paths:
    assert path.is_file() and not path.is_symlink()
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    name = path.relative_to(ROOT).as_posix()
    observed[name] = hashlib.sha256(payload).hexdigest()
    timestamps[name] = datetime.fromtimestamp(after.st_mtime, UTC).isoformat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        unstable_reads.append(name)
    saved = DEST / "observed_source" / name
    saved.parent.mkdir(parents=True, exist_ok=True)
    with saved.open("xb") as stream:
        stream.write(payload)
archived = {name: hashlib.sha256((BASE / "candidate_source" / name).read_bytes()).hexdigest()
            for name in frozen}
assert archived == frozen
added = [{"path": name, "sha256": observed[name], "mtime_utc": timestamps[name]}
         for name in sorted(observed.keys() - frozen.keys())]
changed = [{"path": name, "expected": frozen[name], "observed": observed[name], "mtime_utc": timestamps[name]}
           for name in sorted(observed.keys() & frozen.keys()) if observed[name] != frozen[name]]
missing = sorted(frozen.keys() - observed.keys())
changed_again = [name for name, identity in observed.items()
                 if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != identity]
result = {
    "schema": "s10-source-drift-observation-v1",
    "observed_at": datetime.now(UTC).isoformat(),
    "scope": "RETAINED_SOURCE_COMPARISON_NOT_NEW_CANDIDATE_FREEZE",
    "expected_code_identity": EXPECTED,
    "frozen_file_count": len(frozen), "observed_file_count": len(observed),
    "added": added, "changed": changed, "missing": missing,
    "unstable_during_read": unstable_reads, "changed_again_by_end_of_capture": changed_again,
    "original_candidate_source_matches_all_frozen_files": True,
    "files": observed,
    "controller_rejection_was_earlier_than_this_observation": True,
    "revision_context": "README describes Kaggle portable engineering preparation and explicitly states that the earlier production plan is no longer a current-source plan.",
    "scientific_execution": False,
}
with (DEST / "SOURCE_DRIFT.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps({key: result[key] for key in ("observed_at", "frozen_file_count", "observed_file_count", "added", "changed", "missing", "unstable_during_read", "changed_again_by_end_of_capture", "original_candidate_source_matches_all_frozen_files")}, indent=2))
