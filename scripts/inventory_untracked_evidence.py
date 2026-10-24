"""Create a path-complete inventory of retained untracked FREM-S14 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

GROUPS = {
    "research/frem_s14_fixture_release_20261021": {
        "provenance": "FREM-S14 fixture-release and Kaggle access session",
        "owner_run": "fixture release and queue-readiness work",
        "date": "2026-10-21",
        "sensitivity": "internal research evidence; screenshots may show account UI but no credential values are intended",
        "disposition": "preserve locally; review receipts individually before any later version-control inclusion",
    },
    "research/frem_s14_kaggle_0_1_8_20261021": {
        "provenance": "FREM-S14 0.1.8 Version 6 execution, failure, publication, and token-lifecycle evidence",
        "owner_run": "private Kaggle Version 6 and publication reconciliation",
        "date": "2026-10-21",
        "sensitivity": "internal research evidence; contains private-run screenshots and token-action receipts, not an authorized credential payload",
        "disposition": "preserve locally; do not bulk-stage; select only independently reviewed receipts",
    },
    "research/frem_s14_kaggle_0_1_9_version8_20261021": {
        "provenance": "downloaded private Kaggle Version 8 output and independent semantic-verification work",
        "owner_run": "private Kaggle Version 8 terminal semantic failure",
        "date": "2026-10-21",
        "sensitivity": "large internal execution evidence with model artifacts and private-run logs; no public redistribution decision",
        "disposition": "preserve locally outside ordinary Git; never overwrite or bulk-stage",
    },
    "research/frem_s14_version5_failure_20261021": {
        "provenance": "FREM-S14 Version 5 pre-qualification fixture failure",
        "owner_run": "private Kaggle Version 5",
        "date": "2026-10-21",
        "sensitivity": "internal failure evidence and screenshots",
        "disposition": "preserve locally; review before selective receipt inclusion",
    },
    "research/frem_s14_version8_verification_20261021T113801Z": {
        "provenance": "separately retained Version 8 verification snapshot",
        "owner_run": "independent Version 8 verification",
        "date": "2026-10-21",
        "sensitivity": "internal verification evidence",
        "disposition": "preserve locally; review before selective receipt inclusion",
    },
}

INTENDED_TRACKED = {
    "PROJECT_EXECUTION_MEGA_CHECKLIST_20261024.md",
    "research/frem_s14_execution_20261024/KAGGLE_ACCESS_READY_RECEIPT.json",
    "research/frem_s14_execution_20261024/SESSION_RECEIPT.json",
    "scripts/inventory_untracked_evidence.py",
    "research/frem_s14_execution_20261024/GATE0_UNTRACKED_INVENTORY.json",
}


def git_untracked(root: Path) -> list[str]:
    completed = subprocess.run(
        ["rtk", "proxy", "git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(item.decode("utf-8", "surrogateescape") for item in completed.stdout.split(b"\0") if item)


def regular_path(root: Path, relative: str) -> Path:
    """Reject aliases and nonregular evidence before opening any bytes."""
    name = PurePosixPath(relative)
    if not relative or name.is_absolute() or ".." in name.parts or str(name) != relative:
        raise ValueError(f"unsafe evidence path: {relative}")
    current = root
    for component in name.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"symlink in evidence path: {relative}")
    if not stat.S_ISREG(current.stat().st_mode):
        raise ValueError(f"nonregular evidence path: {relative}")
    return current


def file_record(root: Path, relative: str, group: str) -> dict:
    candidate = regular_path(root, relative)
    before = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = candidate.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"evidence changed during hashing: {relative}")
    return {"path": relative, "bytes": after.st_size, "sha256": digest.hexdigest(), "group": group}


def inspect_entries(root: Path, paths: list[str], baseline: dict | None = None) -> list[dict]:
    prior = {}
    if baseline is not None:
        for entry in baseline["entries"]:
            relative = entry["path"]
            if relative in prior:
                raise ValueError(f"duplicate baseline path: {relative}")
            prior[relative] = entry
        if len(prior) != baseline["path_count"]:
            raise ValueError("baseline path_count differs from entries")
        paths = sorted(prior)
    entries = []
    for relative in paths:
        group = next((prefix for prefix in GROUPS if relative.startswith(prefix + "/")), None)
        if group is None:
            raise ValueError(f"unclassified untracked path: {relative}")
        entry = file_record(root, relative, group)
        if relative in prior:
            previous = prior[relative]
            if previous["group"] != group or previous["bytes"] != entry["bytes"]:
                raise ValueError(f"baseline metadata differs: {relative}")
            if "sha256" in previous and previous["sha256"] != entry["sha256"]:
                raise ValueError(f"baseline digest differs: {relative}")
        entries.append(entry)
    return entries


def new_output(path: Path) -> Path:
    # Do not resolve symlinks: doing so would conceal the evidence alias.
    path = Path(os.path.abspath(path))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"receipt exists; use a new output: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"symlink in output parent: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline", help="Recheck only a retained inventory, reporting new untracked paths separately"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = new_output(root / args.output)
    paths = [path for path in git_untracked(root) if path not in INTENDED_TRACKED]
    baseline_path = Path(args.baseline) if args.baseline else None
    baseline_bytes = baseline_path.read_bytes() if baseline_path else None
    baseline = json.loads(baseline_bytes) if baseline_bytes is not None else None
    entries = inspect_entries(root, paths, baseline)
    covered = {entry["path"] for entry in entries}
    counts: Counter[str] = Counter()
    bytes_by_group: Counter[str] = Counter()
    for entry in entries:
        counts[entry["group"]] += 1
        bytes_by_group[entry["group"]] += entry["bytes"]

    result = {
        "schema": "frem-s14-gate0-untracked-inventory-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "branch": subprocess.run(
            ["rtk", "proxy", "git", "branch", "--show-current"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "head": subprocess.run(
            ["rtk", "proxy", "git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "scope": "Baseline entries only"
        if baseline
        else "Every untracked path except the declared intended tracked files",
        "baseline": str(baseline_path) if baseline_path else None,
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest() if baseline_bytes is not None else None,
        "baseline_content_hashes_verified": sum("sha256" in entry for entry in baseline["entries"])
        if baseline
        else 0,
        "new_hashes_establish_current_identity_only": baseline is not None
        and any("sha256" not in entry for entry in baseline["entries"]),
        "untracked_paths_outside_baseline": sorted(set(paths) - covered) if baseline else [],
        "intended_tracked_exclusions": sorted(INTENDED_TRACKED),
        "groups": {
            group: {**metadata, "file_count": counts[group], "regular_file_bytes": bytes_by_group[group]}
            for group, metadata in GROUPS.items()
        },
        "path_count": len(entries),
        "all_paths_classified": len(entries) == sum(counts.values()),
        "bulk_stage_authorized": False,
        "credential_values_collected": False,
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {key: result[key] for key in ("path_count", "all_paths_classified", "bulk_stage_authorized")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
