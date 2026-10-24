#!/usr/bin/env python3
"""Create a path-complete inventory of retained untracked FREM-S14 evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    paths = [path for path in git_untracked(root) if path not in INTENDED_TRACKED]
    entries = []
    counts: Counter[str] = Counter()
    bytes_by_group: Counter[str] = Counter()
    for relative in paths:
        group = next(
            (prefix for prefix in GROUPS if relative == prefix or relative.startswith(prefix + "/")), None
        )
        if group is None:
            raise SystemExit(f"unclassified untracked path: {relative}")
        candidate = root / relative
        size = candidate.stat().st_size
        entries.append({"path": relative, "bytes": size, "group": group})
        counts[group] += 1
        bytes_by_group[group] += size

    result = {
        "schema": "frem-s14-gate0-untracked-inventory-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
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
        "scope": "Every pre-existing untracked path after excluding the explicitly intended tracked files listed below.",
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
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("path_count", "all_paths_classified", "bulk_stage_authorized")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
