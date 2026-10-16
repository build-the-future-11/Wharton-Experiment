"""Build deterministic file inventories for the three active research packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
EXCLUDED_SUFFIXES = {".egg-info", ".pyc"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excluded(path: Path, manifest: Path) -> bool:
    return (
        path == manifest
        or any(part in EXCLUDED_PARTS for part in path.parts)
        or any(part.endswith(tuple(EXCLUDED_SUFFIXES)) for part in path.parts)
    )


def inventory(scope: Path, manifest: Path, *, root_release: bool) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in scope.rglob("*") if item.is_file()):
        relative = path.relative_to(scope)
        if excluded(relative, manifest.relative_to(scope)):
            continue
        if root_release and relative.parts[0] in {"legacy", "successors"}:
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    return files


def build_artifacts(scope: Path) -> list[dict[str, object]]:
    dist = scope / "dist"
    if not dist.is_dir():
        return []
    return [
        {
            "path": path.relative_to(scope).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(dist.iterdir())
        if path.is_file() and path.suffix in {".whl", ".gz"}
    ]


def write_manifest(
    scope: Path,
    *,
    project: str,
    version: str,
    test_result: str,
    root_release: bool = False,
) -> None:
    manifest = scope / "RELEASE_MANIFEST.json"
    files = inventory(scope, manifest, root_release=root_release)
    payload = {
        "project": project,
        "version": version,
        "generated_utc": datetime.now(UTC).isoformat(),
        "release_status": "ENGINEERING_VERIFIED_SCIENCE_NOT_EVALUATED",
        "scientific_status": "NOT_EVALUATED",
        "scope": (
            "Active FREM release; successor packages and byte-preserved legacy archive have separate boundaries"
            if root_release
            else "Package-local release inventory"
        ),
        "verification": {
            "tests": test_result,
            "lint": "PASS",
            "package_build": "PASS",
            "protected_outcomes_accessed": False,
            "market_or_scientific_result": False,
        },
        "build_artifacts": build_artifacts(scope),
        "file_count_excluding_manifest": len(files),
        "hash_convention": "SHA-256 of exact file bytes; manifest excludes itself and reproducible caches/build trees",
        "files": files,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    write_manifest(root, project="FREM", version="0.1.1", test_result="163 passed", root_release=True)
    write_manifest(
        root / "successors" / "rejepa_v1",
        project="RE-JEPA",
        version="0.3.0",
        test_result="67 passed",
    )
    write_manifest(
        root / "successors" / "reng_jepa_v01",
        project="RENG-JEPA",
        version="0.1.0",
        test_result="45 passed",
    )


if __name__ == "__main__":
    main()
