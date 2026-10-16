"""Create immutable source inventories; hashing files does not certify tests or research."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path

EXCLUDED_PARTS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "build", "dist", "tmp", ".local_tmp", ".deps", ".mplconfig",
    "data", "outputs", "runs",  # Data and execution evidence have separate manifests.
}
EXCLUDED_SUFFIXES = (".egg-info", ".pyc", ".pyo")


def file_record(path: Path, scope: Path) -> dict:
    """Hash in bounded memory and reject a file that changes while being read."""
    if path.is_symlink():
        raise ValueError(f"Symlink is not a release file: {path}")
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError(f"File changed during inventory: {path}")
    return {"path": path.relative_to(scope).as_posix(), "bytes": after.st_size, "sha256": digest}


def excluded(path: Path, manifest: Path) -> bool:
    return (
        path == manifest
        or path.name == "RELEASE_MANIFEST.json"
        or any(part in EXCLUDED_PARTS or part.endswith(EXCLUDED_SUFFIXES) for part in path.parts)
        or any(part.startswith(".") and part != ".gitignore" for part in path.parts)
    )


def inventory(
    scope: Path, manifest: Path, *, root_release: bool, output_dir: Path | None = None
) -> list[dict]:
    scope = scope.resolve()
    excluded_manifest = manifest.relative_to(scope) if manifest.is_relative_to(scope) else Path()
    files = []
    # Prune before descending: never walk virtualenvs, protected datasets,
    # raw outputs or preserved legacy repositories for a source inventory.
    for directory, names, filenames in os.walk(scope, followlinks=False):
        directory = Path(directory)
        kept = []
        for name in sorted(names):
            path = directory / name
            relative = path.relative_to(scope)
            if excluded(relative, excluded_manifest):
                continue
            if root_release and relative.parts[0] in {"legacy", "successors"}:
                continue
            if output_dir is not None and path == output_dir:
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink directory is not a release source: {path}")
            kept.append(name)
        names[:] = kept
        for name in sorted(filenames):
            path = directory / name
            if not excluded(path.relative_to(scope), excluded_manifest):
                files.append(file_record(path, scope))
    return sorted(files, key=lambda item: item["path"])


def build_artifacts(scope: Path) -> list[dict]:
    dist = scope / "dist"
    if not dist.is_dir():
        return []
    if dist.is_symlink():
        raise ValueError("Build artifact directory must not be a symlink")
    return [file_record(path, scope) for path in sorted(dist.iterdir())
            if path.is_file() and path.suffix in {".whl", ".gz"}]


def build_manifest(scope: Path, *, root_release: bool, output_dir: Path) -> dict:
    scope = scope.resolve()
    project = tomllib.loads((scope / "pyproject.toml").read_text())["project"]
    files = inventory(scope, scope / "RELEASE_MANIFEST.json", root_release=root_release,
                      output_dir=output_dir.resolve())
    if not any(item["path"].startswith("src/") and item["path"].endswith(".py") for item in files):
        raise ValueError(f"No Python source in package: {scope}")
    old = scope / "RELEASE_MANIFEST.json"
    return {
        "schema_version": 2,
        "project": project["name"],
        "version": project["version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "release_status": "SOURCE_INVENTORY_ONLY",
        "scope": "Package source and documentation; raw data, run outputs and caches excluded",
        "verification": {
            "tests": "NOT_RUN_BY_MANIFEST_BUILDER",
            "lint": "NOT_RUN_BY_MANIFEST_BUILDER",
            "package_build": "NOT_RUN_BY_MANIFEST_BUILDER",
            "scientific_status": "NOT_ASSESSED_BY_MANIFEST_BUILDER",
        },
        "previous_manifest": file_record(old, scope) if old.is_file() else None,
        "build_artifacts": build_artifacts(scope),
        "build_artifact_scope": "Existing file identities only; freshness and source correspondence not verified",
        "file_count_excluding_manifest": len(files),
        "hash_convention": "SHA-256 of exact bytes; snapshot output directory and prior manifests excluded",
        "files": files,
    }


def verify_manifest(scope: Path, manifest: dict) -> dict:
    """Verify listed identities, including historical v1 manifests; never parse outcomes."""
    scope = scope.resolve()
    errors = []
    files = manifest["files"]
    if not files:
        errors.append({"type": "empty_inventory"})
    if manifest.get("file_count_excluding_manifest") != len(files):
        errors.append({"type": "count_mismatch"})
    seen = set()
    prior = [manifest["previous_manifest"]] if manifest.get("previous_manifest") else []
    for entry in [*files, *manifest.get("build_artifacts", []), *prior]:
        name = entry["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Manifest path escapes package: {name}")
        if name in seen:
            errors.append({"path": name, "type": "duplicate_path"})
            continue
        seen.add(name)
        path = scope / relative
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != scope):
            raise ValueError(f"Manifest path traverses symlink: {name}")
        if not path.is_file():
            errors.append({"path": name, "type": "missing_file"})
        elif file_record(path, scope) != entry:
            errors.append({"path": name, "type": "identity_mismatch"})
    return {"status": "PASS" if not errors else "FAIL", "entries_checked": len(seen),
            "errors": errors, "scope": "Listed byte identities only; no runtime or scientific verification"}


def write_snapshots(root: Path, output_dir: Path) -> list[Path]:
    root, output_dir = root.resolve(), output_dir.resolve()
    if output_dir == root or root.is_relative_to(output_dir):
        raise ValueError("Output directory must not contain the source tree")
    # Existing outputs are immutable, including partial failed snapshots.
    output_dir.mkdir(parents=True, exist_ok=False)
    scopes = [(root, "FREM.json", True),
              (root / "successors/rejepa_v1", "RE-JEPA.json", False),
              (root / "successors/reng_jepa_v01", "RENG-JEPA.json", False)]
    written = []
    for scope, name, root_release in scopes:
        payload = build_manifest(scope, root_release=root_release, output_dir=output_dir)
        path = output_dir / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path, help="New directory for all three source snapshots")
    action.add_argument("--verify", type=Path, help="Check a manifest against its package --root")
    args = parser.parse_args()
    if args.verify:
        result = verify_manifest(args.root, json.loads(args.verify.read_text()))
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    paths = write_snapshots(args.root, args.output_dir)
    print(json.dumps({"status": "SOURCE_INVENTORIES_CREATED", "paths": [str(p) for p in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
