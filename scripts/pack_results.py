"""Archive one stopped Kaggle run without editing it; preserve failed evidence.

Stdlib only. A checksum inventory proves transport equality, not scientific
validity. Run the separately anchored S14 archive and semantic verifiers next.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def regular_tree(root: Path) -> list[Path]:
    root = root.absolute()
    if root.is_symlink() or not root.is_dir() or any(p.is_symlink() for p in root.parents):
        raise ValueError("run must be a real directory without symlink ancestors")
    files = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            files.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"symlink or special file forbidden: {path}")
    if not files:
        raise ValueError("empty run directory")
    return files


def fingerprint(path: Path) -> dict:
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "bytes": size}


def pack(run: Path, output: Path) -> dict:
    run, output = run.absolute(), output.absolute()
    receipt = Path(str(output) + ".receipt.json")
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing overwrite: {path}")
        if any(p.is_symlink() for p in path.parents):
            raise ValueError("output ancestors must not be symlinks")
        if path.is_relative_to(run):
            raise ValueError("archive and receipt must be outside the run directory")
    files = regular_tree(run)
    before = {str(p.relative_to(run)): fingerprint(p) for p in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "frem-result-transport-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_name": run.name,
        "status": "IN_PROGRESS",
        "scientific_validity": "NOT_ASSESSED",
        "files": before,
    }
    # Reserve the receipt exclusively before creating the archive. On failure
    # retain both it and any partial ZIP so the attempt cannot disappear.
    with receipt.open("x", encoding="utf-8") as log:
        try:
            with zipfile.ZipFile(
                output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
            ) as archive:
                for path in files:
                    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                        raise ValueError("input type changed during packing")
                    archive.write(path, arcname=f"{run.name}/{path.relative_to(run).as_posix()}")
            after = {str(p.relative_to(run)): fingerprint(p) for p in regular_tree(run)}
            if before != after:
                raise ValueError("run changed during packing; stop execution and retain this failure")
            with zipfile.ZipFile(output) as archive:
                wanted = {f"{run.name}/{Path(k).as_posix()}": v for k, v in before.items()}
                if len(archive.namelist()) != len(wanted) or set(archive.namelist()) != set(wanted):
                    raise ValueError("archive inventory mismatch")
                for name, expected in wanted.items():
                    h, size = hashlib.sha256(), 0
                    with archive.open(name) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(block)
                            size += len(block)
                    if {"sha256": h.hexdigest(), "bytes": size} != expected:
                        raise ValueError(f"archive bytes mismatch: {name}")
            record.update(status="PASS_TRANSPORT_ONLY", archive=fingerprint(output))
        except BaseException as error:
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            json.dump(record, log, indent=2, sort_keys=True, allow_nan=False)
            log.write("\n")
            log.flush()
            os.fsync(log.fileno())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = pack(args.run, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "files": len(result["files"]),
                "archive": result["archive"],
                "scientific_validity": "NOT_ASSESSED",
            }
        )
    )


if __name__ == "__main__":
    main()
