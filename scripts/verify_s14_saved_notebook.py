"""Verify downloaded notebook source against the immutable S14 0.2.4 release.

No notebook code is executed. Source strings and arrays of source lines are
equivalent; all cell text, order and types must otherwise match exactly.
This verifies file content, not Kaggle ownership, privacy, version or runtime.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "successors/frem_s14"
RELEASE = PACKAGE / "dist/RELEASE-0.2.4.json"
NOTEBOOK = PACKAGE / "kaggle/FREM_S14_GPU_0_2_4.ipynb"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    result = json.loads(data, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("notebook must be a JSON object")
    return result


def normalized_cells(data: bytes) -> list[dict]:
    notebook = decode(data)
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("not a version-4 notebook")
    result = []
    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown", "raw"}:
            raise ValueError("invalid notebook cell")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("cell source must be text or text lines")
        result.append({"cell_type": cell["cell_type"], "source": source})
    if not result:
        raise ValueError("empty notebook")
    return result


def embedded_archive(cells: list[dict]) -> bytes:
    code = "\n".join(cell["source"] for cell in cells if cell["cell_type"] == "code")
    tree = ast.parse(code)
    nodes = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "RELEASE"
    ]
    if len(nodes) != 1:
        raise ValueError("exactly one RELEASE assignment required")
    node = nodes[0]
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Attribute)
        or not isinstance(node.func.value, ast.Name)
        or node.func.value.id != "base64"
        or node.func.attr != "b64decode"
        or len(node.args) != 1
    ):
        raise ValueError("RELEASE must be a literal base64 archive")
    return base64.b64decode(ast.literal_eval(node.args[0]), validate=True)


def compare(local: bytes, downloaded: bytes, release: dict, archive: bytes) -> dict:
    if release.get("package_version") != "0.2.4":
        raise ValueError("only the immutable 0.2.4 release is accepted")
    if digest(local) != release["notebook_sha256"]:
        raise ValueError("local notebook differs from trusted release record")
    if digest(archive) != release["archive_sha256"]:
        raise ValueError("local archive differs from trusted release record")
    expected = normalized_cells(local)
    observed = normalized_cells(downloaded)
    if expected != observed:
        raise ValueError("saved notebook cell content/order/types differ from release")
    if embedded_archive(observed) != archive:
        raise ValueError("saved embedded archive differs from trusted release")
    return {
        "schema": "frem-s14-saved-source-comparison-v1",
        "status": "PASS_SOURCE_CONTENT_MATCH",
        "package_version": "0.2.4",
        "source_identity": release["source_identity"],
        "archive_sha256": release["archive_sha256"],
        "wheel_sha256": release["wheel_sha256"],
        "local_notebook_sha256": digest(local),
        "downloaded_notebook_sha256": digest(downloaded),
        "normalized_cells_sha256": digest(json.dumps(expected, sort_keys=True).encode()),
        "cells_compared": len(expected),
        "normalization": "join source line arrays only; ignore cell outputs, IDs and metadata",
        "remote_identity_verified": False,
        "runtime_qualified": False,
        "scientific_status": "NOT_EVALUATED",
        "claim_boundary": "Content comparison only. Separate remote account/version/settings evidence and launch authorization are required.",
    }


def read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError(f"regular file with no symlink ancestors required: {path}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("notebook/release exceeds 16 MiB inspection limit")
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-notebook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("receipt exists; choose a new output")
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("output parent cannot be a symlink")
    release = decode(read_regular(RELEASE))
    archive = PACKAGE / "dist/frem_s14_kaggle_bundle-0.2.4.zip"
    result = compare(
        read_regular(NOTEBOOK), read_regular(args.saved_notebook), release, read_regular(archive)
    )
    result.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "saved_notebook_path": str(args.saved_notebook.absolute()),
            "release_record_sha256": digest(read_regular(RELEASE)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
