"""Verify the pinned S14 handoff bytes without importing numerical libraries.

This portable verifier certifies packaging only, never CUDA or scientific gates.
"""

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

NOTEBOOK = "FREM_S14_GPU_0_2_4.ipynb"
ARCHIVE = "frem_s14_kaggle_bundle-0.2.4.zip"
EXPECTED_NOTEBOOK = "e8fb05b52ee26f6a3045280e92bde247f85b381b7702bbc2fce4f4d97755670f"
EXPECTED_ARCHIVE = "ddc04e93b37a086684624a2ad8d03de9269df1fa7f23ed14b9fcc8b2943f7c21"
EXPECTED_WHEEL = "57ae8e68b83eef921d83d58f2976f040a13281f3c505dfe6c98c7c4eb9d96486"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    """Reject ambiguous or nonstandard JSON in the portable trust records."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject_constant(value)
        return result

    return json.loads(
        data, object_pairs_hook=unique_object,
        parse_constant=reject_constant, parse_float=finite_float,
    )


def read(path):
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError("regular file without symlink ancestors required")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("handoff member exceeds inspection bound")
    return path.read_bytes()


def validate(directory):
    directory = Path(directory).absolute()
    notebook, archive = read(directory / NOTEBOOK), read(directory / ARCHIVE)
    if digest(notebook) != EXPECTED_NOTEBOOK or digest(archive) != EXPECTED_ARCHIVE:
        raise ValueError("pinned notebook or archive checksum mismatch")
    release = strict_json(read(directory / "RELEASE-0.2.4.json"))
    if (release["notebook_sha256"], release["archive_sha256"], release["wheel_sha256"]) != (
        EXPECTED_NOTEBOOK,
        EXPECTED_ARCHIVE,
        EXPECTED_WHEEL,
    ) or release["package_version"] != "0.2.4":
        raise ValueError("release metadata differs from pinned identity")
    metadata = strict_json(read(directory / "kernel-metadata-0.2.4.json"))
    expected = {
        "id": "ryangomez010/frem-s14-gpu-benchmark-20261019",
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": ["borismarjanovic/price-volume-data-for-all-us-stocks-etfs"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    for key, value in expected.items():
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise ValueError(f"target metadata differs: {key}")
    cells = strict_json(notebook)["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    tree = ast.parse(code)
    assignments = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    if ast.literal_eval(assignments["MODE"]) != "historical_suite":
        raise ValueError("wrong workload mode")
    if base64.b64decode(ast.literal_eval(assignments["RELEASE"].args[0]), validate=True) != archive:
        raise ValueError("embedded archive differs")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        payload = strict_json(bundle.read("PAYLOAD.json"))
        if len(bundle.namelist()) != len(payload) + 1 or set(bundle.namelist()) != set(payload) | {
            "PAYLOAD.json"
        }:
            raise ValueError("payload inventory mismatch")
        for name, spec in payload.items():
            data = bundle.read(name)
            if len(data) != spec["bytes"] or digest(data) != spec["sha256"]:
                raise ValueError(f"payload bytes differ: {name}")
        wheel = bundle.read("frem_s14-0.2.4-py3-none-any.whl")
        if digest(wheel) != EXPECTED_WHEEL:
            raise ValueError("wheel differs")
        with zipfile.ZipFile(io.BytesIO(wheel)) as installed:
            intake = ast.parse(installed.read("frem_s14/intake.py"))
            source_hashes = next(
                ast.literal_eval(n.value)
                for n in intake.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "SOURCE_SHA256"
            )
    inputs = strict_json(read(directory / "input_manifest.json"))
    assets = inputs["assets"]
    if (
        not isinstance(assets, list)
        or any(not isinstance(asset, str) for asset in assets)
        or len(assets) != len(source_hashes)
        or inputs["sha256"] != source_hashes
        or set(assets) != set(source_hashes)
    ):
        raise ValueError("input manifest differs from frozen member allowlist")
    return {
        "status": "PASS_STATIC_PACKAGE_ONLY",
        "notebook_sha256": EXPECTED_NOTEBOOK,
        "archive_sha256": EXPECTED_ARCHIVE,
        "wheel_sha256": EXPECTED_WHEEL,
        "payload_members": len(payload),
        "dataset_members": len(source_hashes),
        "runtime_verified": False,
        "manual_launch_ready": False,
        "scientific_status": "NOT_EVALUATED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("receipt exists")
    if any(p.is_symlink() for p in args.output.absolute().parents):
        raise ValueError("receipt ancestors must not be symlinks")
    result = validate(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
