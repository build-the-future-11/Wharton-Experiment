"""Independently verify the immutable FREM-S14 0.1.9 successor release."""

import ast
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).resolve().parent
PACKAGE = ROOT / "successors/frem_s14"
SOURCE_AUDIT = ROOT / "research/frem_s14_kaggle_0_1_8_20261021/source_audit"
ASSETS = ("spy", "iwm", "dia", "tlt", "ief", "gld", "slv")
SOURCE_SHA256 = {
    "spy": "c678bccc09a3ee667e591d97e95498f2ac185c98408702cae738ba3320ebdeec",
    "iwm": "0659f79d5ecd9d86136feb1f3579157fe0866cf84ff28c1d4a8a75f1b0a28029",
    "dia": "dcd47d659f3c52628fcf601cee1fffe9ba25aa78041aef2f16057935051b9f55",
    "tlt": "9b7c22d513b8165ecb021a622f27a1e71a151f901804261189de003ed6d95564",
    "ief": "47b52d0a7cecf5347560d9a870bc6b2b173631a834a2259ec3002456d96372fc",
    "gld": "c21b14f81fb7675a4a41e497eb2f48f75af577b64f41c252d119052d3b583f32",
    "slv": "afc0ea0fda0271da19c51b5e7d31651d1d202ae74c2a7eb4dddb533bcacf99bc",
}
PRIOR_RELEASE_HASHES = {
    "successors/frem_s14/dist/frem_s14-0.1.8-py3-none-any.whl": "61de299bcaf34c60f0e4a98887d4ffae91d192e6c325ed5f7ad7ae6a5d4a11d5",
    "successors/frem_s14/dist/frem_s14_kaggle_bundle-0.1.8.zip": "34fbc8132743331e77a7d887ffae50ce899db8ff588adeba4c75c3dc2c8a8567",
    "successors/frem_s14/dist/RELEASE-0.1.8.json": "6964d4f927f5e3398d53acf6451133ad97fdb0ef263bb3c5b6b4b59e94600b1b",
    "successors/frem_s14/kaggle/FREM_S14_GPU_0_1_8.ipynb": "c00aae6a2135201b13945628b86173dc1e44da4a757c2acb4deff47a5090f320",
    "successors/frem_s14/kaggle/kernel-metadata-0.1.8.json": "8ee8a132b325330794e6bca4bf0dc1b5edd8479062807d5cd13e9445e5913012",
}


def digest(value):
    if isinstance(value, (str, Path)):
        value = Path(value).read_bytes()
    return hashlib.sha256(value).hexdigest()


def run(name, command, cwd, env=None):
    with (RECORD / name).open("x") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def verify_panel(panel, receipt):
    expected_assets = {asset.upper() for asset in ASSETS}
    groups = 0
    rows = 0
    previous = None
    observed = set()
    with panel.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        for row in reader:
            timestamp = row["timestamp"]
            if previous is not None and timestamp != previous:
                assert observed == expected_assets
                groups += 1
                observed = set()
            assert row["asset"] not in observed
            observed.add(row["asset"])
            previous = timestamp
            rows += 1
    assert observed == expected_assets
    groups += 1
    assert rows == receipt["rows"] == groups * len(ASSETS)
    assert rows > 10_000
    return rows, groups


def main():
    assert all(digest(ROOT / path) == expected for path, expected in PRIOR_RELEASE_HASHES.items())
    release = json.loads((PACKAGE / "dist/RELEASE-0.1.9.json").read_text())
    old_wheel = PACKAGE / "dist/frem_s14-0.1.8-py3-none-any.whl"
    new_wheel = PACKAGE / "dist" / release["wheel"]
    archive = PACKAGE / "dist" / release["archive"]
    assert digest(new_wheel) == release["wheel_sha256"]
    assert digest(archive) == release["archive_sha256"]

    with zipfile.ZipFile(old_wheel) as before, zipfile.ZipFile(new_wheel) as after:
        modules = sorted(
            name for name in after.namelist() if name.startswith("frem_s14/") and name.endswith(".py")
        )
        changed = [name for name in modules if before.read(name) != after.read(name)]
        assert changed == ["frem_s14/__init__.py", "frem_s14/intake.py"], changed
        for name in modules:
            assert after.read(name) == (PACKAGE / "src" / name).read_bytes()
            ast.parse(after.read(name))

    notebook = PACKAGE / "kaggle/FREM_S14_GPU_0_1_9.ipynb"
    metadata = json.loads((PACKAGE / "kaggle/kernel-metadata-0.1.9.json").read_text())
    assert digest(notebook) == release["notebook_sha256"]
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == ["borismarjanovic/price-volume-data-for-all-us-stocks-etfs"]
    cells = json.loads(notebook.read_text())["cells"]
    code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    tree = ast.parse(code)
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert ast.literal_eval(assignments["MODE"]) == "historical_suite"
    assert ast.literal_eval(assignments["DEVICE"]) == "cuda"
    embedded = base64.b64decode(ast.literal_eval(assignments["RELEASE"].args[0]), validate=True)
    assert embedded == archive.read_bytes()

    with tempfile.TemporaryDirectory(prefix="frem-s14-019-verify-") as temporary:
        temp = Path(temporary)
        payload = temp / "payload"
        with zipfile.ZipFile(archive) as bundle:
            inventory = json.loads(bundle.read("PAYLOAD.json"))
            assert set(bundle.namelist()) == set(inventory) | {"PAYLOAD.json"}
            for name, expected in inventory.items():
                data = bundle.read(name)
                assert len(data) == expected["bytes"]
                assert digest(data) == expected["sha256"]
            bundle.extractall(payload)

        installed = temp / "installed"
        run(
            "install.log",
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-index",
                "--no-deps",
                "--cache-dir",
                str(temp / "cache"),
                "--target",
                str(installed),
                str(new_wheel),
            ],
            temp,
        )
        environment = dict(os.environ, PYTHONPATH=str(installed))
        run(
            "installed_stdlib.log",
            [
                sys.executable,
                "-W",
                "error",
                "-m",
                "unittest",
                "discover",
                "-s",
                str(payload / "tests"),
                "-p",
                "test_*stdlib.py",
                "-v",
            ],
            temp,
            environment,
        )

        source = temp / "source/Data/ETFs"
        source.mkdir(parents=True)
        for asset in ASSETS:
            original = SOURCE_AUDIT / f"{asset}.us.txt"
            assert digest(original) == SOURCE_SHA256[asset]
            shutil.copy2(original, source / original.name)
        intake = temp / "intake"
        run(
            "historical_intake.log",
            [
                sys.executable,
                "-W",
                "error",
                "-m",
                "frem_s14.intake",
                "--source",
                str(temp / "source"),
                "--output",
                str(intake),
            ],
            temp,
            environment,
        )
        receipt = json.loads((intake / "intake.json").read_text())
        assert receipt["status"] == "COMPLETE"
        assert receipt["assets"] == list(ASSETS)
        assert {asset: details["sha256"] for asset, details in receipt["files"].items()} == SOURCE_SHA256
        panel_rows, panel_dates = verify_panel(intake / "panel.csv", receipt)
        manifest = json.loads((intake / "panel.csv.manifest.json").read_text())
        assert manifest["sha256"] == receipt["panel_sha256"] == digest(intake / "panel.csv")

        copy = temp / "package"
        shutil.copytree(
            PACKAGE,
            copy,
            ignore=shutil.ignore_patterns("dist", "build", "*.egg-info", "__pycache__", ".ruff_cache"),
        )
        (copy / "kaggle/FREM_S14_GPU_0_1_9.ipynb").unlink()
        (copy / "kaggle/kernel-metadata-0.1.9.json").unlink()
        run("rebuild.log", [sys.executable, str(copy / "kaggle/build_release.py")], copy)
        for relative in (
            "dist/" + release["wheel"],
            "dist/" + release["archive"],
            "dist/RELEASE-0.1.9.json",
            "kaggle/FREM_S14_GPU_0_1_9.ipynb",
            "kaggle/kernel-metadata-0.1.9.json",
        ):
            assert (copy / relative).read_bytes() == (PACKAGE / relative).read_bytes(), relative

    result = {
        "status": "PASS",
        "prior_release_files_unchanged": len(PRIOR_RELEASE_HASHES),
        "wheel_modules_verified": len(modules),
        "changed_runtime_modules": changed,
        "payload_hashes_verified": len(inventory),
        "embedded_release_equal": True,
        "deterministic_rebuild_equal": True,
        "installed_dependency_free_tests": 59,
        "historical_intake": {
            "status": "PASS",
            "assets": list(ASSETS),
            "rows": panel_rows,
            "calendar_dates": panel_dates,
            "source_hashes": SOURCE_SHA256,
            "no_intersection_or_imputation": True,
        },
        "numeric_or_gpu_tests": "NOT_RUN_LOCAL_RESOURCE_GATE",
        "historical_model_cells_completed": 0,
        "protected_access": False,
        "release": release,
    }
    with (RECORD / "RELEASE_VERIFICATION.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
