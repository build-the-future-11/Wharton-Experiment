"""Reproducible offline release archive and hash-bound Kaggle notebook."""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hash_bytes(data):
    return sha256(data).hexdigest()


def cell(code=None, markdown=None):
    if markdown is not None:
        return {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(True)}
    return {
        "cell_type": "code",
        "metadata": {},
        "source": code.splitlines(True),
        "execution_count": None,
        "outputs": [],
    }


def main():
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        environment = {**os.environ, "SOURCE_DATE_EPOCH": "1792368000", "PYTHONHASHSEED": "0"}
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", temporary],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        wheel = next(Path(temporary).glob("*.whl"))
        members = {wheel.name: wheel.read_bytes()}
        for pattern in ("tests/*.py", "configs/*.json", "*.md"):
            for path in sorted(ROOT.glob(pattern)):
                members[str(path.relative_to(ROOT))] = path.read_bytes()
        inventory = {
            name: {"sha256": hash_bytes(data), "bytes": len(data)} for name, data in members.items()
        }
        members["PAYLOAD.json"] = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
        archive = destination / "frem_s13_kaggle_bundle.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, data in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 19, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, data)
        (destination / wheel.name).write_bytes(wheel.read_bytes())
    archive_hash = hash_bytes(archive.read_bytes())
    setup = """import hashlib, importlib.metadata, json, os, pathlib, subprocess, sys, time, zipfile
from packaging.requirements import Requirement
from packaging.version import Version
MODE = "qualify"
DEVICE = "cuda"
if MODE not in {"qualify", "development"}:
    raise ValueError("MODE must be qualify or development")
EXPECTED = "ARCHIVE_SHA"
matches = [p for p in pathlib.Path("/kaggle/input").rglob("frem_s13_kaggle_bundle.zip") if p.is_file() and not p.is_symlink()]
if len(matches) != 1 or hashlib.sha256(matches[0].read_bytes()).hexdigest() != EXPECTED:
    raise RuntimeError("Attach exactly the hash-verified S13 release ZIP")
RUN = pathlib.Path("/kaggle/working") / ("frem-s13-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
RUN.mkdir(exist_ok=False)
PAYLOAD = RUN / "payload"
PAYLOAD.mkdir()
with zipfile.ZipFile(matches[0]) as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("duplicate archive members")
    inventory = json.loads(archive.read("PAYLOAD.json"))
    if set(names) != set(inventory) | {"PAYLOAD.json"}:
        raise ValueError("payload inventory mismatch")
    for name, expected in inventory.items():
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe archive member")
        data = archive.read(name)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("payload hash mismatch")
        target = PAYLOAD / name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(data)
INSTALL = RUN / "installed"
wheel = next(PAYLOAD.glob("*.whl"))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(INSTALL), str(wheel)], check=True)
if not (3, 11) <= sys.version_info[:2] < (3, 14):
    raise RuntimeError("Python 3.11-3.13 required")
dist = next(importlib.metadata.distributions(path=[str(INSTALL)]))
for spec in dist.requires or []:
    requirement = Requirement(spec)
    if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
        continue
    version = importlib.metadata.version(requirement.name)
    if Version(version) not in requirement.specifier:
        raise RuntimeError(f"Incompatible installed dependency: {spec}; observed {version}")
ENV = dict(os.environ, PYTHONPATH=str(INSTALL), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
           FREM_S13_TEST_DEVICE=DEVICE, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
(RUN / "delivery.json").write_text(json.dumps({"archive_sha256": EXPECTED, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "mode": MODE}, indent=2))
def run(name, args, timeout):
    log = RUN / (name + ".log")
    started = time.monotonic()
    with log.open("x") as stream:
        try:
            result = subprocess.run([sys.executable, *args], env=ENV, cwd=PAYLOAD, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    (RUN / (name + ".json")).write_text(json.dumps({"returncode": code, "wall_seconds": time.monotonic()-started, "command": args}, indent=2))
    print(log.read_text()[-16000:])
    if code:
        raise RuntimeError(f"{name} failed ({code}); retained {log}")
print("Verified release and isolated install:", RUN)
""".replace("ARCHIVE_SHA", archive_hash)
    test = """run("preflight", ["-m", "frem_s13", "preflight", "--output", str(RUN/"host-admission.json")], 30)
run("tests", ["-m", "pytest", str(PAYLOAD/"tests"), "-W", "error", "-q"], 1800)
run("qualify", ["-W", "error", "-m", "frem_s13", "qualify", "--device", DEVICE, "--output", str(RUN/"qualification")], 1900)
run("verify_qualification", ["-m", "frem_s13", "verify", "--bundle", str(RUN/"qualification")], 300)
"""
    train = """if MODE == "development":
    panels = [p for p in pathlib.Path("/kaggle/input").rglob("panel.csv") if p.is_file() and not p.is_symlink()]
    if len(panels) != 1 or not pathlib.Path(str(panels[0])+".manifest.json").is_file():
        raise RuntimeError("Attach one audited panel.csv and its manifest")
    run("development", ["-W", "error", "-m", "frem_s13", "run", "--config", str(PAYLOAD/"configs/development.json"), "--data", str(panels[0]), "--qualification", str(RUN/"qualification"), "--output", str(RUN/"development")], 14500)
    run("verify_development", ["-m", "frem_s13", "verify", "--bundle", str(RUN/"development")], 600)
else:
    print("Generated qualification only; no market dataset opened.")
print("Retain/download all files under", RUN)
"""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            cell(
                markdown="# FREM-S13 GPU qualification and development\n\nDefault: all generated tests and 15-arm qualification. No market data is used unless MODE is explicitly set to development. Attach the exact release ZIP. Existing CUDA dependencies must satisfy the wheel requirements. Qualification is not a scientific result.\n"
            ),
            cell(code=setup),
            cell(code=test),
            cell(code=train),
        ],
    }
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"s13-{index}"
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"notebook-cell-{index}", "exec")
    notebook_path = ROOT / "kaggle/FREM_S13_GPU.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    receipt = {
        "archive": archive.name,
        "archive_sha256": archive_hash,
        "wheel": wheel.name,
        "wheel_sha256": hash_bytes((destination / wheel.name).read_bytes()),
        "notebook_sha256": hash_bytes(notebook_path.read_bytes()),
        "members": len(members),
        "runtime_qualified": False,
        "status": "PACKAGED_NOT_NUMERICALLY_QUALIFIED",
    }
    (destination / "RELEASE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
