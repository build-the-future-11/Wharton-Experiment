"""Build the reviewed two-phase Kaggle production notebook."""

from __future__ import annotations

import json
from pathlib import Path


def code(source: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Wharton FREM-S10 Kaggle production contract v1\n",
            "Phase 1 provisions the exact environment while Internet is enabled. Phase 2 must run only after the Kaggle Internet setting is disabled. Scientific values remain AES-encrypted until current verification passes. A failed or consumed scientific stage is terminal.\n",
        ],
    },
    code(
        '''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, tomllib, urllib.request, zipfile

BUNDLE_SHA256 = "851b1cb12a1091f21fb45bcabc9fef490e35d26202702cc30c18c313bbc94a42"
ROOT = Path("/kaggle/working/frem-s10-retrospective-20261017")
UPLOAD = Path("/kaggle/working/frem-s10-production-upload")
matches = list(Path("/kaggle/input").rglob("s10-kaggle-production-bundle.zip"))
assert len(matches) == 1, matches
BUNDLE = matches[0]
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == BUNDLE_SHA256
assert not ROOT.exists() and not UPLOAD.exists()
with zipfile.ZipFile(BUNDLE) as archive:
    archive.extractall(UPLOAD)
PAYLOAD = UPLOAD / "payload"
SOURCE = ROOT / "source"
SOURCE.parent.mkdir(parents=True)
shutil.copytree(PAYLOAD / "successors", SOURCE / "successors")
runtime = ROOT / "runtime"
env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "UV_", "PIP_")) and k != "VIRTUAL_ENV"}
env.update(UV_PROJECT_ENVIRONMENT=str(runtime), UV_NO_CONFIG="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
uv = shutil.which("uv")
assert uv
project = PAYLOAD / "successors/frem_s10"
subprocess.run([uv, "sync", "--frozen", "--project", str(project), "--no-install-project", "--group", "quality", "--group", "build", "--python", "3.12"], check=True, env=env, timeout=300)
python = runtime / "bin/python"
subprocess.run([uv, "pip", "install", "--python", str(python), "--no-deps", "-e", str(SOURCE / "successors/frem_s10")], check=True, env=env, timeout=120)
lock = tomllib.loads((SOURCE / "successors/frem_s10/uv.lock").read_text())
numpy = next(row for row in lock["package"] if row["name"] == "numpy" and row["version"] == "2.5.4")
wheel = next(row for row in numpy["wheels"] if "cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" in row["url"])
wheelhouse = SOURCE / "research/frem_s10_wheelhouse"
wheelhouse.mkdir(parents=True)
target = wheelhouse / Path(wheel["url"]).name
urllib.request.urlretrieve(wheel["url"], target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == wheel["hash"].removeprefix("sha256:")
metadata = subprocess.check_output([str(python), "-I", "-B", "-c", "import json,sys,importlib.metadata;print(json.dumps({'python':sys.version,'numpy':importlib.metadata.version('numpy')}))"], text=True)
print(metadata)
print(json.dumps({"stage": "ENVIRONMENT_READY", "bundle": str(BUNDLE), "root": str(ROOT), "instruction": "Disable Internet in Kaggle Settings before running the next cell."}))
'''
    ),
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Required boundary transition\n", "Disable **Internet** in Kaggle Settings. Keep accelerator set to **None**. Then run the next cell. The live boundary probe will refuse execution if networking remains available.\n"],
    },
    code(
        '''import os, subprocess
contract_env = dict(os.environ, FREM_S10_EXECUTION_CONTRACT="s10-kaggle-production-v1", FREM_S10_SEALED_INPUT=str(BUNDLE), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
verification = subprocess.run([str(python), "-B", str(PAYLOAD / "production_runner.py"), "verify", "--root", str(ROOT), "--payload", str(PAYLOAD), "--sealed", str(BUNDLE)], env=contract_env, text=True, capture_output=True, timeout=1800)
print(verification.stdout)
print(verification.stderr)
verification.check_returncode()
'''
    ),
    code(
        '''import subprocess
SCIENTIFIC_KEY = "827a829a38165e9889d89109904bfab44c569c9a9adecc0cef286281b187b3cd"
execution = subprocess.run([str(python), "-B", str(PAYLOAD / "production_runner.py"), "execute", "--root", str(ROOT), "--payload", str(PAYLOAD), "--key", SCIENTIFIC_KEY], env=contract_env, text=True, capture_output=True, timeout=7200)
print(execution.stdout)
print(execution.stderr)
execution.check_returncode()
'''
    ),
    code(
        '''import hashlib, json, tarfile
terminal = json.loads((ROOT / "state-terminal.json").read_text())
archive = Path("/kaggle/working/frem-s10-retrospective-20261017-results.tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    tar.add(ROOT, arcname=ROOT.name)
print(json.dumps({"terminal_stage": terminal["stage"], "archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "scientific_attempt_consumed": terminal["scientific_attempt_consumed"]}, sort_keys=True))
'''
    ),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("tools/kaggle/FREM_S10_KAGGLE_PRODUCTION.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
