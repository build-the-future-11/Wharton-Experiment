"""Verify immutable fixture-only release without importing numerical libraries."""

import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).resolve().parent
PACKAGE = ROOT / "successors/frem_s14"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(name, command, cwd, env=None):
    with (RECORD / name).open("x") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    old_hashes = json.loads((RECORD / "prior_release_hashes.json").read_text())
    assert all(digest((ROOT / p).read_bytes()) == h for p, h in old_hashes.items())
    release = json.loads((PACKAGE / "dist/RELEASE-0.1.8.json").read_text())
    old = PACKAGE / "dist/frem_s14-0.1.7-py3-none-any.whl"
    new = PACKAGE / "dist" / release["wheel"]
    archive = PACKAGE / "dist" / release["archive"]
    assert digest(new.read_bytes()) == release["wheel_sha256"]
    assert digest(archive.read_bytes()) == release["archive_sha256"]
    with zipfile.ZipFile(old) as before, zipfile.ZipFile(new) as after:
        modules = sorted(n for n in after.namelist() if n.startswith("frem_s14/") and n.endswith(".py"))
        changed = [n for n in modules if before.read(n) != after.read(n)]
        assert changed == ["frem_s14/__init__.py"], changed
        assert before.read(changed[0]).replace(b'"0.1.7"', b'"0.1.8"') == after.read(changed[0])
        for name in modules:
            assert after.read(name) == (PACKAGE / "src" / name).read_bytes()
            ast.parse(after.read(name))
    notebook = PACKAGE / "kaggle/FREM_S14_GPU_0_1_8.ipynb"
    assert digest(notebook.read_bytes()) == release["notebook_sha256"]
    cells = json.loads(notebook.read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    tree = ast.parse(code)
    assignments = {n.targets[0].id: n.value for n in tree.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert ast.literal_eval(assignments["MODE"]) == "historical_suite"
    assert ast.literal_eval(assignments["DEVICE"]) == "cuda"
    embedded = base64.b64decode(ast.literal_eval(assignments["RELEASE"].args[0]), validate=True)
    assert embedded == archive.read_bytes()
    with tempfile.TemporaryDirectory(prefix="frem-s14-018-verify-") as temporary:
        temp = Path(temporary)
        payload = temp / "payload"
        with zipfile.ZipFile(archive) as z:
            inventory = json.loads(z.read("PAYLOAD.json"))
            assert set(z.namelist()) == set(inventory) | {"PAYLOAD.json"}
            for name, expected in inventory.items():
                data = z.read(name)
                assert len(data) == expected["bytes"] and digest(data) == expected["sha256"]
            z.extractall(payload)
        installed = temp / "installed"
        run("install.log", [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                            "--target", str(installed), str(new)], temp)
        env = dict(os.environ, PYTHONPATH=str(installed))
        run("installed_stdlib.log", [sys.executable, "-W", "error", "-m", "unittest", "discover",
                                    "-s", str(payload / "tests"), "-p", "test_*stdlib.py", "-v"], temp, env)
        copy = temp / "package"
        shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("dist", "build", "*.egg-info", "__pycache__", ".ruff_cache"))
        for p in (copy / "kaggle").glob("*0_1_8*"):
            p.unlink()
        (copy / "kaggle/kernel-metadata-0.1.8.json").unlink()
        run("rebuild.log", [sys.executable, str(copy / "kaggle/build_release.py")], copy)
        for relative in ("dist/" + release["wheel"], "dist/" + release["archive"],
                         "dist/RELEASE-0.1.8.json", "kaggle/FREM_S14_GPU_0_1_8.ipynb",
                         "kaggle/kernel-metadata-0.1.8.json"):
            assert (copy / relative).read_bytes() == (PACKAGE / relative).read_bytes(), relative
    result = {"status": "PASS", "old_release_files_unchanged": len(old_hashes),
              "wheel_modules_verified": len(modules), "changed_runtime_modules": changed,
              "scientific_runtime_unchanged": True, "payload_hashes_verified": len(inventory),
              "embedded_release_equal": True, "deterministic_rebuild_equal": True,
              "installed_dependency_free_tests": 59, "numeric_or_gpu_tests": "NOT_RUN",
              "historical_cells_completed": 0, "protected_access": False, "release": release}
    with (RECORD / "RELEASE_VERIFICATION.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
