"""Static integrity verification for the immutable FREM-S14 0.2.3 release."""

import ast
import base64
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).resolve().parent
PACKAGE = ROOT / "successors/frem_s14"
VERSION = "0.2.3"


def digest(value):
    if isinstance(value, (str, Path)):
        value = Path(value).read_bytes()
    return hashlib.sha256(value).hexdigest()


def source_identity():
    modules = {path.name: digest(path) for path in sorted((PACKAGE / "src/frem_s14").glob("*.py"))}
    return digest(json.dumps(modules, sort_keys=True, allow_nan=False).encode())


def literal_assignment(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment: {name}")


def suite_seconds(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "suite_config":
            returned = next(child for child in ast.walk(node) if isinstance(child, ast.Return))
            assert isinstance(returned.value, ast.Call)
            for keyword in returned.value.keywords:
                if keyword.arg == "max_seconds":
                    return ast.literal_eval(keyword.value)
    raise AssertionError("suite_config max_seconds is missing")


def main():
    release_path = PACKAGE / f"dist/RELEASE-{VERSION}.json"
    release = json.loads(release_path.read_text())
    assert release["package_version"] == VERSION
    assert release["source_identity"] == source_identity()

    wheel = PACKAGE / "dist" / release["wheel"]
    archive = PACKAGE / "dist" / release["archive"]
    notebook = PACKAGE / f"kaggle/FREM_S14_GPU_{VERSION.replace('.', '_')}.ipynb"
    metadata = json.loads((PACKAGE / f"kaggle/kernel-metadata-{VERSION}.json").read_text())
    assert digest(wheel) == release["wheel_sha256"]
    assert digest(archive) == release["archive_sha256"]
    assert digest(notebook) == release["notebook_sha256"]
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [
        "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    ]

    with zipfile.ZipFile(wheel) as installed:
        modules = sorted(
            name
            for name in installed.namelist()
            if name.startswith("frem_s14/") and name.endswith(".py")
        )
        for name in modules:
            assert installed.read(name) == (PACKAGE / "src" / name).read_bytes()
            ast.parse(installed.read(name))
        assert literal_assignment(installed.read("frem_s14/__init__.py"), "__version__") == VERSION
        metadata_name = next(
            name for name in installed.namelist() if name.endswith(".dist-info/METADATA")
        )
        installed_metadata = installed.read(metadata_name).decode()
        assert f"Version: {VERSION}\n" in installed_metadata
        installed_suite_seconds = suite_seconds(installed.read("frem_s14/suite.py"))

    with zipfile.ZipFile(archive) as bundle:
        inventory = json.loads(bundle.read("PAYLOAD.json"))
        assert set(bundle.namelist()) == set(inventory) | {"PAYLOAD.json"}
        assert len(inventory) + 1 == release["members"]
        for name, expected in inventory.items():
            payload = bundle.read(name)
            assert len(payload) == expected["bytes"]
            assert digest(payload) == expected["sha256"]
            if name.endswith(".py"):
                ast.parse(payload)
        declared = json.loads(bundle.read("configs/historical_suite.json"))
        assert declared["config"]["max_seconds"] == installed_suite_seconds == 4000
        assert b"0.2.3 semantic-reproducibility release candidate" in bundle.read("README.md")
        assert b"Version 0.2.3 inherits" in bundle.read("SPECIFICATION.md")
        embedded_archive = archive.read_bytes()

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
    assert embedded == embedded_archive

    result = {
        "status": "PASS_STATIC_RELEASE_VERIFICATION",
        "release": release,
        "wheel_modules_verified": len(modules),
        "module_and_wheel_version_equal": True,
        "payload_hashes_verified": len(inventory),
        "fold_budget_contract_seconds": installed_suite_seconds,
        "embedded_release_equal": True,
        "notebook_syntax_verified": True,
        "private_gpu_internet_off": True,
        "runtime_qualified": False,
        "numeric_or_gpu_tests": "NOT_RUN_LOCAL_RESOURCE_GATE",
        "historical_model_cells_completed": 0,
        "protected_access": False,
    }
    destination = RECORD / "RELEASE_VERIFICATION.json"
    if destination.exists():
        raise FileExistsError("release verification already exists; do not overwrite evidence")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
