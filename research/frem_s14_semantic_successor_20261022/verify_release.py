"""Static integrity verification for the immutable FREM-S14 0.2.0 release."""

import ast
import base64
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).resolve().parent
PACKAGE = ROOT / "successors/frem_s14"


def digest(value):
    if isinstance(value, (str, Path)):
        value = Path(value).read_bytes()
    return hashlib.sha256(value).hexdigest()


def source_identity():
    modules = {
        path.name: digest(path)
        for path in sorted((PACKAGE / "src/frem_s14").glob("*.py"))
    }
    return digest(json.dumps(modules, sort_keys=True, allow_nan=False).encode())


def main():
    release_path = PACKAGE / "dist/RELEASE-0.2.0.json"
    release = json.loads(release_path.read_text())
    assert release["package_version"] == "0.2.0"
    assert release["source_identity"] == source_identity()

    wheel = PACKAGE / "dist" / release["wheel"]
    archive = PACKAGE / "dist" / release["archive"]
    notebook = PACKAGE / "kaggle/FREM_S14_GPU_0_2_0.ipynb"
    metadata = json.loads((PACKAGE / "kaggle/kernel-metadata-0.2.0.json").read_text())
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
        modules = sorted(name for name in installed.namelist() if name.startswith("frem_s14/") and name.endswith(".py"))
        for name in modules:
            assert installed.read(name) == (PACKAGE / "src" / name).read_bytes()
            ast.parse(installed.read(name))

    with zipfile.ZipFile(archive) as bundle:
        inventory = json.loads(bundle.read("PAYLOAD.json"))
        assert set(bundle.namelist()) == set(inventory) | {"PAYLOAD.json"}
        # ``members`` is recorded after adding PAYLOAD.json; inventory lists its
        # hashed payload siblings only, to avoid a self-referential digest.
        assert len(inventory) + 1 == release["members"]
        for name, expected in inventory.items():
            payload = bundle.read(name)
            assert len(payload) == expected["bytes"]
            assert digest(payload) == expected["sha256"]
            if name.endswith(".py"):
                ast.parse(payload)

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

    result = {
        "status": "PASS_STATIC_RELEASE_VERIFICATION",
        "release": release,
        "wheel_modules_verified": len(modules),
        "payload_hashes_verified": len(inventory),
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
