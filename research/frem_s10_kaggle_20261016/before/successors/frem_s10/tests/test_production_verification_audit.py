"""Adversarial finite provenance checks without recursive test or scientific execution."""

import importlib.machinery
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject
from frem_s10.runtime import provenance, verification
from frem_s10.runtime.declarations import BOUND_ROLES, DESIGN


def _package(root: Path) -> Path:
    package = root / provenance.PACKAGE
    (package / "src/frem_s10").mkdir(parents=True)
    (package / "tests").mkdir()
    for name, contents in {
        "src/frem_s10/__init__.py": '"""Generated package fixture."""\n',
        "src/frem_s10/operator.py": "VALUE = 1\n",
        "tests/test_one.py": "def test_one():\n    assert True\n",
        "pyproject.toml": '[project]\nname = "frem-s10"\nversion = "0.1.0"\n',
        "README.md": "Generated package fixture.\n",
        "MANIFEST.in": "recursive-include tests *.py\ninclude uv.lock\n",
        "uv.lock": 'version = 1\n[[package]]\nname="numpy"\nversion="2.5.4"\n'
        '[[package]]\nname="pytest"\nversion="8.4.2"\n',
        "src/frem_s10/science.json": "{}\n",
        "src/frem_s10/py.typed": "",
    }.items():
        (package / name).write_text(contents)
    return package


def test_code_inventory_detects_modified_added_and_omitted_python(tmp_path: Path) -> None:
    package = _package(tmp_path)
    ref = provenance.create_code_manifest(tmp_path, tmp_path)
    provenance.audit_code(ref, artifact_root=tmp_path, source_root=tmp_path)
    operator = package / "src/frem_s10/operator.py"
    original = operator.read_bytes()
    operator.write_text("VALUE = 2\n")
    with pytest.raises(IntegrityError):
        provenance.audit_code(ref, artifact_root=tmp_path, source_root=tmp_path)
    operator.write_bytes(original)
    extra = package / "src/frem_s10/extra.py"
    extra.write_text("VALUE = 3\n")
    with pytest.raises(IntegrityError):
        provenance.audit_code(ref, artifact_root=tmp_path, source_root=tmp_path)
    extra.unlink()
    operator.unlink()
    with pytest.raises(IntegrityError):
        provenance.audit_code(ref, artifact_root=tmp_path, source_root=tmp_path)


@pytest.mark.parametrize("suffix", [importlib.machinery.EXTENSION_SUFFIXES[0], ".pyc"])
def test_importable_nonpython_member_cannot_escape_code_inventory(tmp_path: Path, suffix: str) -> None:
    package = _package(tmp_path)
    ref = provenance.create_code_manifest(tmp_path, tmp_path)
    # This is inert text. Its importable filename must still invalidate the freeze.
    (package / f"src/frem_s10/operator{suffix}").write_bytes(b"never executed native/bytecode fixture")
    with pytest.raises(IntegrityError):
        provenance.audit_code(ref, artifact_root=tmp_path, source_root=tmp_path)


@dataclass(frozen=True)
class _Distribution:
    metadata: dict[str, str]
    version: str


@dataclass(frozen=True)
class _LocatedDistribution(_Distribution):
    root: Path

    def locate_file(self, path: str) -> Path:
        return self.root / path


def test_editable_build_metadata_does_not_change_observed_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    (package / "uv.lock").write_text('version=1\n[[package]]\nname="frem-s10"\nversion="0.1.0"\n')
    installed_root = Path(sys.executable).absolute().parent.parent / "lib/python3.12/site-packages"
    installed = _LocatedDistribution({"Name": "frem-s10"}, "0.1.0", installed_root)
    built = _LocatedDistribution({"Name": "frem-s10"}, "0.1.0", package / "src")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [installed])
    before = provenance._installed_versions(tmp_path)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [installed, built])
    assert provenance._installed_versions(tmp_path) == before == [("frem-s10", "0.1.0")]
    unbound = _LocatedDistribution({"Name": "frem-s10"}, "0.1.0", tmp_path / "unbound")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [installed, unbound])
    with pytest.raises(IntegrityError, match="duplicate"):
        provenance._installed_versions(tmp_path)


def test_environment_rejects_non_numpy_locked_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    spec = importlib.machinery.ModuleSpec(
        "frem_s10", loader=None, origin=str(package / "src/frem_s10/__init__.py")
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [_Distribution({"Name": "numpy"}, "2.5.4"), _Distribution({"Name": "pytest"}, "999.0")],
    )
    with pytest.raises(IntegrityError):
        provenance.observe_environment(tmp_path)


def test_environment_receipt_cannot_claim_another_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: JSONObject = {"schema": "generated-environment-fixture", "identity": "one"}
    monkeypatch.setattr(provenance, "observe_environment", lambda root: observed)
    ref = provenance.create_environment_receipt(tmp_path, tmp_path, evidence_scope="GENERATED_ENGINEERING")
    provenance.audit_environment(
        ref, artifact_root=tmp_path, source_root=tmp_path, evidence_scope="GENERATED_ENGINEERING"
    )
    with pytest.raises(IntegrityError, match="declaration mismatch"):
        provenance.audit_environment(
            ref, artifact_root=tmp_path, source_root=tmp_path, evidence_scope="RETROSPECTIVE_DEVELOPMENT"
        )
    changed: JSONObject = {"schema": "generated-environment-fixture", "identity": "two"}
    monkeypatch.setattr(provenance, "observe_environment", lambda root: changed)
    with pytest.raises(IntegrityError, match="differ"):
        provenance.audit_environment(
            ref, artifact_root=tmp_path, source_root=tmp_path, evidence_scope="GENERATED_ENGINEERING"
        )


def test_structural_v1_pass_never_certifies_production_verification(tmp_path: Path) -> None:
    bindings: JSONObject = {role: str(index) * 64 for index, role in enumerate(BOUND_ROLES)}
    old = provenance.save_object(
        tmp_path,
        {
            "schema": "s10-test-result-v1",
            "status": "PASS",
            "code_sha256": bindings["code"],
            "environment_sha256": bindings["environment"],
        },
    )
    ref = provenance.save_object(
        tmp_path,
        {
            "schema": "s10-verification-receipt-v1",
            "design_id": DESIGN,
            "operation": "prepare-precision",
            "evidence_scope": "RETROSPECTIVE_DEVELOPMENT",
            "bindings": bindings,
            "status": "PASS",
            "logs": [old],
        },
    )
    with pytest.raises((ContractError, IntegrityError)):
        verification.audit_verification(ref, artifact_root=tmp_path, source_root=tmp_path)


def test_fixed_command_inventory_includes_every_source_and_test_scope(tmp_path: Path) -> None:
    commands = dict(verification.verification_commands(tmp_path, tmp_path / "checks"))
    assert tuple(commands) == (
        "collection",
        "tests",
        "lint",
        "format",
        "types",
        "packaging",
        "mutations",
        "boundary_probe",
    )
    assert "--collect-only" in commands["collection"]
    assert commands["tests"][commands["tests"].index("-W") + 1] == "error"
    for name in ("lint", "format", "types"):
        assert commands[name][-2:] == ("successors/frem_s10/src/frem_s10", "successors/frem_s10/tests")
    assert "--check" in commands["format"]
    env = verification._environment()
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "PYTHONPATH" not in env
    assert all(env[key] == "1" for key in provenance.THREAD_ENV)


def test_controller_imports_without_io_mutation_network_threads_or_numpy(tmp_path: Path) -> None:
    script = """import sys
blocked = {'socket.connect', 'socket.bind', 'subprocess.Popen', 'os.mkdir', 'os.remove', 'os.rename'}
def audit(event, args):
    if event in blocked:
        raise AssertionError('import-time side effect: ' + event)
sys.addaudithook(audit)
import frem_s10.runtime.provenance
import frem_s10.runtime.verification
import frem_s10.runtime.qualification
import frem_s10.runtime.qualification_worker
import frem_s10.runtime.pipeline
import frem_s10.runtime.worker
import frem_s10.runtime.receipts
assert 'numpy' not in sys.modules
assert not any(name.startswith('frem_s10.model.') or name.startswith('frem_s10.experiment.') for name in sys.modules)
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
