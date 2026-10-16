"""Generated metadata faults for the exact-candidate mutation verifier; no nested runs."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from frem_s10.errors import IntegrityError, ResourceLimitError
from frem_s10.identities import JSONObject, checked_json, object_json
from frem_s10.runtime import mutation_check as mutation
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.provenance import PACKAGE

NODE = "tests/test_generated.py::test_fixed_assertion"


def _observation(target: Path, *, mutant: bool) -> JSONObject:
    return {
        "schema": "s10-mutation-observations-v1",
        "origin": str(target / "src/frem_s10/__init__.py"),
        "collected": [NODE],
        "phases": {NODE: {"setup": "passed", "call": "failed" if mutant else "passed", "teardown": "passed"}},
        "exceptions": {NODE: {"call": {"type": "builtins.AssertionError", "message": "expected"}}}
        if mutant
        else {},
        "exit_code": int(mutant),
    }


@pytest.mark.parametrize("mutant", [False, True])
def test_mutation_observation_contract(tmp_path: Path, mutant: bool) -> None:
    value = _observation(tmp_path, mutant=mutant)
    mutation._observations(value, tmp_path, (NODE,), mutant=mutant)
    if mutant:
        value["exceptions"] = {
            NODE: {"call": {"type": "_pytest.outcomes.Failed", "message": "DID NOT RAISE expected"}}
        }
        mutation._observations(value, tmp_path, (NODE,), mutant=True)


@pytest.mark.parametrize(
    "fault", ["import", "setup", "teardown", "missing_node", "wrong_origin", "skip", "exit", "empty", "xfail"]
)
def test_nonassertion_or_wrong_phase_does_not_kill(tmp_path: Path, fault: str) -> None:
    value = _observation(tmp_path, mutant=True)
    phases = object_json(object_json(value["phases"])[NODE])
    if fault == "import":
        value["exceptions"] = {NODE: {"call": {"type": "builtins.ImportError", "message": "missing"}}}
    elif fault in {"setup", "teardown", "skip"}:
        phases["call" if fault == "skip" else fault] = "skipped" if fault == "skip" else "failed"
    elif fault == "missing_node":
        value["collected"] = []
    elif fault == "wrong_origin":
        value["origin"] = str(tmp_path / "original/__init__.py")
    elif fault == "exit":
        value["exit_code"] = 2
    elif fault == "empty":
        value["exceptions"] = {}
    else:
        value["exceptions"] = {NODE: {"call": {"type": "_pytest.outcomes.Failed", "message": "XFAIL"}}}
    with pytest.raises(IntegrityError):
        mutation._observations(value, tmp_path, (NODE,), mutant=True)


@pytest.mark.parametrize("fault", ["changed", "extra", "missing", "symlink"])
def test_isolated_copy_detects_tampering(tmp_path: Path, fault: str) -> None:
    source, target = tmp_path / "source", tmp_path / "copy"
    file = source / PACKAGE / "src/frem_s10/__init__.py"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"VALUE = 1\n")
    inventory: JSONObject = {"src/frem_s10/__init__.py": hashlib.sha256(file.read_bytes()).hexdigest()}
    mutation._copy(source, target, inventory)
    for name in ("command.json", "observations.json", "pytest.log"):
        (target / name).write_text("generated metadata")
    mutation._audit_copy(target, inventory)
    copied = target / "src/frem_s10/__init__.py"
    if fault == "changed":
        copied.write_text("VALUE = 2\n")
    elif fault == "extra":
        (target / "injected.py").write_text("VALUE = 2\n")
    elif fault == "missing":
        copied.unlink()
    else:
        copied.unlink()
        copied.symlink_to(file)
    with pytest.raises(IntegrityError):
        mutation._audit_copy(target, inventory)
    assert file.read_bytes() == b"VALUE = 1\n"


def test_mutation_refuses_host_before_test_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    monkeypatch.setattr(
        mutation, "check_host", lambda: {"status": "FAIL", "reason": "generated host refusal"}
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("test dispatch occurred before resource admission")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ResourceLimitError):
        mutation.run_mutation_check(source, output)
    assert {path.name for path in output.iterdir()} == {"host.json", "failure.json"}
    assert mutation._small(output / "failure.json")["error_type"] == "ResourceLimitError"


@pytest.mark.parametrize("elapsed", [-1, 601, True, "1"])
def test_mutation_budget_is_fixed_and_fail_closed(elapsed: object) -> None:
    value: JSONObject = {"budget_seconds": 600.0, "wall_seconds": 1.0}
    if isinstance(elapsed, (int, str)):
        value["wall_seconds"] = elapsed
    with pytest.raises(IntegrityError):
        mutation._audit_budget(value)


def test_reviewed_mutation_anchor_requires_exact_unique_source(tmp_path: Path) -> None:
    assert checked_json(mutation._recipes()) == mutation._recipes()
    assert len(mutation.MUTANTS) == 11
    file = tmp_path / PACKAGE / "src/frem_s10/generated.py"
    file.parent.mkdir(parents=True)
    file.write_text("VALUE = 1\n")
    recipe = mutation.Mutant("generated", "generated.py", "VALUE = 1", "VALUE = 2", NODE)
    assert mutation._mutated(tmp_path, recipe) == b"VALUE = 2\n"
    file.write_text("VALUE = 1\nVALUE = 1\n")
    with pytest.raises(IntegrityError, match="not unique"):
        mutation._mutated(tmp_path, recipe)


def test_rehashed_mutation_receipt_cannot_omit_fixed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mutation, "_inventory", lambda source: {})
    host = publish_json(tmp_path, "host.json", {"status": "PASS"})
    publish_json(
        tmp_path,
        "receipt.json",
        {
            "schema": "s10-semantic-mutants-v2",
            "status": "PASS",
            "scope": "GENERATED_ENGINEERING",
            "scientific_support_claimed": False,
            "parent_resource_and_network_proof_required": True,
            "source_files": {},
            "mutations": [],
            "host_sha256": host,
            "runs": [],
            "wall_seconds": 1.0,
            "budget_seconds": 600.0,
        },
    )
    with pytest.raises(IntegrityError):
        mutation.audit_mutation_receipt(tmp_path, tmp_path)
