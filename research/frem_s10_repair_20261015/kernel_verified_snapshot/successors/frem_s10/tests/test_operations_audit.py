"""Adversarial metadata and durable-evidence cases; no market observations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from frem_s10.cli import main, plan
from frem_s10.errors import (
    ArtifactIOError,
    ContractError,
    ExecutionStateError,
    IntegrityError,
    ResourceLimitError,
    S10Error,
)
from frem_s10.identities import JSONValue, parse_json, scientific_digest
from frem_s10.runtime.artifacts import publish_json, read_verified_json
from frem_s10.runtime.consumption import consume
from frem_s10.runtime.resources import MemorySample, assess, parse_vm_stat

SCIENCE = Path(__file__).parents[1] / "src/frem_s10/science.json"


def test_typed_identity_has_no_scalar_object_or_type_collision() -> None:
    values: list[JSONValue] = [
        1.0,
        1,
        True,
        "1",
        ["binary64", "0x1.0000000000000p+0"],
        {"binary64": "0x1.0000000000000p+0"},
        None,
        "null",
    ]
    assert len({scientific_digest(v) for v in values}) == len(values)
    assert scientific_digest({"a": 2, "b": 3}) == scientific_digest({"b": 3, "a": 2})
    assert scientific_digest(0.0) != scientific_digest(-0.0)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "[] trailing"])
def test_json_rejects_ambiguous_or_nonfinite_text(text: str) -> None:
    with pytest.raises(ContractError):
        parse_json(text)


def test_publish_is_immutable_verified_and_corruption_is_rejected(tmp_path: Path) -> None:
    digest = publish_json(tmp_path, "receipt.json", {"state": "FAILED"})
    before = (tmp_path / "receipt.json").read_bytes()
    with pytest.raises(ArtifactIOError):
        publish_json(tmp_path, "receipt.json", {"state": "COMPLETE"})
    assert (tmp_path / "receipt.json").read_bytes() == before
    assert read_verified_json(tmp_path / "receipt.json", digest) == {"state": "FAILED"}
    (tmp_path / "receipt.json").write_bytes(b"{}")
    with pytest.raises(IntegrityError):
        read_verified_json(tmp_path / "receipt.json", digest)
    assert not list(tmp_path.glob(".pending-*"))


@pytest.mark.parametrize("name", ["../escape.json", "/escape.json", "..", ""])
def test_publish_rejects_traversal(tmp_path: Path, name: str) -> None:
    with pytest.raises(ArtifactIOError):
        publish_json(tmp_path, name, {})


def test_publish_and_read_reject_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ArtifactIOError):
        publish_json(link, "receipt.json", {})
    publish_json(target, "source.json", {})
    (target / "link.json").symlink_to(target / "source.json")
    with pytest.raises(IntegrityError):
        read_verified_json(target / "link.json", "x")


def test_fault_before_atomic_publication_keeps_no_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise OSError("simulated publication fault")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(ArtifactIOError):
        publish_json(tmp_path, "result.json", {"success": True})
    assert not (tmp_path / "result.json").exists()
    assert not list(tmp_path.glob(".pending-*"))


def test_memory_threshold_pageouts_and_required_counters() -> None:
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 16000.\nPages speculative: 16768.\nPageouts: 5.\n"
    sample = parse_vm_stat(text, "2026-10-15T00:00:00Z")
    assert sample.available_mib == 512
    assert assess((sample, sample, sample))["status"] == "PASS"
    low = MemorySample(sample.timestamp, 16384, 16000, 16767, 5)
    growing = MemorySample(sample.timestamp, 16384, 16000, 16768, 6)
    assert assess((sample, low, sample))["status"] == "BLOCKED"
    assert assess((sample, sample, growing))["status"] == "BLOCKED"
    with pytest.raises(ResourceLimitError):
        parse_vm_stat(text.replace("Pageouts", "Pageins"), "t")
    with pytest.raises(ResourceLimitError):
        parse_vm_stat(text + "Pageouts: 5.\n", "t")


def test_plan_enumerates_complete_unique_matrix_without_claiming_run() -> None:
    result = plan(SCIENCE)
    assert result["planned_ledgers"] == 192
    cells = result["cells"]
    assert isinstance(cells, list)
    assert len(set(cells)) == 192
    assert sum("/strategy/" in cell for cell in cells) == 128
    assert sum("/benchmark/" in cell for cell in cells) == 48
    assert sum("/ablation/" in cell for cell in cells) == 16
    assert result["execution_available"] is False
    assert result["market_ledgers"] == 0


@pytest.mark.parametrize("value", [True, 3, 1.0])
def test_plan_rejects_even_equal_python_boolean_or_numeric_type_changes(
    tmp_path: Path, value: int | float | bool
) -> None:
    data = json.loads(SCIENCE.read_text())
    data["schema_version"] = value
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(data))
    with pytest.raises(S10Error):
        plan(path)


def test_cli_receipt_cannot_overwrite_and_has_no_run_route(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    args = ["plan", "--manifest", str(SCIENCE), "--output", str(output)]
    assert main(args) == 0
    assert main(args) == 2
    with pytest.raises(SystemExit) as error:
        main(["run"])
    assert error.value.code == 2


def test_metadata_import_does_not_import_numpy_or_predecessor() -> None:
    code = (
        "import sys; import frem_s10.cli; assert 'numpy' not in sys.modules; assert 'frem' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_consumption_is_per_operation_persistent_and_cannot_be_reused(tmp_path: Path) -> None:
    freeze_id = "a" * 64
    consume(tmp_path, operation="precision", freeze_id=freeze_id, attempt_id="first")
    with pytest.raises(ExecutionStateError):
        consume(tmp_path, operation="precision", freeze_id=freeze_id, attempt_id="retry")
    assert json.loads((tmp_path / f"precision-{freeze_id}.json").read_text())["attempt_id"] == "first"
    consume(tmp_path, operation="campaign", freeze_id=freeze_id, attempt_id="different-operation")
    assert len(list(tmp_path.glob("*.json"))) == 2
    with pytest.raises(ContractError):
        consume(tmp_path, operation="resume", freeze_id=freeze_id, attempt_id="x")
