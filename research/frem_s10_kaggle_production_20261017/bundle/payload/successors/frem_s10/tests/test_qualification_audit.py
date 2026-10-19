"""Small qualification metadata/fault checks; no generated model campaign executes."""

import hashlib
import json
import sys
from pathlib import Path
from typing import BinaryIO

import pytest

from frem_s10.errors import IntegrityError, ResourceLimitError
from frem_s10.identities import JSONObject, JSONValue
from frem_s10.runtime import qualification
from frem_s10.runtime.journal import Journal
from frem_s10.runtime.provenance import save_object


def _inventory_fixture(root: Path) -> tuple[JSONObject, Path]:
    directory = root / "archive"
    directory.mkdir()
    child = directory / "generation.json"
    child.write_bytes(b"original payload")
    from frem_s10.runtime.artifacts import publish_json

    digest = publish_json(
        directory,
        "manifest.json",
        {"files": [{"name": child.name, "sha256": hashlib.sha256(child.read_bytes()).hexdigest()}]},
    )
    return {"path": "archive/manifest.json", "sha256": digest}, child


def test_controller_inventory_does_not_read_payload_but_deep_audit_detects_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, child = _inventory_fixture(tmp_path)
    child.write_bytes(b"modified payload")
    original = Path.open

    def guarded_open(path: Path, mode: str = "r") -> BinaryIO:
        if path == child:
            raise AssertionError("controller opened a retained numerical payload")
        assert mode == "rb"
        return original(path, "rb")

    with monkeypatch.context() as context:
        context.setattr(Path, "open", guarded_open)
        qualification._inventory(reference, tmp_path)
    with pytest.raises(IntegrityError, match="bytes differ"):
        qualification._verify_inventory_payloads(reference, tmp_path)


def test_inventory_rejects_extra_missing_and_symlink_payloads(tmp_path: Path) -> None:
    reference, child = _inventory_fixture(tmp_path)
    extra = child.parent / "extra.json"
    extra.write_text("{}")
    with pytest.raises(IntegrityError, match="extra"):
        qualification._inventory(reference, tmp_path)
    extra.unlink()
    child.unlink()
    with pytest.raises(IntegrityError, match="missing"):
        qualification._inventory(reference, tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    child.symlink_to(outside)
    with pytest.raises(IntegrityError):
        qualification._inventory(reference, tmp_path)


def _resource_reference(root: Path, elapsed: float, *, pageouts: int = 0) -> JSONObject:
    path = root / "resources.jsonl"
    journal = Journal(path)
    journal.append(
        {
            "elapsed_seconds": elapsed,
            "owned_rss_mib": 10.0,
            "output_bytes": 10,
            "free_plus_speculative_mib": 1024.0,
            "pageouts": pageouts,
        }
    )
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_resource_observation_cannot_postdate_successful_completion(tmp_path: Path) -> None:
    reference = _resource_reference(tmp_path, 2.0)
    completion: JSONObject = {
        "resource_observations": 1,
        "elapsed_seconds": 1.0,
        "maximum_observed_owned_rss_mib": 10.0,
    }
    with pytest.raises(IntegrityError, match="resource bounds"):
        qualification._observed_resources(reference, tmp_path, completion)


def test_resource_observation_requires_exact_peak_and_positive_rss(tmp_path: Path) -> None:
    reference = _resource_reference(tmp_path, 0.5)
    completion: JSONObject = {
        "resource_observations": 1,
        "elapsed_seconds": 1.0,
        "maximum_observed_owned_rss_mib": 11.0,
    }
    with pytest.raises(IntegrityError, match="peak"):
        qualification._observed_resources(reference, tmp_path, completion)
    completion["maximum_observed_owned_rss_mib"] = 10.0
    qualification._observed_resources(reference, tmp_path, completion)


def _stress() -> JSONObject:
    rows: list[JSONValue] = [
        {
            "atoms": n,
            "cost_bps": cost,
            "robust": robust,
            "wall_seconds": 0.1,
            "certificate": {"candidate_count": 610, "optimality_gap": 0.0},
        }
        for n in (64, 128, 260, 520)
        for cost in (0, 10, 20, 40)
        for robust in (False, True)
    ]
    return {
        "schema": "s10-generated-policy-stress-v1",
        "evidence_scope": "GENERATED_ENGINEERING",
        "rows": rows,
    }


def test_stress_metadata_requires_every_fixed_axis_and_generated_scope() -> None:
    original = _stress()
    qualification._audit_stress(original)
    rows = original["rows"]
    assert isinstance(rows, list)
    shortened = {**original, "rows": rows[:-1]}
    with pytest.raises(IntegrityError, match="complete fixed"):
        qualification._audit_stress(shortened)
    with pytest.raises(IntegrityError, match="generated"):
        qualification._audit_stress({**original, "evidence_scope": "RETROSPECTIVE_DEVELOPMENT"})


def test_deep_stress_comparison_detects_changed_certificate_but_allows_elapsed_change() -> None:
    retained, recomputed = _stress(), _stress()
    rows = recomputed["rows"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["wall_seconds"] = 5.0
    qualification._compare_stress(retained, recomputed)
    rows[0]["certificate"] = {"candidate_count": 610, "optimality_gap": 1e-11}
    with pytest.raises(IntegrityError, match="recomputation"):
        qualification._compare_stress(retained, recomputed)


def test_structural_qualification_pass_without_request_chain_is_rejected(tmp_path: Path) -> None:
    reference = save_object(tmp_path, {"schema": "s10-qualification-receipt-v1", "status": "PASS"})
    from frem_s10.errors import ContractError

    with pytest.raises((IntegrityError, ContractError)):
        qualification.audit_qualification(
            reference,
            artifact_root=tmp_path,
            source_root=tmp_path,
            code_sha256="a" * 64,
            environment_sha256="b" * 64,
            calendar_sha256="c" * 64,
        )


@pytest.mark.parametrize("rejection", ["returned", "raised"])
def test_qualification_dispatch_uses_empty_bytecode_prefix_and_preserves_unconsumed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rejection: str
) -> None:
    for name in ("source", "outputs", "consumption"):
        (tmp_path / name).mkdir()
    reference = save_object(tmp_path, {"fixture": "no numerical execution"})
    launch: JSONObject = {
        "schema": "s10-qualification-launch-v1",
        "request": reference,
        "artifact_root": str(tmp_path),
        "source_root": str(tmp_path / "source"),
        "consumption_root": str(tmp_path / "consumption"),
        "output_root": str(tmp_path / "outputs"),
        "attempt_id": "bounded-fake-attempt",
    }
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch))
    observed: dict[str, object] = {}

    def validate(request: JSONObject, *, artifact_root: Path, source_root: Path) -> None:
        assert request == {"fixture": "no numerical execution"}
        assert artifact_root == tmp_path and source_root == tmp_path / "source"

    def execute(**kwargs: object) -> JSONObject:
        observed.update(kwargs)
        if rejection == "raised":
            from frem_s10.runtime.artifacts import publish_json

            output = tmp_path / "outputs/bounded-fake-attempt"
            output.mkdir()
            refused: JSONObject = {
                "state": "REJECTED_UNCONSUMED",
                "consumed": False,
                "worker_dispatched": False,
                "error_type": "ResourceLimitError",
            }
            journal = Journal(output / "timeline.jsonl")
            journal.append({"state": "PREFLIGHT"})
            journal.append(refused)
            publish_json(output, "completion.json", refused)
            raise ResourceLimitError("bounded generated refusal fixture")
        return {"state": "REJECTED_UNCONSUMED", "consumed": False}

    monkeypatch.setattr(qualification, "validate_request", validate)
    monkeypatch.setattr(qualification, "execute_operation", execute)
    result = qualification.run_qualification(path)
    command = observed["command"]
    assert isinstance(command, tuple)
    assert command[:6] == (
        sys.executable,
        "-B",
        "-X",
        f"pycache_prefix={tmp_path / 'outputs/bounded-fake-attempt/empty_bytecode'}",
        "-m",
        "frem_s10.runtime.qualification_worker",
    )
    assert result["status"] == "REJECTED_UNCONSUMED"
    assert result["consumed"] is False
    assert result["scientific_support_claimed"] is False


@pytest.mark.parametrize("wrong", ["order", "marker", "extra"])
def test_lifecycle_rejects_wrong_order_marker_or_extra_terminal_state(tmp_path: Path, wrong: str) -> None:
    marker = save_object(tmp_path, {"scope": "GENERATED_ENGINEERING", "fixture": "consumption-link-only"})
    path = tmp_path / "timeline.jsonl"
    journal = Journal(path)
    states = ["PREFLIGHT", "CONSUMED", "RUNNING", "COMPLETE"]
    if wrong == "order":
        states[1:3] = ["RUNNING", "CONSUMED"]
    if wrong == "extra":
        states.append("COMPLETE")
    for state in states:
        event: JSONObject = {"state": state}
        if state == "CONSUMED":
            event["marker_sha256"] = "f" * 64 if wrong == "marker" else marker["sha256"]
        journal.append(event)
    receipt: JSONObject = {
        "timeline": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "consumption": marker,
    }
    with pytest.raises(IntegrityError, match="lifecycle"):
        qualification._audit_lifecycle(receipt, tmp_path)
