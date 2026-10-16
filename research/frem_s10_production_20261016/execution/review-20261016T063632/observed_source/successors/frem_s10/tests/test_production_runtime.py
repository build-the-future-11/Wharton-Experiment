"""Bounded generated production join, semantic result, and early refusal checks."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from test_declarations_audit import Fixture

from frem_s10.analysis.bootstrap import assess_precision, pilot_indices
from frem_s10.cli import main
from frem_s10.errors import ArtifactIOError, ExecutionStateError, IntegrityError
from frem_s10.experiment.precision import PrecisionScores
from frem_s10.identities import (
    JSONObject,
    canonical_bytes,
    checked_json,
    digest_bytes,
    object_json,
    parse_json,
)
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.consumption import consume
from frem_s10.runtime.pipeline import (
    audit_operation_review,
    audit_precision_prerequisite,
    output_allowance,
    validate_production_declaration,
)
from frem_s10.runtime.results import audit_precision, publish_precision
from frem_s10.runtime.worker import execute_worker


def generated_scores() -> PrecisionScores:
    candidate = 1.0 + np.arange(24, dtype=np.float64) / 1000
    equal = candidate + 0.05 + np.sin(np.arange(24)) / 500
    global_stack = candidate + 0.03 + np.cos(np.arange(24)) / 500
    return PrecisionScores(
        "generated-input",
        tuple(f"date-{i}" for i in range(24)),
        tuple((f"c-{i}", f"e-{i}", f"g-{i}") for i in range(24)),
        candidate,
        equal,
        global_stack,
        96,
        "generated-final-state",
        assess_precision(candidate, equal, global_stack, 24, 25),
        "GENERATED_ENGINEERING",
        tuple(pilot_indices(24).items()),
        (False,) * 24,
        (None,) * 24,
    )


def test_saved_precision_regenerates_fixed_status_and_preserves_every_array(tmp_path: Path) -> None:
    scores = generated_scores()
    identity = publish_precision(tmp_path, scores, projected_a=24, projected_b=25)
    audit = audit_precision(tmp_path, identity)
    assert audit["status"] == scores.result.status
    assert audit["result"] == parse_json(json.dumps(asdict(scores.result)))
    assert audit["evidence_scope"] == "GENERATED_ENGINEERING"
    assert len(tuple(tmp_path.glob("bootstrap-*.json"))) == 2
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ArtifactIOError):
        publish_precision(tmp_path, scores, projected_a=24, projected_b=25)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize("defect", ("score", "boolean", "status", "stream"))
def test_precision_resigned_corruption_cannot_pass(tmp_path: Path, defect: str) -> None:
    identity = publish_precision(tmp_path, generated_scores(), projected_a=24, projected_b=25)
    value = object_json(parse_json((tmp_path / "precision.json").read_bytes()))
    if defect in ("score", "boolean"):
        values = object_json(value["candidate"])["values"]
        assert isinstance(values, list)
        values[0] = True if defect == "boolean" else 5.0
    elif defect == "status":
        object_json(value["result"])["status"] = "POSITIVE"
    else:
        streams = value["bootstrap"]
        assert isinstance(streams, list)
        reference = object_json(streams[0])
        path = tmp_path / str(reference["path"])
        stream = object_json(parse_json(path.read_bytes()))
        observed = stream["values"]
        assert isinstance(observed, list) and isinstance(observed[0], list)
        observed[0][0] = True
        raw = canonical_bytes(stream) + b"\n"
        path.write_bytes(raw)
        reference["sha256"] = digest_bytes(raw)
    raw = canonical_bytes(value) + b"\n"
    (tmp_path / "precision.json").write_bytes(raw)
    assert identity != digest_bytes(raw)
    with pytest.raises(IntegrityError):
        audit_precision(tmp_path, digest_bytes(raw))


def test_structural_pass_cannot_bypass_real_operation_producer(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path, "RETROSPECTIVE_DEVELOPMENT")
    declaration = fixture.operation("prepare-precision")
    with pytest.raises(IntegrityError, match="v2 local audit"):
        validate_production_declaration(
            declaration, artifact_root=tmp_path, source_root=fixture.source, consumption_root=fixture.markers
        )
    assert not tuple(fixture.markers.iterdir())


def test_worker_refuses_consumed_identity_from_another_output_before_import_or_values(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    reference = fixture.operation("prepare-precision")
    output = tmp_path / "attempt"
    output.mkdir()
    consume(
        fixture.markers,
        operation="precision",
        freeze_id=str(reference["sha256"]),
        attempt_id="attempt",
        output_path=str(tmp_path / "different-output"),
    )
    with pytest.raises(ExecutionStateError, match="output identity"):
        execute_worker(
            declaration_path=tmp_path / str(reference["path"]),
            declaration_sha256=str(reference["sha256"]),
            artifact_root=tmp_path,
            source_root=fixture.source,
            consumption_root=fixture.markers,
            attempt_id="attempt",
            output=output,
        )
    assert not tuple(output.iterdir())
    assert len(tuple(fixture.markers.iterdir())) == 1


def test_worker_start_boundary_is_exclusive_and_retains_marker(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    reference = fixture.operation("prepare-precision")
    output = tmp_path / "attempt"
    output.mkdir()
    consume(
        fixture.markers,
        operation="precision",
        freeze_id=str(reference["sha256"]),
        attempt_id="attempt",
        output_path=str(output),
    )
    publish_json(output, "worker-started.json", {"prior": "retained"})
    with pytest.raises(ArtifactIOError):
        execute_worker(
            declaration_path=tmp_path / str(reference["path"]),
            declaration_sha256=str(reference["sha256"]),
            artifact_root=tmp_path,
            source_root=fixture.source,
            consumption_root=fixture.markers,
            attempt_id="attempt",
            output=output,
        )
    assert len(tuple(fixture.markers.iterdir())) == 1


def test_cli_requires_explicit_execution_and_never_accepts_resume() -> None:
    with pytest.raises(SystemExit) as missing:
        main(["prepare-precision", "--manifest", "unused.json"])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as unknown:
        main(["resume", "--manifest", "unused.json"])
    assert unknown.value.code == 2


def test_cli_module_entry_cannot_succeed_silently() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "frem_s10.cli", "plan", "--manifest", "missing-plan-file.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert object_json(parse_json(result.stdout))["status"] == "REJECTED_UNCONSUMED"


def test_review_cannot_swap_independently_verified_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frem_s10.runtime import verification

    reference: JSONObject = {
        "path": "review.json",
        "sha256": publish_json(
            tmp_path,
            "review.json",
            {
                "schema": "s10-operation-review-v2",
                "bindings": {
                    "code": "a",
                    "configuration": "b",
                    "data": "c",
                    "calendar": "d",
                    "environment": "e",
                    "verification": "f" * 64,
                },
                "verification": {"path": "unused.json", "sha256": "f" * 64},
                "operation": "development-campaign",
                "evidence_scope": "RETROSPECTIVE_DEVELOPMENT",
            },
        ),
    }

    def fake_audit(reference: object, *, artifact_root: Path, source_root: Path) -> JSONObject:
        return {
            "bindings": {"code": "a", "configuration": "b", "data": "c", "calendar": "d", "environment": "e"},
            "operation": "prepare-precision",
            "evidence_scope": "RETROSPECTIVE_DEVELOPMENT",
        }

    monkeypatch.setattr(verification, "audit_verification", fake_audit)
    with pytest.raises(IntegrityError, match="review bindings"):
        audit_operation_review(reference, artifact_root=tmp_path, source_root=tmp_path)


def test_historical_structural_precision_cannot_authorize_production_campaign(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    pilot = fixture.operation("prepare-precision")
    declaration: JSONObject = {"precision": fixture.precision_receipt(pilot)}
    with pytest.raises(IntegrityError, match="supervised v2"):
        audit_precision_prerequisite(declaration, artifact_root=tmp_path, source_root=fixture.source)


def test_storage_allowances_cannot_relax_fixed_compute_limits() -> None:
    assert output_allowance("prepare-precision") == 512 * 2**20
    assert output_allowance("development-campaign") == 2 * 2**30
    with pytest.raises(IntegrityError):
        output_allowance("confirmatory")


def test_complete_provenance_inventory_is_accepted_by_freeze_and_remains_hash_bound(tmp_path: Path) -> None:
    from test_packaging_audit import _fixture

    from frem_s10.runtime.declarations import _code
    from frem_s10.runtime.provenance import PACKAGE, code_document

    _fixture(tmp_path)
    manifest = code_document(tmp_path)
    _code(manifest, tmp_path)
    rows = manifest["files"]
    assert isinstance(rows, list)
    names = {Path(str(object_json(row)["path"])).name for row in rows}
    assert {
        "README.md",
        "MANIFEST.in",
        "uv.lock",
        "pyproject.toml",
        "science.json",
        "py.typed",
        "test_fixture.py",
    } <= names
    (tmp_path / PACKAGE / "MANIFEST.in").write_text("# changed build inventory\n")
    with pytest.raises(IntegrityError, match="current code differs"):
        _code(manifest, tmp_path)


def _saved_execution(tmp_path: Path, defect: str | None) -> JSONObject:
    from frem_s10.runtime.journal import Journal
    from frem_s10.runtime.resources import MemorySample, assess

    output, markers = tmp_path / "attempt", tmp_path / "markers"
    output.mkdir()
    markers.mkdir()
    identity = "a" * 64
    marker_id = consume(
        markers, operation="precision", freeze_id=identity, attempt_id="attempt", output_path=str(output)
    )

    def save(name: str, payload: JSONObject) -> JSONObject:
        return {"path": "attempt/" + name, "sha256": publish_json(output, name, payload)}

    result: JSONObject = {
        "operation": "prepare-precision",
        "declaration_sha256": identity,
        "evidence_scope": "GENERATED_ENGINEERING",
        "scientific_disposition": "PRECISION_PASS",
        "consumption": {"path": "markers/precision-" + identity + ".json", "sha256": marker_id},
    }
    result["worker_evidence"] = save(
        "worker-evidence.json",
        {
            "operation": "prepare-precision",
            "declaration_sha256": identity,
            "evidence_scope": "GENERATED_ENGINEERING",
            "scientific_disposition": "PRECISION_PASS",
        },
    )
    result["supervisor"] = save(
        "completion.json",
        {
            "state": "COMPLETE",
            "operation": "precision",
            "freeze_id": identity,
            "attempt_id": "attempt",
            "worker_exit_code": 0,
            "consumed": True,
            "elapsed_seconds": 2.0,
            "maximum_observed_owned_rss_mib": 16.0,
            "wall_limit_seconds": 600,
            "rss_limit_mib": 256,
            "output_limit_bytes": 512 * 2**20,
            "resource_observations": 1,
        },
    )
    host = assess(tuple(MemorySample(f"2026-10-16T00:00:0{i}+00:00", 4096, 262144, 0, 10) for i in range(3)))
    result["preflight"] = save("preflight.json", object_json(parse_json(json.dumps(host))))
    row: JSONObject = {
        "elapsed_seconds": 1.0,
        "owned_rss_mib": 16.0,
        "output_bytes": 100,
        "free_plus_speculative_mib": 1024.0,
        "pageouts": 10,
        "owned_processes": [{"pid": 123, "start_identity": "generated"}],
    }
    faults: dict[str, tuple[str, float | int]] = {
        "rss": ("owned_rss_mib", 257.0),
        "pageouts": ("pageouts", 11),
        "free": ("free_plus_speculative_mib", 511.0),
        "elapsed": ("elapsed_seconds", 3.0),
        "zero_rss": ("owned_rss_mib", 0.0),
        "negative_rss": ("owned_rss_mib", -1.0),
        "peak": ("owned_rss_mib", 8.0),
    }
    if defect is not None:
        field, value = faults[defect]
        row[field] = value
    for name, events in (
        ("resources", (row,)),
        (
            "timeline",
            (
                {"state": "PREFLIGHT"},
                {"state": "CONSUMED", "marker_sha256": marker_id},
                {"state": "RUNNING"},
                {"state": "COMPLETE"},
            ),
        ),
    ):
        path = output / f"{name}.jsonl"
        journal = Journal(path)
        for event in events:
            journal.append(dict(event))
        result[name] = {"path": f"attempt/{name}.jsonl", "sha256": digest_bytes(path.read_bytes())}
    return result


def test_execution_receipt_audits_actual_marker_and_resource_chain(tmp_path: Path) -> None:
    from frem_s10.runtime.receipts import audit_execution

    result = _saved_execution(tmp_path, None)
    assert audit_execution(result, artifact_root=tmp_path)["scientific_disposition"] == "PRECISION_PASS"


@pytest.mark.parametrize("defect", ("rss", "pageouts", "free", "elapsed", "zero_rss", "negative_rss", "peak"))
def test_resigned_sampled_resource_failure_cannot_be_complete(tmp_path: Path, defect: str) -> None:
    from frem_s10.runtime.receipts import audit_execution

    result = _saved_execution(tmp_path, defect)
    with pytest.raises(IntegrityError):
        audit_execution(result, artifact_root=tmp_path)


def test_resigned_lifecycle_cannot_run_before_consumption(tmp_path: Path) -> None:
    from frem_s10.runtime.receipts import audit_execution

    result = _saved_execution(tmp_path, None)
    reference = object_json(result["timeline"])
    path = tmp_path / str(reference["path"])
    events = [object_json(parse_json(line))["event"] for line in path.read_bytes().splitlines()]
    events[1], events[2] = events[2], events[1]
    previous, lines = "0" * 64, []
    for index, event in enumerate(events):
        line = canonical_bytes({"sequence": index, "previous": previous, "event": event}) + b"\n"
        previous = digest_bytes(line)
        lines.append(line)
    raw = b"".join(lines)
    path.write_bytes(raw)
    reference["sha256"] = digest_bytes(raw)
    with pytest.raises(IntegrityError, match="consume the exact marker"):
        audit_execution(result, artifact_root=tmp_path)


def test_fixed_worker_dispatch_keeps_resource_refusal_unconsumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frem_s10.runtime import pipeline
    from frem_s10.runtime.supervisor import Budget

    fixture = Fixture(tmp_path)
    reference = fixture.operation("prepare-precision")
    output = tmp_path / "outputs"
    output.mkdir()

    def admitted(
        reference: object, *, artifact_root: Path, source_root: Path, consumption_root: Path
    ) -> JSONObject:
        return {
            "operation": "prepare-precision",
            "declaration": fixture.read(object_json(checked_json(reference))),
            "declaration_sha256": reference_identity,
        }

    reference_identity = str(reference["sha256"])

    def blocked(**kwargs: object) -> JSONObject:
        command = kwargs["command"]
        assert isinstance(command, tuple) and command[1:3] == ("-B", "-X")
        assert command[3] == f"pycache_prefix={output / 'bounded-refusal' / 'empty_bytecode'}"
        assert command[4:6] == ("-m", "frem_s10.runtime.worker")
        assert kwargs["freeze_id"] == reference_identity
        budget = kwargs["budget"]
        assert isinstance(budget, Budget)
        assert (budget.wall_seconds, budget.rss_mib, budget.output_bytes) == (600, 256, 512 * 2**20)
        return {"state": "REJECTED_UNCONSUMED", "consumed": False}

    monkeypatch.setattr(pipeline, "validate_production_declaration", admitted)
    monkeypatch.setattr(pipeline, "execute_operation", blocked)
    manifest: JSONObject = {
        "schema": "s10-launch-manifest-v1",
        "declaration": reference,
        "artifact_root": str(tmp_path),
        "source_root": str(fixture.source),
        "consumption_root": str(fixture.markers),
        "output_root": str(output),
        "attempt_id": "bounded-refusal",
    }
    path = tmp_path / "launch.json"
    path.write_bytes(canonical_bytes(manifest))
    result = pipeline.run_declared(path, expected_operation="prepare-precision")
    assert result["state"] == "REJECTED_UNCONSUMED" and not tuple(fixture.markers.iterdir())
    manifest["command"] = ["arbitrary-executable"]
    path.write_bytes(canonical_bytes(manifest))
    from frem_s10.errors import ContractError

    with pytest.raises(ContractError):
        pipeline.run_declared(path, expected_operation="prepare-precision")


def test_generated_pilot_worker_phase_archives_and_regenerates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_precision_inputs_audit import fixture as numerical_fixture
    from test_production_data_audit import _request

    from frem_s10.data import production
    from frem_s10.data.admission import calendar_identity
    from frem_s10.data.precision import PrecisionInputs, prepare_precision_inputs
    from frem_s10.experiment.model_archive import ModelArchiveWriter, verify_model_archive
    from frem_s10.runtime.worker import _run_precision

    metadata, numerical, output = (tmp_path / name for name in ("metadata", "numeric", "output"))
    for path in (metadata, numerical, output):
        path.mkdir()
    reference = production.create_source_admission(
        _request(metadata), artifact_root=metadata, source_root=metadata, output_name="admitted"
    )
    source = production.validate_source_admission(
        dict(reference), artifact_root=metadata, source_root=metadata
    )
    reader, identities, contract, dates = numerical_fixture(numerical)
    inputs = prepare_precision_inputs(
        reader,
        identities,
        pilot_contract=contract,
        projected_sessions=dates,
        projected_calendar_id=calendar_identity(dates),
        evidence_scope="GENERATED_ENGINEERING",
    )

    def constructed_loader(admission: production.SourceAdmission, *, source_root: Path) -> PrecisionInputs:
        assert admission is source
        return inputs

    # Replace only the source boundary with a separately validated generated view.
    # Kernel, archive, score serialization and saved-statistic audit all execute.
    monkeypatch.setattr(production, "load_precision_inputs", constructed_loader)
    archive = output / "archive"
    archive.mkdir()
    writer = ModelArchiveWriter(
        archive,
        experiment_id="generated-pilot-join",
        evidence_scope="GENERATED_ENGINEERING",
        binding_id="e" * 64,
        variants=("primary",),
    )
    result, admitted_inputs = _run_precision(source, numerical, output, writer)
    archive_id = writer.finish()
    verified = verify_model_archive(archive, archive_id)
    assert verified["evidence_scope"] == "GENERATED_ENGINEERING"
    assert result["scientific_disposition"] in (
        "PRECISION_PASS",
        "PRECISION_INADEQUATE",
        "PRECISION_UNIDENTIFIABLE",
    )
    evidence = object_json(result["evidence"])
    from frem_s10.runtime.results import audit_precision_archive

    audit = audit_precision_archive(
        output / "precision", str(evidence["sha256"]), archive, archive_id, admitted_inputs
    )
    assert audit["status"] == result["scientific_disposition"]
    assert not (output / "campaign").exists()
    path = output / "precision/precision.json"
    values = object_json(parse_json(path.read_bytes()))
    flags = values["cold_start_blocks"]
    assert isinstance(flags, list)
    flags[0] = not flags[0]
    changed = canonical_bytes(values) + b"\n"
    path.write_bytes(changed)
    # Re-signing both result and identity can pass the scalar-only calculation;
    # the separately retained actual weekly transition still rejects the lie.
    audit_precision(output / "precision", digest_bytes(changed))
    with pytest.raises(IntegrityError, match="diagnostics"):
        audit_precision_archive(
            output / "precision", digest_bytes(changed), archive, archive_id, admitted_inputs
        )
