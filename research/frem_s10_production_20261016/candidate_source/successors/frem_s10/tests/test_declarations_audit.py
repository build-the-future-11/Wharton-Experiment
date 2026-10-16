"""Constructed metadata binding checks; none can authorize scientific execution."""

from importlib.resources import files
from pathlib import Path

import pytest

from frem_s10.errors import ContractError, ExecutionStateError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, digest_bytes, object_json, parse_json
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.declarations import DESIGN, validate_declaration


class Fixture:
    def __init__(self, root: Path, scope: str = "GENERATED_ENGINEERING") -> None:
        self.root = root
        self.source = root / "source"
        self.markers = root / "markers"
        self.source.mkdir()
        self.markers.mkdir()
        self.scope = scope
        (self.source / "kernel.py").write_text("# Generated source identity fixture only.\n")
        self.code = self.save(
            "code",
            {
                "schema": "s10-code-manifest-v1",
                "files": [
                    {
                        "path": "kernel.py",
                        "sha256": digest_bytes((self.source / "kernel.py").read_bytes()),
                    }
                ],
            },
        )
        self.config = self.save(
            "configuration", parse_json(files("frem_s10").joinpath("science.json").read_bytes())
        )
        self.common = {"code": self.code, "configuration": self.config}
        for role in ("calendar", "environment"):
            self.common[role] = self.save(
                role,
                {
                    "schema": f"s10-{role}-receipt-v1",
                    "design_id": DESIGN,
                    "evidence_scope": scope,
                    "status": "PASS",
                    "identity": "f" * 64,
                },
            )

    def save(self, name: str, data: JSONValue) -> JSONObject:
        path = name + ".json"
        return {"path": path, "sha256": publish_json(self.root, path, data)}

    def read(self, reference: JSONObject) -> JSONObject:
        return object_json(parse_json((self.root / str(reference["path"])).read_bytes()))

    def operation(self, operation: str, precision: JSONValue = None) -> JSONObject:
        stem = "pilot" if operation == "prepare-precision" else "campaign"
        cutoff = "2010-01-01" if stem == "pilot" else "2017-01-01"
        refs = dict(self.common)
        refs["data"] = self.save(
            stem + "-data",
            {
                "schema": "s10-data-manifest-v1",
                "assets": ["SPY", "TLT", "GLD"],
                "source_hashes": ["a" * 64, "b" * 64, "c" * 64],
                "values_before": cutoff,
            },
        )
        bindings: JSONObject = {role: ref["sha256"] for role, ref in refs.items()}
        log = self.save(
            stem + "-test",
            {
                "schema": "s10-test-result-v1",
                "status": "PASS",
                "code_sha256": bindings["code"],
                "environment_sha256": bindings["environment"],
            },
        )
        base: JSONObject = {
            "design_id": DESIGN,
            "operation": operation,
            "evidence_scope": self.scope,
            "bindings": bindings,
            "status": "PASS",
        }
        refs["verification"] = self.save(
            stem + "-verification", base | {"schema": "s10-verification-receipt-v1", "logs": [log]}
        )
        review = self.save(
            stem + "-review",
            base
            | {
                "schema": "s10-operation-review-v1",
                "bindings": bindings | {"verification": refs["verification"]["sha256"]},
            },
        )
        refs["admission"] = self.save(
            stem + "-admission",
            base
            | {
                "schema": "s10-admission-receipt-v1",
                "review": review,
                "verification_sha256": refs["verification"]["sha256"],
            },
        )
        return self.save(
            stem,
            {
                "schema": "s10-operation-declaration-v1",
                "design_id": DESIGN,
                "operation": operation,
                "evidence_scope": self.scope,
                "created_at": "2026-10-15T12:00:00+00:00",
                "values_before": cutoff,
                "artifacts": dict(refs),
                "precision": precision,
                "budget": {
                    "wall_seconds": 600 if stem == "pilot" else 3600,
                    "rss_mib": 256,
                    "workers": 1,
                    "threads": 1,
                    "weekly_origins": 1000,
                    "ledgers": 0 if stem == "pilot" else 192,
                },
            },
        )

    def precision_receipt(self, pilot: JSONObject) -> JSONObject:
        consumed = self.save(
            "pilot-consumed",
            {
                "schema": "s10-operation-consumption-v1",
                "operation": "prepare-precision",
                "declaration_sha256": pilot["sha256"],
                "state": "CONSUMED",
            },
        )
        completed = self.save(
            "pilot-completed",
            {
                "schema": "s10-precision-completion-v1",
                "operation": "prepare-precision",
                "declaration_sha256": pilot["sha256"],
                "state": "COMPLETE",
                "precision_status": "PRECISION_PASS",
                "evidence_scope": self.scope,
                "consumption": consumed,
            },
        )
        return self.save(
            "precision",
            {
                "schema": "s10-precision-prerequisite-v1",
                "state": "COMPLETE",
                "status": "PRECISION_PASS",
                "evidence_scope": self.scope,
                "declaration": pilot,
                "completion": completed,
            },
        )

    def validate(self, reference: JSONObject) -> bool:
        result = validate_declaration(
            reference, artifact_root=self.root, source_root=self.source, consumption_root=self.markers
        )
        assert result.status == "STRUCTURE_VALID_SCIENTIFIC_GATE_CLOSED"
        assert result.unconsumed and result.blockers
        return result.authorizes_scientific_execution


def test_generated_precision_and_bound_campaign_remain_non_authorizing(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    pilot = fixture.operation("prepare-precision")
    assert fixture.validate(pilot) is False
    campaign = fixture.operation("development-campaign", fixture.precision_receipt(pilot))
    assert fixture.validate(campaign) is False
    assert pilot["sha256"] != campaign["sha256"]


def test_self_asserted_real_pass_is_not_scientific_admission(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path, "RETROSPECTIVE_DEVELOPMENT")
    assert fixture.validate(fixture.operation("prepare-precision")) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("design_id", "FREM-S10-DESIGN-3-CL15"),
        ("values_before", "2025-01-01"),
        ("operation", "confirmatory"),
        ("evidence_scope", "CONFIRMED"),
        ("precision", True),
    ],
)
def test_invalid_operation_scope_or_circular_precision_rejected(
    tmp_path: Path,
    field: str,
    value: JSONValue,
) -> None:
    fixture = Fixture(tmp_path)
    declaration = fixture.read(fixture.operation("prepare-precision"))
    with pytest.raises(ContractError):
        fixture.validate(fixture.save("bad", declaration | {field: value}))


def test_stale_code_and_modified_artifact_hash_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    pilot = fixture.operation("prepare-precision")
    (fixture.source / "kernel.py").write_text("# changed\n")
    with pytest.raises(IntegrityError, match="current code"):
        fixture.validate(pilot)
    (tmp_path / "pilot.json").write_text("{}")
    with pytest.raises(IntegrityError, match="hash"):
        fixture.validate(pilot)


def test_consumed_marker_and_ambiguous_symlink_refuse_without_mutation(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    pilot = fixture.operation("prepare-precision")
    marker = fixture.markers / ("precision-" + str(pilot["sha256"]) + ".json")
    marker.symlink_to("missing-target")
    with pytest.raises(ExecutionStateError):
        fixture.validate(pilot)
    assert marker.is_symlink()


def test_campaign_requires_completed_precision_and_rejects_wrong_binding(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    pilot = fixture.operation("prepare-precision")
    passing = fixture.precision_receipt(pilot)
    campaign = fixture.operation("development-campaign", passing)
    changed = fixture.read(campaign)
    with pytest.raises(ContractError):
        fixture.validate(fixture.save("missing", changed | {"precision": None}))
    prerequisite = fixture.read(passing)
    failed = fixture.save("failed-precision", prerequisite | {"status": "PRECISION_INADEQUATE"})
    with pytest.raises(IntegrityError, match="passing precision"):
        fixture.validate(fixture.save("failed-campaign", changed | {"precision": failed}))
    completion = fixture.read(object_json(prerequisite["completion"]))
    bad_completion = fixture.save("wrong-completion", completion | {"declaration_sha256": "0" * 64})
    wrong = fixture.save("wrong-precision", prerequisite | {"completion": bad_completion})
    with pytest.raises(IntegrityError, match="exact pilot"):
        fixture.validate(fixture.save("wrong-campaign", changed | {"precision": wrong}))


def test_boolean_authority_unknown_fields_paths_and_budget_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    pilot = fixture.operation("prepare-precision")
    declaration = fixture.read(pilot)
    with pytest.raises(ContractError, match="schema"):
        fixture.validate(fixture.save("boolean-authority", declaration | {"reviewed": True}))
    with pytest.raises(ContractError, match="traversal"):
        fixture.validate({"path": "../outside.json", "sha256": "a" * 64})
    budget = object_json(declaration["budget"])
    with pytest.raises(ContractError, match="budget"):
        fixture.validate(fixture.save("oversized", declaration | {"budget": budget | {"rss_mib": 257}}))


def test_environment_change_cannot_reuse_previous_verification(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    declaration = fixture.read(fixture.operation("prepare-precision"))
    artifacts = object_json(declaration["artifacts"])
    environment = fixture.read(object_json(artifacts["environment"]))
    altered = fixture.save("new-environment", environment | {"identity": "0" * 64})
    replacement = fixture.save(
        "stale-verification",
        declaration
        | {
            "artifacts": artifacts | {"environment": altered},
        },
    )
    with pytest.raises(IntegrityError, match="current identities"):
        fixture.validate(replacement)


def test_config_threshold_change_and_missing_review_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    declaration = fixture.read(fixture.operation("prepare-precision"))
    artifacts = object_json(declaration["artifacts"])
    config = fixture.read(fixture.config)
    altered = fixture.save(
        "lower-margin",
        config
        | {
            "success": object_json(config["success"])
            | {
                "new_distribution_relative_margin": 0.04,
            }
        },
    )
    with pytest.raises(IntegrityError, match="exact installed"):
        fixture.validate(
            fixture.save(
                "changed-config",
                declaration
                | {
                    "artifacts": artifacts | {"configuration": altered},
                },
            )
        )
    (tmp_path / "pilot-review.json").unlink()
    with pytest.raises(IntegrityError, match="regular"):
        fixture.validate(fixture.save("missing-review", declaration))
