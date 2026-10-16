"""Catalog failure cases use constructed identity lists, never scientific authority."""

import json
from datetime import date
from pathlib import Path

import pytest
from conftest import Observations, observed_outcome

from frem_s10.errors import ContractError
from frem_s10.identities import JSONObject, object_json
from frem_s10.runtime.catalog import (
    REQUIREMENT_IDS,
    expand_test_families,
    required_obligations,
    validate_catalog,
    validate_operation_catalog,
)


def catalog() -> JSONObject:
    return {
        "obligations": [
            {
                "test_id": f"TEST-{index:03}",
                "requirement_ids": ["REQ-generated"],
                "status": "VERIFIED",
                "nodeids": [f"tests/generated.py::case_{index}"],
            }
            for index in range(1, 97)
        ]
    }


def test_complete_constructed_registry_is_not_scientific_authorization() -> None:
    nodes = tuple(f"tests/generated.py::case_{i}" for i in range(1, 97))
    result = validate_catalog(
        catalog(),
        requirements=("REQ-generated",),
        collected_nodeids=nodes,
        test_outcomes=dict.fromkeys(nodes, "PASSED"),
        as_of=date(2026, 10, 16),
    )
    assert result.complete and result.verified == 96
    assert result.authorizes_scientific_execution is False


@pytest.mark.parametrize(
    "defect",
    (
        "missing",
        "duplicate",
        "orphan",
        "requirement",
        "skip",
        "xfail",
        "no_receipt",
        "expired",
        "current_exception",
    ),
)
def test_catalog_gaps_never_disappear_behind_passing_counts(defect: str) -> None:
    document = catalog()
    rows = document["obligations"]
    assert isinstance(rows, list)
    nodes = tuple(f"tests/generated.py::case_{i}" for i in range(1, 97))
    outcomes = dict.fromkeys(nodes, "PASSED")
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = rows[0]
    elif defect == "orphan":
        object_json(rows[0])["nodeids"] = ["tests/missing.py::not_collected"]
    elif defect == "requirement":
        object_json(rows[0])["requirement_ids"] = ["REQ-not-declared"]
    elif defect in ("skip", "xfail", "no_receipt"):
        outcomes[nodes[0]] = {"skip": "SKIPPED", "xfail": "XFAILED", "no_receipt": "NOT_RUN"}[defect]
    else:
        document["quality_exceptions"] = [
            {
                "test_id": "TEST-001",
                "reason": "generated exception",
                "expires_on": "2026-10-15" if defect == "expired" else "2026-10-17",
            }
        ]
    result = validate_catalog(
        document,
        requirements=("REQ-generated",),
        collected_nodeids=nodes,
        test_outcomes=outcomes,
        as_of=date(2026, 10, 16),
    )
    assert not result.complete and result.gaps


def test_outcome_for_uncollected_test_is_rejected() -> None:
    with pytest.raises(ContractError, match="orphaned"):
        validate_catalog(
            catalog(),
            requirements=("REQ-generated",),
            collected_nodeids=(),
            test_outcomes={"unknown": "PASSED"},
            as_of=date(2026, 10, 16),
        )


def operation_plan() -> JSONObject:
    return {
        "schema": "s10-operation-catalog-plan-v1",
        "design": "FREM-S10-design4-CL16",
        "requirement_ids": list(REQUIREMENT_IDS),
        "obligations": [
            {
                "test_id": f"TEST-{index:03}",
                "requirement_ids": [REQUIREMENT_IDS[(index - 1) % 60]],
                "test_families": [f"tests/generated.py::case_{index}"],
                "coverage_review": "BOUNDED_COMPLETE",
            }
            for index in range(1, 97)
        ],
    }


def test_operation_scope_does_not_create_a_pilot_qualification_cycle() -> None:
    pilot = required_obligations("prepare-precision", target_platform="macOS")
    campaign = required_obligations("run-development-campaign", target_platform="macOS")
    assert "TEST-090" not in pilot and "TEST-090" in campaign
    assert "TEST-043" not in pilot and "TEST-043" in campaign
    assert "TEST-062" in pilot and "TEST-062" in campaign
    assert "TEST-081" not in pilot and "TEST-081" in campaign
    assert "TEST-089" not in campaign
    assert "TEST-089" in required_obligations("prepare-precision", target_platform="Linux")
    assert "TEST-091" not in pilot and "TEST-093" in pilot and "TEST-093" in campaign
    nodes = tuple(f"tests/generated.py::case_{i}" for i in range(1, 97))
    result = validate_operation_catalog(
        operation_plan(),
        operation="prepare-precision",
        target_platform="macOS",
        requirements=REQUIREMENT_IDS,
        collected_nodeids=nodes,
        test_outcomes=dict.fromkeys(nodes, "PASSED"),
        externally_verified_obligations=("TEST-086", "TEST-093", "TEST-096"),
    )
    assert result.complete and result.verified == len(pilot)
    assert not result.authorizes_scientific_execution


def test_external_receipt_never_waives_required_generated_checks() -> None:
    nodes = tuple(f"tests/generated.py::case_{i}" for i in range(1, 97))
    outcomes = dict.fromkeys(nodes, "PASSED")
    missing = validate_operation_catalog(
        operation_plan(),
        operation="run-development-campaign",
        target_platform="macOS",
        requirements=REQUIREMENT_IDS,
        collected_nodeids=nodes,
        test_outcomes=outcomes,
    )
    assert not missing.complete and len(missing.gaps) == 5
    passed = validate_operation_catalog(
        operation_plan(),
        operation="run-development-campaign",
        target_platform="macOS",
        requirements=REQUIREMENT_IDS,
        collected_nodeids=nodes,
        test_outcomes=outcomes,
        externally_verified_obligations=("TEST-081", "TEST-086", "TEST-090", "TEST-093", "TEST-096"),
    )
    assert passed.complete
    outcomes[nodes[89]] = "FAILED"
    rejected = validate_operation_catalog(
        operation_plan(),
        operation="run-development-campaign",
        target_platform="macOS",
        requirements=REQUIREMENT_IDS,
        collected_nodeids=nodes,
        test_outcomes=outcomes,
        externally_verified_obligations=("TEST-081", "TEST-086", "TEST-090", "TEST-093", "TEST-096"),
    )
    assert not rejected.complete and any("TEST-090:FAILED" in gap for gap in rejected.gaps)
    with pytest.raises(ContractError, match="operation-specific"):
        validate_operation_catalog(
            operation_plan(),
            operation="run-development-campaign",
            target_platform="macOS",
            requirements=REQUIREMENT_IDS,
            collected_nodeids=nodes,
            test_outcomes=outcomes,
            externally_verified_obligations=("TEST-003",),
        )


def test_all_parameter_cases_and_exact_requirement_registry_are_mandatory() -> None:
    family = "tests/generated.py::case_1"
    nodes = (family + "[low]", family + "[high]", *(f"tests/generated.py::case_{i}" for i in range(2, 97)))
    assert expand_test_families((family,), nodes) == (family + "[high]", family + "[low]")
    outcomes = dict.fromkeys(nodes, "PASSED")
    outcomes[family + "[high]"] = "SKIPPED"
    result = validate_operation_catalog(
        operation_plan(),
        operation="prepare-precision",
        target_platform="macOS",
        requirements=REQUIREMENT_IDS,
        collected_nodeids=nodes,
        test_outcomes=outcomes,
        externally_verified_obligations=("TEST-086", "TEST-093", "TEST-096"),
    )
    assert not result.complete and any("SKIPPED" in gap for gap in result.gaps)
    with pytest.raises(ContractError, match="exact frozen"):
        validate_operation_catalog(
            operation_plan(),
            operation="prepare-precision",
            target_platform="macOS",
            requirements=REQUIREMENT_IDS[:-1],
            collected_nodeids=nodes,
            test_outcomes=outcomes,
        )


def test_catalog_cannot_relabel_another_design_as_current() -> None:
    plan = operation_plan()
    plan["design"] = "FREM-S10-design3"
    nodes = tuple(f"tests/generated.py::case_{i}" for i in range(1, 97))
    with pytest.raises(ContractError, match="exact design"):
        validate_operation_catalog(
            plan,
            operation="prepare-precision",
            target_platform="macOS",
            requirements=REQUIREMENT_IDS,
            collected_nodeids=nodes,
            test_outcomes=dict.fromkeys(nodes, "PASSED"),
            externally_verified_obligations=("TEST-086", "TEST-093", "TEST-096"),
        )


@pytest.mark.parametrize(
    "phase,status",
    (
        ("setup", "FAILED"),
        ("call", "FAILED"),
        ("teardown", "FAILED"),
        ("setup", "SKIPPED"),
        ("call", "XFAILED"),
    ),
)
def test_pytest_observations_retain_all_phase_failures(phase: str, status: str) -> None:
    phases = dict.fromkeys(("setup", "call", "teardown"), "PASSED")
    phases[phase] = status
    assert observed_outcome(phases) == status
    assert observed_outcome({"setup": "PASSED"}) == "FAILED"


def test_receipt_plugin_registers_with_pytest_fixture_discovery(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    plugin = Observations(tmp_path / "unused.json")
    manager = request.config.pluginmanager
    try:
        assert manager.register(plugin, "constructed-s10-observations") is not None
        assert manager.get_plugin("constructed-s10-observations") is plugin
    finally:
        manager.unregister(plugin=plugin)
    assert not plugin.path.exists()


def test_pytest_observation_receipt_is_explicit_and_exclusive(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    path = tmp_path / "observations.json"
    observations = Observations(path, ("tests/generated.py::case[one]",))
    assert not path.exists()
    for phase in ("setup", "call", "teardown"):
        observations.pytest_runtest_logreport(
            pytest.TestReport(
                nodeid=observations.collected[0],
                location=("generated.py", 1, "case"),
                keywords={},
                outcome="passed",
                longrepr=None,
                when=phase,
            )
        )
    observations.pytest_sessionfinish(request.session, pytest.ExitCode.OK)
    saved = json.loads(path.read_text())
    assert saved["outcomes"] == {observations.collected[0]: "PASSED"} and saved["exit_code"] == 0
    with pytest.raises(FileExistsError):
        observations.pytest_sessionfinish(request.session, pytest.ExitCode.OK)
