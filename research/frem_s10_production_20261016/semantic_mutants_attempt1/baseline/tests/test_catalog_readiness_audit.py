"""Catalog failure cases use constructed identity lists, never scientific authority."""

from datetime import date

import pytest

from frem_s10.errors import ContractError
from frem_s10.identities import JSONObject, object_json
from frem_s10.runtime.catalog import validate_catalog


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
