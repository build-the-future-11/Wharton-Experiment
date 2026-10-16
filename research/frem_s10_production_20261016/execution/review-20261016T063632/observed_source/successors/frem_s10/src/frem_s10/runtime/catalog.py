"""Fail-closed catalog completeness checks over explicitly supplied test evidence.

This checks traceability and outcomes, not whether a reviewer chose a scientifically
sufficient test. It never authorizes execution or upgrades engineering to empirical
evidence; source-bound verification/admission must bind the supplied receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from frem_s10.errors import ContractError
from frem_s10.identities import JSONObject, object_json

_PILOT_EXCLUDED = frozenset((*range(41, 62), 67, 68, 69, 71, 81, 83, 90))
REQUIREMENT_IDS = tuple(f"REQ-{index:03}" for index in range(1, 61))
_RELEASE_ONLY = frozenset((91,))


def required_obligations(operation: str, *, target_platform: str) -> tuple[str, ...]:
    """List operation-relevant obligations without changing scientific thresholds.

    Precision does not exercise portfolio rules, campaign ablations/H3 or the
    full192-cell runner. Linux parity is required only for a claimed Linux
    target. CL09 maximum-shape qualification follows a passing pilot and is a
    campaign prerequisite, never a circular prerequisite to the pilot itself.
    """
    if operation not in ("prepare-precision", "run-development-campaign"):
        raise ContractError("unknown catalog operation")
    if target_platform not in ("macOS", "Linux"):
        raise ContractError("catalog requires the explicitly verified target platform")
    excluded = _PILOT_EXCLUDED if operation == "prepare-precision" else frozenset()
    return tuple(
        f"TEST-{index:03}"
        for index in range(1, 97)
        if index not in excluded | _RELEASE_ONLY and (index != 89 or target_platform == "Linux")
    )


def expand_test_families(families: tuple[str, ...], collected: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve each exact function identity to all collected parametrized cases.

    Substring, basename, fuzzy and count matching are deliberately unsupported.
    Missing families remain explicit failures, never silently disappear.
    """
    if not families or len(set(families)) != len(families):
        raise ContractError("nonempty distinct exact test families required")
    expanded: set[str] = set()
    for family in families:
        if "::" not in family or "[" in family or not family.startswith("tests/"):
            raise ContractError("catalog families require tests/path.py::test_function identities")
        matches = {node for node in collected if node == family or node.startswith(family + "[")}
        if not matches:
            raise ContractError(f"catalog test family was not collected: {family}")
        expanded.update(matches)
    return tuple(sorted(expanded))


def validate_operation_catalog(
    plan: JSONObject,
    *,
    operation: str,
    target_platform: str,
    requirements: tuple[str, ...],
    collected_nodeids: tuple[str, ...],
    test_outcomes: dict[str, str],
    externally_verified_obligations: tuple[str, ...] = (),
) -> CatalogReadiness:
    """Validate a source-bound plan against current exact pytest observations.

    A coverage review is a statement about a test's designed scope, never proof
    it passed. Every required family must resolve to all actual parameter cases
    and every case must pass in the supplied current execution. The receipt
    producer additionally binds this plan, source, environment and observations.
    TEST086 requires an actual audited build/install receipt and TEST096 requires
    current-source baseline and the fixed normative targeted mutant executions;
    TEST093 requires the actual restricted full local verification invocation.
    Campaign TEST081
    and TEST090 require saved-report regeneration and maximum-shape qualification.
    The caller audits these actual receipts first; this pure metadata function
    does not grant authority to its arguments or waive any required node test.
    """
    required = set(required_obligations(operation, target_platform=target_platform))
    rows = _plan_rows(plan)
    _validate_operation_evidence(
        plan, requirements, collected_nodeids, test_outcomes, required, externally_verified_obligations
    )
    gaps: list[str] = []
    verified = 0
    covered: set[str] = set()
    for row in rows:
        requirement_ids = _strings(row.get("requirement_ids"), "requirement IDs")
        covered.update(requirement_ids)
        families = _strings(row.get("test_families"), "test families")
        review = row.get("coverage_review")
        if review not in ("BOUNDED_COMPLETE", "PARTIAL", "UNIMPLEMENTED"):
            raise ContractError("catalog coverage review is missing or unsupported")
        if row["test_id"] not in required:
            continue
        if (
            row["test_id"] in ("TEST-081", "TEST-086", "TEST-090", "TEST-093", "TEST-096")
            and row["test_id"] not in externally_verified_obligations
        ):
            gaps.append(f"{row['test_id']}:MISSING_AUDITED_EXTERNAL_EXECUTION_RECEIPT")
        if row["test_id"] in externally_verified_obligations:
            review = "BOUNDED_COMPLETE"
        if not families:
            gaps.append(f"{row['test_id']}:MISSING_TEST_FAMILIES")
            continue
        expanded = expand_test_families(families, collected_nodeids)
        current: JSONObject = {
            "test_id": row["test_id"],
            "requirement_ids": list(requirement_ids),
            "nodeids": list(expanded),
            "status": "VERIFIED" if review == "BOUNDED_COMPLETE" else "PARTIAL",
        }
        row_gaps, _ = _row_gaps(current, requirements, collected_nodeids, test_outcomes)
        gaps.extend(row_gaps)
        verified += int(not row_gaps)
    gaps.extend(f"{identity}:UNMAPPED_REQUIREMENT" for identity in sorted(set(requirements) - covered))
    return CatalogReadiness(not gaps and verified == len(required), verified, tuple(gaps))


def _validate_operation_evidence(
    plan: JSONObject,
    requirements: tuple[str, ...],
    collected: tuple[str, ...],
    outcomes: dict[str, str],
    required: set[str],
    external: tuple[str, ...],
) -> None:
    if requirements != REQUIREMENT_IDS or plan.get("requirement_ids") != list(REQUIREMENT_IDS):
        raise ContractError("exact frozen60-requirement registry required")
    allowed = {"TEST-081", "TEST-086", "TEST-090", "TEST-093", "TEST-096"} & required
    if len(set(external)) != len(external) or set(external) - allowed:
        raise ContractError(
            "only operation-specific packaging/report/shape/isolation/mutation receipts accepted"
        )
    if len(set(collected)) != len(collected) or set(outcomes) - set(collected):
        raise ContractError("pytest observations contain duplicate or uncollected identities")


def _plan_rows(plan: JSONObject) -> tuple[JSONObject, ...]:
    if (plan.get("schema"), plan.get("design")) != (
        "s10-operation-catalog-plan-v1",
        "FREM-S10-design4-CL16",
    ):
        raise ContractError("operation catalog plan schema or exact design mismatch")
    rows_value = plan.get("obligations")
    if not isinstance(rows_value, list):
        raise ContractError("operation catalog obligations required")
    rows = tuple(object_json(value) for value in rows_value)
    identities = tuple(row.get("test_id") for row in rows)
    expected = {f"TEST-{index:03}" for index in range(1, 97)}
    if any(not isinstance(identity, str) for identity in identities):
        raise ContractError("catalog obligation identities must be strings")
    if len(identities) != 96 or set(identities) != expected:
        raise ContractError("operation catalog requires exactly all96 distinct obligation IDs")
    return rows


@dataclass(frozen=True, slots=True)
class CatalogReadiness:
    """Exact outstanding obligations, independent of aggregate passing test counts."""

    complete: bool
    verified: int
    gaps: tuple[str, ...]
    authorizes_scientific_execution: bool = False


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"catalog {name} must be a string list")
    return tuple(value)


def _exceptions(catalog: JSONObject, as_of: date) -> list[str]:
    exceptions = catalog.get("quality_exceptions", [])
    if not isinstance(exceptions, list):
        raise ContractError("catalog quality exceptions must be an explicit list")
    gaps: list[str] = []
    for raw in exceptions:
        entry = object_json(raw)
        if set(entry) != {"test_id", "reason", "expires_on"} or any(
            not isinstance(value, str) or not value for value in entry.values()
        ):
            raise ContractError("catalog quality exception schema is incomplete")
        try:
            expiry = date.fromisoformat(str(entry["expires_on"]))
        except ValueError as exc:
            raise ContractError("quality exception requires an ISO expiry date") from exc
        state = "EXPIRED_EXCEPTION" if expiry < as_of else "UNVERIFIED_EXCEPTION"
        gaps.append(f"{entry['test_id']}:{state}:{entry['reason']}")
    return gaps


def validate_catalog(
    catalog: JSONObject,
    *,
    requirements: tuple[str, ...],
    collected_nodeids: tuple[str, ...],
    test_outcomes: dict[str, str],
    as_of: date,
) -> CatalogReadiness:
    """Require all 96 obligations, exact live nodes, passing outcomes and no waivers.

    Passed/skipped/xfail are not interchangeable. An expired or current exception
    stays visible and cannot turn missing verification into a complete catalog.
    """
    if len(set(requirements)) != len(requirements) or not requirements:
        raise ContractError("nonempty distinct requirement registry required")
    if len(set(collected_nodeids)) != len(collected_nodeids):
        raise ContractError("collected test registry contains duplicate identities")
    if set(test_outcomes) - set(collected_nodeids):
        raise ContractError("test outcomes contain orphaned uncollected node identities")
    obligations = catalog.get("obligations")
    if not isinstance(obligations, list):
        raise ContractError("catalog obligations must be an explicit list")
    rows = tuple(object_json(raw) for raw in obligations)
    ids = [str(row.get("test_id")) for row in rows]
    expected = {f"TEST-{index:03}" for index in range(1, 97)}
    gaps = _exceptions(catalog, as_of)
    gaps.extend(f"{identity}:MISSING" for identity in sorted(expected - set(ids)))
    gaps.extend(f"{identity}:ORPHAN" for identity in sorted(set(ids) - expected))
    gaps.extend(f"{identity}:DUPLICATE" for identity in sorted(set(ids)) if ids.count(identity) != 1)
    verified = 0
    covered: set[str] = set()
    for row in rows:
        row_gaps, requirement_ids = _row_gaps(row, requirements, collected_nodeids, test_outcomes)
        gaps.extend(row_gaps)
        covered.update(requirement_ids)
        verified += int(not row_gaps and row.get("test_id") in expected)
    gaps.extend(f"{identity}:UNMAPPED_REQUIREMENT" for identity in sorted(set(requirements) - covered))
    return CatalogReadiness(not gaps and verified == 96, verified, tuple(gaps))


def _row_gaps(
    row: JSONObject, requirements: tuple[str, ...], collected: tuple[str, ...], outcomes: dict[str, str]
) -> tuple[list[str], tuple[str, ...]]:
    identity = row.get("test_id")
    requirement_ids = _strings(row.get("requirement_ids"), "requirement IDs")
    nodeids = _strings(row.get("nodeids"), "test nodes")
    status = row.get("status")
    if status not in ("VERIFIED", "PARTIAL", "PENDING"):
        raise ContractError("catalog status must be VERIFIED/PARTIAL/PENDING")
    gaps: list[str] = []
    if status != "VERIFIED":
        gaps.append(f"{identity}:{status}")
    if not requirement_ids or len(set(requirement_ids)) != len(requirement_ids):
        gaps.append(f"{identity}:MISSING_OR_DUPLICATE_REQUIREMENTS")
    if set(requirement_ids) - set(requirements):
        gaps.append(f"{identity}:ORPHAN_REQUIREMENT")
    if not nodeids or len(set(nodeids)) != len(nodeids):
        gaps.append(f"{identity}:MISSING_OR_DUPLICATE_TESTS")
    for node in nodeids:
        if node not in collected:
            gaps.append(f"{identity}:ORPHAN_NODE:{node}")
        elif outcomes.get(node) != "PASSED":
            gaps.append(f"{identity}:{outcomes.get(node, 'NO_EXECUTION_RECEIPT')}:{node}")
    return gaps, requirement_ids
