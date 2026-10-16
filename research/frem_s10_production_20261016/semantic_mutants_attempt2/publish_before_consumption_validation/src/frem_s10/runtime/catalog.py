"""Fail-closed catalog completeness checks over explicitly supplied test evidence.

This checks traceability and outcomes, not whether a reviewer chose a scientifically
sufficient test. It never authorizes execution or upgrades engineering to empirical
evidence; source-bound verification/admission must bind the supplied receipts.
"""

from dataclasses import dataclass
from datetime import date

from frem_s10.errors import ContractError
from frem_s10.identities import JSONObject, object_json


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
