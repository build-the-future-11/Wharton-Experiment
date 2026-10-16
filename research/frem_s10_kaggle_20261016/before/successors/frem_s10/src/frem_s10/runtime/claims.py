"""Validate the existing saved report's bounded claims without fitting or arrays.

This is a provenance and claim-scope check, not a substitute for supervised
numerical regeneration. A hash-linked computation is neither an independently
reproduced experiment nor empirical source admission.
"""

import hashlib
from pathlib import Path

from frem_s10.errors import IntegrityError
from frem_s10.identities import JSONObject, JSONValue, object_json
from frem_s10.runtime.artifacts import read_verified_json
from frem_s10.runtime.declarations import _hash, _schema

DISCLAIMER = (
    "This is a calculation over the supplied evidence. It does not establish source admission, "
    "empirical support, confirmation, the original NLL claim or future profitability."
)


def _comparison_registry(raw: JSONValue, key: str, expected: set[tuple[str, int]]) -> None:
    if not isinstance(raw, list):
        raise IntegrityError("saved report requires the complete comparison registry")
    actual: list[tuple[str, int]] = []
    for item in raw:
        row = object_json(item)
        name, block = row.get(key), row.get("block")
        if not isinstance(name, str) or type(block) is not int:
            raise IntegrityError("saved report comparison identity is invalid")
        actual.append((name, block))
    if len(actual) != len(expected) or set(actual) != expected:
        raise IntegrityError("saved report omitted or duplicated a required comparison")


def validate_report_claims(audit: JSONObject, report: JSONObject) -> JSONObject:
    """Return only the claim map justified by the existing typed report envelope.

    The producer calls this after numerical regeneration; the report CLI calls it
    after independently reading the exact audit/report hashes. Free-form prose is
    not treated as a new claim or evidence kind. The canonical scope and disclaimer
    must remain present; scientific, independent and runtime-test attestations are
    never inferred from these two computational artifact types.
    """
    _schema(
        audit,
        {
            "schema",
            "evidence_sha256",
            "evidence_scope",
            "status",
            "cells",
            "primary",
            "ablations",
            "bootstrap",
            "source_authority_approved",
            "numerical_disposition",
            "report",
        },
        "campaign report audit",
    )
    _schema(report, {"schema", "markdown", "evidence_sha256", "evidence_scope"}, "rendered report")
    identity = _hash(audit["evidence_sha256"])
    scope = audit["evidence_scope"]
    if (
        audit["schema"] != "s10-campaign-audit-v1"
        or audit["status"] != "COMPLETE"
        or type(audit["cells"]) is not int
        or audit["cells"] != 192
        or audit["source_authority_approved"] is not False
        or scope not in ("GENERATED_ENGINEERING", "KERNEL_REPLAY")
        or report["schema"] != "s10-rendered-report-v1"
        or report["evidence_sha256"] != identity
        or report["evidence_scope"] != scope
    ):
        raise IntegrityError("report requires complete saved computational evidence with matching scope")
    reference = _schema(audit["report"], {"path", "sha256"}, "report reference")
    if reference["path"] != "report.json":
        raise IntegrityError("canonical saved report reference required")
    _hash(reference["sha256"])
    primary = _schema(
        audit["primary"],
        {
            "disposition",
            "h1",
            "h2",
            "h3",
            "reason",
            "evidence_scope",
            "original_nll_hypothesis_satisfied",
            "confirmation_claim",
            "profitability_guarantee",
        },
        "saved primary claims",
    )
    ablations = _schema(
        audit["ablations"],
        {"comparisons", "evidence_scope", "confirmation_claim", "changes_primary"},
        "saved ablation claims",
    )
    numeric_scope = (
        "GENERATED_ENGINEERING_ANALYSIS"
        if scope == "GENERATED_ENGINEERING"
        else "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    )
    disposition = primary["disposition"]
    if (
        disposition not in ("SUPPORTED", "MIXED", "NEGATIVE", "INCONCLUSIVE", "INVALID")
        or audit["numerical_disposition"] != disposition
        or primary["evidence_scope"] != numeric_scope
        or ablations["evidence_scope"] != numeric_scope
        or any(
            primary[key] is not False
            for key in ("original_nll_hypothesis_satisfied", "confirmation_claim", "profitability_guarantee")
        )
        or ablations["confirmation_claim"] is not False
        or ablations["changes_primary"] is not False
    ):
        raise IntegrityError("computational report cannot certify empirical or independent scientific claims")
    _comparison_registry(
        ablations["comparisons"], "ablation_id", {(f"A{i}", b) for i in range(1, 9) for b in (2, 4)}
    )
    if disposition == "INVALID":
        if primary["h1"] is not None or primary["h2"] is not None or primary["h3"] or not primary["reason"]:
            raise IntegrityError("invalid evidence cannot carry supported primary endpoints")
    else:
        for name, controls in (("h1", ("global_stack", "equal")), ("h2", ("equal100", "buy_hold100"))):
            hypothesis = _schema(primary[name], {"status", "comparisons", "point_gates"}, "hypothesis")
            _comparison_registry(
                hypothesis["comparisons"], "control", {(c, b) for c in controls for b in (2, 4)}
            )
        h3 = primary["h3"]
        if not isinstance(h3, list) or [object_json(row).get("fold_id") for row in h3] != [
            "A",
            "B",
            "POOLED",
        ]:
            raise IntegrityError("saved report must retain every descriptive fold")
    _bootstrap_references(audit["bootstrap"])
    text_scope = (
        "GENERATED_ENGINEERING" if scope == "GENERATED_ENGINEERING" else "SOURCE_AUTHORITY_UNVERIFIED"
    )
    markdown = report["markdown"]
    required = (
        "# S10 computational evidence summary\n\n",
        f"Evidence scope: **{text_scope}**. Inventory SHA-256: `{identity}`.",
        f"Conditional numerical disposition: **{disposition}**.",
        DISCLAIMER,
    )
    if (
        not isinstance(markdown, str)
        or not markdown.startswith(required[0])
        or any(markdown.count(part) != 1 for part in required[1:])
    ):
        raise IntegrityError("canonical computational report scope, evidence link and disclaimer required")
    return {
        "evidence_kind": "SAVED_COMPUTATIONAL_EVIDENCE",
        "evidence_sha256": identity,
        "report_sha256": reference["sha256"],
        "evidence_scope": scope,
        "conditional_numerical_disposition": disposition,
        "source_authority_approved": False,
        "empirical_support_claimed": False,
        "confirmation_claimed": False,
        "independent_reproduction_claimed": False,
        "runtime_test_pass_claimed": False,
    }


def _bootstrap_references(raw: JSONValue) -> None:
    if not isinstance(raw, list) or len(raw) != 4:
        raise IntegrityError("all four saved uncertainty evidence links required")
    seen: set[tuple[str, int]] = set()
    for item in raw:
        row = _schema(item, {"fold", "block", "path", "sha256"}, "bootstrap reference")
        fold, block = row["fold"], row["block"]
        if fold not in ("A", "B") or type(block) is not int or block not in (2, 4):
            raise IntegrityError("unknown saved uncertainty axis")
        assert isinstance(fold, str)
        if row["path"] != f"bootstrap-{fold}-{block}.json" or (fold, block) in seen:
            raise IntegrityError("duplicate or noncanonical uncertainty evidence link")
        _hash(row["sha256"])
        seen.add((fold, block))


def read_saved_report(directory: Path, audit_sha256: str) -> tuple[JSONObject, JSONObject]:
    """Verify the exact local evidence references, then return bounded report claims.

    This metadata-only export does not rerun statistics or attest external authority.
    Full saved-array verification remains in the supervised saved-audit operation.
    """
    audit = object_json(read_verified_json(directory / "audit.json", audit_sha256))
    reference = _schema(audit.get("report"), {"path", "sha256"}, "report reference")
    if reference["path"] != "report.json":
        raise IntegrityError("canonical saved report reference required")
    report = object_json(
        read_verified_json(directory / "report.json", _hash(reference["sha256"]), maximum_bytes=8_388_608)
    )
    claims = validate_report_claims(audit, report)
    # Stream only these four explicitly referenced artifacts. No arrays are
    # decoded and no market source, unreferenced path or model is accessed.
    bootstrap = audit["bootstrap"]
    assert isinstance(bootstrap, list)
    for raw in bootstrap:
        row = object_json(raw)
        path = directory / str(row["path"])
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 67_108_864:
            raise IntegrityError("bounded retained uncertainty evidence is missing")
        with path.open("rb") as stream:
            identity = hashlib.file_digest(stream, "sha256").hexdigest()
        if identity != row["sha256"]:
            raise IntegrityError("retained uncertainty evidence hash differs")
    claims["analysis_sha256"] = _hash(audit_sha256)
    return report, claims
