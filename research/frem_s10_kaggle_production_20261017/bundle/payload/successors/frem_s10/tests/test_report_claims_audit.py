"""Small constructed report envelopes cannot acquire unsupported evidence kinds."""

from pathlib import Path

import pytest

from frem_s10.cli import main
from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, object_json
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.claims import DISCLAIMER, read_saved_report, validate_report_claims


def _documents(directory: Path, scope: str = "GENERATED_ENGINEERING") -> tuple[JSONObject, JSONObject]:
    """Typed metadata only, deliberately no empirical or runtime-test evidence."""
    numeric_scope = (
        "GENERATED_ENGINEERING_ANALYSIS"
        if scope == "GENERATED_ENGINEERING"
        else "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    )
    primary: JSONObject = {
        "disposition": "NEGATIVE",
        "h1": {
            "status": "FAIL",
            "comparisons": [{"control": c, "block": b} for c in ("global_stack", "equal") for b in (2, 4)],
            "point_gates": [],
        },
        "h2": {
            "status": "FAIL",
            "comparisons": [{"control": c, "block": b} for c in ("equal100", "buy_hold100") for b in (2, 4)],
            "point_gates": [],
        },
        "h3": [{"fold_id": fold} for fold in ("A", "B", "POOLED")],
        "reason": None,
        "evidence_scope": numeric_scope,
        "original_nll_hypothesis_satisfied": False,
        "confirmation_claim": False,
        "profitability_guarantee": False,
    }
    bootstrap: list[JSONValue] = []
    for fold in ("A", "B"):
        for block in (2, 4):
            name = f"bootstrap-{fold}-{block}.json"
            identity = publish_json(directory, name, {"scope": "constructed-reference-only", "values": []})
            bootstrap.append({"fold": fold, "block": block, "path": name, "sha256": identity})
    text_scope = (
        "GENERATED_ENGINEERING" if scope == "GENERATED_ENGINEERING" else "SOURCE_AUTHORITY_UNVERIFIED"
    )
    report: JSONObject = {
        "schema": "s10-rendered-report-v1",
        "markdown": (
            "# S10 computational evidence summary\n\n"
            f"Evidence scope: **{text_scope}**. Inventory SHA-256: `{'a' * 64}`.\n\n"
            "Conditional numerical disposition: **NEGATIVE**.\n" + DISCLAIMER
        ),
        "evidence_sha256": "a" * 64,
        "evidence_scope": scope,
    }
    audit: JSONObject = {
        "schema": "s10-campaign-audit-v1",
        "evidence_sha256": "a" * 64,
        "evidence_scope": scope,
        "status": "COMPLETE",
        "cells": 192,
        "primary": primary,
        "ablations": {
            "comparisons": [{"ablation_id": f"A{i}", "block": b} for i in range(1, 9) for b in (2, 4)],
            "evidence_scope": numeric_scope,
            "confirmation_claim": False,
            "changes_primary": False,
        },
        "bootstrap": bootstrap,
        "source_authority_approved": False,
        "numerical_disposition": "NEGATIVE",
        "report": {"path": "report.json", "sha256": "b" * 64},
    }
    return audit, report


def _publish(directory: Path, audit: JSONObject, report: JSONObject) -> str:
    identity = publish_json(directory, "report.json", report)
    audit["report"] = {"path": "report.json", "sha256": identity}
    return publish_json(directory, "audit.json", audit)


@pytest.mark.parametrize("scope", ["GENERATED_ENGINEERING", "KERNEL_REPLAY"])
def test_existing_report_claim_map_stays_computational_and_hash_bound(tmp_path: Path, scope: str) -> None:
    audit, report = _documents(tmp_path, scope)
    identity = _publish(tmp_path, audit, report)
    loaded, claims = read_saved_report(tmp_path, identity)
    assert loaded == report
    assert claims["evidence_kind"] == "SAVED_COMPUTATIONAL_EVIDENCE"
    assert claims["conditional_numerical_disposition"] == "NEGATIVE"
    assert claims["evidence_sha256"] == "a" * 64 and claims["analysis_sha256"] == identity
    for name in (
        "source_authority_approved",
        "empirical_support_claimed",
        "confirmation_claimed",
        "independent_reproduction_claimed",
        "runtime_test_pass_claimed",
    ):
        assert claims[name] is False


@pytest.mark.parametrize(
    "defect",
    [
        "static",
        "independent",
        "runtime",
        "confirmation",
        "empirical",
        "scope",
        "source",
        "disclaimer",
        "missing_control",
        "missing_reference",
    ],
)
def test_resigned_unsupported_report_claims_are_rejected(tmp_path: Path, defect: str) -> None:
    audit, report = _documents(tmp_path)
    if defect == "static":
        audit["schema"] = "s10-static-parse-result-v1"
    elif defect == "independent":
        audit["independent_reproduction_claimed"] = True
    elif defect == "runtime":
        report["runtime_test_pass_claimed"] = True
    elif defect == "confirmation":
        object_json(audit["primary"])["confirmation_claim"] = True
    elif defect == "empirical":
        audit["source_authority_approved"] = True
    elif defect == "scope":
        report["evidence_scope"] = "CONFIRMED_EMPIRICAL"
    elif defect == "source":
        report["evidence_sha256"] = "c" * 64
    elif defect == "disclaimer":
        report["markdown"] = str(report["markdown"]).replace(
            DISCLAIMER, "Independently confirmed empirical success."
        )
    elif defect == "missing_control":
        object_json(object_json(audit["primary"])["h1"])["comparisons"] = []
    else:
        audit["bootstrap"] = []
    identity = _publish(tmp_path, audit, report)  # Consistently re-signed; not merely a stale hash.
    with pytest.raises((ContractError, IntegrityError)):
        read_saved_report(tmp_path, identity)


@pytest.mark.parametrize("defect", ["missing_report", "missing_uncertainty", "altered_uncertainty"])
def test_missing_or_corrupted_retained_evidence_blocks_report_export(tmp_path: Path, defect: str) -> None:
    audit, report = _documents(tmp_path)
    identity = _publish(tmp_path, audit, report)
    if defect == "missing_report":
        (tmp_path / "report.json").unlink()
    elif defect == "missing_uncertainty":
        (tmp_path / "bootstrap-A-2.json").unlink()
    else:
        (tmp_path / "bootstrap-A-2.json").write_text('{"changed": true}\n')
    with pytest.raises(IntegrityError):
        read_saved_report(tmp_path, identity)


def test_report_cli_rejects_forged_independent_claim_before_publication(tmp_path: Path) -> None:
    source = tmp_path / "saved"
    source.mkdir()
    audit, report = _documents(source)
    report["independent_reproduction_claimed"] = True
    identity = _publish(source, audit, report)
    request: JSONObject = {
        "schema": "s10-saved-evidence-request-v1",
        "directory": str(source),
        "sha256": identity,
        "output_directory": str(tmp_path / "export"),
    }
    publish_json(tmp_path, "request.json", request)
    assert main(["report", "--manifest", str(tmp_path / "request.json")]) != 0
    assert not (tmp_path / "export/report.json").exists()


def test_claim_check_requires_exact_existing_audit_and_report_fields(tmp_path: Path) -> None:
    audit, report = _documents(tmp_path)
    del audit["source_authority_approved"]
    with pytest.raises(ContractError):
        validate_report_claims(audit, report)
