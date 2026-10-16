"""Build an authored test plan from static source, never an observed-pass claim."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
TESTS = ROOT / "successors/frem_s10/tests"
review = json.loads((DEST / "VERIFICATION_CATALOG_REVIEW.json").read_text())
requirements_path = ROOT / "research/frem_s10_20261015/design_v2/requirements.json"
requirements = json.loads(requirements_path.read_text())
known = {}
added = {}
for path in sorted(TESTS.glob("test_*.py")):
    for item in ast.parse(path.read_text()).body:
        if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
            continue
        family = f"tests/{path.name}::{item.name}"
        known[family] = path
        match = re.search(r"TEST-(\d{3}(?:/\d{3})*)", ast.get_docstring(item) or "")
        if match:
            for number in match.group(1).split("/"):
                added.setdefault("TEST-" + number, set()).add(family)

def add(identity, module, *names):
    families = [f"tests/{module}.py::{name}" for name in names]
    assert all(family in known for family in families), families
    added.setdefault(identity, set()).update(families)

for identity in ("TEST-006", "TEST-007", "TEST-073", "TEST-075"):
    for family in known:
        if family.startswith("tests/test_production_data_audit.py::"):
            added.setdefault(identity, set()).add(family)
for identity in ("TEST-070", "TEST-073", "TEST-076", "TEST-078", "TEST-080", "TEST-095"):
    for family in known:
        if family.startswith("tests/test_production_runtime.py::"):
            added.setdefault(identity, set()).add(family)
for identity in ("TEST-073", "TEST-084", "TEST-085", "TEST-086", "TEST-087", "TEST-092", "TEST-095"):
    for family in known:
        if family.startswith("tests/test_production_verification_audit.py::"):
            added.setdefault(identity, set()).add(family)
for family in known:
    if family.startswith("tests/test_catalog_readiness_audit.py::"):
        added.setdefault("TEST-094", set()).add(family)
for identity in ("TEST-021", "TEST-022", "TEST-073", "TEST-090"):
    for family in known:
        if family.startswith("tests/test_model_archive_audit.py::"):
            added.setdefault(identity, set()).add(family)
add("TEST-081", "test_report_audit", "test_report_retains_all_comparisons_and_rejects_unsupported_claim_scope", "test_report_rejects_missing_or_duplicate_required_rows")
add("TEST-002", "test_data_audit", "test_context_is_asset_major_and_excludes_current_open_label")
add("TEST-002", "test_readback_audit", "test_resigned_header_axes_cannot_relabel_saved_values")
add("TEST-073", "test_declarations_audit", "test_stale_code_and_modified_artifact_hash_rejected", "test_environment_change_cannot_reuse_previous_verification", "test_config_threshold_change_and_missing_review_rejected")
for identity in ("TEST-076", "TEST-080", "TEST-084"):
    for family in known:
        if family.startswith("tests/test_production_gate_catalog.py::"):
            added.setdefault(identity, set()).add(family)
for identity, module in (("TEST-095", "test_report_claims_audit"), ("TEST-092", "test_preservation_audit"), ("TEST-093", "test_production_gate_catalog"), ("TEST-096", "test_mutation_check_audit")):
    for family in known:
        if family.startswith(f"tests/{module}.py::"):
            added.setdefault(identity, set()).add(family)
add("TEST-062", "test_analysis_adapter_audit", "test_analysis_bundle_has_exact_axes_ablation_order_and_shared_streams", "test_unreported_control_cell_cannot_hide_bad_pairing")
add("TEST-062", "test_endpoints_audit", "test_pooled_score_bootstrap_uses_original_date_weights_and_shared_indices")
add("TEST-096", "test_model_audit", "test_issued_forecast_is_not_recomputed_after_library_changes")
add("TEST-096", "test_catalog_numerics_audit", "test_tiny_negative_simplex_face_is_corrected_only_within_tolerance")
add("TEST-096", "test_supervisor_audit", "test_cancel_terminates_owned_child")
add("TEST-096", "test_ledger_audit", "test_new_buyers_do_not_receive_opening_dividend_and_repeated_marks_charge_nothing", "test_entering_share_dividend_before_endpoint_sale_and_negative_path")
add("TEST-096", "test_math_audit", "test_exact_fee_roots_and_batch_agree_with_decimal")
add("TEST-096", "test_experiment_audit", "test_exact_192_registry_and_no_combined_or_cost_shifted_ablations")

# These cases now have explicit authored assertions that address the recorded
# missing case. No status in this plan means any newly added test has executed.
new_complete = {2, 3, 4, 6, 7, 9, 12, 13, 17, 18, 20, 21, 22, 25, 33, 36, 39, 43, 44, 47, 50, 56, 57, 60, 61, 62, 70, 72, 73, 74, 75, 76, 77, 79, 80, 82, 84, 85, 87, 88, 92, 95}
external = {
    81: "Actual saved-report regeneration must be independently audited through completed qualification.",
    86: "Actual fixed package build, archive, offline clean install, import and CLI receipt must be independently audited.",
    90: "Actual fixed maximum-shape resource qualification must be independently audited after a passing pilot.",
    93: "Actual full local verification under denied network/data access and an empty protected fixture must be independently audited; local verification does not establish remote CI or Linux parity.",
    96: "Fresh current-source baseline and all fixed normative isolated targeted mutant executions must be independently audited; the earlier four-mutation receipt is historical only.",
}
rows = []
for original in review["obligations"]:
    identity = original["test_id"]
    number = int(identity[-3:])
    families = set(node.split("[", 1)[0] for node in original["nodeids"])
    families.update(original["queued_test_families"])
    families.update(added.get(identity, set()))
    missing = families - set(known)
    assert not missing, (identity, missing)
    complete = original["status"] == "VERIFIED" or number in new_complete
    coverage = "BOUNDED_COMPLETE" if complete else "PARTIAL" if families else "UNIMPLEMENTED"
    if number in external:
        coverage = "PARTIAL"
    rows.append({
        "test_id": identity,
        "requirement_ids": original["requirement_ids"],
        "test_families": sorted(families),
        "coverage_review": coverage,
        "execution_state": "PENDING_CURRENT_SOURCE_BOUND_OBSERVATIONS",
        "original_review_status": original["baseline_status"],
        "remaining_evidence": (
            external[number] if number in external else
            "Authored bounded assertions; every parameter case must pass current exact-source verification."
            if complete else original["remaining_evidence"]
        ),
    })
result = {
    "schema": "s10-operation-catalog-plan-v1",
    "design": "FREM-S10-design4-CL16",
    "requirement_ids": [row["id"] for row in requirements["requirements"]],
    "requirement_registry_sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
    "evidence_scope": "GENERATED_ENGINEERING_PLAN_NOT_EXECUTION",
    "historical_catalog": review["baseline"],
    "historical_catalog_sha256": review["baseline_sha256"],
    "preservation_audits": [
        {"path": str((DEST / name).relative_to(ROOT)), "sha256": hashlib.sha256((DEST / name).read_bytes()).hexdigest()}
        for name in ("historical_manifest_audit_attempt1.json", "preservation_audit_v3.json")
    ],
    "historical_mutation_receipt": {
        "path": str((DEST / "semantic_mutants_attempt2/receipt.json").relative_to(ROOT)),
        "sha256": hashlib.sha256((DEST / "semantic_mutants_attempt2/receipt.json").read_bytes()).hexdigest(),
        "current_source_execution": False,
    },
    "obligations": rows,
    "stage_rules": {
        "prepare-precision_exclusions": [*(f"TEST-{i:03}" for i in range(41, 62)), "TEST-067", "TEST-068", "TEST-069", "TEST-071", "TEST-081", "TEST-083", "TEST-090"],
        "both_operations_release_only": ["TEST-091"],
        "TEST-089": "Required only when the actual verified target is Linux; no Linux evidence inferred from macOS.",
        "TEST-081": "Campaign requires actual saved-report regeneration independently audited through qualification, plus listed generated contract tests.",
        "TEST-086": "Both operations require actual fixed build/archive/offline install/import/CLI evidence independently audited by the verifier, plus listed contract tests.",
        "TEST-090": "Campaign requires actual qualified maximum shapes/resource counts independently audited, plus listed bounded shape/archive checks; excluded from pilot to preserve CL09 ordering.",
        "TEST-091": "No optimization benefit is claimed. Original before/after allocations/time obligation remains open and is not replaced by passing equivalence tests.",
        "TEST-093": "Both operations require actual restricted full local verification with denied network, denied raw/research/output data reads except exact evidence work and retained build wheel, and an empty protected fixture. No remote CI or Linux parity is claimed.",
        "TEST-096": "Both operations require independently audited current-source baseline plus fixed normative targeted mutants; every listed generated guard test must also pass. The fixed list retains the four earlier faults and adds the seven missing normative cases within20 mutants and600seconds.",
    },
    "authorizes_scientific_execution": False,
}
output = DEST / "CATALOG_CANDIDATE_PLAN.json"
output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
print(json.dumps({"authored_scope_counts": dict(Counter(row["coverage_review"] for row in rows)), "observed_current_passes": 0, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
