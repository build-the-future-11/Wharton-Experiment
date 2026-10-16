"""Save this blocked local session truthfully; never run or rewrite research."""

import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from frem_s10.identities import parse_json
from frem_s10.runtime.preservation import audit_preservation
from frem_s10.runtime.provenance import audit_code, audit_environment

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
REL = DEST.relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    assert not (DEST / "RECEIPT.json").exists(), "preserve completed session receipt"
    plan = json.loads((DEST / "EXECUTION_PLAN.json").read_text())
    attempt = json.loads((DEST / "controller_attempt1.json").read_text())
    assert attempt["status"] == "BLOCKED" and attempt["consumed"] is False
    assert digest(DEST / "EXECUTION_PLAN.json") == attempt["plan_sha256"]
    host = json.loads((ROOT / attempt["host"]["path"]).read_text())
    assert digest(ROOT / attempt["host"]["path"]) == attempt["host"]["sha256"]
    assert host["status"] == "BLOCKED"
    assert not list(Path(plan["consumption_root"]).iterdir())
    assert not list(Path(plan["output_root"]).iterdir())
    automation_path = Path("/Users/ryan/.codex/automations/frem-s10-guarded-research-continuation/automation.toml")
    automation = tomllib.loads(automation_path.read_text())
    assert automation["status"] == "ACTIVE" and automation["kind"] == "heartbeat"
    assert attempt["plan_sha256"] in automation["prompt"]
    write(DEST / "automation_receipt.json", {
        "schema": "s10-conditional-continuation-schedule-v1",
        "automation_id": automation["id"], "status": automation["status"],
        "kind": automation["kind"], "rrule": automation["rrule"],
        "target_thread_id": automation["target_thread_id"],
        "observed_configuration_sha256": digest(automation_path),
        "configuration_is_live_and_may_change": True,
        "plan_sha256": attempt["plan_sha256"],
        "created_by_app_tool": True,
        "execution_requires_running_local_app_and_host": True,
    })

    status = "IMPLEMENTED_SCOPED_VERIFIED_FULL_VERIFICATION_RESOURCE_BLOCKED"
    session = {
        "status": status,
        "report": f"{REL}/REPORT.md", "checklist": f"{REL}/FLAW_CHECKLIST.md",
        "receipt": f"{REL}/RECEIPT.json", "execution_plan": f"{REL}/EXECUTION_PLAN.json",
        "plan_sha256": attempt["plan_sha256"], "code": plan["artifacts"]["code"],
        "candidate_identity": f"{REL}/CANDIDATE_IDENTITY.json",
        "source_identity_basis": "EXACT_COMPLETE_CONTENT_HASH_SNAPSHOT_ON_DISCLOSED_DIRTY_CHECKOUT",
        "source_admission": f"{REL}/source_admission_attempt1/admission.json",
        "data_scope": "RETROSPECTIVE_DEVELOPMENT_CURRENT_VINTAGE_2005_2016",
        "sessions_admitted": 3021, "early_closes": 25,
        "point_in_time_certified": False, "observed_fills_certified": False,
        "redistribution_rights_established": False,
        "catalog_plan": f"{REL}/CATALOG_CANDIDATE_PLAN.json",
        "authored_catalog_coverage_not_pass_counts": {
            "BOUNDED_COMPLETE": 89, "PARTIAL": 6, "UNIMPLEMENTED_LINUX": 1,
        },
        "complete_current_candidate_verification": False,
        "actual_current_candidate_build": False,
        "actual_current_candidate_eleven_mutation_campaign": False,
        "actual_full_scale_qualification": False,
        "scientific_status": "NOT_EVALUATED", "scientific_freeze": False,
        "scientific_attempt_consumed": False, "market_fits": 0, "market_ledgers": 0,
        "protected_access": False, "positive_market_result_established": False,
        "independent_empirical_replication": False, "full_research_complete": False,
        "latest_controller_receipt": f"{REL}/controller_attempt1.json",
        "latest_resource_receipt": attempt["host"]["path"],
        "resource_samples_mib": [item["available_mib"] for item in host["samples"]],
        "resource_reasons": host["reasons"],
        "automation": {"id": automation["id"], "status_at_session_end": "ACTIVE",
                       "interval_minutes": 30, "receipt": f"{REL}/automation_receipt.json"},
        "remaining_execution_gates": [
            "Fresh unchanged512MiB/non-rising-pageouts host admission",
            "Complete actual exact-source suite/static/package/eleven-mutant verification",
            "One-use pre2010 precision with a passing identifiable result",
            "One-use full-calendar generated qualification including maximum support/action shapes",
            "One exact frozen development campaign and retained numerical analysis",
        ],
        "remaining_external_claim_gates": [
            "Untouched point-in-time confirmation", "Independent empirical reproduction",
            "Broad-universe and physical-execution validity", "Data redistribution rights",
            "Algorithmic novelty", "Final empirical paper and research release",
        ],
        "conservative_stop": "A mutation-host refusal after earlier verification commands stops for review; it is not silently retried.",
        "next_action": "Use only the frozen plan through the existing controller. The active heartbeat checks every30minutes and pauses on source drift, nonretryable failure or terminal result.",
    }
    state_path = ROOT / "RESEARCH_STATE.json"
    state = json.loads(state_path.read_text())
    assert "s10_production_20261016" not in state
    assert state["scientific_status"] == "NOT_EVALUATED"
    state["updated"] = "2026-10-16"
    state["phase"] = "s10_production_integrated_full_verification_resource_blocked"
    state["current_iteration"] = "FREM-S10"
    state["current_request_scope"] = "Complete production integration, verification and source integrity admission; run the exact fixed research sequence once all resource and prerequisite gates pass. Preserve all old results."
    state["compute_authorized_this_session"] = "Local reversible integration and bounded generated metadata tests executed. The actual fixed controller refused fresh host admission before complete verification or science. Conditional one-use precision, generated qualification and development remain authorized only through the exact saved plan and unchanged gates; no S6 waiver applies."
    state["s10_production_20261016"] = session
    state["decisions"].append("S10 production20261016: exact candidate plan and retrospective source integrity admission saved; complete current-source verification and all numerical work remain resource gated. All ten normative semantic faults plus retained omitted-exit fault required. Conditional heartbeat never retries consumed or valid negative work.")
    state["failures"].append({"date": "2026-10-16", "stage": "S10_fixed_production_controller_attempt1", "status": "BLOCKED_UNCONSUMED", "detail": "101.25/138.234375/136.5MiB below512MiB; pageouts867775/867809/867879. Stopped before full verification and scientific execution.", "receipt": f"{REL}/controller_attempt1.json"})
    state["history"].append({"date": "2026-10-16", "event": "S10 production path integrated; bounded source/catalog/packaging/receipt/mutation-contract tests and actual isolation probe pass. Admitted3021 retrospective sessions with25 early closes. Exact immutable candidate plan saved; actual controller blocked unconsumed before verification. Scheduled resource-gated continuation. All old results and scientific boundaries retained.", "receipt": f"{REL}/RECEIPT.json"})
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

    handoff = f"""

## 16 October 2026 S10 production integration (current)

Read `{REL}/REPORT.md`, `FLAW_CHECKLIST.md`, `RECEIPT.json`, `CANDIDATE_IDENTITY.json` and `EXECUTION_PLAN.json`. This section supersedes older readiness descriptions above without rewriting their historical evidence. The exact plan is `{attempt['plan_sha256']}`; code identity is `{plan['artifacts']['code']['sha256']}`. A complete source snapshot is retained on the disclosed dirty checkout; the old Git base does not contain this candidate.

Production source/calendar admission, exact verification/catalog receipts, prospective declarations, consumption, workers, weekly archives, saved numerical analysis, generated qualification and persistent continuation are integrated. Small scoped tests/static checks and the actual network/data/output/research denial probe passed. All96 obligations are mapped:89 authored bounded-complete,6partial,1Linux-unimplemented; these are coverage assessments, not current full-suite passes. Current full-suite/build/eleven-mutant execution is still pending. Preserve every failed scoped attempt. Eleven fixed mutants cover all ten normative faults plus omitted exit cost. TEST093 has no waiver. TEST091 optimization timing remains open; no speed improvement is claimed.

Source integrity admission covers3021 SPY/TLT/GLD sessions in2005–2016 and25 early closes, with separately pinned community calendar bytes. It is retrospective current-vintage evidence: no PIT certification, observed fills, vendor latency, payment-date cash settlement or redistribution rights are established. Pilot values stay before2010, campaign before2017, and protected confirmation remains closed. The ES claim concerns total-return law only, not separate price/dividend identification.

The actual controller attempt exited3 before verification:101.25/138.234375/136.5MiB and rising pageouts, against the unchanged512MiB three-sample rule. No S10 scientific freeze, consumed attempt, market fit, market ledger or protected access occurred. The local heartbeat `{automation['id']}` is ACTIVE at this handoff and checks every30minutes using only the exact plan. It remains quiet while blocked and pauses on nonretryable failure, source/environment drift or terminal result. The app and host must remain running. Do not weaken guards, kill unrelated work, acquire new data or rerun a valid negative study.

Resume with `rtk proxy successors/frem_s10/.venv/bin/python -B -m frem_s10 advance --manifest {REL}/EXECUTION_PLAN.json --execute`. Its fixed order is complete verification, one precision pilot, generated full-scale qualification, then one development campaign. Nested sandbox and /bin/ps require the supported execution context. If mutation preflight loses headroom after earlier engineering commands, the retained failure stops for review; automatic retry is deliberately conservative. Independent replication, unused confirmation and empirical paper/release remain incomplete.
"""
    with (ROOT / "docs/CODEX_HANDOFF.md").open("a") as stream:
        stream.write(handoff)
    with (ROOT / "docs/STATIC_VERIFICATION.md").open("a") as stream:
        stream.write(f"""

## 16 October 2026 S10 production candidate

Source identity `{plan['artifacts']['code']['sha256']}` and plan `{attempt['plan_sha256']}` are retained under `{REL}`. Scoped runtime32, catalog metadata22, mutation-contract23, preservation11, packaging39 and data-boundary29 checks passed in their cited command receipts; these sets overlap and must not be added into a whole-suite count. Root three-file Ruff/format/strict-mypy passed (`root_quality_attempt5.json`), and the final restricted controller-import case passed (`final_controller_import_attempt1.log`). The actual sandbox boundary probe denied network and historical data/outputs/research. These observations do not establish current complete-suite, current offline build, eleven-mutant execution, full-scale qualification or empirical support.

The current candidate's fixed full verification is queued behind unchanged host admission. Actual controller attempt1 exited3 unconsumed at101.25/138.234375/136.5MiB with rising pageouts. The preceding274-test full-suite result belongs only to the preserved15October candidate. Historical preservation checks cover486 references/482unique immutable files plus the prior repair211evidence files/79source files, with mutable canonical-history entries explicitly treated separately. See the final readback and `{REL}/RECEIPT.json`. No positive market result or independent empirical verification is claimed.
""")
    with (DEST / "REPORT.md").open("a") as stream:
        stream.write(f"""

## Final session state

The candidate is saved as an exact source snapshot, with plan SHA256 `{attempt['plan_sha256']}` and code identity `{plan['artifacts']['code']['sha256']}`. All96 obligations are mapped;89 are assessed as authored bounded-complete,6remain partial and1is unimplemented Linux parity. These are designed-coverage classifications, not a claim that this candidate's entire suite passed. The current eleven-mutant producer's23 bounded contract checks and strict static checks passed, and the final controller import-side-effect case passed. Actual complete-suite/build/mutation and maximum-scale execution remain required.

The real controller invocation exited3 at its verification resource gate. Its fresh observations were101.25,138.234375 and136.5MiB; pageouts increased867775→867809→867879. No verification workload, numerical research worker, scientific freeze or consumed scientific attempt began. The exact host receipt and returned outcome are preserved in `controller_attempt1.json` and the controller journal.

The app created heartbeat `{automation['id']}`, ACTIVE at session close, to check every30minutes against this exact plan. It advances only through fresh resource and existing prerequisite gates, stays quiet on unchanged resource refusal, and pauses on source drift, nonretryable failure or terminal result. The computer and Codex app must remain running. A resource refusal inside mutation preparation after earlier verification commands conservatively stops for review; it is not an automatic repeat of the engineering suite.

Canonical state and handoff were updated additively. Every prior source snapshot, invalid or failed check, completed negative/inconclusive result and decision remains retained. This is completed production integration with bounded verification and source integrity admission; complete current-candidate verification and scientific execution are still blocked by resources.
""")

    audit_code(plan["artifacts"]["code"], artifact_root=ROOT, source_root=ROOT)
    audit_environment(plan["artifacts"]["environment"], artifact_root=ROOT, source_root=ROOT,
                      evidence_scope="RETROSPECTIVE_DEVELOPMENT")
    catalog = parse_json((DEST / "CATALOG_CANDIDATE_PLAN.json").read_bytes())
    audit_preservation(catalog["preservation_audits"], ROOT)
    prior = "0" * 64
    count = 0
    for line in (ROOT / "research/events.jsonl").read_text().splitlines():
        record = json.loads(line)
        recorded = record.pop("sha256")
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert record["previous"] == prior and hashlib.sha256(encoded).hexdigest() == recorded
        prior, count = recorded, count + 1
    source_inventory = json.loads((DEST / "candidate_source_manifest.json").read_text())
    write(DEST / "FINAL_READBACK.json", {
        "status": "PASS", "scope": "SOURCE_ENVIRONMENT_HISTORY_AND_PRESERVATION_METADATA_ONLY",
        "journal_records": count, "journal_head": prior, "code_identity": plan["artifacts"]["code"],
        "historical_references": 486, "historical_unique_files": 482,
        "old_repair_evidence_files": 211, "old_repair_source_files": 79,
        "mismatches": [], "scientific_execution": False,
    })
    files = [path for path in DEST.rglob("*") if path.is_file()
             and "execution" not in path.relative_to(DEST).parts
             and "__pycache__" not in path.parts]
    files += [ROOT / name for name in ("RESEARCH_STATE.json", "docs/CODEX_HANDOFF.md", "docs/STATIC_VERIFICATION.md", "research/events.jsonl")]
    write(DEST / "RECEIPT.json", {
        "schema": "s10-production-session-v1", "created_at": datetime.now(UTC).isoformat(),
        "status": status, "science": "NOT_EVALUATED", "source": plan["artifacts"]["code"],
        "environment": plan["artifacts"]["environment"],
        "execution_plan_sha256": attempt["plan_sha256"],
        "actual_controller_result": attempt, "scheduled_continuation": automation["id"],
        "full_current_candidate_verification": False, "scientific_attempt_consumed": False,
        "protected_access": False, "original_thresholds_unchanged": True,
        "retained_source_inventory": source_inventory,
        "files": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)},
        "mutable_root_file_hashes_are_session_snapshots": True,
        "controller_journal_is_append_only_and_excluded_from_fixed_session_file_inventory": True,
    })
    print(json.dumps({"status": status, "receipt_sha256": digest(DEST / "RECEIPT.json"),
                      "retained_session_files": len(files), "journal_head": prior,
                      "scientific_execution": False}, indent=2))


if __name__ == "__main__":
    main()
