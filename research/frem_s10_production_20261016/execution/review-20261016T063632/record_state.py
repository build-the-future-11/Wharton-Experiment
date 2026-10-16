"""Record the observed source rejection; do not change or execute the candidate."""
import hashlib
import json
import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path

root = Path.cwd()
directory = Path(__file__).resolve().parent
relative = directory.relative_to(root).as_posix()
key = "s10_source_rejection_20261016T063632"
path = root / "RESEARCH_STATE.json"
raw = path.read_bytes()
state = json.loads(raw)
assert key not in state
source = json.loads((directory / "SOURCE_DRIFT.json").read_text())
automation_path = Path("/Users/ryan/.codex/automations/frem-s10-guarded-research-continuation/automation.toml")
automation = tomllib.loads(automation_path.read_text())
assert automation["status"] == "PAUSED"
for name in ("consumption", "outputs"):
    assert not list((directory.parent / name).iterdir())
record = {
    "status": "FIXED_CANDIDATE_SOURCE_DRIFT_REVIEW_REQUIRED",
    "scheduled_at": "2026-10-16T06:36:32.319Z",
    "controller_exit_code": 2, "controller_status": "REJECTED_UNCONSUMED",
    "reason": "complete current code inventory differs from verified snapshot",
    "report": f"{relative}/REPORT.md", "receipt": f"{relative}/RECEIPT.json",
    "source_comparison": f"{relative}/SOURCE_DRIFT.json",
    "rejection": "research/frem_s10_production_20261016/execution/heartbeat-20261016T063632.json",
    "observed_source_file_count": source["observed_file_count"],
    "added_files": len(source["added"]), "modified_files": len(source["changed"]),
    "missing_files": len(source["missing"]),
    "preserved_original_candidate_files": 118,
    "original_candidate_matches_frozen_manifest": True,
    "comparison_observed_at": source["observed_at"],
    "comparison_is_not_exact_rejection_time_reconstruction": True,
    "automation_status": "PAUSED", "automatic_retry_permitted": False,
    "controller_invocations_this_wakeup": 1,
    "source_changed_by_this_heartbeat": False,
    "scientific_attempt_consumed": False, "protected_access": False,
    "next_action": "Review ongoing portability work and establish a separate exact candidate, full verification and supported platform qualification before explicit resumption; do not silently update or reuse the original plan.",
}
state[key] = record
state["updated"] = "2026-10-16"
if state.get("phase") == "s10_production_integrated_full_verification_resource_blocked":
    state["phase"] = "s10_fixed_candidate_source_drift_review_required"
production = state["s10_production_20261016"]
production["status"] = record["status"]
production["latest_controller_receipt"] = record["rejection"]
production["latest_gate_failure"] = key
production["next_action"] = record["next_action"]
production["automation"]["current_status"] = "PAUSED"
production["automation"]["pause_reason"] = "SOURCE_INVENTORY_DRIFT"
production["automation"]["pause_receipt"] = record["receipt"]
state["failures"].append({"date": "2026-10-16", "stage": "S10_scheduled_fixed_source_admission", "status": "REJECTED_UNCONSUMED_REVIEW_REQUIRED", "detail": record["reason"], "receipt": record["rejection"]})
state["decisions"].append("Pause the fixed S10 continuation on16October2026 source inventory mismatch. Preserve the original118-file candidate, fourteen additions and seven modified-file observations. No automatic source repair, replacement freeze or retry; portability revision requires separate review and verification.")
state["history"].append({"date": "2026-10-16", "event": "Scheduled S10 controller rejected changed current source before execution. Subsequent capture found14 added/7 modified/0 missing files, while all118 archived original candidate files still match. Automation paused; original consumption and output namespaces remain empty. New portability work remains untouched.", "receipt": record["receipt"]})
assert path.read_bytes() == raw, "canonical state changed concurrently; preserve and reconcile"
temporary = directory / "updated_state.json"
with temporary.open("x") as stream:
    json.dump(state, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
os.replace(temporary, path)
handoff = f"""

## 16 October 2026 12:06 IST fixed S10 source rejection

The once-per-wakeup controller rejected current source with `REJECTED_UNCONSUMED` (exit2), before verification or scientific execution. Read `{relative}/REPORT.md` and `SOURCE_DRIFT.json`. The app automation `frem-s10-guarded-research-continuation` is now PAUSED; earlier ACTIVE descriptions above are historical.

The original plan and all118 archived candidate source files remain intact. The later metadata-only capture found132 current files:14added,7modified,0missing. README documents Kaggle portability work and explicitly says the old plan is no longer a current-source plan. These observations are timestamped after rejection and are not exact rejection-time reconstruction or authorship evidence. Current changes were retained, not repaired or rolled back.

Do not invoke the old plan again against the changed working tree or adopt a new candidate automatically. Review the portability revision, create a separately identified candidate and complete its verification/platform qualification before an explicitly resumed sequence. No scientific attempt was consumed and no protected outcomes were accessed. This is an engineering admission rejection, not a scientific result. Record-decision evidence is `{relative}/decision.json`.
"""
with (root / "docs/CODEX_HANDOFF.md").open("a") as stream:
    stream.write(handoff)
with (root / "docs/STATIC_VERIFICATION.md").open("a") as stream:
    stream.write(f"""

## 16 October 2026 fixed-candidate source identity rejection

The06:36UTC heartbeat ran the controller once; exit2 `REJECTED_UNCONSUMED` reported a complete current-code inventory mismatch. The plan and original manifest still match their fixed hashes. A later source capture found14additions/7modifications/0missing files; all118 preserved original candidate files match. This metadata audit is not verification of the new portability revision and does not transfer previous passing tests to it. No tests, numerical workload or scientific worker ran during this heartbeat. Automation pause was confirmed from the app and persisted configuration. See `{relative}/RECEIPT.json` and timestamped source comparison. Canonical state and append-only decision history record the stop; old results are unchanged.
""")

def digest(candidate: Path) -> str:
    with candidate.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

plan = root / "research/frem_s10_production_20261016/EXECUTION_PLAN.json"
assert digest(plan) == "852e85d6e4889687d7ae555afdadb21b1a563a8392c206a17ca75ebb3a32eba2"
old_receipt = root / "research/frem_s10_production_20261016/RECEIPT.json"
assert digest(old_receipt) == "ad631cfa8b7d073256f442721f032c5419ddbefa5895afd0324abe558933996c"
files = [p for p in directory.rglob("*") if p.is_file()]
files += [root / name for name in ("RESEARCH_STATE.json", "docs/CODEX_HANDOFF.md", "docs/STATIC_VERIFICATION.md", "research/events.jsonl", record["rejection"])]
receipt = record | {"schema": "s10-source-rejection-session-v1", "recorded_at": datetime.now(UTC).isoformat(),
    "original_production_receipt_unchanged": True,
    "files": {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)},
    "canonical_file_hashes_are_session_snapshots": True}
with (directory / "RECEIPT.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
print(json.dumps({"status": record["status"], "automation": automation["status"], "original_receipt_unchanged": True,
    "receipt": record["receipt"], "receipt_sha256": digest(directory / "RECEIPT.json")}, indent=2))
