"""Create one immutable prospective plan after all candidate source edits finish.

Metadata only: no numerical library, market value decoding, fitting or evaluation.
The plan itself does not establish verification, consumption or scientific support.
"""

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from frem_s10.identities import parse_json
from frem_s10.runtime.catalog import required_obligations
from frem_s10.runtime.provenance import (
    create_code_manifest,
    create_environment_receipt,
    expected_code_files,
    read_object,
    save_object,
)

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent


def reference(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.relative_to(ROOT)), "sha256": identity}


def main() -> None:
    output = DEST / "EXECUTION_PLAN.json"
    snapshot = DEST / "candidate_source"
    if output.exists() or snapshot.exists():
        raise RuntimeError("existing candidate must remain immutable; create a versioned successor plan")
    catalog = DEST / "CATALOG_CANDIDATE_PLAN.json"
    plan = json.loads(catalog.read_text())
    mandatory = set(required_obligations("run-development-campaign", target_platform="macOS"))
    external = {"TEST-081", "TEST-086", "TEST-090", "TEST-093", "TEST-096"}
    gaps = [row["test_id"] for row in plan["obligations"] if row["test_id"] in mandatory - external
            and row["coverage_review"] != "BOUNDED_COMPLETE"]
    if gaps:
        raise RuntimeError("authored candidate coverage is still incomplete: " + ", ".join(gaps))
    code = create_code_manifest(ROOT, ROOT)
    environment = create_environment_receipt(ROOT, ROOT, evidence_scope="RETROSPECTIVE_DEVELOPMENT")
    configuration = save_object(ROOT, parse_json((ROOT / "successors/frem_s10/src/frem_s10/science.json").read_bytes()))
    source_request = json.loads((DEST / "source_request.json").read_text())
    clock = source_request["calendar"]
    calendar = save_object(ROOT, {"schema": "s10-calendar-receipt-v1", "design_id": "FREM-S10-DESIGN-4-CL16",
        "evidence_scope": "RETROSPECTIVE_DEVELOPMENT", "status": "PASS", "identity": clock["sha256"]})
    hashes = [item["sha256"] for item in source_request["sources"]]
    data = {label: save_object(ROOT, {"schema": "s10-data-manifest-v1", "assets": ["SPY", "TLT", "GLD"],
            "source_hashes": hashes, "values_before": cutoff})
            for label, cutoff in (("data_pilot", "2010-01-01"), ("data_campaign", "2017-01-01"))}
    owned = {}
    for role, name in (("consumption_root", "consumption"), ("output_root", "outputs"), ("work_root", "controller")):
        path = DEST / "execution" / name
        path.mkdir(parents=True, exist_ok=False)
        owned[role] = str(path)
    artifacts = {"code": code, "environment": environment, "configuration": configuration,
        "calendar": calendar, "clock": clock,
        "source_admission": reference(DEST / "source_admission_attempt1/admission.json"),
        "catalog": reference(catalog), **data}
    document = {"schema": "s10-continuation-plan-v1", "artifact_root": str(ROOT), "source_root": str(ROOT),
        **owned, "attempt_prefix": "FREM-S10-production-20261016", "artifacts": artifacts}
    snapshot.mkdir()
    for source in expected_code_files(ROOT):
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (DEST / "candidate_source_manifest.json").write_text(json.dumps(read_object(code, ROOT), indent=2) + "\n")
    output.write_text(json.dumps(document, indent=2) + "\n")
    identity = {"created_at": datetime.now(UTC).isoformat(), "execution_plan": reference(output), "code": code,
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "snapshot": str(snapshot.relative_to(ROOT)),
        "source_commit_contains_changes": False, "source_identity_basis": "EXACT_COMPLETE_CONTENT_HASH_SNAPSHOT",
        "scientific_freeze": False, "scientific_consumption": False, "protected_access": False,
        "verification": "PENDING_ACTUAL_FIXED_COMMANDS_AFTER_FRESH_RESOURCE_ADMISSION"}
    (DEST / "CANDIDATE_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"plan": reference(output), "code": code, "environment": environment, "status": "PROSPECTIVE_PLAN_ONLY"}, indent=2))


if __name__ == "__main__":
    main()
