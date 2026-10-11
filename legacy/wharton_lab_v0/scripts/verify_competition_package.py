"""Fail on stale artifacts; separately report human/source submission blockers."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = ROOT / "audit/astra_2026-09-27"
    qa = json.loads((audit / "ARTIFACT_QA.json").read_text())
    evidence = json.loads((ROOT / "runs/competition_stress/RESULTS.json").read_text())
    errors = []
    for group in [qa["source_sha256"], evidence["source_sha256"]]:
        for p, sha in group.items():
            if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha:
                errors.append(f"stale artifact source: {p}")
    for p, meta in qa["pdfs"].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != meta["sha256"]:
            errors.append(f"PDF bytes changed: {p}")
    sources = json.loads((ROOT / "protocol/RECOVERED_DOCUMENTS.json").read_text())
    for item in sources["sources"]:
        p = ROOT / item["file"]
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"missing or changed recovered source: {item['file']}")
    questions = re.findall(r"^### (\d+)\.", (ROOT / "reports/competition_package/JUDGE_QA.md").read_text(), re.M)
    if questions != [f"{i:02d}" for i in range(1, 61)]: errors.append("Expected exactly 60 unique ordered questions")
    expected = {(s["id"], p, c) for s in evidence["config"]["scenarios"] for p in evidence["config"]["policies"] for c in evidence["config"]["cost_bps"]}
    actual = {(r["scenario"], r["policy"], r["cost_bps"]) for r in evidence["results"]}
    if actual != expected or len(evidence["results"]) != len(expected): errors.append("Incomplete or duplicate stress matrix")
    for r in evidence["results"]:
        if len(r["trajectory"]) != evidence["config"]["months"]: errors.append("Missing trajectory months")
        if abs(r["total_paid"] + r["unpaid_at_end"] - r["total_obligations"]) > 1e-9:
            errors.append("Payment accounting discrepancy")
    gate = json.loads((ROOT / "protocol/SUBMISSION_GATES.json").read_text())
    blockers = [k for k, done in gate["gates"].items() if done is not True]
    result = {"artifact_consistency": "FAIL" if errors else "PASS", "errors": errors,
              "stress_cells": len(actual), "questions": len(questions),
              "submission_status": "BLOCKED" if blockers else "REQUIRES_FINAL_HUMAN_SIGNOFF",
              "blockers": blockers,
              "note": "Passing artifact checks does not certify scientific validity, client suitability, authorship, eligibility or authority."}
    (audit / "PACKAGE_GATE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
