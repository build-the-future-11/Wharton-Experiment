"""Fail-closed checks for stubs, manifest, and evidence."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

STUB_MARKERS = ("TODO", "FIXME", "STUB", "TBD", "XXX")
PRIMARY_CLAIM_RE = re.compile(r"primary.*(beats|outperforms|superior)", re.I)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_completion_audit(repo_root: Path | str | None = None) -> list[str]:
    root = Path(repo_root) if repo_root else _repo_root()
    errors: list[str] = []

    manifest = root / "protocol" / "EXPERIMENT_MANIFEST.csv"
    if not manifest.exists():
        errors.append("Missing protocol/EXPERIMENT_MANIFEST.csv")
    else:
        with manifest.open(newline="") as f:
            rows = list(csv.DictReader(f))
        completed = [r for r in rows if r.get("status") == "COMPLETED"]
        for r in completed:
            rp = r.get("result_path", "")
            if not rp:
                errors.append(f"COMPLETED row {r['run_id']} missing result_path")
                continue
            receipt = root / rp
            if not receipt.exists():
                errors.append(f"Missing receipt for {r['run_id']}: {rp}")

    reports_dir = root / "reports" / "generated"
    if not reports_dir.exists():
        errors.append("Missing reports/generated (run scripts/build_reports.sh)")

    for card in sorted((root / "model_cards").glob("M*.md")):
        text = card.read_text()
        if any(m in text for m in STUB_MARKERS):
            errors.append(f"Stub marker in {card.name}")

    for md in (root / "reports").rglob("*.md"):
        if "generated" in md.parts:
            body = md.read_text()
            if PRIMARY_CLAIM_RE.search(body) and "evidence_run_id" not in body:
                errors.append(f"Primary claim without evidence in {md.relative_to(root)}")

    registry = root / "protocol" / "MODEL_REGISTRY.yaml"
    if not registry.exists():
        errors.append("Missing protocol/MODEL_REGISTRY.yaml")

    return errors


def main() -> int:
    errors = run_completion_audit()
    if errors:
        for e in errors:
            print(f"AUDIT FAIL: {e}", file=sys.stderr)
        return 1
    print("AUDIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
