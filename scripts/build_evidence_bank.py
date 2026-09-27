"""Rebuild reports/evidence_bank/{EVIDENCE_TABLE.csv, ARTIFACT_INDEX.json} from source artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "reports" / "evidence_bank"

LEGACY = "legacy_lean_matrix"
LEGACY_STATUS = "LEAKAGE_CONTAMINATED"


def _matrix_rows() -> list[dict]:
    with (ROOT / "protocol" / "EXPERIMENT_MANIFEST.csv").open(newline="") as f:
        manifest = [r for r in csv.DictReader(f) if r["status"] == "COMPLETED"]
    groups: dict[tuple, list[float]] = defaultdict(list)
    metric: dict[tuple, str] = {}
    for r in manifest:
        if r["variant"] != "default":
            continue
        val = json.loads((ROOT / r["result_path"]).read_text()).get("metrics", {}).get("primary_value")
        if not isinstance(val, (int, float)):
            continue
        key = (r["profile"], r["model_id"], r["dataset"])
        groups[key].append(float(val))
        metric[key] = r["primary_metric"]
    rows = []
    for (profile, mid, ds), vals in sorted(groups.items()):
        rows.append(
            {
                "family": "matrix",
                "generation": LEGACY,
                "status": LEGACY_STATUS,
                "profile": profile,
                "model_id": mid,
                "dataset": ds,
                "metric": metric[(profile, mid, ds)],
                "n_cells": len(vals),
                "n_distinct_values": len({round(v, 15) for v in vals}),
                "value": sum(vals) / len(vals),
                "artifact": "protocol/EXPERIMENT_MANIFEST.csv",
            }
        )
    return rows


def _lockbox_rows() -> list[dict]:
    res = json.loads((ROOT / "runs" / "lockbox" / "LOCKBOX_RESULTS.json").read_text())
    status = {"synthetic_track_c": "INVALID_LEAKAGE", "etf_track_a": "NEGATIVE"}
    rows = []
    for cell in res["cells"]:
        for fam, mid, m in (
            ("lockbox_selected", cell["selected_forecast"], cell["selected_metrics"]),
            ("lockbox_baseline", "B_PERSISTENCE", cell["persistence"]),
        ):
            rows.append(
                {
                    "family": fam,
                    "generation": "lockbox_single_open",
                    "status": status[cell["dataset"]],
                    "profile": "lockbox",
                    "model_id": mid,
                    "dataset": cell["dataset"],
                    "metric": m["metric_name"],
                    "n_cells": 1,
                    "n_distinct_values": 1,
                    "value": m["primary_value"],
                    "artifact": "runs/lockbox/LOCKBOX_RESULTS.json",
                }
            )
    return rows


def _repaired_rows() -> list[dict]:
    path = ROOT / "runs" / "repaired" / "REPAIRED_RESULTS.json"
    if not path.exists():
        return []
    res = json.loads(path.read_text())
    chall, bench = res["config"]["primary_contrast"]
    rows = []
    for c in res["contrasts"]:
        if c["challenger"] != chall or c["benchmark"] != bench:
            continue
        sig = c["bh_q"] <= 0.05 and c["oos_r2_vs_benchmark"] > 0
        rows.append(
            {
                "family": "repaired_primary",
                "generation": res["generation"],
                "status": "EXPLORATORY_SIGNIFICANT" if sig else "EXPLORATORY_NULL",
                "profile": "repaired",
                "model_id": f"{chall}_vs_{bench}",
                "dataset": c["dataset"],
                "metric": "oos_r2_vs_benchmark",
                "n_cells": c["n_folds"],
                "n_distinct_values": c["n_folds"],
                "value": c["oos_r2_vs_benchmark"],
                "artifact": "runs/repaired/REPAIRED_RESULTS.json",
            }
        )
    return rows


def _artifact_index() -> dict:
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    out = {}
    for rel in sorted(files):
        p = ROOT / rel
        if not p.is_file() or rel == "reports/evidence_bank/ARTIFACT_INDEX.json":
            continue
        out[rel] = {"bytes": p.stat().st_size, "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
    return out


def main() -> int:
    rows = _matrix_rows() + _lockbox_rows() + _repaired_rows()
    BANK.mkdir(parents=True, exist_ok=True)
    with (BANK / "EVIDENCE_TABLE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    index = _artifact_index()
    (BANK / "ARTIFACT_INDEX.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    print(f"EVIDENCE_TABLE.csv: {len(rows)} rows; ARTIFACT_INDEX.json: {len(index)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
