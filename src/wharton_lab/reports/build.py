"""Aggregate runs/*/receipt.json into reports/generated scoreboards."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _mean(xs: list[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


def build_reports(repo_root: Path | str) -> Path:
    root = Path(repo_root)
    out_dir = root / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for receipt in sorted(root.glob("runs/**/receipt.json")):
        data = json.loads(receipt.read_text())
        metrics = data.get("metrics") or {}
        summaries.append(
            {
                "run_id": data.get("run_id"),
                "model_id": data.get("model_id"),
                "profile": data.get("profile"),
                "dataset": data.get("dataset"),
                "variant": data.get("variant"),
                "fold": data.get("fold"),
                "seed": data.get("seed"),
                "metric": metrics.get("metric_name"),
                "value": metrics.get("primary_value"),
                "receipt": str(receipt.relative_to(root)),
                "data_provenance": data.get("data_provenance"),
            }
        )
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))

    # Scoreboards by profile × model (default variants only for clarity)
    by_prof: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_name: dict[str, str] = {}
    for s in summaries:
        if s["value"] is None or not isinstance(s["value"], (int, float)):
            continue
        if s.get("variant") not in (None, "default"):
            continue
        mid = str(s["model_id"])
        prof = str(s.get("profile") or "unknown")
        by_prof[prof][mid].append(float(s["value"]))
        if s.get("metric"):
            metric_name[mid] = str(s["metric"])

    scoreboard_rows = []
    md = [
        "# Experiment summary",
        "",
        f"Runs indexed: {len(summaries)}",
        "",
        "## Status distinctions",
        "",
        "- Implemented ≠ trained ≠ validated.",
        "- Forecasting improvement ≠ investment outperformance.",
        "- Simulated funding success ≠ guarantee.",
        "- Pilot/smoke ≠ FULL completion.",
        "",
        "## Scoreboards (default variant means)",
        "",
    ]
    for prof in sorted(by_prof):
        md.append(f"### Profile `{prof}`")
        md.append("")
        md.append("| model | metric | n | mean |")
        md.append("|---|---|---:|---:|")
        for mid in sorted(by_prof[prof]):
            vals = by_prof[prof][mid]
            mean = _mean(vals)
            mname = metric_name.get(mid, "?")
            md.append(f"| {mid} | {mname} | {len(vals)} | {mean} |")
            scoreboard_rows.append(
                {
                    "profile": prof,
                    "model_id": mid,
                    "metric": mname,
                    "n": len(vals),
                    "mean": mean,
                }
            )
        md.append("")

    md.append("## Receipt index (first 300)")
    md.append("")
    for s in summaries[:300]:
        md.append(
            f"- `{s['run_id']}` {s['model_id']}: {s['metric']}={s['value']} "
            f"(evidence_run_id={s['run_id']})"
        )

    (out_dir / "SUMMARY.md").write_text("\n".join(md) + "\n")
    (out_dir / "scoreboard.json").write_text(json.dumps(scoreboard_rows, indent=2))

    # Manifest rollup
    manifest = root / "protocol" / "EXPERIMENT_MANIFEST.csv"
    if manifest.exists():
        with manifest.open(newline="") as f:
            rows = list(csv.DictReader(f))
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            counts[r.get("profile", "?")][r.get("status", "?")] += 1
        (out_dir / "manifest_status.json").write_text(json.dumps(counts, indent=2, sort_keys=True))

    return out_dir


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    build_reports(root)
    print(f"Wrote reports to {root / 'reports' / 'generated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
