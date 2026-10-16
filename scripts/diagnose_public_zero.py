"""Read-only, standard-library diagnosis of the completed S2 public study.

Parameter swaps are descriptive arithmetic, never selectable forecasts.
No fitting, calibration search, protected data access or outcome-based repair.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.audit_engineering_evidence import digest, nll, require
from scripts.diagnose_frem_loss import describe, shapley_reductions, weighted


def align(rows, controls):
    def key(row):
        return row["time"], row["asset"]

    lookup = {key(row): row for row in controls}
    require(len(lookup) == len(controls), "Duplicate control rows")
    require(len({key(r) for r in rows}) == len(rows), "Duplicate candidate rows")
    require(set(lookup) == {key(r) for r in rows}, "Unpaired rows")
    aligned = [lookup[key(row)] for row in rows]
    require(
        all(
            all(a[k] == b[k] for k in ("actual", "label_time", "asset_id", "target_month"))
            for a, b in zip(rows, aligned)
        ),
        "Outcome or timing mismatch",
    )
    return aligned


def decompose(rows, zero):
    fields = ("prediction", "scale", "student_df")
    scores = {}
    for mask in range(8):
        losses = []
        for a, b in zip(rows, zero):
            values = [float((b if mask & (1 << i) else a)[key]) for i, key in enumerate(fields)]
            losses.append(nll(float(a["actual"]), *values))
        scores[mask] = weighted(rows, losses)
    return dict(zip(fields, shapley_reductions(scores, 3)))


def audit(root):
    run = root / "outputs/frem_s2_development_20261011_attempt3"
    sources, cells, memory = {}, {}, {}
    monthly = {}
    for seed in (11, 29):
        tables = {}
        for receipt in sorted((run / f"seed_{seed}").glob("*/receipt.json")):
            sources[str(receipt.relative_to(root))] = digest(receipt)
            data = json.loads(receipt.read_text())
            # Receipt layout is checked explicitly; do not silently omit models.
            results = data["metrics"]
            for name, result in results.items():
                path = receipt.parent / Path(result["prediction_path"]).name
                require(digest(path) == result["prediction_sha256"], "Prediction hash mismatch")
                sources[str(path.relative_to(root))] = digest(path)
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                require(
                    max(
                        abs(
                            nll(
                                float(r["actual"]),
                                float(r["prediction"]),
                                float(r["scale"]),
                                float(r["student_df"]),
                            )
                            - float(r["nll"])
                        )
                        for r in rows
                    )
                    < 1e-10,
                    "Stored NLL mismatch",
                )
                tables[name] = rows
        require(len(tables) == 12 and "zero" in tables, "Incomplete model grid")
        for name, rows in tables.items():
            zero = align(rows, tables["zero"])
            cells[f"{seed}/{name}"] = {
                **describe(rows),
                "zero_gap_decomposition": decompose(rows, zero),
                "coverage95": weighted(rows, [float(r["coverage_0.95"]) for r in rows]),
            }
            for month in sorted({r["target_month"] for r in rows}):
                subset = [r for r in rows if r["target_month"] == month]
                monthly.setdefault(month, {}).setdefault(name, []).append(
                    weighted(subset, [float(r["nll"]) for r in subset])
                )
        p = run / f"seed_{seed}/frozen_residual/checkpoint/successor.json"
        sources[str(p.relative_to(root))] = digest(p)
        state = json.loads(p.read_text())
        records = state["memory"]["records"]
        rows = tables["frozen_residual"]
        grid = state["calibration"]["memory"]["candidates"]
        best_zero = min(c["validation_nll"] for c in grid if c["weight"] == 0)
        memory[str(seed)] = {
            "occupancy": len(records),
            "records_per_asset": dict(Counter(r["asset_id"] for r in records)),
            "label_time_range": [
                min(r["label_time"] for r in records),
                max(r["label_time"] for r in records),
            ],
            "age_days_at_first_decision": [
                (min(int(r["time"]) for r in rows) - t) / 86400000000000
                for t in [max(r["label_time"] for r in records), min(r["label_time"] for r in records)]
            ],
            "neighbor_counts": dict(Counter(r["memory_neighbors"] for r in rows)),
            "selected": state["calibration"]["memory"]["selected"],
            "validation_penalty_by_weight": {
                str(w): min(c["validation_nll"] for c in grid if c["weight"] == w) - best_zero
                for w in sorted({c["weight"] for c in grid})
            },
            "correction_rms": weighted(rows, [float(r["memory_residual"]) ** 2 for r in rows]) ** 0.5,
            "base_error_rms": cells[f"{seed}/frozen_residual"]["rmse"],
        }
    months = {}
    for month, models in monthly.items():
        scores = {name: sum(values) / len(values) for name, values in models.items()}
        require(all(len(v) == 2 for v in models.values()), "Missing seed")
        months[month] = {
            "nll": scores,
            "winner": min(scores, key=scores.get),
            "zero_beats_frozen": scores["zero"] < scores["frozen_residual"],
            "zero_beats_all": all(scores["zero"] < s for n, s in scores.items() if n != "zero"),
        }
    return {
        "scope": "Descriptive S2 Great Recession only; not new model selection or causal proof",
        "sources_sha256": sources,
        "cells": cells,
        "memory": memory,
        "months": months,
        "new_fits": 0,
        "protected_outcomes_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
