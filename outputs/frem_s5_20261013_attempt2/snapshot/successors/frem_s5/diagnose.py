"""Read-only arithmetic audit of retained S3/S4 results, no model imports."""

import csv
import json
import math
import statistics as st
from pathlib import Path

from successors.frem_portfolio_v1.evidence import sha256, write

ROOT = Path(__file__).resolve().parents[2]
S3 = ROOT / "outputs/frem_s3_20261011_attempt1"
S4 = ROOT / "outputs/frem_s4_20261012_attempt1"


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def corr(x, y):
    mx, my = st.fmean(x), st.fmean(y)
    xx, yy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(xx * yy) if xx * yy else None


def main():
    out = ROOT / "research/frem_s5_20261013/diagnostic.json"
    sources, cells, gates = {}, [], []
    for fold in ("fresh_2023", "fresh_2024"):
        for seed in (11, 29, 47):
            cell = S3 / "public" / fold / str(seed) / "frem"
            receipt = json.loads((cell / "result.json").read_text())
            tables = {}
            for arm in ("off", "combined", "mean", "risk", "random", "recent"):
                p = cell / f"{arm}.csv"
                h = sha256(p)
                if h != receipt["results"][arm]["sha256"]:
                    raise ValueError("Changed original S3 prediction")
                sources[str(p.relative_to(ROOT))] = h
                tables[arm] = rows(p)
            base = tables["off"]
            for arm, table in tables.items():
                if [(r["time"], r["asset"], r["actual"]) for r in base] != [
                    (r["time"], r["asset"], r["actual"]) for r in table
                ]:
                    raise ValueError("Unpaired diagnostic rows")
                sp = cell / f"{arm}_selection.json"
                selection = json.loads(sp.read_text())
                sources[str(sp.relative_to(ROOT))] = sha256(sp)
                null = next(c for c in selection["candidates"] if c["weight"] == 0 and c["power"] == 0)
                correction = [float(r["prediction"]) - float(b["prediction"]) for r, b in zip(table, base)]
                residual = [float(b["actual"]) - float(b["prediction"]) for b in base]
                cells.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "validation_gain": null["nll"] - selection["selected"]["nll"],
                        "evaluation_gain": st.fmean(
                            float(b["nll"]) - float(r["nll"]) for b, r in zip(base, table)
                        ),
                        "correction_residual_correlation": corr(correction, residual),
                        "correction_rms": math.sqrt(st.fmean(v * v for v in correction)),
                        "mean_effective_neighbors": st.fmean(float(r["effective_neighbors"]) for r in table),
                        "mean_age_days": st.fmean(float(r["memory_age_days"]) for r in table),
                        "dates": len({r["time"] for r in table}),
                        "rows": len(table),
                    }
                )
            for arm in ("old_distance", "euclidean_distance", "matched_random"):
                p = S4 / fold / str(seed) / f"{arm}_selection.json"
                sources[str(p.relative_to(ROOT))] = sha256(p)
                s = json.loads(p.read_text())
                c = [r for r in s["candidates"] if r["weight"] > 0]
                gates.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "selected_weight": s["selected"]["weight"],
                        "best_worst_half_gain": max(min(r["half_gains"]) for r in c),
                        "candidates": c,
                    }
                )
    summary = {}
    for fold in ("fresh_2023", "fresh_2024"):
        subset = [c for c in cells if c["fold"] == fold]
        summary[fold] = {
            arm: {
                k: st.fmean(c[k] for c in subset if c["arm"] == arm)
                for k in (
                    "validation_gain",
                    "evaluation_gain",
                    "correction_rms",
                    "mean_effective_neighbors",
                    "mean_age_days",
                )
            }
            for arm in ("combined", "mean", "risk", "random", "recent")
        }
    report = {
        "status": "COMPLETE_RETAINED_EVIDENCE_DIAGNOSIS",
        "cells": cells,
        "s4_gates": gates,
        "summary": summary,
        "source_sha256": sources,
        "interpretation": [
            "S3 nonzero corrections do not consistently generalize.",
            "S4 final zero weights cannot discriminate deployed retrieval metrics.",
            "Correlation and arm comparisons are descriptive, not isolated causal proof.",
            "S5 will test relative expert reliability instead of residual mean correction.",
        ],
        "protected_access": False,
        "new_model_fits": 0,
    }
    write(out, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
