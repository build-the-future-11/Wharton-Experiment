"""Read-only, standard-library audit of completed S3 evidence; no model imports.

Run from the repository root. The destination must not exist.
"""

import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/frem_s3_20261011_attempt1"
HASHES = {}


def read(path):
    payload = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return payload


def js(path):
    return json.loads(read(path))


def rows(path):
    return list(csv.DictReader(read(path).decode().splitlines()))


def nll(row):
    y, mu, scale, nu = (float(row[k]) for k in ("actual", "prediction", "scale", "student_df"))
    assert all(map(math.isfinite, (y, mu, scale, nu))) and scale > 0 and nu > 2
    return (math.lgamma(nu / 2) - math.lgamma((nu + 1) / 2)
            + 0.5 * math.log(nu * math.pi) + math.log(scale)
            + (nu + 1) / 2 * math.log1p(((y - mu) / scale) ** 2 / nu))


def score(table):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in table:
        groups[int(r["time"]) // 86_400_000_000_000][r["asset_id"]].append(float(r["nll"]))
    return statistics.mean(statistics.mean(statistics.mean(v) for v in a.values()) for a in groups.values())


def iso(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat()


def main(destination):
    if destination.exists():
        raise FileExistsError(destination)
    freeze = js(RUN / "freeze.json")
    for name, digest in freeze["sources"].items():
        read(ROOT / name)
        assert HASHES[name] == digest, name
    cells, reconstructed_rows, max_error = [], 0, 0.0
    for fold in ("fresh_2023", "fresh_2024"):
        for seed in (11, 29, 47):
            cell = RUN / "public" / fold / str(seed) / "frem"
            receipt, fit = js(cell / "result.json"), js(cell / "fit.json")
            states = [js(cell / f"combined_{phase}.json") for phase in ("initial", "validated", "final")]
            selection = js(cell / "combined_selection.json")
            validation = rows(cell / "validation_base.csv")
            tables, scores = {}, {}
            for arm, result in receipt["results"].items():
                path = cell / f"{arm}.csv"
                table = rows(path)
                assert HASHES[str(path.relative_to(ROOT))] == result["sha256"], path
                assert len(table) == result["rows"]
                for row in table:
                    max_error = max(max_error, abs(nll(row) - float(row["nll"])))
                    reconstructed_rows += 1
                tables[arm], scores[arm] = table, score(table)
                assert abs(scores[arm] - result["nll"]) < 1e-10
            identity = lambda r: tuple(r[k] for k in ("time", "asset_id", "actual", "label_time"))
            reference = [identity(r) for r in tables["off"]]
            assert len(reference) == len(set(reference))
            for table in tables.values():
                assert [identity(r) for r in table] == reference
            raw = rows(cell / "raw_base.csv")
            assert [identity(r) for r in raw] == reference
            assert abs(score(raw) - receipt["raw_nll"]) < 1e-10
            for row in raw:
                max_error = max(max_error, abs(nll(row) - float(row["nll"])))
                reconstructed_rows += 1
            off_choice = next(x for x in selection["candidates"] if x["weight"] == x["power"] == 0)
            combined = tables["combined"]
            final = states[-1]
            keys = [r["key"] for r in final["records"]]
            dims = {len(k) for k in keys}
            assert len(dims) == 1
            dimension = next(iter(dims))
            norms = [math.sqrt(sum(x * x for x in k)) for k in keys]
            assert max(abs(x - 1) for x in norms) < 1e-5
            cells.append({
                "fold": fold, "seed": seed,
                "partition_rows": fit["partition_rows"],
                "partition_max_label_research_coordinates": [iso(t) for t in fit["partition_max_label"]],
                "base_selected_epoch": fit["base"]["selected_epoch"],
                "base_completed_epochs": fit["base"]["completed_epochs"],
                "base_calibration": fit["calibration"]["selected"],
                "validation_months": len({r["time"] for r in validation}),
                "combined_selection": selection["selected"],
                "validation_gain_over_off": off_choice["nll"] - selection["selected"]["nll"],
                "mean_gate": statistics.mean(float(r["memory_gate"]) for r in combined),
                "mean_risk_power": statistics.mean(float(r["risk_power"]) for r in combined),
                "mean_effective_neighbors": statistics.mean(float(r["effective_neighbors"]) for r in combined),
                "neighbors_observed": sorted({int(r["neighbors"]) for r in combined}),
                "mean_neighbor_age_days": statistics.mean(float(r["memory_age_days"]) for r in combined),
                "final_bank_per_asset": dict(collections.Counter(r["asset"] for r in final["records"])),
                "final_bank_matured": final["matured"], "final_bank_evicted": final["evicted"],
                "key_dimension": dimension, "final_key_norm_range": [min(norms), max(norms)],
                "unit_key_mean_squared_distance_bound": 4 / dimension,
                "unit_key_similarity_lower_bound": math.exp(-4 / dimension),
                "nll_by_arm": scores, "raw_nll": score(raw),
                "memory_gain_by_arm": {a: scores["off"] - s for a, s in scores.items()},
                "calibration_penalty": scores["off"] - score(raw),
                "combined_gain_over_matched_random": scores["random"] - scores["combined"],
            })
    assert max_error < 1e-10
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_DESCRIPTIVE_AUDIT_NO_MODEL_CHANGES",
        "scope": "Six saved S3 FREM cells, all 14 arms, plus raw references; no new predictions, fits or protected data",
        "frozen_sources_verified": len(freeze["sources"]),
        "prediction_rows_reconstructed": reconstructed_rows,
        "max_nll_reconstruction_error": max_error,
        "cells": cells,
        "interpretation": "Snapshot diagnostics and algebraic bounds, not a causal intervention or proof that proposed changes help",
        "protected_outcomes_accessed": False,
        "sources_sha256": HASHES,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"cells", "sources_sha256"}}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
