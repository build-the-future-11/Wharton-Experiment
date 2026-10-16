"""Independent standard-library score/identity audit; never fits or selects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def density_nll(row):
    y, mu, s, nu = (float(row[k]) for k in ("actual", "prediction", "scale", "student_df"))
    if not all(math.isfinite(x) for x in (y, mu, s, nu)) or s <= 0 or nu <= 2:
        raise ValueError("Invalid saved distribution")
    return (
        math.lgamma(nu / 2)
        - math.lgamma((nu + 1) / 2)
        + 0.5 * math.log(nu * math.pi)
        + math.log(s)
        + (nu + 1) / 2 * math.log1p(((y - mu) / s) ** 2 / nu)
    )


def audit(root, run):
    freeze = json.loads((run / "freeze.json").read_text())
    if any(sha(root / name) != value for name, value in freeze["sources"].items()):
        raise ValueError("Frozen source changed")
    predecessor = json.loads((root / "research/frem_s3_20261011/predecessor_hashes.json").read_text())
    if any(sha(root / name) != value for name, value in predecessor.items()):
        raise ValueError("Predecessor changed")
    for receipt in run.glob("mechanism/*/*/result.json"):
        record = json.loads(receipt.read_text())
        if sha(receipt.parent / "generated.csv") != record["source_sha256"]:
            raise ValueError("Generated source changed")
    for receipt in run.glob("public/*/*/*/result.json"):
        record = json.loads(receipt.read_text())
        for name, entry in record["results"].items():
            if sha(receipt.parent / f"{name}.csv") != entry["sha256"]:
                raise ValueError("Public predictions changed")
    max_error = 0.0
    rows_checked = 0
    scores = {}
    paths = {}
    for path in sorted(run.rglob("*.csv")):
        if path.name in {"generated.csv", "residuals.csv", "validation_base.csv"}:
            continue
        with path.open() as stream:
            reader = csv.DictReader(stream)
            if "nll" not in (reader.fieldnames or []):
                continue
            by_date = {}
            seen = set()
            for row in reader:
                identity = (row["time"], row["asset"])
                if identity in seen:
                    raise ValueError("Duplicate saved observation")
                seen.add(identity)
                loss = density_nll(row)
                max_error = max(max_error, abs(loss - float(row["nll"])))
                by_date.setdefault(row["time"], []).append(loss)
                rows_checked += 1
                if int(row["label_time"]) <= int(row["time"]):
                    raise ValueError("Invalid label timing")
            scores[str(path.relative_to(run))] = sum(sum(v) / len(v) for v in by_date.values()) / len(by_date)
            paths[str(path.relative_to(run))] = sha(path)
    if max_error > 1e-9:
        raise ValueError("Saved score reconstruction failed")
    processes = [json.loads(p.read_text()) for p in run.rglob("process.json")]
    if any(p["returncode"] or p["termination"] for p in processes):
        raise ValueError("Failed process in claimed complete run")
    return {
        "status": "VERIFIED",
        "prediction_rows": rows_checked,
        "maximum_nll_error": max_error,
        "scores": scores,
        "prediction_sha256": paths,
        "old_artifacts_unchanged": len(predecessor),
        "frozen_sources_unchanged": len(freeze["sources"]),
        "maximum_sampled_rss_mib": max(p["sampled_peak_rss_mib"] for p in processes),
        "processes": len(processes),
        "protected_outcomes_accessed": False,
        "scope": "Score and artifact verification, not independent experiment replication",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.run)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"scores", "prediction_sha256"}}))
