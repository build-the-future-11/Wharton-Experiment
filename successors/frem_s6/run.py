"""Resource-first, immutable, explicitly invoked S6 development launcher.

No scientific package or data payload is imported/read before admission.
There is no confirmation mode and no automatic expansion of the fixed matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from scripts.public_equity_backtests import memory_snapshot

from .check import source_identities, write


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resource_admission():
    snapshots = []
    for _ in range(3):
        snapshots.append(memory_snapshot())
        time.sleep(1)
    admitted = min(s["available_mib"] for s in snapshots) >= 512 and snapshots[-1]["pageouts"] <= snapshots[0]["pageouts"]
    return {"checked_at": datetime.now(UTC).isoformat(), "samples": snapshots, "admitted": admitted}


def preflight(dataset, protocol, integrity):
    resources = resource_admission()
    snapshots = resources["samples"]
    reasons = []
    if not resources["admitted"]:
        reasons.append("RESOURCE_ADMISSION_FAILED: 512MiB floor or pageouts")
    if dataset is None or not dataset.is_file() or not Path(str(dataset) + ".manifest.json").is_file():
        reasons.append("MISSING_ADMITTED_WINDOW_DATA_AND_MANIFEST")
    if protocol is None or not protocol.is_file():
        reasons.append("MISSING_FROZEN_DEVELOPMENT_PROTOCOL")
    else:
        p = json.loads(protocol.read_text())
        required = ("data_sha256", "fold_rationale", "precision_analysis", "spent_date_inventory",
                    "primary_endpoint", "economic_policy", "limitations")
        if p.get("status") != "FROZEN_DEVELOPMENT" or any(not p.get(k) for k in required):
            reasons.append("INCOMPLETE_DEVELOPMENT_PROTOCOL")
        if p.get("cells") != 24 or p.get("seeds") != [11, 29, 47] or p.get("protected_start") != 1735689600:
            reasons.append("PROTOCOL_MATRIX_OR_PROTECTED_BOUNDARY_MISMATCH")
    if integrity is None or not (integrity / "receipt.json").is_file():
        reasons.append("MISSING_CURRENT_INTEGRITY_RECEIPT")
    else:
        r = json.loads((integrity / "receipt.json").read_text())
        hashes = integrity / "source_hashes.json"
        if r.get("returncode") != 0 or not r.get("sources_unchanged") or not hashes.is_file():
            reasons.append("INTEGRITY_NOT_PASSED")
        elif json.loads(hashes.read_text()) != source_identities():
            reasons.append("SOURCE_CHANGED_SINCE_INTEGRITY_CHECK")
    return {"checked_at": datetime.now(UTC).isoformat(), "samples": snapshots, "reasons": reasons,
            "status": "ADMITTED" if not reasons else "BLOCKED_BEFORE_SCIENTIFIC_IMPORT",
            "market_fits_started": 0, "protected_access": False}


def worker(output):
    freeze = json.loads((output / "freeze.json").read_text())
    admission = json.loads((output / "admission.json").read_text())
    if admission["status"] != "ADMITTED" or source_identities() != freeze["source_hashes"]:
        raise ValueError("worker requires current admitted source freeze")
    write(output / "worker_started.json", {"started_at": datetime.now(UTC).isoformat()})
    for name in ("dataset", "manifest", "protocol"):
        path = Path(freeze[name]["path"])
        if digest(path) != freeze[name]["sha256"]:
            raise ValueError(f"{name} changed after freeze")
    current_resources = resource_admission()
    write(output / "worker_admission.json", current_resources)
    if not current_resources["admitted"]:
        raise ValueError("worker fresh resource admission failed before scientific imports")
    from .data import load_dataset
    from .experiment import (
        economics_from_records,
        ensemble_records,
        paired_date_interval,
        planned_cells,
        run_cell,
        validate_protocol,
    )
    admitted = load_dataset(Path(freeze["dataset"]["path"]))
    data = {**admitted["metadata"], **admitted["partitions"]}
    protocol = json.loads(Path(freeze["protocol"]["path"]).read_text())
    if protocol["data_sha256"] != freeze["dataset"]["sha256"]:
        raise ValueError("protocol dataset identity mismatch")
    validate_protocol(protocol, data, with_xgboost=freeze["with_xgboost"])
    families, receipts = {}, {}
    for name, config in planned_cells(data["train"]["windows"].shape[-1], with_xgboost=freeze["with_xgboost"]):
        raw, receipt = run_cell(data, config, output / name)
        families.setdefault(name.split("_", 1)[1], []).append(raw)
        receipts[name] = receipt
    combined, comparisons, economic_ensembles = {}, {}, {}
    for family, rows in families.items():
        if len(rows) != 3:
            raise ValueError("missing optimization seed")
        combined[family] = ensemble_records(rows)
        economic_ensembles[family] = economics_from_records(combined[family], data["currency"])
        candidate = [r["date_nll"] for r in combined[family] if r["variant"] == "memory"]
        controls = sorted({r["variant"] for r in combined[family]} - {"memory"})
        comparisons[family] = {}
        for control in controls:
            values = [r["date_nll"] for r in combined[family] if r["variant"] == control]
            diffs = [b - a for a, b in zip(candidate, values, strict=True)]
            comparisons[family][control] = {str(block): paired_date_interval(diffs, block_dates=block)
                                            for block in (3, 6)}
    if source_identities() != freeze["source_hashes"]:
        raise ValueError("source changed during execution")
    write(output / "seed_ensembles.json", combined)
    write(output / "comparisons.json", comparisons)
    write(output / "ensemble_economics.json", economic_ensembles)
    write(output / "completion.json", {"status": "FIXED_DEVELOPMENT_MATRIX_COMPLETE", "fits": 24,
                                       "receipts": receipts, "protected_access": False,
                                       "confirmation": "not performed", "novelty": "unestablished",
                                       "laura_allocation": "not certified"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--integrity", type=Path)
    parser.add_argument("--with-xgboost", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        start = time.monotonic()

        def peak_rss():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if platform.system() == "Darwin" else 1024)

        def guard(*_):
            if peak_rss() > 768 or time.monotonic() - start > 7200:
                raise RuntimeError("S6 fixed development resource ceiling exceeded")

        signal.signal(signal.SIGALRM, guard)
        signal.setitimer(signal.ITIMER_REAL, .05, .05)
        try:
            worker(args.output)
            guard()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            write(args.output / "worker_resources.json", {"peak_rss_mib": peak_rss(),
                                                          "seconds": time.monotonic() - start,
                                                          "monitor": "50ms SIGALRM sampled peak checks; native calls cooperative"})
        return
    if not args.execute:
        raise ValueError("explicit --execute required; inspect README and protocol first")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        admission = preflight(args.dataset, args.protocol, args.integrity)
    except Exception as error:
        write(args.output / "failure.json", {"stage": "preflight", "error": repr(error),
                                              "traceback": traceback.format_exc(), "market_fits_started": 0})
        raise
    write(args.output / "admission.json", admission)
    if admission["reasons"]:
        print(json.dumps(admission, indent=2))
        return
    freeze = {"created_at": datetime.now(UTC).isoformat(), "source_hashes": source_identities(),
              "with_xgboost": args.with_xgboost, "python": sys.version, "platform": platform.platform()}
    for name, path in (("dataset", args.dataset), ("manifest", Path(str(args.dataset) + ".manifest.json")),
                       ("protocol", args.protocol)):
        freeze[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    write(args.output / "freeze.json", freeze)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (args.output / "worker.log").open("x") as stream:
        try:
            completed = subprocess.run([sys.executable, "-m", "successors.frem_s6.run", "--worker",
                                        "--output", str(args.output)], env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=7200, check=False)
            if completed.returncode:
                raise RuntimeError(f"worker failed with {completed.returncode}")
        except Exception as error:
            write(args.output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
