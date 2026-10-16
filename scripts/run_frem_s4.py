"""Single-worker, explicitly authorized S4 development experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scripts.public_equity_backtests import ROOT, digest, memory_snapshot, save
from scripts.run_frem_rebuild import bounded_process

SPEC = ROOT / "research/frem_s4_20261012/specification.json"
SOURCE = ROOT / "outputs/frem_s3_20261011_attempt1"
ARMS = ("off", "old_distance", "euclidean_distance", "matched_random")


def identities():
    files = [
        SPEC,
        Path(__file__),
        ROOT / "scripts/run_frem_s3.py",
        ROOT / "scripts/run_frem_rebuild.py",
        ROOT / "scripts/public_equity_backtests.py",
        ROOT / "configs/frem_s3_public.json",
        *sorted((ROOT / "src/frem").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
        *sorted((SOURCE / "public_data").rglob("*.json")),
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def worker(output, fold, seed):
    if json.loads((output / "freeze.json").read_text())["sources"] != identities():
        raise ValueError("Source or data changed after freeze")
    import torch

    from frem.s3 import balanced, losses, run_replay, seed_memory
    from frem.s4 import arm, scale_keys, select
    from frem.s4_forecaster import S4Base
    from scripts.run_frem_s3 import load_public

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cell = output / fold / str(seed)
    (train, validation, evaluation), months, split, coverage = load_public(SOURCE, fold)
    save(cell / "split.json", split)
    save(cell / "coverage.json", coverage)
    base = S4Base("frem", seed)
    save(cell / "fit.json", base.fit(train))
    version = base.save(cell / "checkpoint")
    valid = base.predict(validation)
    test = base.predict(evaluation)
    base.residual_frame.to_csv(cell / "residuals.csv", index=False)
    valid.to_csv(cell / "validation_base.csv", index=False)
    raw = base.raw(evaluation)
    raw["nll"] = losses(raw.actual, raw.prediction, raw.scale, raw.student_df)
    raw.to_csv(cell / "raw.csv", index=False)
    original_keys = [base.keys(p) for p in (base.residual_data, validation, evaluation)]
    results = {}
    for name in ARMS:
        cfg = arm(name, seed)
        rk, vk, ek = [scale_keys(k, name) for k in original_keys]
        initial = seed_memory(base.residual_frame, rk, cfg, version, int(valid.time.min()))
        selected, state = select(initial, valid, vk, cfg)
        pred, final = run_replay(state, test, ek, cfg, selected["selected"]["weight"], 0.0, keep_state=True)
        pred.to_csv(cell / f"{name}.csv", index=False)
        save(cell / f"{name}_selection.json", selected)
        save(cell / f"{name}_state.json", final)
        results[name] = {
            "nll": balanced(pred.nll, pred.time, pred.asset_id),
            "selected": selected["selected"],
            "rows": len(pred),
            "sha256": digest(cell / f"{name}.csv"),
        }
    save(
        cell / "result.json",
        {
            "fold": fold,
            "seed": seed,
            "results": results,
            "raw_nll": balanced(raw.nll, raw.time, raw.asset_id),
            "protected_access": False,
            "scope": "Spent-window public development; no confirmation",
        },
    )
    print(json.dumps({"completed": str(cell)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fold")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args()
    output = a.output.resolve()
    if a.worker:
        worker(output, a.fold, a.seed)
        return
    if not a.execute:
        raise ValueError("Explicit --execute required")
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(SPEC.read_text())
    start = time.monotonic()
    save(
        output / "resource_admission.json",
        {
            "snapshot": memory_snapshot(),
            "normal_floor_mib": 512,
            "deviation": "Explicit user renewed slow execution like S2/S3 on 12 October; normal guard unchanged",
            "limits": spec["budget"],
        },
    )
    save(
        output / "freeze.json",
        {"sources": identities(), "specification": spec, "frozen_at_ns": time.time_ns()},
    )
    receipts = []
    try:
        receipts.append(
            bounded_process(
                [sys.executable, "-m", "pytest", "-q", "tests"],
                output / "tests",
                min(600, 1800 - (time.monotonic() - start)),
            )
        )
        for fold in spec["folds"]:
            for seed in spec["seeds"]:
                remaining = 1800 - (time.monotonic() - start)
                if remaining <= 0:
                    raise RuntimeError("Total budget exhausted")
                receipts.append(
                    bounded_process(
                        [
                            sys.executable,
                            "-m",
                            "scripts.run_frem_s4",
                            "--output",
                            str(output),
                            "--worker",
                            "--fold",
                            fold,
                            "--seed",
                            str(seed),
                        ],
                        output / fold / str(seed),
                        min(300, remaining),
                    )
                )
        records = [
            json.loads((output / f / str(s) / "result.json").read_text())
            for f in spec["folds"]
            for s in spec["seeds"]
        ]
        summary = {}
        for fold in spec["folds"]:
            subset = [r for r in records if r["fold"] == fold]
            nll = {name: sum(r["results"][name]["nll"] for r in subset) / 3 for name in ARMS}
            summary[fold] = {
                "nll": nll,
                "gain_vs_off": {name: nll["off"] - v for name, v in nll.items()},
                "euclidean_gain_vs_old": nll["old_distance"] - nll["euclidean_distance"],
            }
        if identities() != json.loads((output / "freeze.json").read_text())["sources"]:
            raise ValueError("Final identity mismatch")
        save(
            output / "completion.json",
            {
                "status": "COMPLETE",
                "fits": 6,
                "prediction_cells": 24,
                "summary": summary,
                "records": records,
                "processes": receipts,
                "elapsed_seconds": time.monotonic() - start,
                "scope": spec["scope"],
                "protected_access": False,
            },
        )
        print(json.dumps(summary), flush=True)
    except Exception as e:
        save(
            output / "failure.json",
            {
                "status": "FAILED_PRESERVED",
                "error": repr(e),
                "processes": receipts,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        raise


if __name__ == "__main__":
    main()
