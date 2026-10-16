"""Immutable S5-R1 runner with an enforced lightweight worker budget."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics as st
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

from scripts.public_equity_backtests import memory_snapshot
from successors.frem_portfolio_v1.evidence import ensemble_periods, sha256, write

from .experiment import FOLDS, ROOT, SEEDS, inputs, market_cell, mechanisms, portfolio

SPEC = ROOT / "research/frem_s5_20261013/specification.json"


def identities():
    paths = [
        SPEC,
        ROOT / "research/frem_s5_20261013/diagnostic.json",
        *sorted((ROOT / "successors/frem_s5").glob("*.py")),
        *sorted((ROOT / "tests/s5").glob("*.py")),
        ROOT / "successors/frem_portfolio_v1/engine.py",
        ROOT / "successors/frem_portfolio_v1/evidence.py",
        ROOT / "scripts/public_equity_backtests.py",
        ROOT / "outputs/frem_s3_20261011_attempt1/public_data/manifest.json",
        ROOT / "configs/frem_s3_public.json",
        *inputs(),
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def verify(out):
    if json.loads((out / "freeze.json").read_text())["identities"] != identities():
        raise ValueError("Frozen source/data identity changed")


def save(out, name, value):
    # Dataclasses in saved prediction artifacts are explicitly serialized.
    payload = json.loads(
        json.dumps(
            value,
            allow_nan=False,
            default=lambda x: asdict(x) if is_dataclass(x) else TypeError(type(x).__name__),
        )
    )
    write(out / name, payload)


def worker(out):
    verify(out)
    mech = mechanisms()
    save(out, "mechanisms.json", mech)
    if not mech["gate_passed"]:
        save(
            out,
            "completion.json",
            {"status": "MECHANISM_GATE_FAILED", "market_cells": 0, "base_fits": 0, "protected_access": False},
        )
        return
    records = []
    for fold in FOLDS:
        families = []
        for seed in SEEDS:
            cells = market_cell(fold, seed)
            families.append(cells)
            economics = portfolio(cells)
            save(out, f"{fold}/{seed}/forecasts.json", cells)
            save(out, f"{fold}/{seed}/portfolio.json", economics)
            records.append({"fold": fold, "seed": seed, "nll": {m: r["nll"] for m, r in cells.items()}})
        combined = {
            m: {"periods": ensemble_periods([c[m]["periods"] for c in families])} for m in families[0]
        }
        economics = portfolio(combined)
        save(out, f"{fold}/ensemble/portfolio.json", economics)
        save(out, f"{fold}/ensemble/forecasts.json", combined)
    summary = {}
    for fold in FOLDS:
        ps = json.loads((out / fold / "ensemble/portfolio.json").read_text())
        nll = {m: st.fmean(r["nll"][m] for r in records if r["fold"] == fold) for m in records[0]["nll"]}
        summary[fold] = {
            "seed_mean_nll": nll,
            "primary_10bps": {name: r["relative_wealth"] for name, r in ps["10"]["comparisons"].items()},
            "terminal_wealth_10bps": {name: r["terminal_wealth"] for name, r in ps["10"]["ledgers"].items()},
        }
    broad = all(
        all(
            summary[f]["primary_10bps"][b] > 0
            for b in ("equal_weight", "buy_hold", "recent", "random", "linear")
        )
        for f in FOLDS
    )
    stretch = all(
        all(summary[f]["primary_10bps"][b] >= 0.06 for b in ("equal_weight", "buy_hold")) for f in FOLDS
    )
    memory = all(
        all(
            summary[f]["seed_mean_nll"][b] - summary[f]["seed_mean_nll"]["context"] > 0.01
            for b in ("equal", "recent", "random", "linear")
        )
        for f in FOLDS
    )
    verify(out)
    save(
        out,
        "completion.json",
        {
            "status": "COMPLETE_EXPLORATORY_REPLAY",
            "market_cells": 6,
            "base_fits": 0,
            "generated_cells": 12,
            "summary": summary,
            "broad_benchmark_point_win": broad,
            "stretch_6pct_met": stretch,
            "memory_practical_point_gain_all_controls": memory,
            "protected_access": False,
            "fresh_training_status": "BLOCKED_RESOURCE_AND_FRESH_STAGE_PROTOCOL",
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (2**20 if platform.system() == "Darwin" else 1024),
            "limitation": "Previously inspected survivor histories and retained forecasts; no protected confirmation, new base fit, calibrated funding probability or investment claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    a = parser.parse_args()
    out = a.output.resolve()
    if a.worker:
        worker(out)
        return
    if not a.execute:
        raise ValueError("Explicit --execute required")
    out.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for _ in range(3):
        snapshots.append(memory_snapshot())
        time.sleep(1)
    admitted = (
        min(s["available_mib"] for s in snapshots) >= 512
        and snapshots[-1]["pageouts"] <= snapshots[0]["pageouts"]
    )
    write(
        out / "resource_admission.json",
        {
            "samples": snapshots,
            "normal_floor_mib": 512,
            "fresh_training_admitted": admitted,
            "fresh_training_started": False,
            "bounded_replay": "Separate stdlib artifact analysis at128MiB/300seconds; no override of training guard",
        },
    )
    write(
        out / "freeze.json",
        {
            "frozen_at": datetime.now(UTC).isoformat(),
            "identities": identities(),
            "specification": json.loads(SPEC.read_text()),
            "python": sys.version,
            "platform": platform.platform(),
            "protected_access": False,
        },
    )
    # Runtime integrity checks before any generated or market outcome evaluation.
    with (out / "tests.log").open("x") as log:
        tested = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/s5", "-v"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    if tested.returncode:
        write(out / "failure.json", {"stage": "integrity", "returncode": tested.returncode})
        return
    start = time.monotonic()
    peak = 0.0
    with (out / "worker.log").open("x") as log:
        proc = subprocess.Popen(
            [
                "nice",
                "-n",
                "10",
                sys.executable,
                "-m",
                "successors.frem_s5.run",
                "--output",
                str(out),
                "--worker",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while proc.poll() is None:
                current = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True, check=False
                )
                if current.stdout.strip():
                    peak = max(peak, float(current.stdout.strip()) / 1024)
                if time.monotonic() - start > 300 or peak > 128:
                    raise RuntimeError("Lightweight worker resource ceiling reached")
                time.sleep(0.2)
            if proc.returncode:
                raise RuntimeError(f"Worker failed: {proc.returncode}")
            verify(out)
            write(
                out / "process.json",
                {
                    "returncode": proc.returncode,
                    "elapsed_seconds": time.monotonic() - start,
                    "peak_sampled_rss_mib": peak,
                    "threads": 1,
                },
            )
        except Exception as error:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            write(
                out / "failure.json",
                {"error": repr(error), "traceback": traceback.format_exc(), "peak_sampled_rss_mib": peak},
            )
            raise
    print((out / "completion.json").read_text())


if __name__ == "__main__":
    main()
