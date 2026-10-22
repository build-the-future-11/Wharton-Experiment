"""Immutable three-fold expanding-history historical proxy benchmark."""

import argparse
import csv
import time
import traceback
from itertools import zip_longest
from pathlib import Path

from .cli import campaign, source_identity
from .config import ARMS, ModelConfig, RunConfig
from .runtime import (
    cell_inventory,
    digest,
    manifest,
    read_json,
    retain_input,
    timestamp,
    write_json,
)

FOLD_ENDS = ("2011-01-01", "2015-01-01", "2018-01-01")


def verify_fold_prefix(source, fold, end):
    """Compare every retained field against the fixed UTC prefix, in order."""
    cutoff = timestamp(end + "T00:00:00Z")
    with open(source, newline="") as src, open(fold, newline="") as dst:
        expected, actual = csv.DictReader(src), csv.DictReader(dst)
        if expected.fieldnames != actual.fieldnames or not expected.fieldnames:
            raise ValueError("fold input header differs")
        rows = (row for row in expected if timestamp(row["timestamp"]) < cutoff)
        for wanted, observed in zip_longest(rows, actual):
            if wanted != observed or wanted is None or None in wanted or None in wanted.values():
                raise ValueError("fold input is not the exact declared chronological prefix")


def suite_config(device="cuda"):
    return RunConfig(
        model=ModelConfig(input_dim=9, horizon=20),
        device=device,
        arms=ARMS,
        seeds=(11, 29, 47),
        epochs=10,
        patience=3,
        embargo=20,
        batch_size=128,
        # The prior 0.1.9 candidate completed Fold 2's cells but exhausted this
        # per-fold ceiling during mandatory semantic readback.  This successor
        # reserves a distinct full-suite verifier stage while allowing a fold to
        # finish the same required in-run readback; the notebook's unchanged
        # 13,500-second ceiling remains the outer non-negotiable limit.
        max_seconds=4000,
        bootstrap_reps=2000,
        block_dates=20,
        max_weight=0.25,
    )


def run_suite(panel, qualification, output, device="cuda"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    panel = Path(panel)
    cfg = suite_config(device)
    start, results = time.monotonic(), []
    try:
        source_meta = read_json(str(panel) + ".manifest.json")
        if source_meta["sha256"] != digest(panel) or source_meta["purpose"] != "historical_proxy":
            raise ValueError("suite requires exact historical proxy intake")
        panel = retain_input(panel, root / "source_panel.csv")
        write_json(
            root / "plan.json",
            {
                "source_identity": source_identity(),
                "panel_sha256": digest(panel),
                "fold_ends": list(FOLD_ENDS),
                "config": cfg.to_dict(),
                "expected_cells": len(FOLD_ENDS) * len(ARMS) * len(cfg.seeds),
                "selection": "No search or outcome-driven asset/fold/horizon replacement",
                "claims": "exploratory historical adjusted-close proxy; no investment efficacy",
                "inference": "Seed-average date-block bootstrap; Holm within and across all fold comparisons",
                "protected_access": False,
            },
        )
        for fold_index, end in enumerate(FOLD_ENDS):
            data = root / f"fold{fold_index}_panel.csv"
            with panel.open(newline="") as src, data.open("x", newline="") as dst:
                reader = csv.DictReader(src)
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    if timestamp(row["timestamp"]) < timestamp(end + "T00:00:00Z"):
                        writer.writerow(row)
            meta = dict(source_meta, sha256=digest(data))
            write_json(str(data) + ".manifest.json", meta)
            verify_fold_prefix(panel, data, end)
            directory = root / f"fold{fold_index}"
            try:
                campaign(directory, cfg, data, qualification)
                results.append({"fold": fold_index, "status": "COMPLETE"})
            except Exception as error:
                results.append(
                    {
                        "fold": fold_index,
                        "status": "FAILED",
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
        if any(r["status"] != "COMPLETE" for r in results):
            raise RuntimeError("declared benchmark family incomplete; failures retained")
        # All model comparisons remain paired within each fold. No pooling overlap.
        from .evaluation import holm

        comparisons, last = [], None
        for i in range(len(FOLD_ENDS)):
            split = read_json(root / f"fold{i}/splits.json")["test"]
            if last is not None and split["first_decision"] <= last:
                raise ValueError("overlapping evaluation decisions across folds")
            last = split["last_label"]
            family = read_json(root / f"fold{i}/comparisons.json")["comparisons"]
            comparisons.extend(dict(v, fold=i, arm=k) for k, v in family.items())
        calculable = [v for v in comparisons if "p_two_sided" in v]
        for row, corrected in zip(
            calculable, holm([v["p_two_sided"] for v in calculable]), strict=True
        ):
            row["p_holm_entire_suite"] = float(corrected)
        write_json(root / "comparisons.json", comparisons)
        write_json(
            root / "status.json",
            {
                "status": "COMPLETE",
                "folds": results,
                "scientific_status": "HISTORICAL_PROXY_ONLY",
                "wall_seconds": time.monotonic() - start,
            },
        )
    except BaseException as error:
        write_json(
            root / "failure.json",
            {
                "status": "FAILED",
                "folds": results,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.monotonic() - start,
            },
        )
        raise
    finally:
        cells = []
        for i in range(len(FOLD_ENDS)):
            path = root / f"fold{i}/cells.json"
            inventory = read_json(path) if path.exists() else cell_inventory(cfg, [])
            cells.extend(dict(row, fold=i) for row in inventory)
        write_json(root / "cells.json", cells)
        write_json(root / "manifest.json", manifest(root))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run_suite(args.data, args.qualification, args.output, args.device)


if __name__ == "__main__":
    main()
