"""User-authorized sequential public six-window campaign; no protected evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from scripts.public_equity_backtests import ROOT, digest, memory_snapshot, resource_gate, save
from scripts.run_frem_rebuild import bounded_process

CONFIG = ROOT / "configs/public_equity_crises.json"
AMENDMENT = ROOT / "research/decisions/public_slow_execution_20261011.json"


def load_fold(fold_name):
    from scripts.public_equity_backtests import build_research_windows, fold_indices, monthly_prices

    source = ROOT / "data/public_equity_crises_v1"
    cfg = json.loads((ROOT / "configs/public_equity_crises.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest["complete"] or json.loads((source / "request.json").read_text()) != cfg:
        raise ValueError("Public source acquisition is incomplete or request identity changed")
    expected = set(cfg["symbols"])
    if {row["symbol"] for row in manifest["records"]} != expected or len(manifest["records"]) != len(
        expected
    ):
        raise ValueError("Source listing set differs from the predeclared universe")
    prices = {}
    for record in manifest["records"]:
        path = source / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("Public source hash mismatch")
        chart = json.loads(path.read_bytes())["chart"]["result"][0]
        if chart["meta"]["symbol"].upper() != record["symbol"].upper():
            raise ValueError("Source symbol mismatch")
        prices[record["symbol"]] = monthly_prices(chart, cfg, quarantine_zero_quote_months=True)
    data, months, coverage = build_research_windows(prices, cfg["context_months"])
    for symbol, price in prices.items():
        coverage[symbol]["quarantined_zero_quote_months"] = price.attrs.get(
            "quarantined_zero_quote_months", []
        )
    fold = next(row for row in cfg["folds"] if row["name"] == fold_name)
    indices, split = fold_indices(data, months, fold, cfg)
    return cfg, [data.subset(ix) for ix in indices], split, coverage, months[indices[2]]


def worker(output, fold, name):
    import torch

    from frem.baselines import make_baseline
    from frem.evaluation import evaluate_predictions
    from frem.model import ModelConfig
    from frem.training import Trainer, TrainingConfig
    from scripts.public_equity_backtests import local_stock_ledgers

    torch.set_num_threads(1)
    cfg, (train, validation, evaluation), split, coverage, months = load_fold(fold)
    cell = output / fold / name
    save(cell / "split.json", split)
    save(cell / "coverage.json", coverage)
    started = time.monotonic()
    if name in {"frem", "no_memory"}:
        mc = ModelConfig(
            input_dim=4,
            context_length=cfg["context_months"],
            horizon=1,
            reservoir_seed=cfg["seed"],
            use_memory=name == "frem",
            **cfg["model"],
        )
        training = dict(cfg["training"], batch_size=32)
        predictor = Trainer(mc, TrainingConfig(seed=cfg["seed"], device="cpu", **training))
        history = predictor.fit(train, validation)
    else:
        predictor = make_baseline(
            name,
            seed=cfg["seed"],
            max_wall_seconds=cfg["training"]["max_wall_seconds"],
            n_estimators=200,
            max_depth=3,
        )
        history = predictor.fit(train)
    fit_seconds = time.monotonic() - started
    predictor.save(cell / "checkpoint")
    save(cell / "fit.json", history)
    predictions = predictor.predict(evaluation)
    predictions["target_month"] = months
    predictions["model"] = name
    predictions["seed"] = cfg["seed"]
    predictions.to_csv(cell / "predictions.csv", index=False)
    metrics = evaluate_predictions(predictions, student_df=None)
    save(
        cell / "local_stock_ledgers.json",
        local_stock_ledgers(predictions, cfg["hypothetical_cost_per_side_bps"]),
    )
    receipt = {
        "status": "COMPLETE",
        "fold": fold,
        "model": name,
        "seed": cfg["seed"],
        "metrics": metrics,
        "fit_wall_seconds": fit_seconds,
        "prediction_sha256": digest(cell / "predictions.csv"),
        "protected_outcomes_accessed": False,
    }
    save(cell / "receipt.json", receipt)
    if fit_seconds > cfg["training"]["max_wall_seconds"]:
        raise TimeoutError("Retained fit exceeded declared budget")
    print(json.dumps(receipt), flush=True)


def summarize(output):
    import numpy as np
    import pandas as pd

    from frem.evaluation import score_predictions
    from scripts.public_equity_backtests import paired_month_bootstrap

    cfg = json.loads(CONFIG.read_text())
    cells, folds = [], []
    for fold in cfg["folds"]:
        name = fold["name"]
        scored = {}
        reference = None
        for model in cfg["models"]:
            cell = output / name / model
            receipt = json.loads((cell / "receipt.json").read_text())
            process = json.loads((cell / "process.json").read_text())
            if process["returncode"] or process["termination"]:
                raise ValueError("Failed cell cannot enter complete summary")
            if digest(cell / "predictions.csv") != receipt["prediction_sha256"]:
                raise ValueError("Prediction identity mismatch")
            predictions = pd.read_csv(cell / "predictions.csv")
            scored[model] = score_predictions(predictions, student_df=None)
            rows = predictions.sort_values(["time", "asset"])
            signature = rows[["time", "asset", "actual"]].reset_index(drop=True)
            if reference is not None and not signature.equals(reference):
                raise ValueError("Incomplete model-family pairing")
            reference = signature
            reconstructed = float(scored[model].groupby("time")["nll"].mean().mean())
            receipt["date_balanced_nll"] = reconstructed
            receipt["sampled_peak_rss_mib"] = process["sampled_peak_rss_mib"]
            cells.append(receipt)
        comparison = paired_month_bootstrap(scored["frem"], scored["no_memory"], cfg["bootstrap"])
        save(output / name / "comparison.json", comparison)
        folds.append({"fold": name, "comparison": comparison})
    result = {
        "status": "COMPLETE",
        "completed_cells": len(cells),
        "expected_cells": 24,
        "cells": cells,
        "folds": folds,
        "scope": cfg["scope"],
        "limitations": cfg["public_source_limitations"],
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "protected_outcomes_accessed": False,
        "canonical_scientific_status": "NOT_EVALUATED",
    }
    if len(cells) != 24 or not all(np.isfinite(c["date_balanced_nll"]) for c in cells):
        raise ValueError("Incomplete or nonfinite matrix")
    save(output / "summary.json", result)
    print(json.dumps({"status": "COMPLETE", "cells": len(cells), "folds": folds}), flush=True)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text())
    started = time.monotonic()
    files = [
        Path(__file__),
        CONFIG,
        AMENDMENT,
        ROOT / "scripts/run_frem_rebuild.py",
        ROOT / "scripts/public_equity_backtests.py",
        *sorted((ROOT / "src/frem").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]
    try:
        admission = {"passed": True, "samples": resource_gate(512)}
    except RuntimeError as error:
        admission = {"passed": False, "error": str(error)}
    save(
        output / "declaration.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "config": cfg,
            "amendment": json.loads(AMENDMENT.read_text()),
            "normal_admission": admission,
            "source_identity": {str(p.relative_to(ROOT)): digest(p) for p in files},
            "manifest_sha256": digest(ROOT / "data/public_equity_crises_v1/manifest.json"),
            "protected_outcomes_accessed": False,
        },
    )
    completed = []
    try:
        bounded_process(
            [sys.executable, "-m", "pytest", "-q", "-W", "error::DeprecationWarning"],
            output / "verification",
            timeout=180,
        )
        for fold in cfg["folds"]:
            for model in cfg["models"]:
                remaining = cfg["budget"]["max_wall_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Campaign deadline reached")
                cell = output / fold["name"] / model
                save(
                    cell / "started.json", {"memory": memory_snapshot(), "fold": fold["name"], "model": model}
                )
                bounded_process(
                    [
                        sys.executable,
                        "-m",
                        "scripts.run_public_slow",
                        "--output",
                        str(output),
                        "--worker",
                        model,
                        "--fold",
                        fold["name"],
                    ],
                    cell,
                    min(360, remaining),
                )
                completed.append({"fold": fold["name"], "model": model})
                print(json.dumps({"completed": len(completed), **completed[-1]}), flush=True)
        bounded_process(
            [sys.executable, "-m", "scripts.run_public_slow", "--output", str(output), "--summarize"],
            output / "analysis",
            timeout=180,
        )
    except BaseException as error:
        save(
            output / "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "completed_cells": completed,
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", choices=["frem", "no_memory", "ridge", "xgboost"])
    parser.add_argument("--fold", choices=[f["name"] for f in json.loads(CONFIG.read_text())["folds"]])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if not args.fold:
            parser.error("--worker requires --fold")
        worker(args.output.resolve(), args.fold, args.worker)
    elif args.summarize:
        summarize(args.output.resolve())
    elif args.execute:
        run(args.output.resolve())
    else:
        parser.error("Explicit --execute required")
