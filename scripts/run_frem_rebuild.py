"""Bounded, sequential FREM-S2 development runner; imports science only in workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from scripts.public_equity_backtests import ROOT, digest, memory_snapshot, resource_gate, save

MODELS = (
    "legacy_frem",
    "legacy_no_memory",
    "standardized_frem",
    "standardized_no_memory",
    "frozen_residual",
    "zero",
    "ridge",
    "xgboost",
    "gru",
    "dlinear",
    "patch_transformer",
)
SEEDS = (11, 29)
SPEC = ROOT / "research/frem_rebuild_20261011/specification.json"
AMENDMENT = ROOT / "research/frem_rebuild_20261011/slow_execution_amendment.json"
SOURCE_AMENDMENT = ROOT / "research/frem_rebuild_20261011/source_quarantine_amendment.json"


def source_identity() -> dict:
    files = [
        Path(__file__),
        SPEC,
        AMENDMENT,
        SOURCE_AMENDMENT,
        ROOT / "scripts/public_equity_backtests.py",
        ROOT / "configs/public_equity_crises.json",
        *sorted((ROOT / "src/frem").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def bounded_process(command: list[str], directory: Path, timeout: float, rss_limit_mib: float = 768) -> dict:
    """Monitor one low-priority child and retain stdout, exit status and resource limits."""
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    temp = ROOT / ".tmp/frem_s2"
    temp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temp)
    env["PYTHONUNBUFFERED"] = "1"
    start, peak, reason = time.monotonic(), 0.0, None
    with (directory / "console.log").open("x") as log:
        process = subprocess.Popen(
            ["nice", "-n", "10", *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while process.poll() is None:
                raw = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True, check=False
                ).stdout.strip()
                if raw:
                    peak = max(peak, float(raw) / 1024)
                if peak > rss_limit_mib:
                    reason = "process_rss_limit"
                elif time.monotonic() - start > timeout:
                    reason = "wall_limit"
                if reason:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
    receipt = {
        "command": command,
        "returncode": process.returncode,
        "termination": reason,
        "elapsed_seconds": time.monotonic() - start,
        "sampled_peak_rss_mib": peak,
        "rss_ceiling_mib": rss_limit_mib,
        "wall_ceiling_seconds": timeout,
        "threads": 1,
        "nice": 10,
        "memory_snapshot_after": memory_snapshot(),
    }
    save(directory / "process.json", receipt)
    if process.returncode or reason:
        raise RuntimeError(f"Child failed: {directory}; exit={process.returncode}; reason={reason}")
    return receipt


def load_fold():
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
    fold = next(row for row in cfg["folds"] if row["name"] == "great_recession")
    indices, split = fold_indices(data, months, fold, cfg)
    return cfg, [data.subset(ix) for ix in indices], split, coverage, months[indices[2]]


def worker(output: Path, name: str, seed: int) -> None:
    import numpy as np
    import torch

    from frem.baselines import make_baseline
    from frem.evaluation import evaluate_predictions, score_predictions
    from frem.frozen_residual import FrozenResidualForecaster
    from frem.model import ModelConfig
    from frem.training import Trainer, TrainingConfig

    declaration = json.loads((output / "declaration.json").read_text())
    if declaration["source_identity"] != source_identity() or name not in MODELS or seed not in SEEDS:
        raise ValueError("Worker identity or matrix changed after declaration")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = output / f"seed_{seed}" / name
    start = time.monotonic()
    predictor = None
    try:
        cfg, (train, validation, evaluation), split, coverage, months = load_fold()
        save(directory / "split.json", split)
        save(directory / "coverage.json", coverage)
        mc = ModelConfig(
            input_dim=4,
            context_length=12,
            horizon=1,
            reservoir_seed=seed,
            use_memory=name.endswith("frem"),
            return_parameterization="train_standardized" if name.startswith("standardized") else "legacy",
            **cfg["model"],
        )
        tc = TrainingConfig(
            seed=seed, epochs=12, batch_size=32, early_stopping_patience=3, max_wall_seconds=120, device="cpu"
        )
        if name == "frozen_residual":
            predictor = FrozenResidualForecaster(mc, tc)
            fit = predictor.fit(train, validation)
        elif name.startswith(("legacy_", "standardized_")):
            predictor = Trainer(mc, tc)
            fit = predictor.fit(train, validation)
        else:
            predictor = make_baseline(
                name,
                seed=seed,
                epochs=12,
                batch_size=32,
                max_wall_seconds=120,
                device="cpu",
                n_estimators=200,
                max_depth=3,
            )
            fit = predictor.fit(train)
        save(directory / "fit.json", fit)
        predictor.save(directory / "checkpoint")
        metrics = {}
        variants = [name, "frozen_no_memory"] if name == "frozen_residual" else [name]
        for variant in variants:
            frame = (
                predictor.predict(evaluation, use_memory=False)
                if variant == "frozen_no_memory"
                else predictor.predict(evaluation)
            )
            frame["seed"], frame["model"], frame["target_month"] = seed, variant, months
            scored = score_predictions(frame, student_df=None)
            path = directory / f"{variant}_predictions.csv"
            scored.to_csv(path, index=False)
            metrics[variant] = {
                **evaluate_predictions(frame, student_df=None),
                "prediction_sha256": digest(path),
                "prediction_path": str(path),
            }
        save(
            directory / "receipt.json",
            {
                "status": "COMPLETE",
                "model": name,
                "seed": seed,
                "metrics": metrics,
                "wall_seconds": time.monotonic() - start,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                },
                "scope": cfg["scope"],
                "protected_outcomes_accessed": False,
            },
        )
        print(
            json.dumps(
                {"model": name, "seed": seed, "mean_nll": {k: v["mean_nll"] for k, v in metrics.items()}}
            ),
            flush=True,
        )
    except BaseException as error:
        partial = getattr(predictor, "fit_report", None)
        if isinstance(predictor, FrozenResidualForecaster):
            partial = {"successor": partial, "base_history": predictor.base.history}
        save(
            directory / "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_fit_report": partial,
                "history": getattr(predictor, "history", None),
            },
        )
        raise


def summarize(output: Path) -> dict:
    """Require the full seed/model family; report descriptive paired log scores."""
    by_model = {}
    for seed in SEEDS:
        for name in MODELS:
            directory = output / f"seed_{seed}" / name
            receipt = json.loads((directory / "receipt.json").read_text())
            for variant, result in receipt["metrics"].items():
                path = Path(result["prediction_path"])
                if digest(path) != result["prediction_sha256"]:
                    raise ValueError("Saved prediction identity mismatch")
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                by_model.setdefault(variant, {})[seed] = {
                    (row["time"], row["asset"]): (float(row["actual"]), float(row["nll"])) for row in rows
                }
                if len(by_model[variant][seed]) != len(rows):
                    raise ValueError("Duplicate evaluation rows")
    reference = by_model["frozen_residual"][SEEDS[0]]
    scores = {}
    for name, seeds in by_model.items():
        if set(seeds) != set(SEEDS):
            raise ValueError("Incomplete seed family")
        dates = {}
        for key, (actual, _) in reference.items():
            losses = []
            for seed in SEEDS:
                if set(seeds[seed]) != set(reference) or seeds[seed][key][0] != actual:
                    raise ValueError("Incomplete or mismatched paired outcomes")
                losses.append(seeds[seed][key][1])
            dates.setdefault(key[0], []).append(statistics.mean(losses))
        scores[name] = statistics.mean(statistics.mean(values) for values in dates.values())
    return {
        "status": "COMPLETE",
        "fit_cells": len(MODELS) * len(SEEDS),
        "prediction_cells": len(by_model) * len(SEEDS),
        "date_balanced_nll": scores,
        "same_checkpoint_memory_gain_nats": scores["frozen_no_memory"] - scores["frozen_residual"],
        "scope": "Exploratory public survivor sample, 19 crisis months, two optimization seeds",
        "sota_established": False,
        "significance_test_performed": False,
        "protected_outcomes_accessed": False,
    }


def run(output: Path, *, slow: bool) -> None:
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        normal_admission = {"passed": True, "samples": resource_gate(512)}
    except RuntimeError as error:
        normal_admission = {"passed": False, "error": str(error)}
    save(
        output / "declaration.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_identity": source_identity(),
            "normal_admission": normal_admission,
            "user_authorized_slow_mode": slow,
            "models": MODELS,
            "seeds": SEEDS,
            "fit_budget": 22,
            "matrix_wall_seconds": 3600,
            "source_manifest_sha256": digest(ROOT / "data/public_equity_crises_v1/manifest.json"),
            "scientific_freeze": False,
            "phase": "exploratory_development",
        },
    )
    completed = []
    try:
        if not normal_admission["passed"] and not slow:
            raise RuntimeError("Normal resource admission failed; no scientific stack imported")
        bounded_process(
            [sys.executable, "-m", "pytest", "-q", "-W", "error::DeprecationWarning"],
            output / "verification",
            timeout=180,
        )
        for seed in SEEDS:
            for name in MODELS:
                remaining = 3600 - (time.monotonic() - start)
                if remaining < 1:
                    raise TimeoutError("Matrix wall budget exhausted")
                if not slow:
                    resource_gate(512)
                directory = output / f"seed_{seed}" / name
                save(
                    directory / "started.json",
                    {"time": datetime.now(UTC).isoformat(), "memory_snapshot": memory_snapshot()},
                )
                bounded_process(
                    [
                        sys.executable,
                        "-m",
                        "scripts.run_frem_rebuild",
                        "--output",
                        str(output),
                        "--worker",
                        name,
                        "--seed",
                        str(seed),
                    ],
                    directory,
                    min(150, remaining),
                )
                completed.append({"model": name, "seed": seed})
                print((directory / "console.log").read_text().strip(), flush=True)
        save(output / "summary.json", summarize(output))
    except BaseException as error:
        save(
            output / "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "completed_cells": completed,
                "incomplete_matrix": True,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--slow", action="store_true", help="Use recorded user authorization below normal memory floor"
    )
    parser.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output.resolve(), args.worker, args.seed)
    elif args.execute:
        run(args.output.resolve(), slow=args.slow)
    else:
        print(json.dumps({"models": MODELS, "seeds": SEEDS, "fit_cells": 22, "specification": str(SPEC)}))


if __name__ == "__main__":
    main()
