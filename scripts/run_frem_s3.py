"""Frozen, sequential S3 mechanism and new-sample public experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from scripts.public_equity_backtests import ROOT, digest, memory_snapshot, save
from scripts.run_frem_rebuild import bounded_process

SPEC = ROOT / "research/frem_s3_20261011/specification.json"
CONFIG = ROOT / "configs/frem_s3_public.json"
FAMILIES = ("null", "mean_recurrence", "risk_recurrence", "abrupt", "missing_delayed")
PATH_SEEDS = (101, 211, 307)
SEEDS = (11, 29, 47)


def identity():
    files = [
        SPEC,
        CONFIG,
        Path(__file__),
        ROOT / "scripts/public_equity_backtests.py",
        ROOT / "scripts/run_frem_rebuild.py",
        *sorted((ROOT / "src/frem").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def verify(output):
    if json.loads((output / "freeze.json").read_text())["sources"] != identity():
        raise ValueError("S3 source/protocol changed after freeze")


def synthetic(family, seed):
    import numpy as np
    import pandas as pd

    from frem.s3 import DAY

    rng = np.random.default_rng(seed)
    records, keys = [], []
    for t in range(420):
        for asset in range(4):
            state = int(rng.integers(0, 3)) - 1
            mean = 0.02 * state if family in {"mean_recurrence", "missing_delayed"} else 0.0
            std = 0.008 + 0.024 * (state == 1) if family == "risk_recurrence" else 0.02
            if family == "abrupt" and t >= 285:
                mean, std = -0.025, 0.04
            # Student-t innovations match the oracle density, independently per path.
            y = mean + std * np.sqrt(3 / 5) * rng.standard_t(5)
            delay = int(rng.integers(2, 9)) if family == "missing_delayed" else 2
            if family == "missing_delayed" and rng.random() < 0.15:
                continue
            records.append(
                {
                    "time": int(t * DAY),
                    "label_time": int((t + delay) * DAY),
                    "asset_id": asset,
                    "asset": f"synthetic_{asset}",
                    "prediction": 0.0,
                    "scale": 0.02 * np.sqrt(3 / 5),
                    "student_df": 5.0,
                    "actual": y,
                    "oracle_mu": mean,
                    "oracle_scale": std * np.sqrt(3 / 5),
                }
            )
            keys.append([float(state), float(asset) / 4])
    frame = pd.DataFrame(records)
    return frame, np.asarray(keys)


def mechanism_worker(output, family, seed):
    from frem.s3 import DAY, KeyMetric, arms, balanced, losses, run_replay, seed_memory, select_replay

    verify(output)
    cell = output / "mechanism" / family / str(seed)
    frame, keys = synthetic(family, seed)
    frame.to_csv(cell / "generated.csv", index=False)
    # Gaps cover all label delays; no training/validation outcomes overlap the next stage.
    masks = [
        frame.label_time < 180 * DAY,
        (frame.time >= 180 * DAY) & (frame.label_time < 280 * DAY),
        frame.time >= 280 * DAY,
    ]
    train, validation, test = [frame.loc[m].reset_index(drop=True) for m in masks]
    tk, vk, ek = [keys[m] for m in masks]
    metric = KeyMetric().fit(
        tk,
        ((train.actual - train.prediction) / train.scale).to_numpy(),
        train.label_time.to_numpy(),
        int(validation.time.min()),
    )
    scores, calibrations = {}, {}
    for config in arms(seed):
        train_keys, valid_keys, test_keys = (
            [metric.transform(k) for k in (tk, vk, ek)] if config.name == "learned" else (tk, vk, ek)
        )
        initial = seed_memory(
            train, train_keys, config, f"synthetic-fixed-base-{seed}", int(validation.time.min())
        )
        selection, state = select_replay(initial, validation, valid_keys, config)
        choice = selection["selected"]
        predictions = run_replay(state, test, test_keys, config, choice["weight"], choice["power"])
        predictions.to_csv(cell / f"{config.name}.csv", index=False)
        scores[config.name] = balanced(predictions.nll, predictions.time, predictions.asset_id)
        calibrations[config.name] = selection
    scores["oracle"] = balanced(
        losses(test.actual, test.oracle_mu, test.oracle_scale, test.student_df), test.time, test.asset_id
    )
    save(
        cell / "result.json",
        {
            "family": family,
            "seed": seed,
            "scores": scores,
            "calibrations": calibrations,
            "scope": "Generated mechanism evidence, not financial evidence",
            "source_sha256": digest(cell / "generated.csv"),
        },
    )
    print(json.dumps({"family": family, "seed": seed, "complete": True}), flush=True)


def mechanism_summary(output):
    records = [
        json.loads((output / "mechanism" / f / str(s) / "result.json").read_text())
        for f in FAMILIES
        for s in PATH_SEEDS
    ]
    means = {}
    for family in FAMILIES:
        subset = [r for r in records if r["family"] == family]
        means[family] = {
            name: sum(r["scores"][name] for r in subset) / len(subset) for name in subset[0]["scores"]
        }
    null_harm = means["null"]["combined"] - means["null"]["off"]
    recurrence_gain = means["mean_recurrence"]["off"] - means["mean_recurrence"]["combined"]
    result = {
        "status": "COMPLETE",
        "cells": len(records),
        "means": means,
        "null_harm": null_harm,
        "recurrence_gain": recurrence_gain,
        "gate_passed": null_harm <= 0.01 and recurrence_gain > 0,
        "scope": "Mechanism plausibility only; no market promotion without separate evidence",
    }
    save(output / "mechanism_summary.json", result)
    return result


def load_public(output, fold_name):
    from scripts.public_equity_backtests import build_research_windows, fold_indices, monthly_prices

    cfg = json.loads(CONFIG.read_text())
    source = output / "public_data"
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest["complete"] or json.loads((source / "request.json").read_text()) != cfg:
        raise ValueError("Incomplete or changed acquisition")
    if {r["symbol"] for r in manifest["records"]} != set(cfg["symbols"]):
        raise ValueError("Unexpected public universe")
    prices = {}
    for record in manifest["records"]:
        path = source / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("Raw source identity changed")
        chart = json.loads(path.read_bytes())["chart"]["result"][0]
        if chart["meta"]["symbol"].upper() != record["symbol"].upper():
            raise ValueError("Symbol mismatch")
        prices[record["symbol"]] = monthly_prices(chart, cfg)  # strict, no data-driven quarantine
    data, months, coverage = build_research_windows(prices, cfg["context_months"])
    fold = next(f for f in cfg["folds"] if f["name"] == fold_name)
    indices, split = fold_indices(data, months, fold, cfg)
    return [data.subset(i) for i in indices], months[indices[2]], split, coverage


def public_worker(output, fold, seed, name):
    import torch

    from frem.s3 import arms, balanced, losses, run_replay, save_replay, seed_memory, select_replay
    from frem.s3_forecaster import S3Base

    verify(output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cell = output / "public" / fold / str(seed) / name
    (train, validation, evaluation), months, split, coverage = load_public(output, fold)
    save(cell / "split.json", split)
    save(cell / "coverage.json", coverage)
    base = S3Base(name, seed)
    fit = base.fit(train)
    save(cell / "fit.json", fit)
    version = base.save(cell / "checkpoint")
    valid, test = base.predict(validation), base.predict(evaluation)
    base.residual_frame.to_csv(cell / "residuals.csv", index=False)
    valid.to_csv(cell / "validation_base.csv", index=False)
    raw = base.raw(evaluation)
    raw["nll"] = losses(raw.actual, raw.prediction, raw.scale, raw.student_df)
    raw.to_csv(cell / "raw_base.csv", index=False)
    configs = arms(seed) if name == "frem" else [arms(seed)[0]]
    results = {}
    for config in configs:
        learned = config.retrieval == "learned"
        rk = base.keys(base.residual_data, learned)
        vk = base.keys(validation, learned)
        ek = base.keys(evaluation, learned)
        initial = seed_memory(base.residual_frame, rk, config, version, int(validation.time.min()))
        selection, state = select_replay(initial, valid, vk, config)
        choice = selection["selected"]
        predictions, final_state = run_replay(
            state, test, ek, config, choice["weight"], choice["power"], keep_state=True
        )
        predictions["target_month"] = months
        predictions["seed"] = seed
        predictions["model"] = name
        predictions["arm"] = config.name
        path = cell / f"{config.name}.csv"
        predictions.to_csv(path, index=False)
        save(cell / f"{config.name}_selection.json", selection)
        save_replay(cell / f"{config.name}_initial.json", initial)
        save_replay(cell / f"{config.name}_validated.json", state)
        save_replay(cell / f"{config.name}_final.json", final_state)
        results[config.name] = {
            "nll": balanced(predictions.nll, predictions.time, predictions.asset_id),
            "sha256": digest(path),
            "rows": len(predictions),
            "selected": choice,
            "mean_gate": float(predictions.memory_gate.mean()),
            "mean_risk_power": float(predictions.risk_power.mean()),
        }
    save(
        cell / "result.json",
        {
            "status": "COMPLETE",
            "name": name,
            "fold": fold,
            "seed": seed,
            "base_version": version,
            "results": results,
            "raw_nll": balanced(raw.nll, raw.time, raw.asset_id),
            "protected_access": False,
            "scope": "New-sample public exploration only",
        },
    )
    print(json.dumps({"fold": fold, "seed": seed, "base": name, "complete": True}), flush=True)


def public_summary(output):
    import numpy as np
    import pandas as pd

    from frem.s3 import losses
    from scripts.public_equity_backtests import paired_month_bootstrap

    verify(output)
    cfg = json.loads(CONFIG.read_text())
    cells, comparisons = [], {}
    for fold in cfg["folds"]:
        tables = {}
        for seed in SEEDS:
            for name in (*cfg["models"], "zero", "historical_mean"):
                cell = output / "public" / fold["name"] / str(seed) / name
                receipt = json.loads((cell / "result.json").read_text())
                for arm, result in receipt["results"].items():
                    path = cell / f"{arm}.csv"
                    if digest(path) != result["sha256"]:
                        raise ValueError("Saved prediction mismatch")
                    frame = pd.read_csv(path)
                    if (
                        np.max(
                            np.abs(
                                losses(frame.actual, frame.prediction, frame.scale, frame.student_df)
                                - frame.nll
                            )
                        )
                        > 1e-10
                    ):
                        raise ValueError("Score reconstruction mismatch")
                    tables.setdefault(f"{name}/{arm}", []).append(frame)
                cells.append(receipt)
        reference = tables["frem/off"][0][["time", "asset", "actual", "label_time"]]
        for name, frames in tables.items():
            if len(frames) != 3 or any(
                not f[["time", "asset", "actual", "label_time"]].equals(reference) for f in frames
            ):
                raise ValueError("Incomplete paired seed/asset grid")
        # Average seed losses per observation BEFORE the date-block bootstrap.
        aggregates = {
            name: pd.concat(frames)
            .groupby(["time", "asset"], as_index=False)
            .agg(nll=("nll", "mean"), actual=("actual", "first"))
            for name, frames in tables.items()
        }
        fold_results = {}
        for control in [
            "frem/off",
            "frem/recent",
            "frem/random",
            "zero/off",
            "historical_mean/off",
            "ridge/off",
            "xgboost/off",
        ]:
            fold_results[control] = {}
            for block in [3, 6]:
                settings = {"block_months": block, "repetitions": 2000, "seed": 1901, "confidence": 0.95}
                if len(aggregates[control].time.unique()) < 2 * block:
                    fold_results[control][str(block)] = {
                        "status": "UNAVAILABLE_TOO_FEW_BLOCKS",
                        "settings": settings,
                        "ci": None,
                        "p_value": None,
                    }
                    continue
                result = paired_month_bootstrap(aggregates["frem/combined"], aggregates[control], settings)
                result["effect"] = f"{control} NLL minus S3 combined NLL; positive favors S3"
                result["interpretation"] = (
                    "Exploratory survivor sample; three seeds averaged per observation, not independent markets"
                )
                joined = aggregates[control].merge(
                    aggregates["frem/combined"], on=["time", "asset"], suffixes=("_control", "_candidate")
                )
                joined["gain"] = joined.nll_control - joined.nll_candidate
                delta = joined.groupby("time").gain.mean().to_numpy()
                rng = np.random.default_rng(1901)
                starts = rng.integers(len(delta), size=(2000, int(np.ceil(len(delta) / block))))
                indices = ((starts[..., None] + np.arange(block)) % len(delta)).reshape(2000, -1)[
                    :, : len(delta)
                ]
                draws = delta[indices].mean(axis=1)
                null = 0.01 if control == "frem/off" else 0.0
                result["p_value"] = float(
                    (1 + np.sum(np.abs(draws - delta.mean()) >= abs(delta.mean() - null))) / 2001
                )
                result["null_effect"] = null
                fold_results[control][str(block)] = result
        comparisons[fold["name"]] = {
            "paired": fold_results,
            "nll": {
                name: float(frame.groupby("time").nll.mean().mean()) for name, frame in aggregates.items()
            },
        }
    # One conservative family across both windows and all seven declared comparisons.
    family = [value["3"] for fold in comparisons.values() for value in fold["paired"].values()]
    ordered = sorted(family, key=lambda r: r["p_value"])
    previous = 0.0
    for i, row in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - i) * row["p_value"]))
        row["holm_p_value"] = previous
    save(
        output / "public_summary.json",
        {
            "status": "COMPLETE",
            "base_cells": len(cells),
            "comparisons": comparisons,
            "cells": cells,
            "scope": "Prospectively specified new-sample public exploration; not canonical confirmation",
            "protected_access": False,
        },
    )


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(SPEC.read_text())
    started = time.monotonic()
    save(
        output / "freeze.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "sources": identity(),
            "specification": spec,
            "resource_before": memory_snapshot(),
            "scope": "Execution freeze before any generated/new public outcomes; canonical protected freeze untouched",
        },
    )
    try:
        bounded_process(
            [sys.executable, "-m", "pytest", "-q", "-W", "error::DeprecationWarning"],
            output / "integrity",
            240,
        )
        for family in FAMILIES:
            for seed in PATH_SEEDS:
                remaining = spec["budget"]["max_total_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("S3 campaign ceiling reached")
                cell = output / "mechanism" / family / str(seed)
                bounded_process(
                    [
                        sys.executable,
                        "-m",
                        "scripts.run_frem_s3",
                        "--output",
                        str(output),
                        "--stage",
                        "mechanism",
                        "--family",
                        family,
                        "--seed",
                        str(seed),
                    ],
                    cell,
                    min(360, remaining),
                )
                print(f"Mechanism completed: {family} seed {seed}", flush=True)
        mechanism = mechanism_summary(output)
        if not mechanism["gate_passed"]:
            save(
                output / "completion.json",
                {
                    "status": "STOPPED_AT_MECHANISM_GATE",
                    "mechanism": mechanism,
                    "market_fits": 0,
                    "protected_access": False,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            return
        # Acquire only after the predeclared mechanism gate and code freeze.
        from scripts.public_equity_backtests import acquire

        acquire(json.loads(CONFIG.read_text()), output / "public_data")
        for fold in json.loads(CONFIG.read_text())["folds"]:
            for seed in SEEDS:
                for name in ("frem", "no_jepa", "ridge", "xgboost", "zero", "historical_mean"):
                    remaining = spec["budget"]["max_total_seconds"] - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("S3 campaign ceiling reached")
                    cell = output / "public" / fold["name"] / str(seed) / name
                    bounded_process(
                        [
                            sys.executable,
                            "-m",
                            "scripts.run_frem_s3",
                            "--output",
                            str(output),
                            "--stage",
                            "public",
                            "--fold",
                            fold["name"],
                            "--seed",
                            str(seed),
                            "--model",
                            name,
                        ],
                        cell,
                        min(360, remaining),
                    )
                    print(f"Public completed: {fold['name']} seed {seed} {name}", flush=True)
        bounded_process(
            [sys.executable, "-m", "scripts.run_frem_s3", "--output", str(output), "--stage", "summary"],
            output / "analysis",
            180,
        )
        save(
            output / "completion.json",
            {
                "status": "COMPLETE_EXPLORATORY",
                "market_learned_fits": 24,
                "constant_cells": 12,
                "protected_access": False,
                "elapsed_seconds": time.monotonic() - started,
                "canonical_confirmation": "NOT_EXECUTED",
                "portfolio": "NOT_EXECUTED_NO_ADMITTED_EXECUTION_DATA",
            },
        )
    except BaseException as error:
        save(
            output / "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=["mechanism", "public", "summary"])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.stage == "mechanism":
        mechanism_worker(args.output, args.family, args.seed)
    elif args.stage == "public":
        public_worker(args.output, args.fold, args.seed, args.model)
    elif args.stage == "summary":
        public_summary(args.output)
    elif args.execute:
        run(args.output.resolve())
    else:
        parser.error("Explicit --execute required")
