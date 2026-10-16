"""Explicit, bounded experiment execution. Nothing executes on import."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_text, file_digest, object_digest, run_directory, write_json
from .config import finite_number, positive_integer, project_root, validate_config
from .governance import append_event, consume_confirmatory_freeze, identity, verify_freeze

ABLATIONS: dict[str, dict[str, Any]] = {
    "frem": {},
    "no_memory": {"use_memory": False},
    "no_spectral": {"use_spectral": False},
    "no_reservoir": {"use_reservoir": False},
    "no_jepa": {"use_jepa": False},
    "no_gate": {"use_feature_gate": False},
    "no_controller": {"use_adaptive_controller": False},
    "no_ngram": {"use_ngram": False},
    "fixed_scale": {"use_stochastic": False},
    "fifo_memory": {"memory_retention": "fifo"},
    "uniform_memory": {"memory_retention": "random"},
    "hybrid_xgboost": {"residual_xgboost": True},
}
BASELINES = {"zero", "ridge", "reservoir_ridge", "xgboost", "gru", "dlinear", "patch_transformer"}
BUDGET_NOTE = (
    "Cooperative wall checks include data preparation, fitting, prediction and artifact writes. "
    "Python cannot interrupt a running native tensor, tree, linear-solver or I/O call; "
    "use an external process timeout for a hard wall limit. Training overruns are recorded, "
    "and a matrix or suite overrun fails at the next check, including its final cell."
)


def experiment_plan(config: dict[str, Any], phase: str, *, root: str | Path | None = None) -> dict[str, Any]:
    config = validate_config(config, root=root)
    if phase not in {"development", "confirmatory"}:
        raise ValueError("phase must be development or confirmatory")
    return {
        "phase": phase,
        "executed": False,
        "data_path": config["data"]["path"],
        "protected_start": config["split"]["protected_start"],
        "runs": [{"model": m, "seed": s} for m in config["models"] for s in config["seeds"]],
        "budget": config["budget"],
        "budget_scope": "one explicit matrix invocation; an optional robustness suite has a separate declared budget",
        "wall_budget_note": BUDGET_NOTE,
        "requirement": "Explicit --execute; confirmatory additionally requires an unchanged freeze",
    }


def _remaining(started: float, limit: float, stage: str, *, directory: Path | None = None) -> float:
    elapsed = time.perf_counter() - started
    if elapsed >= limit:
        if directory is not None:
            write_json(
                directory / "budget_failure.json",
                {
                    "status": "EXCEEDED",
                    "stage": stage,
                    "max_wall_seconds": limit,
                    "elapsed_seconds": elapsed,
                    "note": BUDGET_NOTE,
                },
            )
        raise TimeoutError(
            f"Wall-time budget exhausted at {stage}; retain this failed matrix/suite and all completed cells"
        )
    return limit - elapsed


def _archive_protocol(directory: Path, base: Path, study_identity: dict[str, Any]) -> None:
    """Archive exact UTF-8 source bytes, then verify against the run identity."""
    source = base / "research" / "protocol.json"
    expected = study_identity.get("code", {}).get("files", {}).get("research/protocol.json")
    if expected is None:
        raise ValueError("Research protocol was not included in the run source identity")
    atomic_text(directory / "protocol.json", source.read_bytes().decode("utf-8"))
    if file_digest(directory / "protocol.json") != expected:
        raise ValueError("Research protocol changed during run creation or was not archived byte-for-byte")


def prepare_data(config: dict[str, Any], phase: str):
    from .data import add_causal_peer_features, build_windows, load_panel
    from .data_sources import load_macro
    from .splits import chronological_split

    dc = config["data"]
    panel = load_panel(dc["path"], dc.get("column_map"))
    cutoff = pd.Timestamp(config["split"]["protected_start"])
    if cutoff.tzinfo is None:
        raise ValueError("protected_start must include its UTC offset")
    cutoff = cutoff.tz_convert("UTC")
    if phase == "development":
        mask = pd.to_datetime(panel["timestamp"], utc=True) < cutoff
        for name in (
            "available_at",
            "fx_available_at",
            "corporate_actions_available_at",
            "feature_available_at",
        ):
            if name in panel:
                known = pd.to_datetime(panel[name], utc=True, errors="coerce")
                mask &= known.isna() | (known < cutoff)
        panel = panel.loc[mask].copy()
    if dc.get("peer_features", False):
        panel = add_causal_peer_features(
            panel, max_age=dc.get("peer_max_age", "3D"), min_peers=dc.get("peer_min_count", 2)
        )
    macro = load_macro(dc["macro_path"]) if dc.get("macro_path") else None
    if macro is not None and phase == "development":
        macro = macro.loc[pd.to_datetime(macro["available_at"], utc=True) < cutoff].copy()
    windows = build_windows(
        panel,
        context=dc.get("context", 64),
        horizon=dc.get("horizon", 5),
        feature_columns=dc.get("feature_columns"),
        macro=macro,
        base_currency=dc.get("base_currency", "USD"),
        include_missingness=dc.get("include_missingness", True),
    )
    split = chronological_split(
        windows,
        train_fraction=config["split"].get("train_fraction", 0.6),
        validation_fraction=config["split"].get("validation_fraction", 0.2),
        embargo_bars=config["split"].get("embargo_bars", 5),
        protected_start=cutoff,
    )
    return windows, split


def _calibration_split(data, embargo_bars: int):
    dates = np.unique(data.time)
    if len(dates) < 2 * embargo_bars + 8:
        raise ValueError("Hybrid requires a longer separate tuning/calibration interval")
    mid = len(dates) // 2
    boundary = dates[mid]
    tune_idx = np.flatnonzero((data.time < dates[max(0, mid - embargo_bars)]) & (data.label_time < boundary))
    cal_idx = np.flatnonzero(data.time >= dates[min(len(dates) - 1, mid + embargo_bars)])
    if len(tune_idx) < 2 or len(cal_idx) < 2:
        raise ValueError("Purging leaves insufficient hybrid tuning/calibration observations")
    return data.subset(tune_idx), data.subset(cal_idx)


def execute_matrix(
    config: dict[str, Any],
    *,
    phase: str = "development",
    freeze_path: str | Path | None = None,
    root: str | Path | None = None,
) -> Path:
    """Run only when called explicitly by the CLI; preserve all failures.

    Stopping a matrix on any exception avoids silently selecting successful seeds.
    Partial successful artifacts remain for diagnosis but cannot establish support.
    """
    base = Path(root or project_root()).resolve()
    config = validate_config(config, root=base)
    experiment_plan(config, phase, root=base)
    started = time.perf_counter()
    matrix_limit = float(config["budget"]["max_wall_seconds"])
    freeze = None
    if phase == "confirmatory":
        if freeze_path is None:
            raise ValueError("Confirmatory evaluation requires --freeze")
        freeze = verify_freeze(config, base, freeze_path)
    study_identity = identity(config, base)
    if freeze:
        study_identity["freeze_sha256"] = freeze["freeze_sha256"]
        consume_confirmatory_freeze(base, freeze)
    with run_directory(config["output_dir"], config, phase, study_identity) as directory:
        _archive_protocol(directory, base, study_identity)
        write_json(
            directory / "budget.json",
            {
                "matrix_max_wall_seconds": matrix_limit,
                "per_fit_max_wall_seconds": config["training"].get("max_wall_seconds"),
                "requested_cells": len(config["models"]) * len(config["seeds"]),
                "timer_starts_before": "identity hashing and data preparation",
                "note": BUDGET_NOTE,
            },
        )
        _remaining(started, matrix_limit, "before data preparation", directory=directory)
        from .baselines import make_baseline
        from .evaluation import evaluate_predictions
        from .model import ModelConfig
        from .robustness import held_out_assets, perturb_inputs, recent_training_fraction
        from .training import Trainer, TrainingConfig

        data, split = prepare_data(config, phase)
        _remaining(started, matrix_limit, "after data preparation", directory=directory)
        train, validation = data.subset(split.train), data.subset(split.validation)
        evaluation_indices = split.protected if phase == "confirmatory" else split.test
        if len(evaluation_indices) == 0:
            raise ValueError(f"No eligible {phase} evaluation observations")
        evaluation = data.subset(evaluation_indices)
        train = recent_training_fraction(train, config["evaluation"].get("training_fraction", 1.0))
        ood_names = config["evaluation"].get("held_out_assets", [])
        if ood_names:
            train, _ = held_out_assets(train, ood_names)
            validation, _ = held_out_assets(validation, ood_names)
            _, evaluation = held_out_assets(evaluation, ood_names)
        if len(train.x) < 2 or len(validation.x) < 2:
            raise ValueError("Insufficient training or validation observations after exclusions")
        if len(evaluation.x) == 0:
            raise ValueError("No eligible evaluation observations remain after exclusions")
        write_json(
            directory / "split.json",
            {
                "metadata": split.metadata,
                "train_indices": split.train,
                "validation_indices": split.validation,
                "test_indices": evaluation_indices,
                "feature_names": data.feature_names,
                "asset_names": data.asset_names,
                "post_filter_sizes": {
                    "train": len(train.x),
                    "validation": len(validation.x),
                    "evaluation": len(evaluation.x),
                },
            },
        )
        summaries = []
        for name in config["models"]:
            for seed in config["seeds"]:
                _remaining(started, matrix_limit, f"before {name}/seed_{seed}", directory=directory)
                cell = directory / name / f"seed_{seed}"
                cell.mkdir(parents=True, exist_ok=False)
                cell_started = time.perf_counter()
                this_eval = perturb_inputs(
                    evaluation,
                    train,
                    seed=seed,
                    noise_std=config["evaluation"].get("noise_std", 0.0),
                    missing_probability=config["evaluation"].get("missing_probability", 0.0),
                )
                write_json(cell / "input_perturbation.json", this_eval.metadata["input_perturbation"])
                remaining = _remaining(
                    started, matrix_limit, f"after {name}/seed_{seed} input preparation", directory=directory
                )
                fit_limit = min(remaining, config["training"].get("max_wall_seconds") or remaining)
                if name in ABLATIONS:
                    mc = dict(config["model"])
                    mc.update(ABLATIONS[name])
                    mc.update(
                        input_dim=train.x.shape[-1],
                        context_length=train.x.shape[1],
                        horizon=train.future_x.shape[1],
                    )
                    mc.setdefault("reservoir_seed", seed)
                    tc = dict(config["training"])
                    tc["seed"] = seed
                    tc["max_wall_seconds"] = fit_limit
                    write_json(cell / "resolved_config.json", {"model": mc, "training": tc})
                    trainer = Trainer(ModelConfig(**mc), TrainingConfig(**tc))
                    tune, calibration = validation, None
                    if mc.get("residual_xgboost"):
                        tune, calibration = _calibration_split(
                            validation, config["split"].get("embargo_bars", 5)
                        )
                    fit_started = time.perf_counter()
                    try:
                        history = trainer.fit(train, tune, calibration=calibration)
                    except BaseException:
                        write_json(
                            cell / "fit.json",
                            {
                                "status": "INCOMPLETE",
                                "fit_report": trainer.fit_report,
                                "history": trainer.history,
                                "gradient_steps": trainer.gradient_steps,
                            },
                        )
                        raise
                    fit_elapsed = time.perf_counter() - fit_started
                    trainer.save(cell / "checkpoint")
                    write_json(cell / "fit.json", history)
                    predictor = trainer
                else:
                    common = {
                        key: config["training"][key]
                        for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "device")
                        if key in config["training"]
                    }
                    common.update(
                        hidden_dim=config["model"].get("hidden_dim", 64),
                        reservoir_dim=config["model"].get("reservoir_dim", 64),
                        n_estimators=config["training"].get("xgboost_estimators", 200),
                        max_depth=config["training"].get("xgboost_max_depth", 3),
                        max_wall_seconds=fit_limit,
                    )
                    write_json(
                        cell / "resolved_config.json", {"baseline": name, "seed": seed, "parameters": common}
                    )
                    estimator = make_baseline(name, seed=seed, **common)
                    fit_started = time.perf_counter()
                    try:
                        history = estimator.fit(train)
                    except BaseException:
                        write_json(
                            cell / "fit.json", {"status": "INCOMPLETE", "fit_report": estimator.fit_report}
                        )
                        raise
                    fit_elapsed = time.perf_counter() - fit_started
                    if not hasattr(estimator, "save"):
                        raise TypeError(f"Baseline {name} does not implement checkpoint preservation")
                    estimator.save(cell / "checkpoint")
                    write_json(cell / "fit.json", history)
                    predictor = estimator
                # The primary checkpoint is the fitted pre-evaluation state.
                # Explicit online evaluation may create a separate later state.
                fit_budget = {
                    "max_wall_seconds": fit_limit,
                    "elapsed_seconds": fit_elapsed,
                    "soft_limit_overrun": fit_elapsed > fit_limit,
                    "note": BUDGET_NOTE,
                }
                write_json(cell / "fit_budget.json", fit_budget)
                _remaining(started, matrix_limit, f"after {name}/seed_{seed} fitting", directory=directory)
                if name in ABLATIONS:
                    online = config["evaluation"].get("online_update", False)
                    prediction = predictor.predict(this_eval, online_update=online)
                    if online:
                        predictor.save(cell / "online_checkpoint")
                else:
                    prediction = predictor.predict(this_eval)
                prediction["model"] = name
                prediction["seed"] = seed
                prediction.to_csv(cell / "predictions.csv", index=False)
                metrics = evaluate_predictions(prediction, student_df=None)
                metrics["wall_seconds"] = time.perf_counter() - cell_started
                metrics["training_wall_seconds"] = fit_elapsed
                metrics["training_limit_seconds"] = fit_limit
                metrics["training_soft_limit_overrun"] = fit_budget["soft_limit_overrun"]
                write_json(cell / "metrics.json", metrics)
                _remaining(
                    started,
                    matrix_limit,
                    f"after {name}/seed_{seed} evaluation and artifacts",
                    directory=directory,
                )
                summaries.append({"model": name, "seed": seed, "metrics": metrics})
                write_json(directory / "completed_cells.json", summaries)
                append_event(
                    base / "research" / "events.jsonl",
                    {
                        "event": "cell_completed",
                        "phase": phase,
                        "run": str(directory),
                        "model": name,
                        "seed": seed,
                    },
                )
        write_json(
            directory / "summary.json",
            {
                "phase": phase,
                "cells": summaries,
                "matrix_wall_seconds": time.perf_counter() - started,
                "training_soft_limit_overruns": sum(
                    row["metrics"]["training_soft_limit_overrun"] for row in summaries
                ),
                "wall_budget_note": BUDGET_NOTE,
                "scientific_disposition": "REQUIRES_PAIRED_ANALYSIS_AND_EVIDENCE_AUDIT",
            },
        )
        _remaining(started, matrix_limit, "final matrix completion", directory=directory)
    return directory


def suite_plan(
    config: dict[str, Any], suite: dict[str, Any], *, root: str | Path | None = None
) -> dict[str, Any]:
    """Validate all merged variants before the first dataset or model is opened."""
    config = validate_config(config, root=root)
    if not isinstance(suite, dict):
        raise ValueError("Suite must be a JSON object")
    allowed = {
        "schema_version",
        "status",
        "name",
        "description",
        "max_variants",
        "max_total_runs",
        "max_wall_seconds",
        "variants",
    }
    if set(suite) - allowed:
        raise ValueError(f"Unknown suite fields: {sorted(set(suite) - allowed)}")
    maximum_variants = positive_integer(suite.get("max_variants"), "suite.max_variants")
    maximum_runs = positive_integer(suite.get("max_total_runs"), "suite.max_total_runs")
    maximum_seconds = finite_number(suite.get("max_wall_seconds"), "suite.max_wall_seconds")
    if maximum_seconds <= 0:
        raise ValueError("Suite must set a positive max_wall_seconds")
    variants = suite.get("variants", [])
    if not isinstance(variants, list) or not variants or len(variants) > maximum_variants:
        raise ValueError("Empty suite or requested variants exceed declared suite budget")
    allowed_variant = {
        "name",
        "description",
        "output_dir",
        "data",
        "split",
        "model",
        "training",
        "evaluation",
        "budget",
        "models",
        "seeds",
    }
    names: set[str] = set()
    configurations: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) - allowed_variant:
            raise ValueError(
                "Suite variants must be objects containing only supported configuration overrides"
            )
        name = variant.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Each suite variant requires a nonempty name without surrounding whitespace")
        if name in names:
            raise ValueError(f"Duplicate suite variant name: {name}")
        names.add(name)
        cfg = copy.deepcopy(config)
        for section in ("data", "split", "model", "training", "evaluation", "budget"):
            changes = variant.get(section, {})
            if not isinstance(changes, dict):
                raise ValueError(f"Variant {name}.{section} must be an object")
            cfg[section].update(changes)
        for key in ("name", "description", "output_dir", "models", "seeds"):
            if key in variant:
                cfg[key] = copy.deepcopy(variant[key])
        cfg = validate_config(cfg, root=root)
        scientific = {
            key: value for key, value in cfg.items() if key not in {"name", "description", "output_dir"}
        }
        scientific["models"] = sorted(cfg["models"])
        scientific["seeds"] = sorted(cfg["seeds"])
        fingerprint = object_digest(scientific)
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate effective suite configuration: {name}")
        fingerprints.add(fingerprint)
        configurations.append(cfg)
    total_cells = sum(len(cfg["models"]) * len(cfg["seeds"]) for cfg in configurations)
    if total_cells > maximum_runs:
        raise ValueError("Suite exceeds max_total_runs")
    primary_cells = len(config["models"]) * len(config["seeds"])
    return {
        "phase": "development_suite",
        "executed": False,
        "variants": configurations,
        "requested_runs": total_cells,
        "max_total_runs": maximum_runs,
        "max_variants": maximum_variants,
        "max_wall_seconds": maximum_seconds,
        "primary_matrix_requested_runs": primary_cells,
        "requested_runs_if_primary_and_suite_each_execute_once": primary_cells + total_cells,
        "budget_scope": "This optional sensitivity suite is separate from the primary development matrix; limits apply to one explicit invocation, not a lifetime quota.",
        "wall_budget_note": BUDGET_NOTE,
    }


def execute_suite(
    config: dict[str, Any], suite_path: str | Path, *, root: str | Path | None = None
) -> list[str]:
    """Execute a separate, fully prevalidated development sensitivity budget.

    The supplied defaults request 24 primary-development cells and 36 optional
    suite cells: 60 when each is deliberately executed once. No suite executes
    implicitly and no confirmatory settings are generated by this function.
    """
    base = Path(root or project_root()).resolve()
    config = validate_config(config, root=base)
    source = (base / Path(suite_path).expanduser()).resolve()
    suite_bytes = source.read_bytes()
    suite = json.loads(suite_bytes)
    plan = suite_plan(config, suite, root=base)
    results = []
    suite_started = time.perf_counter()
    max_seconds = plan["max_wall_seconds"]
    suite_identity = {
        "kind": "development_suite",
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "base_config_sha256": object_digest(config),
        "plan_sha256": object_digest(plan),
    }
    with run_directory(
        Path(config["output_dir"]) / "suites",
        {"base_config": config, "plan": plan},
        "development_suite",
        suite_identity,
    ) as directory:
        atomic_text(directory / "suite.json", suite_bytes.decode("utf-8"))
        if file_digest(directory / "suite.json") != suite_identity["suite_sha256"]:
            raise ValueError("Suite source changed during creation of its audit artifact")
        write_json(directory / "plan.json", plan)
        for variant in plan["variants"]:
            remaining = _remaining(
                suite_started, max_seconds, f"before suite variant {variant['name']}", directory=directory
            )
            cfg = copy.deepcopy(variant)
            cfg["budget"]["max_wall_seconds"] = min(remaining, cfg["budget"]["max_wall_seconds"])
            cfg = validate_config(cfg, root=base)
            results.append(str(execute_matrix(cfg, phase="development", root=base)))
            write_json(directory / "completed_variants.json", results)
            _remaining(
                suite_started, max_seconds, f"after suite variant {variant['name']}", directory=directory
            )
        write_json(
            directory / "summary.json",
            {
                "runs": results,
                "requested_runs": plan["requested_runs"],
                "wall_seconds": time.perf_counter() - suite_started,
                "budget_scope": plan["budget_scope"],
                "scientific_disposition": "DEVELOPMENT_SENSITIVITY_REQUIRES_ANALYSIS",
            },
        )
        _remaining(suite_started, max_seconds, "final suite completion", directory=directory)
    return results
