"""Command-line interface for data, models, experiments, optimization and reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import fit_linear_baseline
from .checkpoint import load_checkpoint, model_state_sha256, save_checkpoint
from .data import load_data_manifest, validate_point_in_time_frame, verify_snapshot
from .engineering_data import generate_engineering_dataset
from .engram import SpectralEngramMemory
from .evaluation import PurgedWalkForwardSplitter
from .experiments import (
    ExperimentCell,
    ExperimentManifest,
    completion_status,
    run_manifest,
    save_manifest,
)
from .mandate import ClientMandate
from .memory_diagnostics import memory_diagnostics
from .model import REJEPA, REJEPAConfig, student_t_nll
from .portfolio import (
    ConstrainedInstrument,
    ScenarioAllocationPolicy,
    optimize_scenario_cvar,
)
from .reporting import StrategyEvidence, render_strategy_markdown, write_strategy_report
from .simulator import CounterfactualLatentSimulator
from .training import TrainingBatch, TrainingConfig, fit_model, predict_frozen, seed_everything


def _load_training_npz(path: str) -> tuple[TrainingBatch, TrainingBatch]:
    with np.load(path) as payload:
        required = {"context", "future_target", "realized_return"}
        if not required <= set(payload.files):
            raise ValueError(f"Training NPZ requires {sorted(required)}")
        context = payload["context"].copy()
        future = payload["future_target"].copy()
        returns = payload["realized_return"].copy()
        horizon = payload["horizon"].copy() if "horizon" in payload else None
        decision_time = payload["decision_time_ns"].copy() if "decision_time_ns" in payload else None
        outcome_time = payload["outcome_available_ns"].copy() if "outcome_available_ns" in payload else None
    if len(context) < 4:
        raise ValueError("At least four chronological samples are required")
    if decision_time is not None or outcome_time is not None:
        if decision_time is None or outcome_time is None:
            raise ValueError("decision_time_ns and outcome_available_ns must be supplied together")
        if decision_time.shape != (len(context),) or outcome_time.shape != (len(context),):
            raise ValueError("Chronology arrays must align with training samples")
        unique_dates = np.unique(decision_time)
        if len(unique_dates) < 5:
            raise ValueError("At least five decision dates are required")
        split_date = unique_dates[min(len(unique_dates) - 1, max(1, int(len(unique_dates) * 0.8)))]
        train_mask = (decision_time < split_date) & (outcome_time <= split_date)
        validation_mask = decision_time >= split_date
        if train_mask.sum() < 2 or validation_mask.sum() < 2:
            raise ValueError("Purged chronological split leaves too few samples")
        selections = (train_mask, validation_mask)
    else:
        split = max(1, int(len(context) * 0.8))
        if split >= len(context):
            split = len(context) - 1
        selections = (slice(None, split), slice(split, None))
    batches = []
    for indices in selections:
        batches.append(
            TrainingBatch(
                context=np_to_tensor(context[indices]),
                future_target=np_to_tensor(future[indices]),
                realized_return=np_to_tensor(returns[indices]),
                horizon=None if horizon is None else torch.as_tensor(horizon[indices], dtype=torch.long),
            )
        )
    return batches[0], batches[1]


def np_to_tensor(value: np.ndarray):
    return torch.as_tensor(value, dtype=torch.float32)


def _command_data_validate(args: argparse.Namespace) -> dict[str, object]:
    report = validate_point_in_time_frame(pd.read_csv(args.csv))
    result: dict[str, object] = asdict(report)
    if args.manifest:
        manifest = load_data_manifest(args.manifest)
        result["manifest_verified"] = verify_snapshot(args.csv, manifest)
        result["manifest_row_count_matches"] = manifest.row_count == report.row_count
        if not result["manifest_verified"] or not result["manifest_row_count_matches"]:
            raise ValueError("Snapshot manifest does not match the supplied data")
    return result


def _command_synthetic_generate(args: argparse.Namespace) -> dict[str, object]:
    return asdict(
        generate_engineering_dataset(
            args.output,
            seed=args.seed,
            asset_count=args.assets,
            decision_dates=args.dates,
            context_length=args.context,
            horizon=args.horizon,
        )
    )


def _command_baseline_train(args: argparse.Namespace) -> dict[str, object]:
    payload = np.load(args.npz)
    fitted = fit_linear_baseline(payload["features"], payload["target"], kind=args.kind)
    prediction = fitted.predict(payload["features"])
    return {"model": fitted.name, "training_mse": float(np.mean((prediction - payload["target"]) ** 2))}


def _command_model_train(args: argparse.Namespace) -> dict[str, object]:
    train, validation = _load_training_npz(args.npz)
    mechanism_off = args.command == "jepa-train"
    max_horizon = int(torch.as_tensor(train.horizon).max()) if train.horizon is not None else 20
    seed_everything(args.seed)
    model = REJEPA(
        REJEPAConfig(
            input_dim=train.context.shape[-1],
            use_world_model=not mechanism_off,
            use_engram=not mechanism_off,
            max_horizon=max(20, max_horizon),
        )
    )
    result = fit_model(
        model,
        [train],
        [validation],
        TrainingConfig(
            epochs=args.epochs,
            patience=min(args.patience, args.epochs),
            seed=args.seed,
            world_seed=args.seed + 1000,
        ),
    )
    hashes = save_checkpoint(
        args.output,
        model,
        metadata={"phase": "development", "scientific_status": "NOT_EVALUATED"},
    )
    return {"training": asdict(result), "checkpoint_hashes": hashes}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    source = Path(__file__).resolve().parent
    for path in sorted(source.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _command_engineering_smoke(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Engineering smoke output must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, validation = _load_training_npz(args.npz)
    maximum_horizon = int(torch.as_tensor(train.horizon).max()) if train.horizon is not None else 5
    seed_everything(args.seed)
    model = REJEPA(
        REJEPAConfig(
            input_dim=train.context.shape[-1],
            reservoir_dim=16,
            latent_dim=12,
            spectral_bands=3,
            worlds=4,
            world_horizon=maximum_horizon,
            interpreters=2,
            max_horizon=max(20, maximum_horizon),
        )
    )
    training = fit_model(
        model,
        [train],
        [validation],
        TrainingConfig(
            epochs=args.epochs,
            patience=min(args.patience, args.epochs),
            seed=args.seed,
            world_seed=args.seed + 1000,
        ),
    )
    checkpoint = output / "checkpoint"
    checkpoint_hashes = save_checkpoint(
        checkpoint,
        model,
        metadata={
            "phase": "engineering_compatibility",
            "scientific_status": "NOT_EVALUATED",
            "dataset_sha256": _file_sha256(args.npz),
        },
    )
    restored, metadata, _ = load_checkpoint(checkpoint)
    forecast = predict_frozen(
        restored,
        validation.context,
        seed=args.seed + 2000,
        horizon=validation.horizon,
    )
    nll = float(
        student_t_nll(
            validation.realized_return,
            forecast.return_location,
            forecast.return_scale,
            restored.config.student_df,
        ).mean()
    )
    baseline = fit_linear_baseline(
        train.context.numpy().reshape(len(train.context), -1),
        train.realized_return.numpy(),
        kind="ridge",
    )
    baseline_prediction = baseline.predict(validation.context.numpy().reshape(len(validation.context), -1))
    baseline_mse = float(np.mean((baseline_prediction - validation.realized_return.numpy()) ** 2))
    elapsed = time.perf_counter() - started
    dataset_sha256 = _file_sha256(args.npz)
    manifest = ExperimentManifest(
        experiment_id=f"rejepa-v0.3-engineering-seed-{args.seed}",
        phase="engineering_compatibility",
        dataset_sha256=dataset_sha256,
        source_tree_sha256=_source_tree_sha256(),
        protected_start=None,
        cells=(
            ExperimentCell(f"ridge-s{args.seed}", "ridge", args.seed, 0.8),
            ExperimentCell(f"rejepa-full-s{args.seed}", "rejepa_full", args.seed, 0.8),
        ),
    ).freeze()
    save_manifest(output / "manifest.json", manifest)

    def cell_result(cell: ExperimentCell) -> dict[str, object]:
        if cell.model == "ridge":
            return {"validation_mse": baseline_mse, "scientific_use": "PROHIBITED"}
        return {
            "validation_student_t_nll": nll,
            "best_validation_composite_loss": training.best_validation_loss,
            "completed_epochs": training.completed_epochs,
            "scientific_use": "PROHIBITED",
        }

    cell_receipts = run_manifest(manifest, output / "cells", cell_result)
    matrix_status = completion_status(manifest, cell_receipts)
    receipt = {
        "status": "ENGINEERING_COMPATIBILITY_PASS",
        "scientific_status": "NOT_EVALUATED",
        "input_label_required": "SYNTHETIC_ENGINEERING_FIXTURE_NOT_MARKET_EVIDENCE",
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest.sha256,
        "matrix_status": matrix_status,
        "train_samples": len(train.context),
        "validation_samples": len(validation.context),
        "training": asdict(training),
        "validation_student_t_nll": nll,
        "ridge_validation_mse": baseline_mse,
        "checkpoint_metadata": metadata,
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint_state_sha256": model_state_sha256(restored.state_dict()),
        "model_parameters": sum(parameter.numel() for parameter in restored.parameters()),
        "wall_seconds": elapsed,
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "limitations": [
            "Synthetic fixture only; no market performance inference",
            "Single bounded seed; no scientific comparison",
            "No protected outcomes or portfolio claims",
        ],
    }
    (output / "ENGINEERING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt


def _command_simulator_train(args: argparse.Namespace) -> dict[str, object]:
    payload = np.load(args.npz)
    if not {"latent", "target"} <= set(payload.files):
        raise ValueError("Simulator NPZ requires latent and target arrays")
    latent = np_to_tensor(payload["latent"])
    target = np_to_tensor(payload["target"])
    if latent.ndim != 2 or target.shape != latent.shape or len(latent) < 2:
        raise ValueError("latent and target must be aligned rank-two arrays")
    condition = np_to_tensor(payload["condition"]) if "condition" in payload else torch.zeros(len(latent), 8)
    torch.manual_seed(args.seed)
    simulator = CounterfactualLatentSimulator(latent.shape[1])
    optimizer = torch.optim.AdamW(simulator.parameters(), lr=args.learning_rate)
    history: list[float] = []
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        output = simulator(
            latent,
            condition,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        loss = torch.nn.functional.smooth_l1_loss(output.trajectories[:, :, -1, :].mean(dim=1), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Simulator loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(simulator.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(simulator.state_dict(), destination)
    return {"epochs": args.epochs, "final_loss": history[-1], "path": str(destination)}


def _command_memory_experiment(args: argparse.Namespace) -> dict[str, object]:
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    memory = SpectralEngramMemory.from_export(payload["memory"])
    query = np_to_tensor(np.asarray(payload["query"], dtype=float))
    return memory_diagnostics(memory, query, as_of_ns=int(payload["as_of_ns"]))


def _command_portfolio_optimize(args: argparse.Namespace) -> dict[str, object]:
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    instruments = [ConstrainedInstrument(**item) for item in payload["instruments"]]
    policy = ScenarioAllocationPolicy(**payload.get("policy", {}))
    result = optimize_scenario_cvar(
        np.asarray(payload["expected_returns"], dtype=float),
        np.asarray(payload["scenario_returns"], dtype=float),
        np.asarray(payload["uncertainty"], dtype=float),
        np.asarray(payload["current_weights"], dtype=float),
        instruments,
        policy,
        data_age_days=int(payload["data_age_days"]),
    )
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}


def _command_walk_forward(args: argparse.Namespace) -> dict[str, object]:
    frame = pd.read_csv(args.csv)
    if args.date_column not in frame:
        raise ValueError("Date column is absent")
    splitter = PurgedWalkForwardSplitter(args.train, args.validation, args.test, args.embargo)
    folds = splitter.split(frame[args.date_column].to_numpy())
    return {
        "fold_count": len(folds),
        "folds": [
            {
                "train_start": fold.train_dates[0].isoformat(),
                "train_end": fold.train_dates[-1].isoformat(),
                "validation_start": fold.validation_dates[0].isoformat(),
                "test_start": fold.test_dates[0].isoformat(),
                "test_end": fold.test_dates[-1].isoformat(),
            }
            for fold in folds
        ],
    }


def _command_report(args: argparse.Namespace) -> dict[str, object]:
    evidence = StrategyEvidence(
        scientific_status="NOT_EVALUATED",
        admitted_data=False,
        approved_universe_verified=False,
        protected_evaluation_complete=False,
    )
    content = render_strategy_markdown(ClientMandate(), evidence)
    write_strategy_report(args.output, content)
    return {"status": "BLOCKED_EVIDENCE_DRAFT", "path": args.output}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rejepa", description="RENG-JEPA research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("data-validate")
    data.add_argument("csv")
    data.add_argument("--manifest")
    data.set_defaults(handler=_command_data_validate)
    synthetic = subparsers.add_parser("synthetic-generate")
    synthetic.add_argument("output")
    synthetic.add_argument("--seed", type=int, default=11)
    synthetic.add_argument("--assets", type=int, default=6)
    synthetic.add_argument("--dates", type=int, default=96)
    synthetic.add_argument("--context", type=int, default=16)
    synthetic.add_argument("--horizon", type=int, default=5)
    synthetic.set_defaults(handler=_command_synthetic_generate)
    baseline = subparsers.add_parser("baseline-train")
    baseline.add_argument("npz")
    baseline.add_argument("--kind", choices=("ridge", "elastic_net"), default="ridge")
    baseline.set_defaults(handler=_command_baseline_train)
    for name in ("jepa-train", "full-train"):
        training = subparsers.add_parser(name)
        training.add_argument("npz")
        training.add_argument("output")
        training.add_argument("--epochs", type=int, default=20)
        training.add_argument("--patience", type=int, default=5)
        training.add_argument("--seed", type=int, default=11)
        training.set_defaults(handler=_command_model_train)
    simulator = subparsers.add_parser("simulator-train")
    simulator.add_argument("npz")
    simulator.add_argument("output")
    simulator.add_argument("--epochs", type=int, default=20)
    simulator.add_argument("--learning-rate", type=float, default=3e-4)
    simulator.add_argument("--seed", type=int, default=11)
    simulator.set_defaults(handler=_command_simulator_train)
    memory = subparsers.add_parser("memory-experiment")
    memory.add_argument("json")
    memory.set_defaults(handler=_command_memory_experiment)
    portfolio = subparsers.add_parser("portfolio-optimize")
    portfolio.add_argument("json")
    portfolio.set_defaults(handler=_command_portfolio_optimize)
    walk = subparsers.add_parser("walk-forward")
    walk.add_argument("csv")
    walk.add_argument("--date-column", default="event_time")
    walk.add_argument("--train", type=int, required=True)
    walk.add_argument("--validation", type=int, required=True)
    walk.add_argument("--test", type=int, required=True)
    walk.add_argument("--embargo", type=int, default=1)
    walk.set_defaults(handler=_command_walk_forward)
    report = subparsers.add_parser("report-generate")
    report.add_argument("output")
    report.set_defaults(handler=_command_report)
    smoke = subparsers.add_parser("engineering-smoke")
    smoke.add_argument("npz")
    smoke.add_argument("output")
    smoke.add_argument("--epochs", type=int, default=3)
    smoke.add_argument("--patience", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=11)
    smoke.set_defaults(handler=_command_engineering_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except Exception as error:  # noqa: BLE001 - CLI must return a machine-readable failed receipt
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)}))
        return 2
    print(json.dumps({"status": "COMPLETE", "result": result}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
