"""Explicit development and generated qualification; protected execution is unavailable."""

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

from .config import ARMS, ModelConfig, RunConfig
from .runtime import admit, digest, identity, manifest, read_json, verify, write_json


def source_identity():
    return identity({p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    from packaging.specifiers import SpecifierSet

    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("supported Python range is 3.11 through 3.13")
    constraints = {
        "numpy": ">=1.26,<3",
        "scipy": ">=1.12,<2",
        "torch": ">=2.4,<3",
        "xgboost": ">=2.1,<4",
    }
    versions = {name: importlib.metadata.version(name) for name in constraints}
    for name, requirement in constraints.items():
        if versions[name] not in SpecifierSet(requirement):
            raise RuntimeError(
                f"unsupported {name} version {versions[name]}; requires {requirement}"
            )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def qualification_config(device):
    return RunConfig(
        model=ModelConfig(
            context=8,
            horizon=2,
            latent=8,
            reservoir=8,
            initial_neurons=4,
            growth_step=2,
            growth_every=1,
            bands=2,
            memory_capacity=32,
            neighbors=4,
        ),
        seeds=(11,),
        arms=ARMS,
        epochs=2,
        batch_size=32,
        patience=2,
        device=device,
        embargo=2,
        bootstrap_reps=99,
        block_dates=4,
        xgb_rounds=5,
        max_seconds=1800,
    )


def campaign(root, config, data_path=None, qualification=None, resume=None):
    # These settings must precede importing torch / XGBoost / BLAS.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    root = Path(root)
    prior_seconds = 0.0
    if resume:
        prior_root = Path(resume).parent.parent
        if (prior_root / "status.json").exists():
            raise ValueError("completed runs cannot be resumed")
        prior_seconds = float(json.loads((prior_root / "failure.json").read_text())["wall_seconds"])
        if prior_seconds >= config.max_seconds or prior_seconds < 0:
            raise ValueError("prior attempt exhausted the unchanged campaign budget")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config.to_dict())
    start = time.monotonic()
    complete = []
    try:
        admission = admit()
        write_json(root / "admission.json", admission)
        if not admission["passed"]:
            raise RuntimeError(
                "512 MiB/no-rising-pageout admission failed before numerical imports"
            )
        env = environment()
        source = source_identity()
        write_json(root / "environment.json", env)
        write_json(root / "source.json", {"identity": source})
        if data_path is not None:
            if qualification is None:
                raise ValueError(
                    "market development requires a completed exact-source qualification bundle"
                )
            verify(qualification)
            q = Path(qualification)
            receipt = json.loads((q / "status.json").read_text())
            if (
                receipt.get("status") != "COMPLETE"
                or receipt.get("kind") != "generated_qualification"
            ):
                raise ValueError("qualification did not complete")
            if (
                json.loads((q / "source.json").read_text())["identity"] != source
                or json.loads((q / "environment.json").read_text()) != env
                or identity(json.loads((q / "config.json").read_text()))
                != identity(qualification_config(config.device).to_dict())
            ):
                raise ValueError("qualification source/runtime/device/config changed")
        import numpy as np
        import torch

        from .baselines import baseline, hybrid
        from .data import Samples, generated_panel, load_csv
        from .evaluation import holm, paired_bootstrap, portfolio, score_law, summary
        from .training import budget, predict, train_neural
        from .verification import verify_semantics

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        write_json(
            root / "hardware.json",
            {
                "device": config.device,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
            },
        )
        if resume and (
            len(config.arms) != 1
            or len(config.seeds) != 1
            or config.arms[0] in ("zero", "ridge", "xgboost")
        ):
            raise ValueError(
                "epoch resume requires one neural arm and one seed, with original configuration binding"
            )
        panel = load_csv(data_path, config.protected_start) if data_path else generated_panel()
        samples = Samples(panel, config)
        write_json(root / "data.json", panel.provenance)
        write_json(root / "splits.json", samples.split_receipt())
        np.savez(
            root / "preprocessing.npz",
            center=samples.center,
            scale=samples.scale,
            target_scale=samples.target_scale,
        )
        data_id = identity(
            {
                "provenance": panel.provenance,
                "features": digest(data_path)
                if data_path
                else identity({"seed": 391, "length": 180, "assets": 4}),
                "splits": samples.split_receipt(),
            }
        )
        deadline = start + config.max_seconds - prior_seconds
        date_scores = {}
        for arm in config.arms:
            for seed in config.seeds:
                budget(deadline)
                cell = root / f"{arm}-s{seed}"
                cell.mkdir()
                binding = {
                    "source": source,
                    "data": data_id,
                    "config": config.to_dict(),
                    "environment": env,
                    "arm": arm,
                    "seed": seed,
                }
                # Canonical identity avoids tuple/list serialization differences during restore.
                binding = json.loads(json.dumps(binding))
                write_json(cell / "started.json", {"binding": binding})
                cell_start = time.monotonic()
                if config.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                if arm in ("zero", "ridge", "xgboost"):
                    raw = baseline(samples, config, arm, seed, cell, deadline)
                else:
                    model = train_neural(
                        samples, config, arm, seed, cell, deadline, binding, resume
                    )
                    raw = predict(model, samples, samples.test, config.device, deadline)
                    if arm == "hybrid_xgboost":
                        validation_raw = predict(
                            model, samples, samples.validation, config.device, deadline
                        )
                        raw = hybrid(
                            raw, validation_raw, samples, config, seed, cell / "xgboost", deadline
                        )
                    del model
                scores = score_law(raw, samples.y[samples.test], config.model.student_df)
                if not np.allclose(raw["nll"], scores["nll"], atol=3e-5, rtol=3e-5):
                    raise ValueError("independent Student-mixture score reconstruction failed")
                np.savez(
                    cell / "predictions.npz",
                    **raw,
                    y=samples.y[samples.test],
                    times=samples.time[samples.test],
                    ends=samples.end[samples.test],
                    assets=samples.asset[samples.test],
                )
                report = summary(
                    raw,
                    samples.y[samples.test],
                    samples.time[samples.test],
                    config.model.student_df,
                )
                report["wall_seconds"] = time.monotonic() - cell_start
                report["process_peak_rss_bytes"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
                report["rss_scope"] = "process lifetime maximum, not an isolated cell peak"
                report["peak_cuda_bytes"] = (
                    torch.cuda.max_memory_allocated() if config.device == "cuda" else 0
                )
                write_json(cell / "metrics.json", report)
                for multiplier in (1, 2, 4):
                    write_json(
                        cell / f"portfolio_cost{multiplier}.json",
                        portfolio(samples, raw, config, cost_multiplier=multiplier),
                    )
                date_scores.setdefault(arm, []).append(
                    np.array(
                        [
                            scores["nll"][samples.time[samples.test] == d].mean()
                            for d in np.unique(samples.time[samples.test])
                        ]
                    )
                )
                complete.append({"arm": arm, "seed": seed})
        for policy in ("equal_weight", "cash"):
            write_json(root / f"{policy}.json", portfolio(samples, raw, config, policy=policy))
        comparisons = {}
        if "full" in date_scores:
            candidate = np.mean(date_scores["full"], axis=0)
            for arm in config.arms:
                if arm != "full":
                    difference = np.mean(date_scores[arm], axis=0) - candidate
                    if len(difference) < 2 * config.block_dates:
                        comparisons[arm] = {"status": "INSUFFICIENT_DATES_FOR_DECLARED_BLOCK"}
                    else:
                        comparisons[arm] = paired_bootstrap(
                            difference, config.block_dates, config.bootstrap_reps
                        )
            available = [v for v in comparisons.values() if "p_two_sided" in v]
            for value, adjusted in zip(
                available, holm([v["p_two_sided"] for v in available]), strict=True
            ):
                value["p_holm"] = float(adjusted)
        write_json(
            root / "comparisons.json",
            {
                "seed_aggregation": "average within arm before date-block inference",
                "comparisons": comparisons,
                "practical_nll_margin": 0.01,
                "status": "GENERATED_ONLY" if data_path is None else "EXPLORATORY_DEVELOPMENT",
            },
        )
        verify_semantics(root)
        write_json(
            root / "status.json",
            {
                "status": "COMPLETE",
                "kind": "generated_qualification" if data_path is None else "development",
                "complete_cells": complete,
                "expected_cells": len(config.arms) * len(config.seeds),
                "wall_seconds": prior_seconds + time.monotonic() - start,
                "scientific_status": "NOT_EVALUATED" if data_path is None else "DEVELOPMENT_ONLY",
                "protected_access": False,
            },
        )
    except BaseException as error:
        write_json(
            root / "failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "complete_cells": complete,
                "expected_cells": len(config.arms) * len(config.seeds),
                "wall_seconds": prior_seconds + time.monotonic() - start,
            },
        )
        raise
    finally:
        write_json(root / "manifest.json", manifest(root))
    return {"output": str(root), "status": "COMPLETE", "cells": len(complete)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="frem-s13")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--output", required=True)
    qualify = sub.add_parser("qualify")
    qualify.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    qualify.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--qualification", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--resume-checkpoint")
    check = sub.add_parser("verify")
    check.add_argument("--bundle", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = admit()
            write_json(args.output, result)
            print(json.dumps(result))
            return 0 if result["passed"] else 2
        if args.command == "plan":
            cfg = RunConfig.from_dict(read_json(args.config))
            result = {
                "configuration": cfg.to_dict(),
                "cells": len(cfg.arms) * len(cfg.seeds),
                "source_identity": source_identity(),
                "protected_execution": "unsupported",
            }
        elif args.command == "verify":
            result = verify(args.bundle)
            admission = admit()
            if not admission["passed"]:
                raise RuntimeError(
                    "inventory passed; numerical semantic verification resource-blocked"
                )
            from .verification import verify_semantics

            result["semantics"] = verify_semantics(Path(args.bundle))
        elif args.command == "qualify":
            result = campaign(args.output, qualification_config(args.device))
        else:
            cfg = RunConfig.from_dict(read_json(args.config))
            result = campaign(
                args.output, cfg, args.data, args.qualification, args.resume_checkpoint
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
