"""Research commands. Training and network acquisition require explicit flags."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .artifacts import file_digest, json_default, read_json, write_json


def _emit(value) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, default=json_default))


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(
        prog="frem", description="FREM research package: explicit execution, preserved evidence"
    )
    sub = app.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Display a matrix without loading data or fitting models")
    plan.add_argument("--config", required=True)
    plan.add_argument("--phase", choices=["development", "confirmatory"], default="development")
    run = sub.add_parser("run", help="Execute an explicitly authorized research matrix")
    run.add_argument("--config", required=True)
    run.add_argument("--phase", choices=["development", "confirmatory"], default="development")
    run.add_argument("--freeze")
    run.add_argument("--execute", action="store_true")
    suite = sub.add_parser("suite", help="Execute declared development robustness variants")
    suite.add_argument("--config", required=True)
    suite.add_argument("--suite", required=True)
    suite.add_argument("--execute", action="store_true")
    freeze = sub.add_parser("freeze", help="Hash the chosen protocol before protected evaluation")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--rationale", required=True)
    inspect = sub.add_parser("inspect-data", help="Validate only the supplied pre-boundary panel")
    inspect.add_argument("--path", required=True)
    inspect.add_argument("--before", required=True, help="Exclusive protected boundary, with UTC offset")
    inspect.add_argument("--column-map", help="JSON mapping input column name to canonical column name")
    predict = sub.add_parser("predict", help="Frozen forecasts for latest unlabeled causal windows")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--config", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--as-of")
    predict.add_argument("--max-staleness", default="7D")
    predict.add_argument(
        "--device",
        default="cpu",
        help="Explicit inference device; defaults to cpu regardless of checkpoint training device",
    )
    compare = sub.add_parser("compare", help="Paired date-block inference from a completed matrix")
    compare.add_argument("--run", required=True)
    compare.add_argument("--candidate", default="frem")
    compare.add_argument("--control", default="no_memory")
    compare.add_argument("--block-dates", type=int)
    compare.add_argument("--resamples", type=int)
    family = sub.add_parser(
        "compare-family", help="Complete mechanism-family comparison with frozen settings and Holm correction"
    )
    family.add_argument("--run", required=True)
    family.add_argument(
        "--block-dates",
        type=int,
        help="Override must belong to the frozen block family for confirmatory evidence",
    )
    family.add_argument(
        "--resamples", type=int, help="Development-only override; confirmatory settings are frozen"
    )
    report = sub.add_parser("report", help="Generate tables and figures only from retained evidence")
    report.add_argument("--run", required=True)
    report.add_argument("--output")
    reserve = sub.add_parser("reserve", help="Calculate a specified constant-rate annuity-due reserve")
    reserve.add_argument("--payment", type=float, default=50000.0)
    reserve.add_argument("--periods", type=int, default=10)
    reserve.add_argument("--annual-rate", type=float, required=True)
    reserve.add_argument("--output")
    allocation = sub.add_parser("allocate", help="Research-only allocation, subject to a validated mandate")
    allocation.add_argument("--request", required=True)
    allocation.add_argument("--output", required=True)
    allocation.add_argument("--execute", action="store_true")
    decision = sub.add_parser(
        "decide", help="Explicit bounded log-forecast scenarios to a constrained research allocation"
    )
    decision.add_argument("--request", required=True)
    decision.add_argument("--output", required=True)
    decision.add_argument("--execute", action="store_true")
    fred = sub.add_parser("acquire-fred", help="Explicitly acquire and preserve macro vintages")
    fred.add_argument("--request", required=True)
    fred.add_argument("--output", required=True)
    fred.add_argument("--execute", action="store_true")
    event = sub.add_parser("record-decision", help="Append a versioned hypothesis or decision to the journal")
    event.add_argument("--event", required=True, help="JSON event file")
    event.add_argument("--journal", default="research/events.jsonl")
    return app


def _allocation_inputs(request_path: str, command: str):
    """Read only explicitly named local artifacts; reject misspelled controls."""
    from .mandate import Instrument, Mandate

    request = read_json(request_path)
    if not isinstance(request, dict) or request.get("schema_version") != 1:
        raise ValueError("Allocation request must be a JSON object with schema_version=1")
    common = {
        "schema_version",
        "mandate_path",
        "as_of",
        "nav",
        "return_contract",
        "instruments",
        "instrument_master_path",
        "current_weights",
        "cash_return",
        "reserve_cash",
        "decision_context",
        "policy",
    }
    additional = (
        {"expected_returns", "covariance", "scenarios", "scenario_probabilities"}
        if command == "allocate"
        else {"forecast_path", "scenario_config"}
    )
    unknown = set(request) - common - additional
    if unknown:
        raise ValueError(f"Unknown allocation request fields: {sorted(unknown)}")
    required = {
        "mandate_path",
        "as_of",
        "nav",
        "return_contract",
        "current_weights",
        "cash_return",
        "reserve_cash",
    }
    required |= (
        {"expected_returns", "covariance"} if command == "allocate" else {"forecast_path", "scenario_config"}
    )
    if required - set(request):
        raise ValueError(f"Allocation request is missing {sorted(required - set(request))}")
    if ("instruments" in request) == ("instrument_master_path" in request):
        raise ValueError("Supply exactly one of instruments or instrument_master_path")
    base = Path(request_path).resolve().parent
    mandate_path = (base / request["mandate_path"]).resolve()
    mandate = Mandate.from_json(mandate_path)
    provenance = {
        "request_path": str(Path(request_path).resolve()),
        "request_sha256": file_digest(request_path),
        "mandate_path": str(mandate_path),
        "mandate_file_sha256": file_digest(mandate_path),
    }
    if "instrument_master_path" in request:
        master_path = (base / request["instrument_master_path"]).resolve()
        rows = read_json(master_path)
        provenance.update(
            instrument_master_path=str(master_path), instrument_master_file_sha256=file_digest(master_path)
        )
    else:
        rows = request["instruments"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Instrument master must be a nonempty JSON list of dated instrument records")
    master = [Instrument.from_mapping(row) for row in rows]
    return request, mandate, master, provenance, base


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in {"plan", "run", "suite", "freeze"}:
            from .config import load_config, project_root

            config = load_config(args.config)
            if args.command == "plan":
                from .experiments import experiment_plan

                _emit(experiment_plan(config, args.phase))
            elif args.command == "freeze":
                from .governance import freeze_protocol

                _emit(freeze_protocol(config, project_root(), args.output, args.rationale))
            else:
                if not args.execute:
                    raise ValueError(
                        "Execution was not enabled; inspect `frem plan`, then explicitly pass --execute"
                    )
                from .experiments import execute_matrix, execute_suite

                if args.command == "suite":
                    _emit({"runs": execute_suite(config, args.suite)})
                else:
                    _emit(
                        {"run_directory": execute_matrix(config, phase=args.phase, freeze_path=args.freeze)}
                    )
        elif args.command == "inspect-data":
            import pandas as pd

            from .data import load_panel, validate_panel

            mapping = read_json(args.column_map) if args.column_map else None
            frame = load_panel(args.path, mapping)
            boundary = pd.Timestamp(args.before)
            if boundary.tzinfo is None:
                raise ValueError("Inspection boundary requires an explicit UTC offset")
            mask = pd.to_datetime(frame["timestamp"], utc=True) < boundary
            for name in (
                "available_at",
                "fx_available_at",
                "corporate_actions_available_at",
                "feature_available_at",
            ):
                if name in frame:
                    known = pd.to_datetime(frame[name], utc=True, errors="coerce")
                    mask &= known.isna() | (known < boundary)
            _emit(validate_panel(frame.loc[mask].copy()))
        elif args.command == "predict":
            from .config import load_config
            from .data import (
                add_causal_peer_features,
                align_asset_universe,
                build_inference_windows,
                load_panel,
            )
            from .data_sources import load_macro
            from .training import Trainer

            destination = Path(args.output)
            if destination.exists():
                raise FileExistsError("Forecast artifact already exists; choose a new output path")
            cfg = load_config(args.config)
            trainer = Trainer.load(args.checkpoint, device=args.device)
            dc = cfg["data"]
            panel = load_panel(args.input, dc.get("column_map"))
            if dc.get("peer_features"):
                panel = add_causal_peer_features(
                    panel, max_age=dc.get("peer_max_age", "3D"), min_peers=dc.get("peer_min_count", 2)
                )
            macro = load_macro(dc["macro_path"]) if dc.get("macro_path") else None
            windows = build_inference_windows(
                panel,
                context=trainer.model_config.context_length,
                horizon=trainer.model_config.horizon,
                feature_columns=dc.get("feature_columns"),
                macro=macro,
                base_currency=dc.get("base_currency", "USD"),
                include_missingness=dc.get("include_missingness", True),
                as_of=args.as_of,
                max_staleness=args.max_staleness,
            )
            windows = align_asset_universe(windows, trainer.asset_names)
            prediction = trainer.predict(windows, online_update=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_csv(destination, index=False)
            _emit(
                {"forecast_file": destination, "rows": len(prediction), "mode": "frozen_unlabeled_inference"}
            )
        elif args.command == "compare":
            from .reporting import compare_models

            _emit(
                compare_models(
                    args.run,
                    args.candidate,
                    args.control,
                    block_length=args.block_dates,
                    n_resamples=args.resamples,
                )
            )
        elif args.command == "compare-family":
            from .reporting import compare_family

            _emit(compare_family(args.run, block_length=args.block_dates, n_resamples=args.resamples))
        elif args.command == "report":
            from .reporting import render_report

            _emit({"report": render_report(args.run, args.output)})
        elif args.command == "reserve":
            from .mandate import annuity_due_reserve

            result = {
                "payment": args.payment,
                "periods": args.periods,
                "annual_rate": args.annual_rate,
                "reserve": annuity_due_reserve(args.payment, args.periods, args.annual_rate),
                "assumption": "Specified constant net return; first payment occurs immediately; this is not a guaranteed market yield.",
            }
            if args.output:
                write_json(args.output, result)
            _emit(result)
        elif args.command == "allocate":
            if not args.execute:
                raise ValueError("Portfolio optimization requires explicit --execute; no orders are placed")
            if Path(args.output).exists():
                raise FileExistsError("Allocation artifact already exists; choose a new output path")
            import numpy as np

            from .decision import SIMPLE_RETURN_SPACE, ReturnContract
            from .portfolio import PortfolioConfig, optimize_portfolio
            from .risk import RegimeInputs

            request, mandate, master, provenance, _ = _allocation_inputs(args.request, args.command)
            contract = ReturnContract.from_dict(request["return_contract"])
            contract.validate(
                as_of=request["as_of"],
                asset_ids=[value.asset_id for value in master],
                base_currency=mandate.base_currency,
                expected_return_space=SIMPLE_RETURN_SPACE,
            )
            result = optimize_portfolio(
                None
                if request["expected_returns"] is None
                else np.asarray(request["expected_returns"], float),
                np.asarray(request["covariance"], float),
                master,
                mandate,
                as_of=request["as_of"],
                nav=request["nav"],
                current_weights=None
                if request.get("current_weights") is None
                else np.asarray(request["current_weights"], float),
                scenarios=None
                if request.get("scenarios") is None
                else np.asarray(request["scenarios"], float),
                scenario_probabilities=request.get("scenario_probabilities"),
                cash_return=request["cash_return"],
                reserve_cash=request["reserve_cash"],
                decision_context=RegimeInputs(**request["decision_context"])
                if request.get("decision_context")
                else None,
                config=PortfolioConfig.from_dict(request.get("policy", {})),
            )
            payload = asdict(result) if is_dataclass(result) else result
            payload["return_contract"] = contract.to_dict()
            payload["provenance"] = provenance
            write_json(args.output, payload)
            _emit(payload)
        elif args.command == "decide":
            if not args.execute:
                raise ValueError(
                    "Scenario generation and portfolio optimization require explicit --execute; no orders are placed"
                )
            if Path(args.output).exists():
                raise FileExistsError("Decision artifact already exists; choose a new output path")
            import pandas as pd

            from .decision import decide_from_forecasts
            from .portfolio import PortfolioConfig
            from .risk import RegimeInputs

            request, mandate, master, provenance, base = _allocation_inputs(args.request, args.command)
            forecast_path = (base / request["forecast_path"]).resolve()
            forecasts = pd.read_csv(forecast_path, dtype={"asset": str})
            provenance.update(
                forecast_path=str(forecast_path), forecast_file_sha256=file_digest(forecast_path)
            )
            payload = decide_from_forecasts(
                forecasts,
                master,
                mandate,
                contract=request["return_contract"],
                scenario_config=request["scenario_config"],
                as_of=request["as_of"],
                nav=request["nav"],
                current_weights=request["current_weights"],
                cash_return=request["cash_return"],
                reserve_cash=request["reserve_cash"],
                decision_context=RegimeInputs(**request["decision_context"])
                if request.get("decision_context")
                else None,
                config=PortfolioConfig.from_dict(request.get("policy", {})),
            )
            payload["provenance"] = provenance
            write_json(args.output, payload)
            # Full scenario evidence is retained in the file, not dumped to the terminal.
            _emit(
                {
                    "decision_file": str(Path(args.output).resolve()),
                    "status": payload["status"],
                    "allocation": payload.get("allocation"),
                    "blocking_reasons": payload.get("blocking_reasons", []),
                    "scenario_sha256": payload.get("scenario_sha256"),
                    "research_only": True,
                }
            )
        elif args.command == "acquire-fred":
            if not args.execute:
                raise ValueError(
                    "Network acquisition requires --execute; inspect request and provider terms first"
                )
            from .data_sources import FredVintageConfig, fetch_fred_vintages

            _emit(fetch_fred_vintages(FredVintageConfig(**read_json(args.request)), args.output))
        elif args.command == "record-decision":
            from .governance import append_event

            _emit(append_event(args.journal, read_json(args.event)))
        else:
            raise ValueError("Unknown command")
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        ImportError,
    ) as exc:
        print(f"FREM: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
