"""Deterministic DEVELOPMENT plumbing demonstration, never market evidence.

One new artificial seed, anonymous assets, chronological nested Ridge selection,
point-in-time inputs and self-financing accounting. No network, investor mandate,
portfolio optimizer, old outcome files, or protected evaluation are accessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wharton_lab.data.point_in_time import (
    DataManifest, MembershipUpdate, Observation, admit_data_file, build_asof_snapshot,
)
from wharton_lab.models.m11.accounting import simulate_ledger
from wharton_lab.splits.walk_forward import walk_forward_folds

SEED = 2026101908
ASSETS = ("ASSET_A", "ASSET_B", "ASSET_C")
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
LOOKBACK, COST_RATE = 5, 0.001
FEATURES = tuple(f"lag_{i}_return" for i in range(1, 6)) + (
    "trailing_5_mean", "trailing_5_std", "prior_session_cross_asset_mean")
FORECAST_ARMS = ("ridge", "zero_no_change", "historical_mean", "trailing_5_momentum")
CONFIG = dict(seed=SEED, n_days=160, assets=ASSETS, lookback=LOOKBACK,
              outer_min_train=64, outer_test_size=20, outer_step=20, purge_gap=1,
              lockbox_size=0, inner_min_train=32, inner_test_size=16, inner_step=16,
              ridge_alphas=(0.1, 1.0, 10.0), commission_rate=0.0005,
              slippage_rate=0.0005, max_asset_weight=0.4, invested_fraction=0.8,
              max_seconds=60, cpu_threads=1)
PROTOCOL = {
    "evidence_class": "SYNTHETIC_DEVELOPMENT_ONLY_NOT_MARKET_ALPHA",
    "target": "one_next_session_open_to_close_simple_return",
    "clock_UTC": {"decision": "08:55", "execution": "09:00", "event": "16:00", "available": "16:05"},
    "calendar": "Artificial daily sessions; overnight asset returns and cash interest are zero.",
    "selection": "Minimum pooled validation MSE across chronological inner folds; alpha ascending breaks ties.",
    "allocation": "Each forecast arm holds top two ranks at 0.4 each, cash 0.2, regardless of forecast sign; ties use asset order.",
    "allocation_controls": "Separate cash-only and 0.8/3 per-asset equal-weight portfolios.",
    "accounting": "Weights target post-cost wealth; drifted holdings carry forward. Costs charged once on gross buys plus sells.",
    "endpoint": "Marked wealth after last scored return, no terminal liquidation; initial entry costs included.",
    "information": "Train-only scaling. Mature earlier development outcomes may enter later outer training folds.",
    "limitations": ["One constructed seed; no confidence intervals or efficacy inference.",
                    "No licensed market data, client figures, optimizer, lockbox, downloads, or trading.",
                    "Fixed exposure rule is a software control, not an investment recommendation."],
}


def synthetic_returns(n_days: int = 160) -> np.ndarray:
    """Generate only the new fixed development seed; no caller-selected seed."""
    if type(n_days) is not int or n_days < LOOKBACK + 1:
        raise ValueError("n_days must exceed the lookback")
    noise = np.random.default_rng(SEED).normal(0, 0.008, size=(n_days, 4))
    values = np.zeros((n_days, 3))
    for day in range(n_days):
        previous = values[day - 1] if day else np.zeros(3)
        values[day] = np.clip(0.15 * previous + 0.4 * noise[day, 0] + noise[day, 1:], -0.12, 0.12)
    return values


def observations_from_returns(values: np.ndarray) -> tuple[Observation, ...]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all() or np.any(values < -1):
        raise ValueError("expected finite three-asset simple returns >= -1")
    return tuple(Observation(asset, "return", BASE + timedelta(days=day, hours=16),
                             BASE + timedelta(days=day, hours=16, minutes=5), float(value))
                 for day, row in enumerate(values) for asset, value in zip(ASSETS, row))


@dataclass(frozen=True)
class DevelopmentDesign:
    X: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    decisions: tuple[datetime, ...]
    maturity: tuple[datetime, ...]
    asof_records: tuple[dict, ...]


def _observation_dict(record: Observation) -> dict:
    return dict(asset=record.asset, field=record.field, event_time=record.event_time.isoformat(),
                available_time=record.available_time.isoformat(), value=record.value, revision=record.revision)


def build_design(observations: tuple[Observation, ...]) -> DevelopmentDesign:
    """Build five strictly prior lag snapshots and retain every selected vintage."""
    lookup = {(r.event_time, r.asset): r for r in observations}
    events = sorted({r.event_time for r in observations})
    if len(lookup) != len(observations) or len(observations) != len(events) * 3:
        raise ValueError("demo requires one complete three-asset vintage per session")
    if len(events) <= LOOKBACK or any(t != BASE + timedelta(days=i, hours=16) for i, t in enumerate(events)):
        raise ValueError("demo requires consecutive artificial sessions from its fixed origin")
    if any(r.asset not in ASSETS or r.field != "return" for r in observations):
        raise ValueError("unexpected demo asset or field")
    membership = tuple(MembershipUpdate(a, BASE, BASE, True) for a in ASSETS)
    features, labels, decisions, maturity, provenance = [], [], [], [], []
    for origin in range(LOOKBACK, len(events)):
        decision = BASE + timedelta(days=origin, hours=8, minutes=55)
        lag_rows, lag_records = [], []
        for lag in range(LOOKBACK):
            cutoff = decision - timedelta(days=lag)
            snapshot = build_asof_snapshot(observations, decision_time=cutoff, assets=ASSETS,
                                          fields=("return",), membership_updates=membership,
                                          max_age=timedelta(days=1))
            if any(row[0].event_time != events[origin - lag - 1] for row in snapshot.records):
                raise ValueError("a required prior-session observation is unavailable; no fill allowed")
            lag_rows.append([row[0] for row in snapshot.values])
            lag_records.append(dict(lag=lag + 1, query_time=cutoff.isoformat(),
                                    records=[_observation_dict(row[0]) for row in snapshot.records]))
        past = np.asarray(lag_rows)
        features.append(np.column_stack((past.T, past.mean(axis=0), past.std(axis=0),
                                         np.full(3, past[0].mean()))))
        target = [lookup[(events[origin], a)] for a in ASSETS]
        labels.append([r.value for r in target])
        decisions.append(decision)
        maturity.append(max(r.available_time for r in target))
        provenance.append(dict(sample=origin - LOOKBACK, origin=origin,
                               decision_time=decision.isoformat(), inputs=lag_records,
                               label_records=[_observation_dict(r) for r in target]))
    return DevelopmentDesign(np.asarray(features), np.asarray(labels), np.arange(LOOKBACK, len(events)),
                             tuple(decisions), tuple(maturity), tuple(provenance))


def check_label_maturity(design: DevelopmentDesign, indices: slice, cutoff: datetime) -> dict:
    selected = list(range(len(design.y)))[indices]
    if not selected or any(design.maturity[i] > cutoff for i in selected):
        raise ValueError("training or selection label is not mature at the declared fitting cutoff")
    return dict(start=selected[0], stop_exclusive=selected[-1] + 1,
                latest_label_available=max(design.maturity[i] for i in selected).isoformat(),
                cutoff=cutoff.isoformat(), all_labels_mature=True)


def _fit(design, indices, alpha, model_id):
    X, y = design.X[indices].reshape(-1, len(FEATURES)), design.y[indices].ravel()
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(X), y)
    state = dict(model_id=model_id, alpha=alpha, scaler_mean=scaler.mean_.tolist(),
                 scaler_scale=scaler.scale_.tolist(), scaler_variance=scaler.var_.tolist(),
                 scaler_samples=int(scaler.n_samples_seen_), coefficients=model.coef_.tolist(),
                 intercept=float(model.intercept_), features=FEATURES,
                 training_X_sha256=hashlib.sha256(np.asarray(X, dtype="<f8").tobytes()).hexdigest(),
                 training_y_sha256=hashlib.sha256(np.asarray(y, dtype="<f8").tobytes()).hexdigest())
    return scaler, model, state


def evaluate_development(
    design: DevelopmentDesign, *, journal: Callable[[str, dict], None] | None = None,
) -> dict:
    """Nested chronological forecasting; all attempts remain visible, including losses."""
    def record(kind: str, value: dict) -> None:
        if journal is not None:
            journal(kind, value)

    started = time.perf_counter()
    timings = dict(tuning_seconds=0.0, final_training_seconds=0.0, inference_seconds=0.0)
    forecasts, attempts, parameters, boundaries = [], [], [], []
    folds = walk_forward_folds(len(design.y), min_train=64, test_size=20, step=20, purge_gap=1, lockbox_size=0)
    with threadpool_limits(limits=1):
        for outer in folds:
            tr, te = outer.train_indices, outer.test_indices
            cutoff = design.decisions[te.start]
            train_check = check_label_maturity(design, tr, cutoff)
            inner_folds = walk_forward_folds(tr.stop, min_train=32, test_size=16, step=16,
                                            purge_gap=1, lockbox_size=0)
            scores = []
            for alpha in CONFIG["ridge_alphas"]:
                losses = []
                for inner in inner_folds:
                    if time.perf_counter() - started >= CONFIG["max_seconds"]:
                        raise RuntimeError("development demonstration exceeded its wall-time budget")
                    fit_cutoff = design.decisions[inner.test_indices.start]
                    maturity = check_label_maturity(design, inner.train_indices, fit_cutoff)
                    validation_maturity = check_label_maturity(design, inner.test_indices, cutoff)
                    model_id = f"outer_{outer.fold_id}_inner_{inner.fold_id}_alpha_{alpha}"
                    attempt = dict(outer_fold=outer.fold_id, inner_fold=inner.fold_id,
                                   alpha=alpha, model_id=model_id, train=maturity,
                                   validation=validation_maturity,
                                   scoring_cutoff=cutoff.isoformat())
                    record("attempt", dict(stage="inner_selection", status="started", **attempt))
                    tick = time.perf_counter()
                    try:
                        scaler, model, state = _fit(design, inner.train_indices, alpha, model_id)
                        record("model_state", dict(stage="inner_selection", **state))
                        pred = model.predict(scaler.transform(design.X[inner.test_indices].reshape(-1, len(FEATURES))))
                        squared = (pred.reshape(-1, 3) - design.y[inner.test_indices]) ** 2
                    except Exception as exc:
                        record("attempt", dict(stage="inner_selection", status="failed",
                                               error_type=type(exc).__name__, error=str(exc), **attempt))
                        raise
                    timings["tuning_seconds"] += time.perf_counter() - tick
                    losses.extend(squared.mean(axis=1).tolist())
                    parameters.append(dict(stage="inner_selection", **state))
                    attempts.append(dict(**attempt, mse=float(squared.mean()), status="completed"))
                    record("attempt", dict(stage="inner_selection", **attempts[-1]))
                scores.append(dict(alpha=alpha, mse=float(np.mean(losses)), validation_dates=len(losses)))
            chosen = min(scores, key=lambda item: (item["mse"], item["alpha"]))["alpha"]
            final_attempt = dict(stage="outer_final_fit", outer_fold=outer.fold_id,
                                 alpha=chosen, model_id=f"outer_{outer.fold_id}_final", train=train_check)
            record("attempt", dict(status="started", **final_attempt))
            tick = time.perf_counter()
            try:
                scaler, model, state = _fit(design, tr, chosen, f"outer_{outer.fold_id}_final")
                record("model_state", dict(stage="outer_final", **state))
            except Exception as exc:
                record("attempt", dict(status="failed", error_type=type(exc).__name__,
                                       error=str(exc), **final_attempt))
                raise
            record("attempt", dict(status="completed", **final_attempt))
            timings["final_training_seconds"] += time.perf_counter() - tick
            parameters.append(dict(stage="outer_final", **state))
            tick = time.perf_counter()
            prediction = model.predict(scaler.transform(design.X[te].reshape(-1, len(FEATURES)))).reshape(-1, 3)
            predictions = dict(ridge=prediction, zero_no_change=np.zeros_like(prediction),
                               historical_mean=np.tile(design.y[tr].mean(axis=0), (len(prediction), 1)),
                               trailing_5_momentum=design.X[te, :, 5])
            timings["inference_seconds"] += time.perf_counter() - tick
            if not all(np.isfinite(p).all() for p in predictions.values()):
                raise ValueError("nonfinite forecast; no period may be silently dropped")
            boundaries.append(dict(fold=outer.fold_id, train=train_check,
                                   test_start=te.start, test_stop_exclusive=te.stop,
                                   is_lockbox=False, alpha_scores=scores, chosen_alpha=chosen))
            for offset, sample in enumerate(range(te.start, te.stop)):
                for asset_index, asset in enumerate(ASSETS):
                    for arm, pred in predictions.items():
                        forecasts.append(dict(fold=outer.fold_id, sample=sample, origin=int(design.origins[sample]),
                                              asset=asset, model=arm, decision_time=design.decisions[sample].isoformat(),
                                              label_available_time=design.maturity[sample].isoformat(),
                                              forecast=float(pred[offset, asset_index]), actual=float(design.y[sample, asset_index])))
    return dict(forecasts=forecasts, selection_attempts=attempts, model_parameters=parameters,
                folds=boundaries, timings=timings)


def rank_weights(predictions: np.ndarray) -> np.ndarray:
    """Predeclared exposure: top two at 0.4, including negative predictions."""
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != 3 or not np.isfinite(predictions).all():
        raise ValueError("expected a finite three-asset forecast matrix")
    weights = np.zeros_like(predictions)
    for row, order in enumerate(np.argsort(-predictions, axis=1, kind="stable")):
        weights[row, order[:2]] = 0.4
    return weights


def simulate_portfolio_path(returns, weights, *, strategy="fixture", origins=None,
                            decisions=None, maturity=None, fee=COST_RATE) -> list[dict]:
    """Replay each next-session return once, carrying actual drifted positions."""
    returns, weights = np.asarray(returns, dtype=float), np.asarray(weights, dtype=float)
    if returns.ndim != 2 or returns.shape != weights.shape or returns.shape[1] != 3 or not len(returns):
        raise ValueError("matching nonempty three-asset return and weight matrices required")
    cash, holdings, peak, rows = 1.0, np.zeros(3), 1.0, []
    for t in range(len(returns)):
        old_holdings, old_cash = holdings.copy(), cash
        result = simulate_ledger(weights[t:t + 1], returns[t:t + 1][None], cash, holdings, np.zeros(1), fee)
        before, after = float(result["wealth_before"][0, 0]), float(result["wealth_after"][0, 0])
        cost = float(result["trade_fee"][0, 0])
        posttrade = weights[t] * (before - cost)
        traded = float(np.abs(posttrade - old_holdings).sum())
        cash, holdings = float(result["cash_after"][0, 0]), result["final_holdings"][0].copy()
        if not np.isclose(cost, fee * traded, atol=1e-12, rtol=1e-12):
            raise ArithmeticError("trade costs do not reconcile with gross traded notional")
        if abs(after - cash - holdings.sum()) > 1e-10 or abs(result["identity_residual"][0, 0]) > 1e-10:
            raise ArithmeticError("self-financing identity failed")
        peak = max(peak, after)
        row = dict(strategy=strategy, period=t, origin=int(origins[t]) if origins is not None else t,
                   decision_time=decisions[t].isoformat() if decisions is not None else "fixture",
                   execution_time=decisions[t].replace(hour=9, minute=0).isoformat() if decisions is not None else "fixture",
                   label_available_time=maturity[t].isoformat() if maturity is not None else "fixture",
                   wealth_before=before, posttrade_wealth=before - cost, wealth_after=after,
                   cash_before=old_cash, cash_after=cash,
                   gross_traded_notional=traded, commission_cost=cost / 2, slippage_cost=cost / 2,
                   total_cost=cost, market_gain=float(result["market_gain"][0, 0]),
                   gross_return=float(result["market_gain"][0, 0]) / before if before else 0.0,
                   net_return=after / before - 1 if before else 0.0,
                   identity_residual=float(result["identity_residual"][0, 0]),
                   drawdown=after / peak - 1, turnover=traded / before if before else 0.0)
        for i, asset in enumerate(ASSETS):
            row.update({f"weight_{asset}": float(weights[t, i]), f"position_before_{asset}": float(old_holdings[i]),
                        f"position_posttrade_{asset}": float(posttrade[i]), f"position_after_{asset}": float(holdings[i]),
                        f"realized_return_{asset}": float(returns[t, i])})
        rows.append(row)
    return rows


def _portfolio_results(design, evaluation):
    indices = sorted({row["sample"] for row in evaluation["forecasts"]})
    if np.any(np.diff(indices) != 1):
        raise ValueError("demo scoring periods must be contiguous to carry holdings")
    indexed = {(r["model"], r["sample"], r["asset"]): r["forecast"] for r in evaluation["forecasts"]}
    policies = {arm: rank_weights([[indexed[(arm, i, a)] for a in ASSETS] for i in indices]) for arm in FORECAST_ARMS}
    policies.update(cash_only=np.zeros((len(indices), 3)), equal_weight=np.full((len(indices), 3), 0.8 / 3))
    rows, summaries = [], {}
    for arm, weights in policies.items():
        ledger = simulate_portfolio_path(design.y[indices], weights, strategy=arm,
                                        origins=design.origins[indices], decisions=[design.decisions[i] for i in indices],
                                        maturity=[design.maturity[i] for i in indices])
        rows.extend(ledger)
        summaries[arm] = dict(final_wealth=ledger[-1]["wealth_after"],
                              cumulative_return=ledger[-1]["wealth_after"] - 1,
                              max_drawdown=max(0.0, -min(row["drawdown"] for row in ledger)),
                              total_cost=sum(row["total_cost"] for row in ledger),
                              gross_turnover=sum(row["turnover"] for row in ledger))
    mse = {arm: float(np.mean([(r["forecast"] - r["actual"]) ** 2 for r in evaluation["forecasts"] if r["model"] == arm]))
           for arm in FORECAST_ARMS}
    return rows, dict(evidence_class=PROTOCOL["evidence_class"], evaluated_samples=indices,
                      unscored_tail_samples=list(range(indices[-1] + 1, len(design.y))),
                      forecast_mse=mse, portfolios=summaries, folds=evaluation["folds"])


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _csv_bytes(rows) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _run_development_demo_body(output, write, journal, context, started) -> dict:
    """Numerical workflow; the outer wrapper owns durable execution receipts."""
    context["stage"] = "preparation"
    write("protocol.json", _json_bytes(PROTOCOL))
    write("config.json", _json_bytes(CONFIG))
    tick = time.perf_counter()
    observations = observations_from_returns(synthetic_returns(CONFIG["n_days"]))
    raw_rows = [dict(data_kind="synthetic", seed=SEED, **_observation_dict(r)) for r in observations]
    raw = _csv_bytes(raw_rows)
    write("synthetic_returns.csv", raw)
    manifest = DataManifest(
        source=f"local://wharton_lab/research/development_demo.py#seed={SEED}",
        licence="Locally generated synthetic fixture; no third-party market observations",
        data_kind="synthetic", sha256=hashlib.sha256(raw).hexdigest(),
        availability_semantics="Synthetic session outcome event 16:00 UTC; publication 16:05 UTC",
        corporate_action_policy="No corporate actions in the artificial world",
        missingness_policy="Complete generated sessions required; missing values rejected, never filled",
        universe_policy="Three anonymous assets explicitly included from fixed simulation origin")
    write("data_manifest.json", _json_bytes(asdict(manifest)))
    admitted = admit_data_file(output / "synthetic_returns.csv", manifest)
    context["data_sha256"] = admitted.sha256
    parsed = list(csv.DictReader(io.StringIO(admitted.raw.decode(), newline="")))
    if any(r["data_kind"] != "synthetic" or r["seed"] != str(SEED) for r in parsed):
        raise ValueError("demo accepts only its explicitly labelled fixed synthetic fixture")
    observations = tuple(Observation(r["asset"], r["field"], datetime.fromisoformat(r["event_time"]),
                                     datetime.fromisoformat(r["available_time"]), float(r["value"]), r["revision"]) for r in parsed)
    design = build_design(observations)
    preparation_seconds = time.perf_counter() - tick
    context["stage"] = "evaluation"
    evaluation = evaluate_development(design, journal=journal)
    context["stage"] = "portfolio_accounting"
    tick = time.perf_counter()
    ledger, numerical = _portfolio_results(design, evaluation)
    ledger_seconds = time.perf_counter() - tick
    context["stage"] = "result_serialization"
    write("forecasts.csv", _csv_bytes(evaluation["forecasts"]))
    write("selection_attempts.json", _json_bytes(evaluation["selection_attempts"]))
    write("model_parameters.json", _json_bytes(evaluation["model_parameters"]))
    write("asof_records.json", _json_bytes(design.asof_records))
    write("ledger.csv", _csv_bytes(ledger))
    write("numerical_results.json", _json_bytes(numerical))
    hashes = dict(source_sha256=context["source_sha256"],
                  deterministic_artifacts_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in sorted(output.iterdir())
                                                  if p.name != "execution_context.json"},
                  data_sha256=admitted.sha256, config_sha256=hashlib.sha256(_json_bytes(CONFIG)).hexdigest())
    write("hashes.json", _json_bytes(hashes))
    elapsed = time.perf_counter() - started
    if elapsed >= CONFIG["max_seconds"]:
        raise RuntimeError("development demonstration exceeded its total wall-time budget")
    context["stage"] = "run_metadata"
    metadata = dict(evidence_class=PROTOCOL["evidence_class"], created_utc=datetime.now(timezone.utc).isoformat(),
                    device="cpu", numerical_threads=1, python=platform.python_version(), platform=platform.platform(),
                    numpy=importlib.metadata.version("numpy"), sklearn=importlib.metadata.version("scikit-learn"),
                    timings=dict(preparation_seconds=preparation_seconds, ledger_seconds=ledger_seconds,
                                 total_seconds=elapsed, **evaluation["timings"]))
    write("run_metadata.json", _json_bytes(metadata))
    return numerical


def run_development_demo(output_dir: str | Path) -> dict:
    """Retain incremental attempts and a terminal receipt, including on failure.

    Numerical values and the returned dictionary use the unchanged workflow.
    Journals contain no clocks, so successful journals are deterministic. The
    context and terminal receipts record invocation, identity, and wall time.
    """
    started, output = time.perf_counter(), Path(output_dir)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("development output directory must be new or empty")
    else:
        output.mkdir(parents=True, exist_ok=False)

    def write(name, raw):
        with (output / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    journal_names = {"attempt": "selection_attempts.journal.jsonl",
                     "model_state": "model_parameters.journal.jsonl"}
    counts = dict(started=0, completed=0, failed=0, model_states=0)

    def journal(kind: str, value: dict) -> None:
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        with (output / journal_names[kind]).open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if kind == "model_state":
            counts["model_states"] += 1
        elif value["status"] in counts:
            counts[value["status"]] += 1

    context = dict(evidence_class=PROTOCOL["evidence_class"], stage="source_binding",
                   created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
                   process_argv=list(sys.argv), function="run_development_demo",
                   output_dir=str(output.resolve()), source_sha256={}, data_sha256=None,
                   config_sha256=hashlib.sha256(_json_bytes(CONFIG)).hexdigest(),
                   wall_time_budget_seconds=CONFIG["max_seconds"],
                   budget_enforcement="cooperative wall-time checks, not a hard CPU-time limit")
    terminal = dict(status="failed", exit_code=1)
    try:
        root = Path(__file__).resolve().parents[3]
        sources = ("src/wharton_lab/research/development_demo.py", "src/wharton_lab/data/point_in_time.py",
                   "src/wharton_lab/splits/walk_forward.py", "src/wharton_lab/models/m11/accounting.py",
                   "scripts/run_development_demo.py")
        context["source_sha256"] = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources}
        context["environment"] = dict(device="cpu", numerical_threads=1, python=platform.python_version(),
                                      platform=platform.platform(), numpy=importlib.metadata.version("numpy"),
                                      sklearn=importlib.metadata.version("scikit-learn"))
        write("execution_context.json", _json_bytes(context))
        for name in journal_names.values():
            write(name, b"")
        numerical = _run_development_demo_body(output, write, journal, context, started)
    except BaseException as exc:
        terminal.update(error_type=type(exc).__name__, error=str(exc),
                        exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1)
        raise
    else:
        terminal = dict(status="success", exit_code=0)
        context["stage"] = "complete"
        return numerical
    finally:
        write("terminal_receipt.json", _json_bytes(dict(
            **context, **terminal, attempt_counts=counts,
            elapsed_wall_seconds=time.perf_counter() - started,
            finished_utc=datetime.now(timezone.utc).isoformat())))
