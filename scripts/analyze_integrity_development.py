#!/usr/bin/env python3
"""Audit saved DEVELOPMENT evidence without fitting models or loading pickle.

Usage: python scripts/analyze_integrity_development.py --session-dir SESSION --output NEW_DIR
Inputs are the session amendment, declared matrix artifacts, and the complete
demo_v2 pair when available (otherwise demo_primary/demo_replay). Missing or
failed experiments remain visible and never become numerical zeroes. Exit 1
means an integrity check failed; incomplete experiment coverage is reported
separately. This script makes no market, discovery, or replication claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.stats import t as student_t


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite number")
    return result


def aware(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset() is not None, "naive timestamp")
    return result


def close(actual, expected, name):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name + " shape/finiteness")
    require(np.allclose(a, b, rtol=1e-9, atol=1e-11), name + " numerical mismatch")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


class Audit:
    def __init__(self, session):
        self.session = session.resolve()
        self.inputs, self.failures, self.checks = {}, [], []

    def path(self, relative):
        path = (self.session / relative).resolve()
        require(path.is_relative_to(self.session), "input path escapes session")
        return path

    def read(self, path):
        path = path.resolve()
        require(path.is_relative_to(self.session), "input path escapes session")
        raw = path.read_bytes()
        self.inputs[str(path.relative_to(self.session))] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, path):
        return json.loads(self.read(path), parse_constant=reject_constant)

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path).decode(), newline="")))

    def npz(self, path):
        raw = self.read(path)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        for name, value in result.items():
            require(value.dtype.kind in "biufc", "non-numeric array: " + name)
            require(np.isfinite(value).all(), "nonfinite array: " + name)
        return result

    def failure(self, scope, error, kind="integrity"):
        self.failures.append(dict(scope=scope, kind=kind, error=str(error)))

    def check(self, scope, function):
        try:
            result = function()
            self.checks.append(dict(scope=scope, passed=True))
            return result
        except Exception as exc:
            self.failure(scope, f"{type(exc).__name__}: {exc}")
            self.checks.append(dict(scope=scope, passed=False))
            return None


def demo_hashes(audit, directory):
    manifest = audit.json(directory / "hashes.json")
    hashes = manifest["deterministic_artifacts_sha256"]
    required = {"config.json", "protocol.json", "data_manifest.json", "synthetic_returns.csv",
                "forecasts.csv", "ledger.csv", "numerical_results.json", "asof_records.json",
                "selection_attempts.json", "model_parameters.json"}
    require(required <= set(hashes), "missing required deterministic artifact declarations")
    require("run_metadata.json" not in hashes, "variable runtime metadata in deterministic manifest")
    for name, expected in sorted(hashes.items()):
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "artifact path escapes demo directory")
        require(hashlib.sha256(audit.read(path)).hexdigest() == expected, "SHA256 mismatch: " + name)
    require(manifest["data_sha256"] == hashes["synthetic_returns.csv"], "data hash declaration mismatch")
    require(manifest["config_sha256"] == hashes["config.json"], "config hash declaration mismatch")
    data_manifest = audit.json(directory / "data_manifest.json")
    require(data_manifest["data_kind"] == "synthetic", "demo must be explicitly synthetic")
    require(data_manifest["sha256"] == manifest["data_sha256"], "data admission hash mismatch")
    return manifest


def audit_demo(audit, directory):
    config = audit.json(directory / "config.json")
    reported = audit.json(directory / "numerical_results.json")
    forecasts, ledger = audit.csv(directory / "forecasts.csv"), audit.csv(directory / "ledger.csv")
    raw = audit.csv(directory / "synthetic_returns.csv")
    require(forecasts and ledger and raw, "empty demonstration evidence")
    assets = config["assets"]
    raw_values = {(r["asset"], r["available_time"]): finite(r["value"]) for r in raw}
    require(len(raw_values) == len(raw), "duplicate raw asset/vintage")
    require(all(r["data_kind"] == "synthetic" and int(r["seed"]) == config["seed"] for r in raw),
            "raw data kind/seed differs from protocol")
    losses, predictions, actuals, sample_origins = defaultdict(list), {}, {}, {}
    for row in forecasts:
        key = (row["model"], int(row["origin"]), row["asset"])
        require(key not in predictions, "duplicate forecast")
        predictions[key] = finite(row["forecast"])
        actual = finite(row["actual"])
        close(actual, raw_values[(row["asset"], row["label_available_time"])], "forecast label/source")
        require(aware(row["decision_time"]) < aware(row["label_available_time"]), "label already available at decision")
        actual_key = (int(row["origin"]), row["asset"])
        if actual_key in actuals:
            close(actual, actuals[actual_key], "different targets across forecast arms")
        actuals[actual_key] = actual
        sample_origins[int(row["sample"])] = int(row["origin"])
        losses[row["model"]].append((predictions[key] - actual) ** 2)
    mse = {arm: float(np.mean(values)) for arm, values in sorted(losses.items())}
    require(set(mse) == set(reported["forecast_mse"]), "forecast arm set mismatch")
    for arm, value in mse.items():
        close(value, reported["forecast_mse"][arm], "forecast MSE: " + arm)
    require(sorted(sample_origins) == reported["evaluated_samples"], "scored sample set mismatch")
    origins = sorted(sample_origins.values())
    for arm in losses:
        require({(origin, asset) for model, origin, asset in predictions if model == arm}
                == {(origin, asset) for origin in origins for asset in assets}, "unbalanced forecast panel")

    asof = audit.json(directory / "asof_records.json")
    maturity = {}
    for row in asof:
        decision = aware(row["decision_time"])
        maturity[row["sample"]] = max(aware(r["available_time"]) for r in row["label_records"])
        for lag in row["inputs"]:
            query = aware(lag["query_time"])
            require(query <= decision, "as-of query after decision")
            for record in lag["records"]:
                require(aware(record["event_time"]) < query and aware(record["available_time"]) <= query,
                        "unavailable as-of feature")
    attempts = audit.json(directory / "selection_attempts.json")
    checks = [fold["train"] for fold in reported["folds"]]
    checks += [attempt[key] for attempt in attempts for key in ("train", "validation")]
    for check in checks:
        latest = max(maturity[i] for i in range(check["start"], check["stop_exclusive"]))
        require(latest <= aware(check["cutoff"]), "immature fitting/selection label")
        require(latest == aware(check["latest_label_available"]), "label maturity provenance mismatch")

    groups = defaultdict(list)
    for row in ledger:
        groups[row["strategy"]].append(row)
    require(set(groups) == set(reported["portfolios"]), "portfolio strategy set mismatch")
    summaries, recalculated = {}, []
    rate = finite(config["commission_rate"]) + finite(config["slippage_rate"])
    for strategy, rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r["period"]))
        require([int(r["period"]) for r in rows] == list(range(len(origins))), "ledger period gaps/duplicates")
        require([int(r["origin"]) for r in rows] == origins, "ledger/forecast chronology mismatch")
        cash, positions, peak = 1.0, np.zeros(len(assets)), 1.0
        costs, turnovers, drawdowns, residuals, errors = [], [], [], [], []
        for period, row in enumerate(rows):
            origin = int(row["origin"])
            weights = np.array([finite(row["weight_" + a]) for a in assets])
            returns = np.array([finite(row["realized_return_" + a]) for a in assets])
            require(np.all(weights >= 0) and np.all(weights <= config["max_asset_weight"] + 1e-12)
                    and weights.sum() <= 1 + 1e-12, "portfolio exposure violation")
            close(returns, [actuals[(origin, a)] for a in assets], "ledger return/forecast label")
            expected_weights = np.zeros(len(assets))
            if strategy in losses:
                order = np.argsort([-predictions[(strategy, origin, a)] for a in assets], kind="stable")
                expected_weights[order[:2]] = config["max_asset_weight"]
            elif strategy == "equal_weight":
                expected_weights[:] = config["invested_fraction"] / len(assets)
            else:
                require(strategy == "cash_only", "unknown allocation rule")
            close(weights, expected_weights, "predeclared allocation rule")
            before = cash + positions.sum()
            # Independent scalar bisection of post-cost wealth plus its trade fee.
            low, high = 0.0, before
            for _ in range(64):
                middle = (low + high) / 2
                if middle + rate * np.abs(weights * middle - positions).sum() > before:
                    high = middle
                else:
                    low = middle
            postwealth = (low + high) / 2
            postpositions = weights * postwealth
            traded = float(np.abs(postpositions - positions).sum())
            cost = rate * traded
            gain = float(postpositions @ returns)
            cash_after = postwealth - postpositions.sum()
            positions_after = postpositions * (1 + returns)
            after = float(cash_after + positions_after.sum())
            peak = max(peak, after)
            residual = after - (before + gain - cost)
            values = dict(wealth_before=before, posttrade_wealth=postwealth, wealth_after=after,
                          cash_before=cash, cash_after=cash_after, gross_traded_notional=traded,
                          commission_cost=config["commission_rate"] * traded,
                          slippage_cost=config["slippage_rate"] * traded, total_cost=cost,
                          market_gain=gain, gross_return=gain / before,
                          net_return=after / before - 1, drawdown=after / peak - 1,
                          turnover=traded / before, identity_residual=residual)
            for name, value in values.items():
                errors.append(close(finite(row[name]), value, strategy + ":" + name))
            for prefix, value in (("position_before_", positions), ("position_posttrade_", postpositions),
                                  ("position_after_", positions_after)):
                errors.append(close([finite(row[prefix + a]) for a in assets], value, prefix))
            require(aware(row["decision_time"]) < aware(row["execution_time"])
                    < aware(row["label_available_time"]), "execution chronology")
            recalculated.append(dict(strategy=strategy, period=period, origin=origin, **values))
            costs.append(cost); turnovers.append(traded / before)
            drawdowns.append(values["drawdown"]); residuals.append(abs(residual))
            cash, positions = cash_after, positions_after
        summary = dict(final_wealth=after, cumulative_return=after - 1,
                       max_drawdown=max(0.0, -min(drawdowns)), total_cost=sum(costs),
                       gross_turnover=sum(turnovers))
        for name, value in summary.items():
            close(value, reported["portfolios"][strategy][name], "portfolio summary: " + name)
        summaries[strategy] = dict(**summary, max_identity_residual=max(residuals),
                                   max_csv_numerical_discrepancy=max(errors))
    return dict(forecast_mse=mse, portfolios=summaries, forecast_rows=len(forecasts),
                ledger_rows=len(ledger), evaluated_samples=sorted(sample_origins),
                maturity_checks=len(checks), independently_recomputed_ledger=recalculated)


def tail_mean(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) > 0, "empty/nonvector loss")
    # Equivalent finite-sample CVaR through the independent threshold formula.
    return float(min(eta + np.maximum(values - eta, 0).mean() / (1 - alpha) for eta in np.unique(values)))


def matrix_run(audit, spec, kind):
    directory = audit.path(spec["artifacts"])
    expected = len(spec["models"]) * len(spec["worlds"]) * len(spec["seeds"]) * 2
    report = dict(kind=kind, artifacts=spec["artifacts"], expected_cells=expected,
                  result_rows=0, verified_cells=0, protocol=None, rows=[], verified=[])
    if not directory.exists():
        audit.failure(spec["artifacts"], "declared run was not attempted", "not_attempted")
        return report
    protocol = audit.json(directory / "protocol_snapshot.json")
    report["protocol"] = protocol
    for key in ("models", "worlds", "seeds"):
        require(protocol[key] == spec[key], "run/amendment mismatch: " + key)
    require(protocol["n_test"] == spec["n_test"], "run/amendment n_test mismatch")
    source_hash = hashlib.sha256(json.dumps(protocol["source_hashes"], sort_keys=True).encode()).hexdigest()
    require(source_hash == protocol["source_fingerprint"], "source fingerprint does not match hash set")
    path = directory / "results.json"
    if path.exists():
        rows = audit.json(path)
    elif (directory / "cells.jsonl").exists():
        rows = []
        for number, line in enumerate(audit.read(directory / "cells.jsonl").decode().splitlines(), 1):
            try:
                rows.append(json.loads(line, parse_constant=reject_constant))
            except ValueError as exc:
                audit.failure(spec["artifacts"], f"incomplete cell journal line {number}: {exc}", "execution")
        audit.failure(spec["artifacts"], "run ended without results.json; retained journal rows", "execution")
    else:
        rows = []
        audit.failure(spec["artifacts"], "no saved cell results", "execution")
    report["rows"], report["result_rows"] = rows, len(rows)
    seen, datasets = set(), {}
    for row in rows:
        key = row["cell_id"]
        if key in seen:
            audit.failure(key, "duplicate result cell")
            continue
        seen.add(key)
        if row.get("status") != "completed":
            audit.failure(key, row.get("error", row.get("status", "unknown status")), "execution")
            error_path = directory / "raw" / (key + ".error.txt")
            if error_path.exists():
                audit.read(error_path)
            continue
        def verify_cell():
            require(row["model"] in spec["models"] and row["world"] in spec["worlds"]
                    and row["seed"] in spec["seeds"], "undeclared cell")
            require(row["source_fingerprint"] == source_hash, "cell source fingerprint mismatch")
            path = (directory / row["artifact"]).resolve()
            require(path.is_relative_to(directory), "raw artifact escapes run")
            raw = audit.npz(path)
            require(audit.inputs[str(path.relative_to(audit.session))] == row["artifact_sha256"], "raw NPZ SHA256 mismatch")
            dataset_key = (row["world"], row["seed"])
            if dataset_key not in datasets:
                data = audit.npz(directory / "datasets" / f"{row['world']}_{row['seed']}.npz")
                datasets[dataset_key] = hashlib.sha256(np.ascontiguousarray(data["returns"]).tobytes()).hexdigest()
            require(datasets[dataset_key] == row["dataset_sha256"], "dataset values SHA256 mismatch")
            require(raw["model_losses"].shape == raw["baseline_losses"].shape, "unpaired raw losses")
            for array_name, value_name in (("model_losses", "primary_value"), ("baseline_losses", "baseline_value")):
                values = raw[array_name]
                require(values.ndim == 1 and len(values) == row["eval_origins"], "loss observation count")
                score = tail_mean(values) if row.get("reduction") == "cvar95" else float(values.mean())
                close(score, finite(row[value_name]), "reported loss reduction")
            close(row["paired_difference"], row["primary_value"] - row["baseline_value"], "reported paired difference")
            return row
        verified = audit.check(key, verify_cell)
        if verified is not None:
            report["verified"].append(verified)
    report["verified_cells"] = len(report["verified"])
    for model in spec["models"]:
        for world in spec["worlds"]:
            for seed in spec["seeds"]:
                cells = [r for r in rows if (r["model"], r["world"], r["seed"]) == (model, world, seed)]
                for variant_class in ("full", "ablation"):
                    matches = [r for r in cells if (r["variant"] == "full") == (variant_class == "full")]
                    if len(matches) != 1:
                        audit.failure(f"{model}/{world}/{seed}/{variant_class}",
                                      f"expected one cell; observed {len(matches)}", "coverage")
    return report


def interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    result = dict(n_seeds=n, mean=None, minimum=None, maximum=None, ci95_low=None, ci95_high=None,
                  degrees_of_freedom=n - 1 if n else None, values=values.tolist())
    if n:
        result.update(mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()))
    if n >= 2:
        half = float(student_t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
        result.update(ci95_low=result["mean"] - half, ci95_high=result["mean"] + half)
    return result


def paired_summary(audit, runs, matrix):
    verified = [row for run in runs for row in run["verified"]]
    details, summaries = [], []
    for model in matrix["models"]:
        for world in matrix["worlds"]:
            block = [row for row in verified if row["model"] == model and row["world"] == world]
            semantics = {(r["primary_metric"], r["baseline_name"], r["direction"]) for r in block}
            require(len(semantics) <= 1, "mixed task semantics within model/world")
            metric, baseline, direction = next(iter(semantics), (None, None, None))
            contrasts = defaultdict(list)
            for seed in matrix["seeds"]:
                full = [r for r in block if r["seed"] == seed and r["variant"] == "full"]
                ablated = [r for r in block if r["seed"] == seed and r["variant"] != "full"]
                f, a = (full[0] if len(full) == 1 else None), (ablated[0] if len(ablated) == 1 else None)
                item = dict(model=model, world=world, seed=seed, metric=metric, baseline_name=baseline,
                            full=f["primary_value"] if f else None, baseline=f["baseline_value"] if f else None,
                            ablation=a["primary_value"] if a else None, ablation_name=a["variant"] if a else None,
                            full_minus_baseline=None, full_minus_ablation=None)
                if f:
                    item["full_minus_baseline"] = f["primary_value"] - f["baseline_value"]
                    contrasts["full_minus_baseline"].append((seed, item["full_minus_baseline"]))
                if f and a:
                    require(f["dataset_sha256"] == a["dataset_sha256"], "ablation dataset pairing mismatch")
                    require(f["source_fingerprint"] == a["source_fingerprint"], "ablation source pairing mismatch")
                    item["full_minus_ablation"] = f["primary_value"] - a["primary_value"]
                    contrasts["full_minus_ablation"].append((seed, item["full_minus_ablation"]))
                details.append(item)
            for contrast in ("full_minus_baseline", "full_minus_ablation"):
                pairs = contrasts[contrast]
                summaries.append(dict(model=model, world=world, metric=metric, baseline=baseline,
                                      direction=direction, contrast=contrast, seeds=[seed for seed, _ in pairs],
                                      missing_seeds=[seed for seed in matrix["seeds"] if seed not in {s for s, _ in pairs}],
                                      **interval([value for _, value in pairs])))
    return details, summaries


SECONDARY_CONTROLS = {
    # Name, saved prediction/loss array, saved loss cross-check, reported score cross-check.
    "M01": (("training_empirical_quantiles", "historical_quantiles", "historical_losses", "empirical_quantile_pinball"),),
    "M02": (("lag_residual_rank", "lag_residual", None, "lag_residual_spearman"),),
    "M03": (("expanding_historical_mean", "historical_mean", None, None),),
    "M04": (("training_historical_mean", "historical_mean", "historical_mean_losses", "historical_mean_mse"),
            ("same_backbone_no_memory", "same_backbone_prediction", None, None)),
    "M05": (("training_historical_mean", "historical_mean", "historical_mean_losses", "historical_mean_mse"),),
    "M06": (("training_historical_mean", "historical_mean", "historical_mean_losses", "historical_mean_mse"),),
    "M07": (("training_global_covariance", "global_covariance", "global_covariance_losses", "global_covariance_error"),),
    "M08": (("training_historical_mean", "historical_mean", "historical_mean_losses", "historical_mean_mse"),),
    "M09": (("training_historical_mean", "historical_mean", "historical_mean_losses", "historical_mean_mse"),),
    "M10": (("unconditional_training_bootstrap", "bootstrap_prediction", "bootstrap_losses", "unconditional_bootstrap_energy"),),
    "M11": (("equal_weight_policy", "equal_weight_losses", None, "equal_weight_cvar"),),
    "M12": (("training_historical_mean", "historical_mean", "historical_mean_losses", "historical_mean_mse"),),
}


def rescore_secondary(row, raw, specification):
    """Use only controls already saved by the fixed original matrix runner."""
    name, field, stored_losses, reported_score = specification
    model, truth, prediction = row["model"], raw["truth"], raw[field]
    scope = "independent_rescoring_of_saved_control_predictions"
    if model == "M01":
        require(truth.ndim == 1 and prediction.shape == (len(truth), len(raw["quantiles"])), "quantile control shape")
        error = truth[:, None] - prediction
        losses = np.maximum(raw["quantiles"] * error, (raw["quantiles"] - 1) * error).mean(axis=1)
    elif model == "M02":
        require(prediction.shape == truth.shape and truth.ndim == 2, "ranking control shape")
        losses = -np.array([spearmanr(target, forecast).statistic for target, forecast in zip(truth, prediction)])
    elif model == "M07":
        require(prediction.shape == truth.shape and truth.ndim == 3, "covariance control shape")
        losses = np.sqrt(np.square(prediction - truth).sum(axis=(1, 2)))
    elif model == "M10":
        require(prediction.ndim == 3 and truth.shape == (len(prediction), prediction.shape[2]), "bootstrap control shape")
        ensemble = prediction.shape[1]
        require(ensemble > 1, "energy score requires at least two saved draws")
        losses = np.array([cdist(draws, target[None]).mean()
                           - 0.5 * cdist(draws, draws).sum() / (ensemble * (ensemble - 1))
                           for target, draws in zip(truth, prediction)])
    elif model == "M11":
        # The runner saved per-path equal-weight ledger losses, not its complete
        # intermediate equal-weight ledger. State this narrower scope explicitly.
        losses = prediction
        scope = "saved_equal_weight_loss_vector_and_independent_CVaR_reduction_only_no_ledger_reconstruction"
    else:
        require(prediction.shape == truth.shape, "historical-mean/backbone control shape")
        squared = np.square(prediction - truth)
        losses = squared if squared.ndim == 1 else squared.mean(axis=tuple(range(1, squared.ndim)))
    require(losses.ndim == 1 and len(losses) == row["eval_origins"] and np.isfinite(losses).all(),
            "nonfinite or incomplete secondary control loss vector")
    if stored_losses is not None:
        close(losses, raw[stored_losses], "stored secondary control losses: " + name)
    score = tail_mean(losses) if row.get("reduction") == "cvar95" else float(losses.mean())
    if reported_score is not None:
        reported = finite(row["extra_metrics"][reported_score])
        close(score, -reported if model == "M02" else reported, "reported secondary control: " + name)
    return dict(control_value=score, full_minus_control=finite(row["primary_value"]) - score,
                loss_observations=len(losses), verification_scope=scope, status="verified")


def secondary_summary(audit, runs, matrix):
    """Report every saved secondary control, including losses and absent pairs."""
    full = [(run, row) for run in runs for row in run["verified"] if row["variant"] == "full"]
    indexed = {(row["model"], row["world"], row["seed"]): (run, row) for run, row in full}
    require(len(indexed) == len(full), "duplicate full cells for secondary controls")
    details, summaries = [], []
    for model in matrix["models"]:
        require(model in SECONDARY_CONTROLS, "unknown model secondary-control contract")
        for world in matrix["worlds"]:
            group = []
            for seed in matrix["seeds"]:
                entry = indexed.get((model, world, seed))
                row, raw, artifact = None, None, None
                if entry:
                    run, row = entry
                    artifact = str(Path(run["artifacts"]) / row["artifact"])
                    def read_control_artifact():
                        path = audit.path(artifact)
                        arrays = audit.npz(path)
                        require(audit.inputs[artifact] == row["artifact_sha256"], "secondary control artifact hash mismatch")
                        return arrays
                    raw = audit.check("secondary_raw:" + row["cell_id"], read_control_artifact)
                for specification in SECONDARY_CONTROLS[model]:
                    name, field, _, _ = specification
                    item = dict(model=model, world=world, seed=seed, control=name, source_array=field,
                                metric=row["primary_metric"] if row else None, direction="lower",
                                full_value=finite(row["primary_value"]) if row else None,
                                control_value=None, full_minus_control=None, loss_observations=None,
                                verification_scope=None, status="missing_verified_full" if row is None else "not_verified",
                                artifact=artifact)
                    if raw is not None:
                        scored = audit.check("secondary:" + row["cell_id"] + ":" + name,
                                             lambda: rescore_secondary(row, raw, specification))
                        if scored:
                            item.update(scored)
                    details.append(item); group.append(item)
            for name, _, _, _ in SECONDARY_CONTROLS[model]:
                valid = [r for r in group if r["control"] == name and r["status"] == "verified"]
                metrics = {r["metric"] for r in valid}
                scopes = {r["verification_scope"] for r in valid}
                require(len(metrics) <= 1 and len(scopes) <= 1, "mixed secondary-control semantics")
                seeds = [r["seed"] for r in valid]
                summaries.append(dict(model=model, world=world, control=name,
                                      metric=next(iter(metrics), None), direction="lower", contrast="full_minus_control",
                                      full_mean=float(np.mean([r["full_value"] for r in valid])) if valid else None,
                                      control_mean=float(np.mean([r["control_value"] for r in valid])) if valid else None,
                                      verification_scope=next(iter(scopes), None), seeds=seeds,
                                      missing_seeds=[s for s in matrix["seeds"] if s not in seeds],
                                      **interval([r["full_minus_control"] for r in valid])))
    return details, summaries


def compare_matrix_replay(audit, primary_runs, replay):
    primary = {r["cell_id"]: (run, r) for run in primary_runs for r in run["verified"]}
    comparisons = []
    for row in replay["verified"]:
        def compare():
            require(row["cell_id"] in primary, "replay has no verified primary cell")
            run, original = primary[row["cell_id"]]
            left = audit.npz(audit.path(run["artifacts"]) / original["artifact"])
            right = audit.npz(audit.path(replay["artifacts"]) / row["artifact"])
            require(set(left) == set(right), "replay raw array key mismatch")
            require(all(left[k].dtype == right[k].dtype and left[k].shape == right[k].shape
                        and left[k].tobytes(order="C") == right[k].tobytes(order="C") for k in left),
                    "replay raw arrays differ")
            a, b = run["protocol"]["source_hashes"], replay["protocol"]["source_hashes"]
            changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            # Demo journaling may change independently while model sources stay fixed.
            relevant = [k for k in changed if k.startswith("src/wharton_lab/models/")
                        or k in ("src/wharton_lab/research/causal_portfolio.py", "scripts/run_portfolio_research.py")]
            require(not relevant, "replay model/runner source changed: " + repr(relevant))
            return dict(cell_id=row["cell_id"], all_arrays_bitwise_equal=True,
                        raw_npz_bytes_equal=row["artifact_sha256"] == original["artifact_sha256"],
                        all_source_hashes_equal=not changed, changed_source_paths=changed)
        result = audit.check("replay:" + row["cell_id"], compare)
        if result:
            comparisons.append(result)
    return dict(expected_cells=replay["expected_cells"], verified_cells=len(comparisons),
                complete=len(comparisons) == replay["expected_cells"], checks=comparisons)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figures(output, demo, summaries):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return dict(status="unavailable", reason=str(exc), files=[])
    files = []
    if demo:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        groups = defaultdict(list)
        for row in demo["independently_recomputed_ledger"]:
            groups[row["strategy"]].append(row)
        for name, rows in groups.items():
            periods = [r["period"] + 1 for r in rows]
            axes[0].plot([0] + periods, [1.0] + [r["wealth_after"] for r in rows], label=name)
            axes[1].plot([0] + periods, [0.0] + [100 * r["drawdown"] for r in rows])
        axes[0].set(ylabel="Marked wealth (initial = 1)", title="Synthetic development: recorded policies after one charge per trade")
        axes[1].set(xlabel="Scored session", ylabel="Drawdown (%)")
        axes[0].legend(fontsize=8, ncol=2)
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.savefig(output / "demo_wealth_drawdown.png", dpi=150); plt.close(fig)
        files.append("demo_wealth_drawdown.png")
        fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axis.bar(list(demo["forecast_mse"]), list(demo["forecast_mse"].values()), color="#356a9a")
        axis.set(ylabel="MSE in simple-return units squared", title="One constructed development seed; descriptive forecast errors")
        axis.tick_params(axis="x", rotation=15)
        fig.savefig(output / "demo_forecast_mse.png", dpi=150); plt.close(fig)
        files.append("demo_forecast_mse.png")
    models = sorted({r["model"] for r in summaries if r["n_seeds"]})
    if models:
        fig, axes = plt.subplots(math.ceil(len(models) / 4), 4, figsize=(20, 3.5 * math.ceil(len(models) / 4)),
                                 squeeze=False, constrained_layout=True)
        for axis, model in zip(axes.flat, models):
            selected = [r for r in summaries if r["model"] == model and r["n_seeds"]]
            worlds = list(dict.fromkeys(r["world"] for r in selected))
            for row in selected:
                is_ablation = row["contrast"] == "full_minus_ablation"
                y = worlds.index(row["world"]) + (0.12 if is_ablation else -0.12)
                color = "#b15b32" if is_ablation else "#356a9a"
                axis.scatter(row["values"], [y] * row["n_seeds"], color=color, alpha=0.4, s=14)
                axis.plot(row["mean"], y, "s", color=color, markersize=4)
                if row["ci95_low"] is not None:
                    axis.plot([row["ci95_low"], row["ci95_high"]], [y, y], color=color)
            axis.axvline(0, color="black", linewidth=0.7)
            axis.set_yticks(range(len(worlds)), worlds, fontsize=8)
            axis.set_title(model + " | " + selected[0]["metric"], fontsize=9)
            axis.grid(axis="x", alpha=0.2)
        for axis in list(axes.flat)[len(models):]:
            axis.set_visible(False)
        fig.suptitle("Development seed effects: blue = full minus own baseline; orange = full minus ablation\n"
                     "Points: every seed; squares: means; lines: exploratory 95% t intervals. Negative favors full; axes have model-specific units.", fontsize=12)
        fig.savefig(output / "matrix_paired_effects.png", dpi=150); plt.close(fig)
        files.append("matrix_paired_effects.png")
    return dict(status="created", files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.session_dir.is_dir(), "session directory does not exist")
    require(not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir())),
            "output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.session_dir)
    amendment = audit.json(audit.path("development_matrix.json"))
    contract = audit.json(audit.path("contract.json"))
    audit.check("amendment_parent", lambda: require(
        audit.inputs["contract.json"] == amendment["parent_contract_sha256"], "amendment parent hash mismatch"))
    v2 = all(audit.path(f"demo_v2_{arm}/artifacts/hashes.json").is_file() for arm in ("primary", "replay"))
    prefix = "demo_v2" if v2 else "demo"
    demo_directories = [audit.path(f"{prefix}_{arm}/artifacts") for arm in ("primary", "replay")]
    manifests = [audit.check(str(path.relative_to(audit.session)) + ":hashes", lambda path=path: demo_hashes(audit, path))
                 for path in demo_directories]
    if all(manifests):
        audit.check("demo_deterministic_replay", lambda: require(
            manifests[0] == manifests[1], "demo deterministic payload/source manifest differs"))
        audit.check("demo_manifest_bytes_replay", lambda: require(
            audit.inputs[str((demo_directories[0] / "hashes.json").relative_to(audit.session))]
            == audit.inputs[str((demo_directories[1] / "hashes.json").relative_to(audit.session))],
            "demo hash manifest bytes differ"))
    demo = audit.check("demo_csv_recomputation", lambda: audit_demo(audit, demo_directories[0]))
    execution_history = []
    for path in sorted(set(audit.session.glob("*/receipt.json")) | set(audit.session.glob("*/RECORDER_FAILURE.json"))):
        content = audit.json(path)
        relative = str(path.relative_to(audit.session))
        execution_history.append(dict(path=relative, content=content))
        if path.name == "RECORDER_FAILURE.json":
            audit.failure(relative, content.get("cause", content.get("status", "recorder failed")), "recorder")
        elif content.get("timed_out") or content.get("exit_status", 0) != 0:
            audit.failure(relative, f"exit_status={content.get('exit_status')}; timed_out={content.get('timed_out')}", "execution")
    matrix = amendment["matrix"]
    specifications = [("preflight", dict(amendment["preflight"], seeds=[amendment["preflight"]["seed"]]))]
    specifications += [("matrix", dict(run, worlds=[run["world"]], seeds=matrix["seeds"],
                                       models=matrix["models"], n_test=matrix["n_test"])) for run in matrix["runs"]]
    specifications += [("replay", dict(amendment["replay"], seeds=[amendment["replay"]["seed"]]))]
    runs = []
    for kind, spec in specifications:
        result = audit.check(spec["artifacts"] + ":run", lambda spec=spec, kind=kind: matrix_run(audit, spec, kind))
        if result:
            runs.append(result)
    primary = [run for run in runs if run["kind"] == "matrix"]
    paired = audit.check("paired_seed_summary", lambda: paired_summary(audit, primary, matrix))
    details, summaries = paired if paired else ([], [])
    secondary = audit.check("secondary_control_summary", lambda: secondary_summary(audit, primary, matrix))
    secondary_details, secondary_summaries = secondary if secondary else ([], [])
    replay = next((run for run in runs if run["kind"] == "replay"), None)
    replay_report = compare_matrix_replay(audit, primary, replay) if replay else None
    figure_report = audit.check("figures", lambda: figures(args.output, demo, summaries))
    if demo:
        write_csv(args.output / "demo_forecast_metrics.csv", [dict(model=k, mse=v) for k, v in demo["forecast_mse"].items()], ["model", "mse"])
        portfolio_rows = [dict(strategy=k, **v) for k, v in demo["portfolios"].items()]
        write_csv(args.output / "demo_portfolio_metrics.csv", portfolio_rows, list(portfolio_rows[0]))
        rows = demo.pop("independently_recomputed_ledger")
        write_csv(args.output / "demo_ledger_recomputed.csv", rows, list(rows[0]))
    detail_fields = ["model", "world", "seed", "metric", "baseline_name", "full", "baseline", "ablation",
                     "ablation_name", "full_minus_baseline", "full_minus_ablation"]
    write_csv(args.output / "matrix_paired_seeds.csv", details, detail_fields)
    summary_fields = ["model", "world", "metric", "baseline", "direction", "contrast", "n_seeds", "mean",
                      "minimum", "maximum", "ci95_low", "ci95_high", "degrees_of_freedom", "seeds", "missing_seeds", "values"]
    write_csv(args.output / "matrix_paired_summary.csv", [dict(r, **{k: json.dumps(r[k]) for k in ("seeds", "missing_seeds", "values")}) for r in summaries], summary_fields)
    secondary_fields = ["model", "world", "seed", "control", "source_array", "metric", "direction", "full_value",
                        "control_value", "full_minus_control", "loss_observations", "verification_scope", "status", "artifact"]
    write_csv(args.output / "secondary_baseline_seeds.csv", secondary_details, secondary_fields)
    secondary_summary_fields = ["model", "world", "control", "metric", "direction", "contrast", "full_mean", "control_mean",
                                "n_seeds", "mean", "minimum", "maximum", "ci95_low", "ci95_high", "degrees_of_freedom",
                                "verification_scope", "seeds", "missing_seeds", "values"]
    write_csv(args.output / "secondary_baseline_summary.csv",
              [dict(r, **{k: json.dumps(r[k]) for k in ("seeds", "missing_seeds", "values")}) for r in secondary_summaries],
              secondary_summary_fields)
    secondary_report = dict(seed_values=secondary_details, summaries=secondary_summaries,
                            interpretation=["These controls were already retained by the original fixed runner; no model or new baseline is fitted.",
                                            "Full-minus-control is calculated within one model, world, seed, target and loss unit; negative favors full.",
                                            "All stored controls are included; intervals remain descriptive, without p-values or discovery claims.",
                                            "M11 uses saved equal-weight per-path losses and re-reduces CVaR; this analysis does not independently reconstruct that control ledger."])
    write_json(args.output / "secondary_baseline_analysis.json", secondary_report)
    verified = sum(run["verified_cells"] for run in primary)
    report = dict(evidence_class="SYNTHETIC_DEVELOPMENT_ONLY", demo_pair=prefix, demo=demo,
                  matrix_expected_cells=matrix["expected_cells"], matrix_verified_cells=verified,
                  matrix_complete=verified == matrix["expected_cells"], paired_seed_values=details,
                  paired_summaries=summaries, secondary_baselines=secondary_report,
                  matrix_replay=replay_report, figures=figure_report,
                  runs=[{k: v for k, v in run.items() if k not in ("verified", "rows", "protocol")} for run in runs],
                  cell_results=[dict(run=run["artifacts"], **row) for run in runs for row in run["rows"]],
                  checks=audit.checks, failures=audit.failures, execution_history=execution_history,
                  integrity_passed=not any(f["kind"] == "integrity" for f in audit.failures),
                  interpretation=["No aggregate loss, ranking, p-value, or discovery across model task units.",
                                  "Seed within a single world is the independent unit; dates/assets/replays are not independent seeds.",
                                  "Student-t intervals are exploratory; missing pairs are reported and never replaced.",
                                  "Same-host replay checks deterministic arrays, not independent replication.",
                                  "Matrix audit verifies hashes and saved loss reductions; demo accounting is independently reconstructed.",
                                  "Manifest checks validate supplied provenance declarations, not provider/licence/source truth.",
                                  "M11 empirical CVaR95 is a finite-sample development statistic, not population tail-risk evidence."])
    write_json(args.output / "analysis.json", report)
    write_json(args.output / "failures.json", audit.failures)
    write_json(args.output / "input_hashes.json", dict(analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                     inputs_sha256=dict(sorted(audit.inputs.items()))))
    write_json(args.output / "output_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps({k: report[k] for k in ("integrity_passed", "demo_pair", "matrix_complete", "matrix_verified_cells", "matrix_expected_cells")}, allow_nan=False))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
