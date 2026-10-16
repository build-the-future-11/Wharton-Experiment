"""Opt-in public-price exploratory study, separate from canonical market admission.

This adapter never manufactures historical membership or availability metadata.
It uses monthly research coordinates with a blind month and stores an explicit
survivor-sample limitation. No call changes the canonical FREM data contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import resource
import subprocess
import time
import traceback
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(item):
    if hasattr(item, "tolist"):
        return item.tolist()
    raise TypeError(f"Unsupported receipt value: {type(item).__name__}")


def save(path: Path, value: object) -> None:
    """Create receipts once; an existing receipt is never silently replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=json_default)
        handle.write("\n")


def read_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if len(cfg["folds"]) * len(cfg["models"]) > cfg["budget"]["max_cells"]:
        raise ValueError("Requested cells exceed the total campaign cap")
    if cfg["source_end_exclusive"] > cfg["protected_start"]:
        raise ValueError("Source request reaches the protected boundary")
    if len({f["name"] for f in cfg["folds"]}) != len(cfg["folds"]):
        raise ValueError("Duplicate fold names")
    return cfg


def acquire(cfg: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    save(output / "request.json", cfg)
    records = []
    for symbol, market in cfg["symbols"].items():
        started = datetime.now(UTC).isoformat()
        params = {
            "period1": int(datetime.fromisoformat(cfg["source_start"]).replace(tzinfo=UTC).timestamp()),
            "period2": int(
                datetime.fromisoformat(cfg["source_end_exclusive"]).replace(tzinfo=UTC).timestamp()
            ),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(symbol)
        url += "?" + urllib.parse.urlencode(params)
        record = {"symbol": symbol, "market": market, "url": url, "requested_at": started}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                record["http_status"] = response.status
            path = output / f"{symbol}.json"
            with path.open("xb") as handle:
                handle.write(raw)
            record.update(path=path.name, sha256=digest(path))
            data = json.loads(raw)
            if data["chart"].get("error") or not data["chart"].get("result"):
                raise ValueError("Provider returned no chart result")
            chart = data["chart"]["result"][0]
            record.update(
                status="DOWNLOADED",
                currency=chart["meta"]["currency"],
                exchange_timezone=chart["meta"]["exchangeTimezoneName"],
                rows=len(chart.get("timestamp", [])),
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save(output / f"{symbol}.receipt.json", record)
        records.append(record)
        print(json.dumps({"symbol": symbol, "status": record["status"]}), flush=True)
        time.sleep(0.25)
    save(
        output / "manifest.json",
        {
            "study_id": cfg["study_id"],
            "scope": cfg["scope"],
            "created_at": datetime.now(UTC).isoformat(),
            "records": records,
            "complete": all(r["status"] == "DOWNLOADED" for r in records),
            "canonical_market_admission": False,
            "protected_prices_requested": False,
            "limitations": cfg["public_source_limitations"],
            "metadata_policy": "Current quote fields in opaque raw responses are never used or summarized.",
        },
    )


def monthly_prices(chart: dict, cfg: dict, *, quarantine_zero_quote_months: bool = False):
    import numpy as np
    import pandas as pd

    times = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
    local = times.tz_convert(chart["meta"]["exchangeTimezoneName"]).tz_localize(None)
    values = np.asarray(chart["indicators"]["adjclose"][0]["adjclose"], dtype=float)
    if len(values) != len(times) or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Malformed or unordered price observations")
    # All filtering precedes outcome aggregation. No current metadata price is read.
    keep = (local >= pd.Timestamp(cfg["source_start"])) & (local < pd.Timestamp(cfg["source_end_exclusive"]))
    series = pd.Series(values[keep], index=local[keep])
    quarantined_months = []
    if quarantine_zero_quote_months:
        # Explicit exploratory-only missing-data policy. No historical holiday
        # or terminal-loss inference: the WHOLE affected month remains unknown.
        quotes = chart["indicators"]["quote"][0]
        placeholder = values == 0
        for field in ("open", "high", "low", "close", "volume"):
            quote = np.asarray(quotes[field], dtype=float)
            if quote.shape != values.shape:
                raise ValueError("Unaligned raw quote fields")
            placeholder &= quote == 0
        quarantined_months = local[keep & placeholder].to_period("M").unique().tolist()
        # Only the exact all-zero rows bypass the strict invalid-price check.
        # A different invalid row in that same month must still fail.
        series.iloc[np.flatnonzero(placeholder[keep])] = np.nan
    if np.isinf(series).any() or (series.dropna() <= 0).any():
        raise ValueError(
            "Nonpositive/invalid prices: terminal losses require a different model; never drop them"
        )
    # Pick the final source row, including NaN; do not forward-fill a missing close.
    months = series.index.to_period("M")
    final_rows = ~months.duplicated(keep="last")
    result = pd.Series(series.to_numpy()[final_rows], index=months[final_rows])
    for month in quarantined_months:
        result.loc[month] = np.nan
    result.attrs["quarantined_zero_quote_months"] = [str(month) for month in quarantined_months]
    if result.empty:
        raise ValueError("No pre-protected history")
    result = result.reindex(pd.period_range(result.index.min(), result.index.max(), freq="M"))
    return result


def build_research_windows(prices: dict, context: int):
    import numpy as np
    import pandas as pd

    from frem.data import WindowData

    records = []
    coverage = {}
    names = sorted(prices)
    features = ["local_log_return", "trailing_3m_log_return", "trailing_12m_log_return", "trailing_6m_vol"]
    for asset_id, symbol in enumerate(names):
        price = prices[symbol]
        returns = np.log(price / price.shift(1))
        values = pd.DataFrame(
            {
                features[0]: returns,
                features[1]: returns.rolling(3).sum(),
                features[2]: returns.rolling(12).sum(),
                features[3]: returns.rolling(6).std(ddof=1),
            }
        ).to_numpy(dtype=np.float32)
        unavailable = 0
        for j in range(context + 2, len(price)):
            # Target M; decision start M-1; inputs stop M-3. M-2 is blind.
            x = values[j - context - 2 : j - 2]
            future = values[j : j + 1]
            if not np.isfinite(x).all() or not np.isfinite(future).all():
                unavailable += 1
                continue
            target = price.index[j]
            decision = (target - 1).start_time.tz_localize("UTC").value
            maturity = ((target + 1).start_time.tz_localize("UTC") + np.timedelta64(2, "D")).value
            records.append((decision, asset_id, maturity, x, future, float(returns.iloc[j]), str(target)))
        coverage[symbol] = {
            "first_month": str(price.index.min()),
            "last_month": str(price.index.max()),
            "source_months": len(price),
            "missing_months": [str(x) for x in price.index[price.isna()]],
            "windows_unavailable_from_gaps_or_warmup": unavailable,
            "valid_windows": sum(r[1] == asset_id for r in records),
        }
    if not records:
        raise ValueError("No valid research windows")
    records.sort(key=lambda r: (r[0], r[1]))
    data = WindowData(
        x=np.stack([r[3] for r in records]),
        future_x=np.stack([r[4] for r in records]),
        y=np.asarray([r[5] for r in records]),
        time=np.asarray([r[0] for r in records]),
        label_time=np.asarray([r[2] for r in records]),
        asset_id=np.asarray([r[1] for r in records]),
        feature_names=features,
        asset_names=names,
        metadata={"scope": "retrospective_public_research_coordinates", "executable_timestamps": False},
    )
    return data, np.asarray([r[6] for r in records]), coverage


def fold_indices(data, target_months, fold: dict, cfg: dict):
    import numpy as np
    import pandas as pd

    first = pd.Period(fold["return_start"], freq="M")
    last = pd.Period(fold["return_end"], freq="M")
    if first > last or (last + 1).start_time >= pd.Timestamp(cfg["protected_start"]):
        raise ValueError("Invalid fold or protected boundary reached")
    eval_start = (first - 1).start_time.tz_localize("UTC").value
    validation_start = (first - 1 - cfg["validation_months"]).start_time.tz_localize("UTC").value
    train = np.flatnonzero((data.time < validation_start) & (data.label_time < validation_start))
    validation = np.flatnonzero(
        (data.time >= validation_start) & (data.time < eval_start) & (data.label_time < eval_start)
    )
    evaluation = np.flatnonzero((target_months >= str(first)) & (target_months <= str(last)))
    expected = [str(x) for x in pd.period_range(first, last, freq="M")]
    excluded, included = {}, []
    for asset_id, symbol in enumerate(data.asset_names):
        counts = [int(np.sum(data.asset_id[ix] == asset_id)) for ix in (train, validation, evaluation)]
        actual_months = target_months[evaluation[data.asset_id[evaluation] == asset_id]].tolist()
        reasons = []
        if counts[0] < cfg["minimum_train_samples_per_stock"]:
            reasons.append("insufficient pre-validation training")
        if counts[1] < cfg["minimum_validation_samples_per_stock"]:
            reasons.append("insufficient matured validation")
        if actual_months != expected:
            reasons.append("incomplete target-month history; no stitching over gaps")
        if reasons:
            excluded[symbol] = {
                "reasons": reasons,
                "train_validation_test_rows": counts,
                "missing_target_months": sorted(set(expected) - set(actual_months)),
            }
        else:
            included.append(asset_id)
    if not included:
        raise ValueError("No stocks satisfy declared fold coverage")
    selected = tuple(ix[np.isin(data.asset_id[ix], included)] for ix in (train, validation, evaluation))
    return selected, {
        "name": fold["name"],
        "included_stocks": [data.asset_names[i] for i in included],
        "excluded_stocks": excluded,
        "expected_target_months": expected,
        "validation_start_ns": validation_start,
        "evaluation_decision_start_ns": eval_start,
        "max_training_label_ns": int(data.label_time[selected[0]].max()),
        "max_validation_label_ns": int(data.label_time[selected[1]].max()),
        "row_counts": [len(ix) for ix in selected],
        "exclusion_policy": "Predetermined completeness/coverage only; no outcome-based exclusions",
    }


def memory_snapshot() -> dict:
    if platform.system() != "Darwin":
        raise RuntimeError("This local admission implementation is macOS-specific")
    out = subprocess.check_output(["vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+) bytes", out).group(1))
    counts = {key: int(value) for key, value in re.findall(r"^([^:\n]+):\s+(\d+)\.", out, re.MULTILINE)}
    return {
        "available_mib": (counts["Pages free"] + counts["Pages speculative"]) * page_size / 2**20,
        "pageouts": counts["Pageouts"],
    }


def resource_gate(minimum_mib: float) -> list[dict]:
    samples = []
    for _ in range(3):
        samples.append(memory_snapshot())
        time.sleep(1)
    if (
        min(s["available_mib"] for s in samples) < minimum_mib
        or samples[-1]["pageouts"] > samples[0]["pageouts"]
    ):
        raise RuntimeError("RESOURCE_ADMISSION_FAILED " + json.dumps(samples))
    return samples


def local_stock_ledgers(predictions, cost_bps: float):
    import numpy as np

    result = {}
    fee = cost_bps / 10000
    for symbol, rows in predictions.groupby("asset", sort=True):
        rows = rows.sort_values("time")
        # Monthly independent round-trips; cash earns zero. No shared global NAV.
        active = rows["prediction"].to_numpy() > 0
        gross = np.exp(rows["actual"].to_numpy())
        strategy = np.where(active, gross * (1 - fee) / (1 + fee), 1.0)
        always_long = gross * (1 - fee) / (1 + fee)
        wealth = np.r_[1.0, np.cumprod(strategy)]
        result[symbol] = {
            "months": len(rows),
            "long_months": int(active.sum()),
            "net_local_return": float(wealth[-1] - 1),
            "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            "same_cost_monthly_always_long_return": float(np.prod(always_long) - 1),
            "gross_continuous_buy_hold_return": float(np.prod(gross) - 1),
        }
    return result


def paired_month_bootstrap(left, right, settings: dict):
    import numpy as np

    keys = ["time", "asset"]
    merged = left.merge(right, on=keys, suffixes=("_candidate", "_control"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Incomplete prediction pairing")
    if not np.array_equal(merged["actual_candidate"], merged["actual_control"]):
        raise ValueError("Paired outcomes differ")
    merged["difference"] = merged["nll_control"] - merged["nll_candidate"]
    differences = merged.groupby("time")["difference"].mean().to_numpy()
    n, block = len(differences), settings["block_months"]
    if n < 2 * block:
        raise ValueError("Insufficient independent monthly blocks")
    rng = np.random.default_rng(settings["seed"])
    starts = rng.integers(n, size=(settings["repetitions"], math.ceil(n / block)))
    indices = ((starts[..., None] + np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
    draws = differences[indices].mean(axis=1)
    alpha = (1 - settings["confidence"]) / 2
    return {
        "effect": "no_memory NLL minus FREM NLL; positive favors FREM",
        "estimate": float(differences.mean()),
        "ci": np.quantile(draws, [alpha, 1 - alpha]).tolist(),
        "months": n,
        "paired_rows": len(merged),
        "settings": settings,
        "interpretation": "Exploratory interval; single seed, survivor sample, unadjusted multiple windows",
    }


def run(cfg: dict, source: Path, output: Path) -> None:
    # Gate before importing the scientific stack or creating a campaign.
    output.mkdir(parents=True, exist_ok=False)
    try:
        samples = resource_gate(cfg["budget"]["minimum_free_speculative_mib"])
    except RuntimeError as error:
        save(
            output / "failure.json",
            {
                "status": "BLOCKED_BEFORE_MODEL_IMPORT",
                "error": str(error),
                "created_at": datetime.now(UTC).isoformat(),
                "completed_cells": [],
                "config": cfg,
            },
        )
        raise
    import numpy as np
    import torch

    from frem.baselines import make_baseline
    from frem.evaluation import evaluate_predictions, score_predictions
    from frem.model import ModelConfig
    from frem.training import Trainer, TrainingConfig

    torch.set_num_threads(1)
    start = time.monotonic()
    manifest = json.loads((source / "manifest.json").read_text())
    save(
        output / "declaration.json",
        {
            "config": cfg,
            "source_manifest_sha256": digest(source / "manifest.json"),
            "source_request_sha256": digest(source / "request.json"),
            "created_at": datetime.now(UTC).isoformat(),
            "resource_admission": samples,
            "code_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [Path(__file__), *sorted((ROOT / "src/frem").glob("*.py"))]
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "scientific_freeze": False,
            "phase": "exploratory_public_history",
        },
    )
    cells, fold_results = [], []
    try:
        if json.loads((source / "request.json").read_text()) != cfg:
            raise ValueError("Acquisition request differs from the declared configuration")
        if not manifest["complete"]:
            raise ValueError(
                "Acquisition incomplete; preserve failures instead of silently selecting downloads"
            )
        prices = {}
        for item in manifest["records"]:
            path = source / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError("Source hash mismatch")
            chart = json.loads(path.read_bytes())["chart"]["result"][0]
            if chart["meta"]["symbol"].upper() != item["symbol"].upper():
                raise ValueError("Provider symbol does not match the declared listing")
            prices[item["symbol"]] = monthly_prices(chart, cfg)
        data, months, coverage = build_research_windows(prices, cfg["context_months"])
        save(output / "coverage.json", coverage)
        for fold in cfg["folds"]:
            indices, split = fold_indices(data, months, fold, cfg)
            directory = output / fold["name"]
            save(directory / "split.json", split)
            train, validation, evaluation = [data.subset(ix) for ix in indices]
            scored = {}
            for name in cfg["models"]:
                if time.monotonic() - start >= cfg["budget"]["max_wall_seconds"]:
                    raise TimeoutError("Campaign budget exhausted")
                admission = resource_gate(cfg["budget"]["minimum_free_speculative_mib"])
                cell = directory / name
                cell.mkdir()
                cell_start = time.monotonic()
                save(cell / "started.json", {"model": name, "seed": cfg["seed"], "admission": admission})
                if name in {"frem", "no_memory"}:
                    mc = ModelConfig(
                        input_dim=4,
                        context_length=cfg["context_months"],
                        horizon=1,
                        reservoir_seed=cfg["seed"],
                        use_memory=name == "frem",
                        **cfg["model"],
                    )
                    tc = TrainingConfig(seed=cfg["seed"], device="cpu", **cfg["training"])
                    predictor = Trainer(mc, tc)
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
                fit_seconds = time.monotonic() - cell_start
                predictor.save(cell / "checkpoint")
                save(cell / "fit.json", history)
                predictions = predictor.predict(evaluation)
                predictions["target_month"] = months[indices[2]]
                predictions["model"] = name
                predictions["seed"] = cfg["seed"]
                predictions.to_csv(cell / "predictions.csv", index=False)
                metrics = evaluate_predictions(predictions, student_df=None)
                metrics["forecast_units"] = "monthly log adjusted-close return in each stock's local currency"
                scored[name] = score_predictions(predictions, student_df=None)
                ledger = local_stock_ledgers(predictions, cfg["hypothetical_cost_per_side_bps"])
                save(cell / "local_stock_ledgers.json", ledger)
                receipt = {
                    "model": name,
                    "fold": fold["name"],
                    "status": "COMPLETE",
                    "seed": cfg["seed"],
                    "metrics": metrics,
                    "wall_seconds": time.monotonic() - cell_start,
                    "fit_wall_seconds": fit_seconds,
                    "fit_budget_overrun": fit_seconds > cfg["training"]["max_wall_seconds"],
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "prediction_sha256": digest(cell / "predictions.csv"),
                    "median_stock_net_local_return": float(
                        np.median([r["net_local_return"] for r in ledger.values()])
                    ),
                    "scope": cfg["scope"],
                }
                save(cell / "receipt.json", receipt)
                cells.append(receipt)
                print(
                    json.dumps(
                        {
                            "fold": fold["name"],
                            "model": name,
                            "status": "COMPLETE",
                            "mean_nll": metrics["mean_nll"],
                        }
                    ),
                    flush=True,
                )
                del predictor
                if (
                    receipt["fit_budget_overrun"]
                    or time.monotonic() - start > cfg["budget"]["max_wall_seconds"]
                ):
                    raise TimeoutError("Budget overrun retained; remaining campaign cells were not started")
            comparison = paired_month_bootstrap(scored["frem"], scored["no_memory"], cfg["bootstrap"])
            save(directory / "comparison.json", comparison)
            fold_results.append(
                {
                    "fold": fold["name"],
                    "stocks": len(split["included_stocks"]),
                    "excluded": split["excluded_stocks"],
                    "comparison": comparison,
                }
            )
        save(
            output / "summary.json",
            {
                "status": "COMPLETE",
                "completed_cells": len(cells),
                "expected_cells": len(cfg["folds"]) * len(cfg["models"]),
                "cells": cells,
                "folds": fold_results,
                "wall_seconds": time.monotonic() - start,
                "scope": cfg["scope"],
                "limitations": cfg["public_source_limitations"],
                "canonical_scientific_status": "NOT_EVALUATED",
                "protected_outcomes_accessed": False,
            },
        )
    except BaseException as error:
        save(
            output / "failure.json",
            {
                "status": "INCOMPLETE",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "completed_cells": cells,
                "wall_seconds": time.monotonic() - start,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "run"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "acquire":
        acquire(cfg, args.output)
    else:
        if args.data is None:
            parser.error("run requires --data")
        run(cfg, args.data, args.output)


if __name__ == "__main__":
    main()
