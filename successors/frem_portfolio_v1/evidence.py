"""Strict adapters for retained FREM CSVs; no scientific dependency imports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from .engine import Forecast, ensemble


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def next_month(year, month):
    return (year + 1, 1) if month == 12 else (year, month + 1)


def read_predictions(path, expected_hash, *, expected_rows=None):
    if sha256(path) != expected_hash:
        raise ValueError(f"Prediction hash mismatch: {path}")
    rows, seen, error = [], set(), 0.0
    with Path(path).open(newline="") as file:
        for raw in csv.DictReader(file):
            key = int(raw["time"]), raw["asset"]
            if key in seen:
                raise ValueError("Duplicate forecast observation")
            seen.add(key)
            mu, scale, df, actual = (float(raw[k]) for k in ("prediction", "scale", "student_df", "actual"))
            forecast = Forecast.student(raw["asset"], mu, scale, df)
            if not math.isfinite(actual) or int(raw["label_time"]) <= key[0]:
                raise ValueError("Invalid actual or non-future label")
            # Reconstruct Student-t NLL separately from the original SciPy code.
            nll = (
                math.lgamma(df / 2)
                - math.lgamma((df + 1) / 2)
                + 0.5 * math.log(df * math.pi)
                + math.log(scale)
                + (df + 1) / 2 * math.log1p(((actual - mu) / scale) ** 2 / df)
            )
            if raw.get("nll"):
                error = max(error, abs(nll - float(raw["nll"])))
            maturity = datetime.fromtimestamp(int(raw["label_time"]) / 1e9, UTC)
            year, month = (
                (maturity.year - 1, 12) if maturity.month == 1 else (maturity.year, maturity.month - 1)
            )
            target = f"{year:04d}-{month:02d}"
            if raw.get("target_month", target) != target or target >= "2025-01":
                raise ValueError("Unexpected research horizon or protected outcome")
            end_year, end_month = next_month(year, month)
            start = int(datetime(year, month, 1, tzinfo=UTC).timestamp())
            end = int(datetime(end_year, end_month, 1, tzinfo=UTC).timestamp())
            rows.append(
                {
                    "time": start,
                    "end": end,
                    "target_month": target,
                    "available_at": key[0] // 1_000_000_000,
                    "asset": raw["asset"],
                    "forecast": forecast,
                    "actual": actual,
                    "return": math.expm1(actual),
                }
            )
    if not rows or (expected_rows is not None and len(rows) != expected_rows):
        raise ValueError("Missing prediction rows")
    if error > 1e-10:
        raise ValueError("Independent NLL reconstruction mismatch")
    return sorted(rows, key=lambda x: (x["time"], x["asset"])), {
        "path": str(path),
        "sha256": expected_hash,
        "rows": len(rows),
        "max_nll_error": error,
    }


def periods(rows, currency, names=None):
    names = sorted({r["asset"] for r in rows}) if names is None else sorted(names)
    groups = {}
    for row in rows:
        if row["asset"] in names:
            groups.setdefault(row["time"], []).append(row)
    result = []
    for time, group in sorted(groups.items()):
        group.sort(key=lambda x: x["asset"])
        if [r["asset"] for r in group] != names:
            raise ValueError("Incomplete market/month grid; no outcome-based exclusion")
        if len({r["end"] for r in group}) != 1:
            raise ValueError("Unsynchronized horizon")
        result.append(
            {
                "time": time,
                "end": group[0]["end"],
                "available_at": max(r["available_at"] for r in group),
                "currency": currency,
                "forecasts": [r["forecast"] for r in group],
                "returns": {r["asset"]: r["return"] for r in group},
            }
        )
    if not result:
        raise ValueError("Empty market")
    return result


def assert_paired(left, right):
    if len(left) != len(right):
        raise ValueError("Mismatched period count")
    for a, b in zip(left, right, strict=True):
        for field in ("time", "end", "currency", "returns"):
            if a[field] != b[field]:
                raise ValueError(f"Comparison information mismatch: {field}")


def ensemble_periods(families):
    if not families:
        raise ValueError("Empty seed family")
    for family in families[1:]:
        assert_paired(families[0], family)
    return [
        {**column[0], "forecasts": ensemble([p["forecasts"] for p in column])}
        for column in zip(*families, strict=True)
    ]


def load_cases(root, spec):
    """Every declared file/seed/control required; no glob-based winner selection."""
    cases, receipts, failures = [], [], []
    roots = {k: root / v for k, v in spec["existing_model_runs"].items()}
    if load(roots["S4"] / "completion.json")["status"] != "COMPLETE":
        raise ValueError("S4 matrix incomplete")
    s3_manifest = load(roots["S3"] / "public_data/manifest.json")
    if not s3_manifest["complete"] or {r["currency"] for r in s3_manifest["records"]} != {"USD"}:
        raise ValueError("S3 common-currency data not established")
    expected_names = sorted(r["symbol"] for r in s3_manifest["records"])
    for lineage in ("S4", "S3"):
        controls = {"frem": spec["s4_arms"]} if lineage == "S4" else spec["s3_controls"]
        for fold in spec["primary_folds"]:
            common = None
            for model, arms in controls.items():
                for arm in arms:
                    families = []
                    for seed in spec["seeds"]:
                        cell = (
                            roots[lineage] / fold / str(seed)
                            if lineage == "S4"
                            else roots[lineage] / "public" / fold / str(seed) / model
                        )
                        result = load(cell / "result.json")["results"][arm]
                        rows, audit = read_predictions(
                            cell / f"{arm}.csv", result["sha256"], expected_rows=result["rows"]
                        )
                        receipts.append(audit)
                        ps = periods(rows, "USD", expected_names)
                        if common is None:
                            common = ps
                        assert_paired(common, ps)
                        families.append(ps)
                        cases.append(
                            {
                                "lineage": lineage,
                                "fold": fold,
                                "model": model,
                                "arm": arm,
                                "seed": str(seed),
                                "market": "US",
                                "periods": ps,
                            }
                        )
                    cases.append(
                        {
                            "lineage": lineage,
                            "fold": fold,
                            "model": model,
                            "arm": arm,
                            "seed": "ensemble",
                            "market": "US",
                            "periods": ensemble_periods(families),
                        }
                    )
    # Each older listing market remains a separate currency ledger.
    slow_cfg = load(root / "configs/public_equity_crises.json")
    source_manifest = load(root / "data/public_equity_crises_v1/manifest.json")
    currencies = {r["symbol"]: r["currency"] for r in source_manifest["records"]}
    market_names = {}
    for symbol, market in slow_cfg["symbols"].items():
        market_names.setdefault(market, []).append(symbol)
    for fold in slow_cfg["folds"]:
        paired = {}
        for model in slow_cfg["models"]:
            cell = roots["slow"] / fold["name"] / model
            receipt = load(cell / "receipt.json")
            if receipt["status"] != "COMPLETE":
                raise ValueError("Slow cell incomplete")
            rows, audit = read_predictions(cell / "predictions.csv", receipt["prediction_sha256"])
            receipts.append(audit)
            for market, names in sorted(market_names.items()):
                identity = {
                    "lineage": "slow",
                    "fold": fold["name"],
                    "model": model,
                    "arm": "native",
                    "seed": "11",
                    "market": market,
                }
                try:
                    currency = {currencies[n] for n in names}
                    if len(currency) != 1:
                        raise ValueError("Market basket includes mixed currencies")
                    ps = periods(rows, next(iter(currency)), names)
                    if market not in paired:
                        paired[market] = ps
                    assert_paired(paired[market], ps)
                    cases.append({**identity, "periods": ps})
                except ValueError as error:
                    failures.append({**identity, "status": "INCOMPLETE_RETAINED", "error": str(error)})
    return cases, receipts, failures
