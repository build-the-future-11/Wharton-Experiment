"""Fail-closed historical OHLCV adapter and publication-time joins (stdlib only)."""

import argparse
import bisect
import csv
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

from .runtime import digest, timestamp, write_json

ASSETS = ("spy", "iwm", "dia", "tlt", "ief", "gld", "slv")
SOURCE_SHA256 = {
    "spy": "c678bccc09a3ee667e591d97e95498f2ac185c98408702cae738ba3320ebdeec",
    "iwm": "0659f79d5ecd9d86136feb1f3579157fe0866cf84ff28c1d4a8a75f1b0a28029",
    "dia": "dcd47d659f3c52628fcf601cee1fffe9ba25aa78041aef2f16057935051b9f55",
    "tlt": "9b7c22d513b8165ecb021a622f27a1e71a151f901804261189de003ed6d95564",
    "ief": "47b52d0a7cecf5347560d9a870bc6b2b173631a834a2259ec3002456d96372fc",
    "gld": "c21b14f81fb7675a4a41e497eb2f48f75af577b64f41c252d119052d3b583f32",
    "slv": "afc0ea0fda0271da19c51b5e7d31651d1d202ae74c2a7eb4dddb533bcacf99bc",
}
SOURCE = "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
FEATURES = (
    "intraday",
    "overnight",
    "range",
    "logvolume",
    "momentum5",
    "momentum20",
    "volatility20",
    "downside20",
)


def resolve_etf_source(source_root, asset):
    """Resolve one asset from the dataset's declared ``Data/ETFs`` taxonomy.

    The upstream archive can contain the same ticker filename in both its
    ``Stocks`` and ``ETFs`` branches. The cohort is explicitly an ETF cohort,
    so a repository-wide basename match is ambiguous and incorrect. We still
    fail closed if more than one canonical ETF path is mounted.
    """
    filename = asset + ".us.txt"
    all_matches = sorted(Path(source_root).rglob(filename))
    canonical = [
        path
        for path in all_matches
        if len(path.parts) >= 3 and path.parts[-3:] == ("Data", "ETFs", filename)
    ]
    if len(canonical) != 1 or canonical[0].is_symlink() or not canonical[0].is_file():
        raise ValueError(
            f"exactly one regular Data/ETFs/{filename} required, "
            f"found {len(canonical)} canonical among {len(all_matches)} basename matches"
        )
    return canonical[0], all_matches


def read_prices(path, boundary="2025-01-01"):
    if date.fromisoformat(boundary).isoformat() != boundary or boundary > "2025-01-01":
        raise ValueError("invalid or protected price boundary")
    records = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]:
            raise ValueError(f"unexpected OHLCV schema: {path}")
        last = None
        for line, row in enumerate(reader, 2):
            day = date.fromisoformat(row["Date"]).isoformat()
            if day >= boundary:
                continue  # Do not inspect protected values.
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"wrong CSV row width: {path}:{line}")
            if last is not None and day <= last:
                raise ValueError(f"duplicate or unordered date: {path}:{line}")
            last = day
            values = [float(row[key]) for key in ("Open", "High", "Low", "Close", "Volume")]
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"nonfinite price/volume: {path}:{line}")
            opening, high, low, close, volume = values
            if min(opening, high, low, close) <= 0 or volume < 0:
                raise ValueError(f"nonpositive price or negative volume: {path}:{line}")
            if not low <= min(opening, close) <= max(opening, close) <= high:
                raise ValueError(f"inconsistent OHLC envelope: {path}:{line}")
            open_interest = float(row["OpenInt"])
            if (
                not math.isfinite(open_interest)
                or open_interest < 0
                or not open_interest.is_integer()
            ):
                raise ValueError(f"invalid open interest: {path}:{line}")
            records[day] = values
    if not records:
        raise ValueError(f"no admitted historical observations: {path}")
    return records


def prepare(
    source_root,
    output,
    start="2007-01-01",
    stop="2018-01-01",
    expected_hashes=SOURCE_SHA256,
):
    """Frozen seven-ETF successor cohort. No data-driven asset substitution."""
    if (
        date.fromisoformat(start).isoformat() != start
        or date.fromisoformat(stop).isoformat() != stop
        or not "2007-01-01" <= start < stop <= "2025-01-01"
    ):
        raise ValueError("invalid or protected historical interval")
    if expected_hashes is not None and set(expected_hashes) != set(ASSETS):
        raise ValueError("exact seven-ETF source hash allowlist required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "source": SOURCE,
        "assets": list(ASSETS),
        "start": start,
        "stop": stop,
        "status": "STARTED",
        "files": {},
        "rights": "Dataset card lists CC0; provider assertion",
        "limitations": [
            "historical survivor-selected ETF cohort",
            "no verified publication timestamps",
            "adjustments may embed later revisions",
            "no verified terminal distributions",
            "not a tradable or point-in-time-valid performance claim",
        ],
    }
    try:
        tables = {}
        for asset in ASSETS:
            path, matches = resolve_etf_source(source_root, asset)
            observed_hash = digest(path)
            if expected_hashes is not None and observed_hash != expected_hashes[asset]:
                raise ValueError(f"source hash mismatch for {asset}")
            table = read_prices(path, boundary=stop)
            selected = {k: v for k, v in table.items() if start <= k < stop}
            if len(selected) < 200:
                raise ValueError(f"insufficient fixed-cohort observations: {asset}")
            tables[asset] = selected
            receipt["files"][asset] = {
                "sha256": observed_hash,
                "bytes": path.stat().st_size,
                "selected_rows": len(selected),
                "source_path": str(path),
                "basename_matches": [str(candidate) for candidate in matches],
                "selection_rule": "exact Data/ETFs path suffix",
            }
        dates = list(tables[ASSETS[0]])
        for asset, table in tables.items():
            if list(table) != dates:
                raise ValueError(
                    f"calendar mismatch for {asset}; no intersection or forward fill allowed"
                )
        features = {}
        for asset, table in tables.items():
            values = list(table.values())
            returns = [0.0] + [values[i][3] / values[i - 1][3] - 1 for i in range(1, len(values))]
            features[asset] = []
            for i in range(20, len(values)):
                opening, high, low, close, volume = values[i]
                trailing = returns[i - 19 : i + 1]
                mean = math.fsum(trailing) / 20
                variance = math.fsum((x - mean) ** 2 for x in trailing) / 20
                downside = math.fsum(min(x, 0) ** 2 for x in trailing) / 20
                features[asset].append(
                    [
                        returns[i],
                        close / opening - 1,
                        opening / values[i - 1][3] - 1,
                        (high - low) / close,
                        math.log1p(volume),
                        math.fsum(returns[i - 4 : i + 1]) / 5,
                        mean,
                        math.sqrt(variance),
                        math.sqrt(downside),
                    ]
                )
                if not all(math.isfinite(x) for x in features[asset][-1]):
                    raise ValueError(f"derived feature overflow: {asset}:{dates[i]}")
        panel = output / "panel.csv"
        with panel.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "timestamp",
                    "asset",
                    "available_at",
                    "return",
                    *[f"feature_{i:02d}_{name}" for i, name in enumerate(FEATURES)],
                ]
            )
            for i, day in enumerate(dates[20:]):
                # Calendar ordering coordinate only; NOT asserted as a source release timestamp.
                instant = day + "T23:59:59Z"
                for asset in ASSETS:
                    writer.writerow([instant, asset.upper(), instant, *features[asset][i]])
        write_json(
            str(panel) + ".manifest.json",
            {
                "schema": 1,
                "sha256": digest(panel),
                "source": SOURCE,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "license": "Kaggle dataset card: CC0 (provider assertion, checked 2026-10-19)",
                "universe_policy": "Fixed SPY IWM DIA TLT IEF GLD SLV source-calendar successor cohort; QQQ excluded prospectively after pre-market calendar audit; no eligibility claim",
                "return_definition": "source_adjusted_close_ratio_not_verified_total_return",
                "feature_availability": "historical_order_only_unverified_point_in_time",
                "purpose": "historical_proxy",
            },
        )
        receipt.update(
            status="COMPLETE",
            rows=(len(dates) - 20) * len(ASSETS),
            panel_sha256=digest(panel),
            features=["return", *FEATURES],
        )
        return panel
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "intake.json", receipt)


def asof_values(decisions, observations):
    """Join explicitly published vintages: (available_at, observation_at, value).

    Latest *observation* available at the decision is chosen, then its latest
    revision. An old-period revision must not replace a newer-period observation.
    Unknown availability is rejected. No backfill before the first publication.
    """
    rows = []
    seen = set()
    for available, observed, value in observations:
        known, period = timestamp(available), timestamp(observed)
        if known < period or not math.isfinite(value) or (known, period) in seen:
            raise ValueError("invalid, ambiguous or nonfinite vintage")
        seen.add((known, period))
        rows.append((known, period, float(value)))
    rows.sort()
    parsed = [timestamp(x) for x in decisions]
    if parsed != sorted(set(parsed)):
        raise ValueError("decisions must be strictly increasing")
    cursor, latest, periods, result = 0, {}, [], []
    for decision in parsed:
        while cursor < len(rows) and rows[cursor][0] <= decision:
            known, period, value = rows[cursor]
            if period not in latest:
                bisect.insort(periods, period)
            latest[period] = (value, known)
            cursor += 1
        result.append(None if not periods else latest[periods[-1]])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps({"panel": str(prepare(args.source, args.output))}))


if __name__ == "__main__":
    main()
