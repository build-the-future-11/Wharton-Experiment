"""Bounded 2020-2022 intake, with no 2019 or protected quote consumption."""

import argparse
import hashlib
import json
import math
import traceback
import urllib.request
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

from successors.frem_s6_real.prepare import ASSETS, DAY, sha, stamp, write
from successors.frem_s7.prepare import date_of, interval_return


def market_open(day):
    return datetime.fromisoformat(day + "T09:30:00").replace(tzinfo=ZoneInfo("America/New_York")).timestamp()


def acquire(root, freeze):
    root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(freeze.read_text())
    if protocol["evaluation_years"] != [2020, 2021, 2022]:
        raise ValueError("wrong frozen acquisition scope")
    for name, digest in protocol["sources"].items():
        if sha(name) != digest:
            raise ValueError("source drift before acquisition: " + name)
    receipts = []
    try:
        for asset in ASSETS:
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{asset}"
                f"?period1={stamp('2020-01-01')}&period2={stamp('2023-01-01')}"
                "&interval=1d&events=div%2Csplits"
            )
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read(4_000_001)
                server_date = response.headers.get("Date")
            if len(raw) > 4_000_000:
                raise ValueError("oversized provider response")
            chart = json.loads(raw)["chart"]
            if chart["error"] or len(chart["result"]) != 1:
                raise ValueError("provider error")
            result = chart["result"][0]
            times = result["timestamp"]
            if not times or any(not stamp("2020-01-01") <= t < stamp("2023-01-01") for t in times):
                raise ValueError("provider dates outside frozen request")
            splits = result.get("events", {}).get("splits", {})
            if splits:
                raise ValueError("split needs a separately frozen adapter")
            dividends = result.get("events", {}).get("dividends", {})
            if any(not stamp("2020-01-01") <= x["date"] < stamp("2023-01-01") for x in dividends.values()):
                raise ValueError("out-of-scope dividend")
            quotes = result["indicators"]["quote"][0]
            path = root / f"{asset}.json"
            write(
                path,
                {
                    "symbol": asset,
                    "rows": [
                        {"timestamp": t, "open": op, "close": cl}
                        for t, op, cl in zip(times, quotes["open"], quotes["close"], strict=True)
                    ],
                    "dividends": dividends,
                    "split_events": splits,
                },
            )
            receipts.append(
                {
                    "symbol": asset,
                    "url": url,
                    "http_date": server_date,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "retained_sha256": sha(path),
                    "rows": len(times),
                }
            )
            write(root / f"{asset}.receipt.json", receipts[-1])
        write(
            root / "acquisition.json",
            {
                "sources": receipts,
                "freeze_sha256": sha(freeze),
                "retention": "Only bounded historical quotes/actions retained; current metadata discarded without use.",
                "rights": "Publicly accessible; redistribution rights unverified. No rights certification.",
                "scope": "Current-vintage survivor ETFs; not point-in-time or independent confirmation.",
            },
        )
    except BaseException:
        write(root / "acquisition_failure.json", {"traceback": traceback.format_exc(), "completed": receipts})
        raise


def build_segment(root, *, fresh):
    prices, features, dividends = {}, {}, {}
    for asset in ASSETS:
        raw = json.loads((root / f"{asset}.json").read_text())
        rows = raw["rows"]
        low, high = (
            (stamp("2020-01-01"), stamp("2023-01-01"))
            if fresh
            else (stamp("2005-01-01"), stamp("2019-01-01"))
        )
        if any(not low <= r["timestamp"] < high for r in rows):
            raise ValueError("2019/out-of-scope source timestamps rejected before quote use")
        prices[asset], features[asset] = {}, []
        previous, previous_time = None, -math.inf
        for r in rows:
            if r["timestamp"] <= previous_time:
                raise ValueError("unordered or duplicate source observations")
            previous_time = r["timestamp"]
            day, op, cl = date_of(r["timestamp"]), r["open"], r["close"]
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                for v in (op, cl)
            ):
                raise ValueError("invalid quote; no silent omission or filling")
            prices[asset][day] = op
            if previous is not None:
                features[asset].append((stamp(day) + DAY, math.log(cl / previous)))
            previous = cl
        dividends[asset] = []
        for x in raw["dividends"].values():
            # The retained old provider export includes a 2004-12-31 event.
            # No supported holding can own it: exclude by time before amount use.
            if not fresh and x["date"] < low:
                continue
            if not low <= x["date"] < high or not math.isfinite(x["amount"]) or x["amount"] < 0:
                raise ValueError("invalid or out-of-scope dividend")
            dividends[asset].append((date_of(x["date"]), x["amount"]))
    dates = sorted(prices[ASSETS[0]])
    if any(set(p) != set(dates) for p in prices.values()):
        raise ValueError("calendar mismatch; no silent calendar intersection")
    weeks = {}
    for day in dates:
        weeks.setdefault(datetime.fromisoformat(day).isocalendar()[:2], day)
    anchors = list(weeks.values())
    fields = (
        "assets",
        "origin_times",
        "maturity_times",
        "available_times",
        "window_times",
        "window_available",
        "windows",
        "y",
        "simple_returns",
    )
    parts = {p: {k: [] for k in fields} for p in ("train", "calibration", "evaluation")}
    parts["train"].update({k: [] for k in ("future_times", "future_available", "future_windows")})
    exclusions = []
    for start, end in pairwise(anchors):
        origin, maturity = market_open(start), market_open(end)
        part = "evaluation" if fresh else "train" if start < "2017-01-01" else "calibration"
        selected = []
        for asset in ASSETS:
            past = [x for x in features[asset] if x[0] < origin][-8:]
            future = [x for x in features[asset] if x[0] > origin][:8]
            if len(past) < 8 or (
                part == "train"
                and (
                    len(future) < 8
                    or future[-1][0] >= stamp("2017-01-01")
                    or maturity + 4 * 3600 >= stamp("2017-01-01")
                )
            ):
                break
            selected.append((asset, past, future))
        if len(selected) != len(ASSETS):
            exclusions.append({"start": start, "reason": "warmup_or_train_target_purge"})
            continue
        for asset, past, future in selected:
            ret = interval_return(start, end, prices[asset], dividends[asset])
            values = {
                "assets": asset,
                "origin_times": origin,
                "maturity_times": maturity,
                "available_times": maturity + 4 * 3600,
                "window_times": [x[0] for x in past],
                "window_available": [x[0] for x in past],
                "windows": [[x[1]] for x in past],
                "y": math.log1p(ret),
                "simple_returns": ret,
            }
            if part == "train":
                values.update(
                    future_times=[x[0] for x in future],
                    future_available=[x[0] for x in future],
                    future_windows=[[x[1]] for x in future],
                )
            for key, value in values.items():
                parts[part][key].append(value)
    return parts, exclusions


def build(root):
    old, old_exclusions = build_segment(Path("data/s6_real_etf_20261014"), fresh=False)
    new, new_exclusions = build_segment(root, fresh=True)
    old["evaluation"] = new["evaluation"]
    payload = {
        "schema_version": "s7-variable-week-v1",
        "return_units": "log_return",
        "currency": "USD",
        "horizon_seconds": 7 * DAY,
        "protected_start": stamp("2025-01-01"),
        "fit_cutoff": stamp("2017-01-01"),
        "asset_universe": list(ASSETS),
        "partitions": old,
    }
    write(root / "windows.json", payload)
    write(
        root / "coverage.json",
        {
            "dates": {k: len(set(v["origin_times"])) for k, v in old.items()},
            "exclusions": old_exclusions + new_exclusions,
            "evaluation_start": min(old["evaluation"]["origin_times"]),
            "evaluation_end": max(old["evaluation"]["maturity_times"]),
            "2019_rows_used": 0,
            "limitation": "No year-gap bridging. Fresh evaluation starts after eight 2020 daily returns. Last interval requires both endpoints before 2023. Assumed availability; weekly dividend accrual/reinvestment; zero cash yield.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.root, args.freeze)
    build(args.root)
