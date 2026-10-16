"""Acquire bounded real quotes and construct the predeclared weekly benchmark.

Standard-library only. Never downloads the reserved 2019+ historical tail.
Provider metadata is discarded unread; its current quote is not a feature.
"""
import argparse
import hashlib
import json
import math
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
ASSETS = ("SPY", "TLT", "GLD")
DAY = 86400


def stamp(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(root):
    root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for asset in ASSETS:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{asset}"
               f"?period1={stamp('2005-01-01')}&period2={stamp('2019-01-01')}"
               "&interval=1d&events=div%2Csplits")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise ValueError("unexpected response size")
            server_date = response.headers.get("Date")
        chart = json.loads(raw)["chart"]
        if chart["error"] or len(chart["result"]) != 1:
            raise ValueError("provider error")
        result = chart["result"][0]
        # No raw metadata persistence: it can contain current/protected quotes.
        times = result["timestamp"]
        if any(t < stamp('2005-01-01') or t >= stamp('2019-01-01') for t in times):
            raise ValueError("provider returned dates outside the requested boundary")
        quotes = result["indicators"]["quote"][0]
        if any(len(quotes[k]) != len(times) for k in ("open", "close")):
            raise ValueError("provider quote alignment mismatch")
        splits = result.get("events", {}).get("splits", {})
        if splits:
            raise ValueError("unadjusted-price contract excludes splits; no silent correction")
        rows = [{"timestamp": t, "open": op, "close": cl}
                for t, op, cl in zip(times, quotes["open"], quotes["close"], strict=True)]
        path = root / f"{asset}.json"
        write(path, {"symbol": asset, "rows": rows, "split_events": splits,
                     "dividends": result.get("events", {}).get("dividends", {})})
        receipts.append({"symbol": asset, "url": url, "http_date": server_date,
                         "retrieved_at": datetime.now(UTC).isoformat(),
                         "response_sha256": hashlib.sha256(raw).hexdigest(),
                         "retained_sha256": sha(path), "rows": len(rows)})
    write(root / "acquisition.json", {"sources": receipts,
          "retention": "Only requested historical timestamps, unadjusted open/close, dividends and splits. Current metadata discarded without use.",
          "rights": "Public endpoint access; no redistribution license established. Local private research only; obtain provider permission before distribution."})


def build(root):
    prices, features = {}, {}
    for asset in ASSETS:
        rows = json.loads((root / f"{asset}.json").read_text())["rows"]
        prices[asset], features[asset] = {}, []
        previous = None
        for row in rows:
            day = datetime.fromtimestamp(row["timestamp"], UTC).date().isoformat()
            if day in prices[asset]:
                raise ValueError("duplicate daily bar")
            op, cl = row["open"], row["close"]
            valid = all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (op, cl))
            if not valid:
                previous = None
                continue
            prices[asset][day] = op
            if previous is not None:
                features[asset].append((stamp(day) + DAY, math.log(cl / previous)))
            previous = cl
    parts = {p: {k: [] for k in ("assets", "origin_times", "maturity_times", "available_times",
              "window_times", "window_available", "windows", "y", "simple_returns")}
             for p in ("train", "calibration", "evaluation")}
    parts["train"].update({k: [] for k in ("future_times", "future_available", "future_windows")})
    excluded = []
    day = datetime(2005, 1, 3, tzinfo=UTC)
    while day.year < 2019:
        date, end = day.date().isoformat(), (day + timedelta(days=7)).date().isoformat()
        origin = int(day.timestamp()) + 12 * 3600  # before NYSE regular open in either DST regime
        maturity = origin + 7 * DAY
        part = "train" if day.year <= 2014 else "calibration" if day.year <= 2016 else "evaluation"
        selected = []
        for asset in ASSETS:
            past = [x for x in features[asset] if x[0] < origin][-8:]
            future = [x for x in features[asset] if x[0] > origin][:8]
            if (date not in prices[asset] or end not in prices[asset] or len(past) < 8
                    or (part == "train" and (len(future) < 8 or future[-1][0] >= stamp('2015-01-01')))
                    or (part == "train" and maturity + 4*3600 >= stamp('2015-01-01'))
                    or (part == "calibration" and maturity + 4*3600 >= stamp('2017-01-01'))):
                break
            selected.append((asset, past, future))
        if len(selected) != len(ASSETS):
            excluded.append(date)
        else:
            target = parts[part]
            for asset, past, future in selected:
                ratio = prices[asset][end] / prices[asset][date]
                values = dict(assets=asset, origin_times=origin, maturity_times=maturity,
                              available_times=maturity+4*3600, window_times=[x[0] for x in past],
                              window_available=[x[0] for x in past], windows=[[x[1]] for x in past],
                              y=math.log(ratio), simple_returns=ratio-1)
                if part == "train":
                    values.update(future_times=[x[0] for x in future], future_available=[x[0] for x in future],
                                  future_windows=[[x[1]] for x in future])
                for key, value in values.items():
                    target[key].append(value)
        day += timedelta(days=7)
    payload = {"schema_version": "s6-windows-v1", "return_units": "log_return", "currency": "USD",
               "horizon_seconds": 7*DAY, "protected_start": stamp('2025-01-01'),
               "fit_cutoff": stamp('2015-01-01'), "asset_universe": list(ASSETS), "partitions": parts}
    path = root / "windows.json"
    write(path, payload)
    write(Path(str(path)+".manifest.json"), {"schema_version": "s6-manifest-v1", "kind": "admitted_development",
          "sha256": sha(path), "protected_start": stamp('2025-01-01'),
          "last_outcome_time": max(max(p["maturity_times"]) for p in parts.values()),
          "source": "Yahoo chart historical unadjusted daily ETF quotes; acquisition.json binds source URLs and identities.",
          "rights": "Local private research; redistribution rights unverified; no redistribution authorized by this manifest.",
          "point_in_time": "NOT CERTIFIED. Current-vintage retrospective quotes. Midnight after daily bar and Monday 16:00 UTC after outcome open are conservative research availability conventions, NOT observed vendor publication timestamps. No PIT or actual-fill claim."})
    write(root / "coverage.json", {"dates": {k: len(set(v["origin_times"])) for k,v in parts.items()},
          "excluded_mondays": excluded, "selection": "Both Monday endpoint opens and eight past observations required for every asset; train future targets purged at cutoff. Missing endpoints are excluded, never filled.",
          "limitations": "Fixed surviving ETFs; price returns exclude dividends; missing holiday weeks mean economic wealth covers included episodes only, not continuous calendar holding. Retrospective availability assumptions, no historical-vintage certification."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    if args.acquire:
        acquire(args.root)
    build(args.root)
