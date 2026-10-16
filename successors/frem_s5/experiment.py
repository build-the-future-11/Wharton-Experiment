"""Frozen lightweight S5 replay. Retained public evidence, never fresh confirmation."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics as st
from datetime import UTC, datetime
from pathlib import Path

from successors.frem_portfolio_v1.engine import Forecast, backtest, paired_interval, rebalance
from successors.frem_portfolio_v1.evidence import periods, sha256

from .router import Episode, Router, mixture_nll, nll

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/frem_s3_20261011_attempt1/public"
EXPERTS = ("zero", "ridge", "frem", "no_jepa")
FOLDS = ("fresh_2023", "fresh_2024")
SEEDS = (11, 29, 47)


def inputs():
    return [
        SOURCE / f / str(s) / m / name
        for f in FOLDS
        for s in SEEDS
        for m in EXPERTS
        for name in ("validation_base.csv", "off.csv", "fit.json", "result.json")
    ]


def read(path):
    with path.open() as f:
        data = list(csv.DictReader(f))
    result = {}
    for r in data:
        t, maturity = int(r["time"]) // 10**9, int(r["label_time"]) // 10**9
        y, mu, scale, df = (float(r[k]) for k in ("actual", "prediction", "scale", "student_df"))
        loss = nll(y, mu, scale, df)
        if r.get("nll") and abs(loss - float(r["nll"])) > 1e-10:
            raise ValueError("Saved NLL mismatch")
        if maturity <= t or datetime.fromtimestamp(maturity, UTC).year >= 2025:
            raise ValueError("Invalid maturity or protected outcome")
        key = t, r["asset"]
        if key in result:
            raise ValueError("Duplicate forecast")
        result[key] = {
            "time": t,
            "maturity": maturity,
            "asset": r["asset"],
            "actual": y,
            "mu": mu,
            "scale": scale,
            "df": df,
            "loss": loss,
        }
    if not result:
        raise ValueError("Empty forecasts")
    return result


def load(fold, seed, name):
    tables = []
    for expert in EXPERTS:
        cell = SOURCE / fold / str(seed) / expert
        if name == "off.csv":
            receipt = json.loads((cell / "result.json").read_text())["results"]["off"]
            if sha256(cell / name) != receipt["sha256"]:
                raise ValueError("Original expert hash mismatch")
        table = read(cell / name)
        fitted = json.loads((cell / "fit.json").read_text())
        if max(fitted["partition_max_label"]) // 10**9 >= min(t for t, _ in table):
            raise ValueError("Expert fitting overlaps forecast issuance")
        if tables and set(table) != set(tables[0]):
            raise ValueError("Incomplete expert grid")
        tables.append(table)
    dates = sorted({t for t, a in tables[0]})
    names = sorted({a for t, a in tables[0]})
    batches = []
    for t in dates:
        assets = []
        for asset in names:
            rr = [table[t, asset] for table in tables]
            if len({(r["actual"], r["maturity"]) for r in rr}) != 1:
                raise ValueError("Unpaired outcomes/horizons")
            assets.append(rr)
        # Common date-level context uses only issued forecasts, never outcomes.
        mus = [r["mu"] for rr in assets for r in rr]
        scales = [r["scale"] for rr in assets for r in rr]
        raw = (
            st.fmean(mus) / 0.05,
            math.log(st.fmean(scales) / 0.05),
            math.sqrt(st.fmean((m - st.fmean(mus)) ** 2 for m in mus)) / 0.05,
        )
        batches.append(
            {
                "time": t,
                "maturity": max(r["maturity"] for rr in assets for r in rr),
                "assets": assets,
                "raw_context": raw,
            }
        )
    return batches


def scaler(batches):
    center = [st.fmean(b["raw_context"][j] for b in batches) for j in range(3)]
    unit = [
        max(0.1, math.sqrt(st.fmean((b["raw_context"][j] - center[j]) ** 2 for b in batches)))
        for j in range(3)
    ]
    return center, unit


def transform(batch, normalizer):
    center, unit = normalizer
    return tuple(max(-5, min(5, (x - c) / s)) for x, c, s in zip(batch["raw_context"], center, unit))


def advance(router, batch, context):
    weights, diagnostic = router.weights(batch["time"], context)
    # All contemporaneous assets predicted before any outcome becomes feedback.
    losses = [st.fmean(rr[j]["loss"] for rr in batch["assets"]) for j in range(4)]
    scored = st.fmean(mixture_nll([r["loss"] for r in rr], weights) for rr in batch["assets"])
    router.observe(Episode(batch["time"], batch["maturity"], context, tuple(losses)))
    return weights, diagnostic, scored


def selection(validation, mode, seed):
    boundary = len(validation) // 2
    if boundary < 6:
        raise ValueError("At least twelve validation dates required")
    normalizer = scaler(validation[:boundary])
    candidates = []
    for temperature in (1.0,) if mode == "equal" else (0.5, 1.0):
        router = Router(mode, temperature, seed)
        scores = []
        for i, batch in enumerate(validation):
            _, _, score = advance(router, batch, transform(batch, normalizer))
            if i >= boundary:
                scores.append(score)
        candidates.append((st.fmean(scores), -temperature, router))
    chosen = min(candidates, key=lambda c: (c[0], c[1]))
    return (
        chosen[2],
        normalizer,
        {
            "candidates": [{"temperature": -t, "nll": s} for s, t, r in candidates],
            "selected_temperature": -chosen[1],
            "selection_nll": chosen[0],
            "first_selection_date": validation[boundary]["time"],
            "center": normalizer[0],
            "unit": normalizer[1],
        },
    )


def market_cell(fold, seed):
    validation, evaluation = load(fold, seed, "validation_base.csv"), load(fold, seed, "off.csv")
    if validation[-1]["time"] >= evaluation[0]["time"]:
        raise ValueError("Validation/evaluation overlap")
    if max(b["maturity"] for b in validation) >= evaluation[0]["time"]:
        raise ValueError("Temperature selection uses unmatured validation outcomes")
    results = {}
    for mode in (*Router.MODES, *EXPERTS):
        if mode in Router.MODES:
            router, normalizer, selected = selection(validation, mode, seed)
        else:
            selected = {"fixed_expert": mode}
        scores, rows, trace = [], [], []
        for batch in evaluation:
            if mode in Router.MODES:
                w, diag, score = advance(router, batch, transform(batch, normalizer))
            else:
                w = [float(expert == mode) for expert in EXPERTS]
                diag = {}
                score = st.fmean(rr[EXPERTS.index(mode)]["loss"] for rr in batch["assets"])
            scores.append(score)
            trace.append({"time": batch["time"], "weights": w, "diagnostic": diag, "nll": score})
            for rr in batch["assets"]:
                mu = sum(a * r["mu"] for a, r in zip(w, rr))
                variance = sum(
                    a * (r["scale"] ** 2 * r["df"] / (r["df"] - 2) + (r["mu"] - mu) ** 2)
                    for a, r in zip(w, rr)
                )
                r = rr[0]
                maturity = datetime.fromtimestamp(r["maturity"], UTC)
                end = datetime(maturity.year, maturity.month, 1, tzinfo=UTC)
                start = (
                    datetime(end.year - 1, 12, 1, tzinfo=UTC)
                    if end.month == 1
                    else datetime(end.year, end.month - 1, 1, tzinfo=UTC)
                )
                rows.append(
                    {
                        "time": int(start.timestamp()),
                        "end": int(end.timestamp()),
                        "available_at": r["time"],
                        "asset": r["asset"],
                        "return_value": math.expm1(r["actual"]),
                        "forecast": Forecast(r["asset"], mu, variance),
                    }
                )
        for row in rows:
            row["return"] = row.pop("return_value")
        ps = periods(rows, "USD")
        results[mode] = {
            "nll": st.fmean(scores),
            "date_nll": scores,
            "selection": selected,
            "trace": trace,
            "periods": ps,
            "final_state": router.state() if mode in Router.MODES else None,
        }
    return results


def volatility_ledger(ps, cost):
    """Equal basket scaled to 10% annual marginal-vol proxy, capped at full investment.

    Uses zero expert's available variance, identical weights for every asset.
    No covariance diversification credit, no claim of calibrated portfolio risk.
    """
    names = [f.asset for f in ps[0]["forecasts"]]
    holdings, cash = dict.fromkeys(names, 0.0), 100000.0
    rows, fees, peak, drawdown = [], 0.0, 100000.0, 0.0
    fee = cost / 10000
    for p in ps:
        before = sum(holdings.values()) + cash
        vol = math.sqrt(12 * st.fmean(f.variance for f in p["forecasts"]))
        exposure = min(1, 0.10 / max(vol, 1e-8))
        h, c, tr = rebalance(holdings, cash, {k: exposure / len(names) for k in names}, fee)
        holdings = {k: h[k] * (1 + p["returns"][k]) for k in names}
        cash = c
        after = sum(holdings.values()) + cash
        fees += tr["fee"]
        peak = max(peak, after)
        drawdown = max(drawdown, 1 - after / peak)
        rows.append(
            {
                "time": p["time"],
                "end": p["end"],
                "nav_start": before,
                "nav_end": after,
                "net_factor": after / before,
                "fee": tr["fee"],
                "holdings": holdings.copy(),
                "cash": cash,
            }
        )
    liquidation = fee * sum(holdings.values())
    terminal = sum(holdings.values()) + cash - liquidation
    rows[-1]["nav_end"] = terminal
    rows[-1]["net_factor"] = terminal / rows[-1]["nav_start"]
    rows[-1]["liquidation_fee"] = liquidation
    return {
        "terminal_wealth": terminal,
        "fees": fees + liquidation,
        "max_drawdown": max(drawdown, 1 - terminal / peak),
        "periods": rows,
    }


def portfolio(cells):
    output = {}
    for cost in (10, 20, 40):
        ledgers = {name: backtest(record["periods"], cost_bps=cost) for name, record in cells.items()}
        base = cells["equal"]["periods"]
        for name in ("equal_weight", "buy_hold", "cash"):
            ledgers[name] = backtest(base, policy=name, cost_bps=cost)
        ledgers["volatility_managed"] = volatility_ledger(cells["zero"]["periods"], cost)
        candidate = ledgers["context"]
        comparisons = {
            name: {
                "relative_wealth": candidate["terminal_wealth"] / r["terminal_wealth"] - 1,
                "intervals": [
                    paired_interval(
                        [p["net_factor"] for p in candidate["periods"]],
                        [p["net_factor"] for p in r["periods"]],
                        block=b,
                    )
                    for b in (3, 6)
                ],
            }
            for name, r in ledgers.items()
            if name != "context"
        }
        output[str(cost)] = {"ledgers": ledgers, "comparisons": comparisons}
    return output


def mechanisms():
    records = []
    for family in ("null", "recurrence", "abrupt", "delayed"):
        for seed in (101, 211, 307):
            rng = random.Random(seed)
            routers = {m: Router(m, 1.0, seed) for m in Router.MODES}
            scores = {m: [] for m in routers}
            for i in range(240):
                regime = (-1 if (i // 8) % 2 == 0 else 1) if family != "abrupt" else (-1 if i < 120 else 1)
                location = 0 if family == "null" else 2 * regime
                y = location + rng.gauss(0, 0.3 if family != "null" else 1)
                means = (-0.2, 0, 0.2, 0.1) if family == "null" else (-2, 2, 0, 0.5)
                scale = 1 if family == "null" else 0.3
                losses = tuple(nll(y, mu, scale, 5) for mu in means)
                t = 86400 * 30 * (i + 1)
                delay = 5 if family == "delayed" else 1
                for mode, router in routers.items():
                    w, _ = router.weights(t, (float(regime),))
                    if i >= 120:
                        scores[mode].append(mixture_nll(losses, w))
                    router.observe(Episode(t, t + delay * 30 * 86400, (float(regime),), losses))
            records.append(
                {"family": family, "seed": seed, "nll": {m: st.fmean(v) for m, v in scores.items()}}
            )
    means = {
        f: {m: st.fmean(r["nll"][m] for r in records if r["family"] == f) for m in Router.MODES}
        for f in ("null", "recurrence", "abrupt", "delayed")
    }
    gate = means["null"]["context"] - means["null"]["equal"] <= 0.01 and all(
        means["recurrence"][m] - means["recurrence"]["context"] > 0 for m in ("equal", "recent", "random")
    )
    return {
        "records": records,
        "means": means,
        "gate_passed": gate,
        "interpretation": "Generated prescribed signals only; not evidence of market predictability",
    }
