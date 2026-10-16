"""Predeclared summaries; no refits, policy selection or parameter search."""

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from successors.frem_s6.experiment import paired_date_interval
from successors.frem_s6_real.prepare import write


def read(path):
    return json.loads(path.read_text())


def path_summary(ledger):
    rows = ledger["periods"]
    ends = [r["nav_end"] for r in rows]
    ends[-1] = ledger["terminal_wealth"]
    factors = [v / r["nav_start"] for r, v in zip(rows, ends, strict=True)]
    peak, drawdown = 100000.0, 0.0
    for v in ends:
        peak = max(peak, v)
        drawdown = max(drawdown, 1 - v / peak)
    exposure = [1 - r["cash"] / (r["nav_start"] - r["fee"]) for r in rows]
    yearly = {}
    for year in (2020, 2021, 2022):
        selected = [
            f
            for r, f in zip(rows, factors, strict=True)
            if datetime.fromtimestamp(r["time"], UTC).year == year
        ]
        yearly[str(year)] = math.prod(selected) - 1
    return {
        "terminal_wealth": ledger["terminal_wealth"],
        "net_return": ledger["terminal_wealth"] / 100000 - 1,
        "fees": ledger["fees"],
        "max_drawdown": drawdown,
        "average_risky_weight": sum(exposure) / len(exposure),
        "cash_only_dates": sum(x < 1e-12 for x in exposure),
        "trade_dates": sum(r["notional"] > 1e-8 for r in rows),
        "traded_notional": sum(r["notional"] for r in rows) + sum(rows[-1]["holdings"].values()),
        "annual_carried_path_returns": yearly,
        "net_factors": factors,
    }


def main(root, output):
    scores, comparisons = read(root / "scores.json"), read(root / "comparisons.json")
    economics, benchmarks = read(root / "economics.json"), read(root / "benchmarks.json")
    summary = {
        v: {c: {p: path_summary(l) for p, l in policies.items()} for c, policies in costs.items()}
        for v, costs in economics.items()
    }
    bench = {v: {c: path_summary(l) for c, l in costs.items()} for v, costs in benchmarks.items()}
    primary = summary["s8"]["10"]["s8_h4"]
    economic_controls = {
        "equal_h4": summary["equal"]["10"]["s8_h4"],
        "s6_memory_h4": summary["memory"]["10"]["s8_h4"],
        "s8_legacy_cost_aware": summary["s8"]["10"]["cost_aware"],
        "equal_weight_assets": bench["equal_weight"]["10"],
    }
    economic_intervals = {}
    for name, control in economic_controls.items():
        diffs = np.log(primary["net_factors"]) - np.log(control["net_factors"])
        intervals = {}
        for block in (6, 12):
            result = paired_date_interval(diffs, block_dates=block)
            result["relative_terminal_wealth"] = primary["terminal_wealth"] / control["terminal_wealth"] - 1
            if "interval95" in result:
                result["conditional_relative_wealth_interval95"] = [
                    math.expm1(v * len(diffs)) for v in result["interval95"]
                ]
            intervals[str(block)] = result
        economic_intervals[name] = intervals
    predictive_support = all(
        comparisons[v][str(b)].get("interval95", [-math.inf])[0] > 0.01
        for v in ("equal", "memory")
        for b in (6, 12)
    )
    economic_point_win = all(
        primary["terminal_wealth"] > x["terminal_wealth"] for x in economic_controls.values()
    )
    mechanism_point = all(
        scores["s8"]["ensemble_nll"] < scores[x]["ensemble_nll"] for x in ("fixed_half", "s8_global")
    )
    result = {
        "prediction_practical_support_both_controls": predictive_support,
        "economic_point_win_all_four_controls": economic_point_win,
        "soft_update_and_locality_point_advantage": mechanism_point,
        "prediction": scores,
        "paired_prediction": comparisons,
        "economics": summary,
        "benchmarks": bench,
        "economic_primary_intervals": economic_intervals,
        "scientific_scope": "One frozen development matrix; no canonical confirmation or independent retraining.",
    }
    write(output / "SUMMARY.json", result)
    fields = [
        "forecast",
        "policy",
        "cost_bps",
        "nll",
        "terminal_wealth",
        "net_return",
        "fees",
        "max_drawdown",
        "average_risky_weight",
        "cash_only_dates",
        "trade_dates",
        "traded_notional",
        "return_2020",
        "return_2021",
        "return_2022",
    ]
    with (output / "ALL_BASELINES.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        families = [
            (v, c, p, stats, scores[v]["ensemble_nll"])
            for v, costs in summary.items()
            for c, policies in costs.items()
            for p, stats in policies.items()
        ]
        families += [
            (v, c, "benchmark", stats, "") for v, costs in bench.items() for c, stats in costs.items()
        ]
        for v, c, p, stats, nll in families:
            writer.writerow(
                {
                    "forecast": v,
                    "policy": p,
                    "cost_bps": c,
                    "nll": nll,
                    **{k: stats[k] for k in fields[4:12]},
                    **{
                        f"return_{y}": stats["annual_carried_path_returns"][str(y)]
                        for y in (2020, 2021, 2022)
                    },
                }
            )
    with (output / "ANNUAL_PREDICTION.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["forecast", "year", "nll", "equal_minus_forecast_nll", "memory_minus_forecast_nll"])
        for v, row in scores.items():
            for year, value in row["years"].items():
                writer.writerow(
                    [
                        v,
                        year,
                        value,
                        scores["equal"]["years"][year] - value,
                        scores["memory"]["years"][year] - value,
                    ]
                )
    lines = [
        "# S8 frozen development result",
        "",
        f"Prediction practical-support gate: **{predictive_support}**. Economic point-win gate against all four controls: **{economic_point_win}**.",
        "",
        "S7 and all prior results remain closed. This successor was specified and source-frozen before acquiring 2020–2022 ETF quotes; no 2019 data entered its fitting, features, gate updates or evaluation.",
        "",
        "## Why S7 rejected the gains and stayed in cash",
        "",
        "The largest 2018 routing gain was approximately 0.00568 nats. Both chronological half means were positive (about 0.0052 and 0.0062), but both were below the required 0.01. The gate followed its protocol; it treated the threshold for a substantial scientific claim as a threshold for permitting any routing. Positive development means alone did not establish a reliable advantage.",
        "",
        "The saved 2019 signal candidates had positive gross weekly utility on all 51 dates, but all net scores were negative after the two-way fee estimate. Their implied break-even one-way costs were only 0.15–5.41 bps, below the tested 10 bps. The 80% anchor had the same problem (maximum 5.61 bps). With cash as the initial state and no small-position candidates, cash remained the maximizer. This was a conservative one-week surrogate, not evidence the rally was predictable. No new policy or threshold was tested on 2019.",
        "",
        "## Successor and matched comparisons",
        "",
        "S8 softly mixes equal expert densities and the contextual mixture using only delayed mature scores. A fixed-share update retains recovery weight for both strategies. The allocation solves a continuous concave log-moment surrogate and commits holdings for four weekly intervals, with a one-interval ablation. It allows cash and partial entry, charges actual self-financing costs, and never imposes a risky anchor.",
        "",
        "All forecasts share three newly fitted S6 base models (seeds 11,29,47). Training ends before 2017; scale calibration uses 2017; gate initialization uses 2018. The primary evaluation is one continuous 2020–2022 path after the eight-return warmup. The S6 row is the unchanged S6 memory mechanism under the common data and fit contract, not its old 2017–2018 score.",
        "",
        "| Forecast | NLL, lower better | H4 wealth, 10 bps | H1 wealth | S7 cost-aware wealth | S6 signal wealth |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in [
        "s8",
        "equal",
        "memory",
        "s7",
        "fixed_half",
        "s8_global",
        "mixture_ungated",
        "mixture_global",
        "recent",
        "zero",
        "ridge",
        "trend",
        "adaptive_volatility",
        "reservoir_jepa",
        "xgboost_hist",
    ]:
        e = summary[v]["10"]
        lines.append(
            f"| {v} | {scores[v]['ensemble_nll']:.9f} | ${e['s8_h4']['terminal_wealth']:,.2f} | ${e['s8_h1']['terminal_wealth']:,.2f} | ${e['cost_aware']['terminal_wealth']:,.2f} | ${e['signal']['terminal_wealth']:,.2f} |"
        )
    lines += [
        "",
        "Asset benchmarks at 10 bps: "
        + ", ".join(
            f"{v} ${bench[v]['10']['terminal_wealth']:,.2f}" for v in ("cash", "buy_hold", "equal_weight")
        )
        + ".",
        "",
        "## Predeclared uncertainty and mechanism checks",
        "",
    ]
    for v in ("equal", "memory", "fixed_half", "s8_global"):
        lines.append(
            f"- {v} NLL minus S8: {comparisons[v]['6']['mean']:.9f}; 6-date 95% interval {comparisons[v]['6'].get('interval95')}; 12-date interval {comparisons[v]['12'].get('interval95')}."
        )
    for v, intervals in economic_intervals.items():
        lines.append(
            f"- S8 H4 relative wealth versus {v}: {intervals['6']['relative_terminal_wealth']:.6%}; conditional 6-date interval {intervals['6'].get('conditional_relative_wealth_interval95')}; 12-date interval {intervals['12'].get('conditional_relative_wealth_interval95')}."
        )
    lines += [
        "",
        f"Soft-update and locality ablations both beaten at the point estimate: **{mechanism_point}**. No favorable arm replaces S8 as the primary.",
        "",
        "## Economics and year decomposition",
        "",
        f"S8 H4 at 10 bps: mean risky weight {primary['average_risky_weight']:.2%}, {primary['cash_only_dates']} cash-only dates, {primary['trade_dates']} trade dates, fees ${primary['fees']:,.2f}, maximum drawdown {primary['max_drawdown']:.2%}.",
        "",
        "| Origin year | S8 NLL | Fixed-average NLL | S6 memory NLL | S8 H4 net return | Equal-asset net return |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for year in ("2020", "2021", "2022"):
        lines.append(
            f"| {year} | {scores['s8']['years'][year]:.9f} | {scores['equal']['years'][year]:.9f} | {scores['memory']['years'][year]:.9f} | {primary['annual_carried_path_returns'][year]:.3%} | {bench['equal_weight']['10']['annual_carried_path_returns'][year]:.3%} |"
        )
    lines += [
        "",
        "Year rows decompose the same carried path by forecast origin; they are not reset portfolios or selected subgroup tests. Final liquidation belongs to the last interval. All 0/10/20/40 bps outcomes and every crossed baseline are retained in ALL_BASELINES.csv; seed and year NLLs are in SUMMARY.json and ANNUAL_PREDICTION.csv.",
        "",
        "## Evidence limits and preservation",
        "",
        "New to this ETF successor lineage does not mean globally unseen: earlier research already used 2020/2022 on other assets. These are current-vintage survivor ETF histories, with assumed availability, ex-date dividend accrual, weekly reinvestment, zero cash yield and no actual fills. The four-week policy assumes persistent log-moment signals and diagonal risk; its surrogate is not a calibrated expected return. Conditional date-block intervals do not simulate refitting or policy reoptimization. Seeds are not independent market histories.",
        "",
        "The scientific 0.01-nat threshold was preserved. The study stops after the frozen matrix, regardless of sign. Original 2025+ protected outcomes remain untouched. No funding certification, novelty, canonical confirmation, external publication or independent reproduction is established.",
        "",
        "See AUDIT.json for the separate saved-output arithmetic, raw-quote/feature chronology and preservation audit; completion.json and resources.json under the output directory for execution. Earlier failed test collection and calendar fixture checks remain in the research directory with their source snapshots. Every market seed, forecast, checkpoint, gate state, ledger and data acquisition receipt is retained.",
    ]
    with (output / "REPORT.md").open("x") as f:
        f.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "prediction_practical_support": predictive_support,
                "economic_point_win": economic_point_win,
                "primary": {k: v for k, v in primary.items() if k != "net_factors"},
                "nll": {v: scores[v]["ensemble_nll"] for v in ("s8", "equal", "memory")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
