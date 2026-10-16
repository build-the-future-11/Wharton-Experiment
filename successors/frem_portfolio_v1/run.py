"""Opt-in, immutable, standard-library reanalysis of already retained forecasts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from .engine import Forecast, backtest, laura_path, paired_interval, relative_wealth
from .evidence import load, load_cases, sha256, write

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def identity(case):
    return "__".join(str(case[k]) for k in ("lineage", "fold", "market", "model", "arm", "seed"))


def stress(spec):
    result = []
    for name, returns in spec["stress_paths"].items():
        paths = []
        for index in range(12):
            values = (
                {"A": 0.0, "B": 0.0}
                if name == "single_default"
                else {"A": returns[index], "B": returns[index]}
            )
            if name == "single_default" and index == 0:
                values["A"] = -1.0
            paths.append(
                {
                    "time": index,
                    "end": index + 1,
                    "currency": "USD",
                    "forecasts": [Forecast("A", 0.0, 0.04), Forecast("B", 0.0, 0.04)],
                    "returns": values,
                }
            )
        for bps in spec["policy"]["cost_bps"]:
            candidate = backtest(paths, cost_bps=bps)
            benchmark = backtest(paths, cost_bps=bps, policy="equal_weight")
            result.append(
                {
                    "scenario": name,
                    "cost_bps": bps,
                    "candidate_wealth": candidate["terminal_wealth"],
                    "benchmark_wealth": benchmark["terminal_wealth"],
                    "relative_wealth": relative_wealth(
                        candidate["terminal_wealth"], benchmark["terminal_wealth"]
                    ),
                    "kind": "CONSTRUCTED_ACCOUNTING_COUNTEREXAMPLE_NOT_MODEL_PERFORMANCE",
                }
            )
    return result


def render(summary, destination):
    primary = summary["primary_cases"]
    lines = [
        "# FREM portfolio evidence — 12 October 2026",
        "",
        (
            f"**{summary['verdict']}**. The requested 6% minimum is an ending-wealth target. "
            "A positive result in one historical path is not a universal floor or a live investment recommendation."
        ),
        "",
        "## What was executed",
        "",
        (
            f"{summary['forecast_files']} source-hashed forecast files; {summary['forecast_rows']} independently audited rows; "
            f"{summary['completed_policy_cases']} model/seed/window/market/cost cases; "
            f"{summary['ledgers_written']} shared-accounting ledgers. No new fits or protected observations."
        ),
        "",
        (
            "All models used the fixed 80% equal-weight anchor plus up to 20% positive log-signal sleeve. "
            "Unallocated weight stays in zero-return cash. This is a declared unoptimized research policy, not an optimized Laura mandate. "
            "Costs, initial cash and final liquidation are shared; drifted holdings are carried between periods. "
            "The same-basket equal-weight, buy-and-hold and cash controls are retained."
        ),
        "",
        "## Primary: S4 corrected distance versus equal weight",
        "",
        "| Window | Seed | Cost per traded dollar | Uplift | Versus memory off | Max drawdown |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['fold']} | {row['seed']} | {row['cost_bps']} bp | {row['vs_equal_weight']:.2%} | "
            f"{row['vs_memory_off']:.2%} | {row['max_drawdown']:.2%} |"
        )
    lines += [
        "",
        (
            f"Observed range across all {len(primary)} primary cases: "
            f"{summary['primary_range'][0]:.2%} to {summary['primary_range'][1]:.2%}. "
            f"Cases at least 6%: {summary['primary_at_least_6pct']}/{len(primary)}. "
            "Seed outcomes share the same market history and are not independent financial repetitions."
        ),
        "",
        (
            "Intervals in summary.json use predeclared 3- and 6-month paired circular blocks of saved net-wealth increments. "
            "These intervals condition on the realized policy path, omit retraining uncertainty, are not simultaneous, "
            "and cannot certify future coverage. Short windows remain explicitly insufficient for a block length."
        ),
        "",
        "## Across time, currencies and predecessor versions",
        "",
        (
            "All six older crisis/recovery windows and four methods are in comparisons.csv. Each listing-market basket "
            "is separate; currencies are never combined. Two-stock local baskets are narrow accounting illustrations. "
            "S3/S4 forecast source histories are exploratory, retrospective survivor samples with adjusted-price proxies. "
            "Changing checkpoints and training chronology prevents a causal attribution of cross-version portfolio differences."
        ),
        "",
        (
            f"Incomplete declared market/model groups: {len(summary['incomplete_cases'])}. Each is retained in summary.json; "
            "missing prices/assets are not backfilled and incomplete groups do not become passing cases."
        ),
        "",
        "## Why an all-scenario 6% floor fails",
        "",
        (
            "In a zero-return world with zero costs and identical cash flows, a long-only self-financing strategy and "
            "the benchmark both end at their starting wealth. Uplift is exactly zero. No retriever, JEPA loss or "
            "backtest can change this accounting identity. Constructed stress counterexamples are in stress.json; "
            "they test the claim, not model skill."
        ),
        "",
        "## Laura Gao",
        "",
        (
            "Use $300,000 at the beginning of 2027 and $150,000 at the beginning of 2028. Before the first 2033 "
            "payment, establish a reserve for ten $50,000 beginning-year payments through 2042. "
            "The reserve calculation includes the immediate payment (annuity due). At zero growth and zero reserve "
            "discount, wealth is $450,000 against $500,000 of obligations: a $50,000 shortfall. "
            "WInS profits are not added to these contributions. The scenario grid in laura.json uses assumed rates, "
            "not observed yields or calibrated funding probabilities. Facility capacity remains null without an explicit flexibility buffer."
        ),
        "",
        (
            "Current research direction and sourced instrument candidates are in LAURA_CURRENT_RESEARCH.md. "
            "No certified ticker allocation can be produced without current eligible instruments, holdings, prices, "
            "risk/funding policy and admissible forecast inputs. The external quant/ML framework is an explicit dated "
            "input contract in engine.external_forecasts; its source has not yet been identified or integrated."
        ),
        "",
        "## Verification and reproducibility",
        "",
        (
            "The source/config snapshot is in freeze.json, per-file reconstruction in forecast_audit.json, "
            "per-period holdings/cash/costs in ledgers/, and every comparison in comparisons.csv. "
            "The runner refuses existing output directories. Earlier negative experiments and S4 freeze files are preserved."
        ),
        "",
        (
            "These are exploratory reanalyses and engineering tests, not independent training reproduction, "
            "point-in-time market certification, proof of profitability, live fills or public submission readiness."
        ),
    ]
    with destination.open("x") as file:
        file.write("\n".join(lines) + "\n")


def run(output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    try:
        spec = load(PACKAGE / "specification.json")
        sources = {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted(PACKAGE.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts
        }
        write(
            output / "freeze.json",
            {
                "created_at": datetime.now(UTC).isoformat(),
                "specification": spec,
                "sources": sources,
                "scope": "Development reanalysis declaration, not a confirmatory freeze",
            },
        )
        cases, audit, failures = load_cases(ROOT, spec)
        write(output / "forecast_audit.json", audit)
        results, comparisons, ledger_count = {}, [], 0
        controls = {}
        for case in cases:
            case_id = identity(case)
            for bps in spec["policy"]["cost_bps"]:
                if time.monotonic() - started > spec["execution"]["max_wall_seconds"]:
                    raise TimeoutError("Reanalysis budget exhausted")
                candidate = backtest(case["periods"], cost_bps=bps)
                result_key = f"{case_id}__{bps}"
                write(output / "ledgers" / f"{result_key}.json", candidate)
                ledger_count += 1
                results[result_key] = {k: v for k, v in candidate.items() if k != "periods"}
                results[result_key]["factors"] = [r["net_factor"] for r in candidate["periods"]]
                # Control identity includes actual horizon/universe; never mix S3 and local baskets.
                reference_key = (case["lineage"], case["fold"], case["market"], bps)
                if reference_key not in controls:
                    controls[reference_key] = {}
                    for policy in ("equal_weight", "buy_hold", "cash"):
                        value = backtest(case["periods"], policy=policy, cost_bps=bps)
                        controls[reference_key][policy] = value
                        key = "__".join(map(str, reference_key)) + "__" + policy
                        write(output / "ledgers" / f"{key}.json", value)
                        ledger_count += 1
                control = controls[reference_key]
                row = {k: v for k, v in case.items() if k != "periods"}
                row.update(
                    cost_bps=bps,
                    currency=candidate["currency"],
                    terminal_wealth=candidate["terminal_wealth"],
                    max_drawdown=candidate["max_drawdown"],
                    fees=candidate["fees"],
                    vs_equal_weight=relative_wealth(
                        candidate["terminal_wealth"], control["equal_weight"]["terminal_wealth"]
                    ),
                    vs_buy_hold=relative_wealth(
                        candidate["terminal_wealth"], control["buy_hold"]["terminal_wealth"]
                    ),
                    vs_cash=relative_wealth(candidate["terminal_wealth"], control["cash"]["terminal_wealth"]),
                )
                comparisons.append(row)
        for row in comparisons:
            if row["lineage"] in {"S3", "S4"} and row["model"] == "frem":
                ref = {**row, "arm": "off"}
                control = results[f"{identity(ref)}__{row['cost_bps']}"]
                row["vs_memory_off"] = relative_wealth(row["terminal_wealth"], control["terminal_wealth"])
            else:
                row["vs_memory_off"] = None
        primary = [r for r in comparisons if r["lineage"] == "S4" and r["arm"] == "euclidean_distance"]
        expected = len(spec["primary_folds"]) * (len(spec["seeds"]) + 1) * len(spec["policy"]["cost_bps"])
        if len(primary) != expected:
            raise ValueError("Incomplete primary case family")
        for row in primary:
            if row["seed"] != "ensemble":
                continue
            candidate = results[f"{identity(row)}__{row['cost_bps']}"]["factors"]
            ref = controls[(row["lineage"], row["fold"], row["market"], row["cost_bps"])]["equal_weight"]
            benchmark = [p["net_factor"] for p in ref["periods"]]
            row["intervals"] = [
                paired_interval(
                    candidate,
                    benchmark,
                    block=b,
                    repetitions=spec["statistics"]["resamples"],
                    seed=spec["statistics"]["seed"],
                )
                for b in spec["statistics"]["paired_circular_blocks"]
            ]
        stress_cases = stress(spec)
        write(output / "stress.json", stress_cases)
        funding = [
            {"assumed_growth": g, **laura_path([g] * 6, reserve_discount=d)}
            for g in spec["laura"]["growth_sensitivities"]
            for d in spec["laura"]["reserve_discount_sensitivities"]
        ]
        write(output / "laura.json", funding)
        fields = [k for k in comparisons[0] if k != "intervals"]
        with (output / "comparisons.csv").open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(comparisons)
        values = [r["vs_equal_weight"] for r in primary]
        passed = all(v >= spec["target"]["minimum"] for v in values)
        summary = {
            "status": "COMPLETE_EXPLORATORY_REANALYSIS",
            "verdict": "FINITE_PRIMARY_TARGET_OBSERVED_NOT_CONFIRMED"
            if passed
            else "REQUESTED_ALL_CASE_6PCT_TARGET_NOT_MET",
            "primary_range": [min(values), max(values)],
            "primary_at_least_6pct": sum(v >= 0.06 for v in values),
            "primary_cases": primary,
            "completed_policy_cases": len(comparisons),
            "ledgers_written": ledger_count,
            "incomplete_cases": failures,
            "forecast_files": len(audit),
            "forecast_rows": sum(r["rows"] for r in audit),
            "max_nll_reconstruction_error": max(r["max_nll_error"] for r in audit),
            "universal_6pct_guarantee": False,
            "new_model_fits": 0,
            "protected_outcomes_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (2**20 if platform.system() == "Darwin" else 1024),
            "framework_source_integrated": False,
            "current_allocation_status": "REQUIRES_INPUTS",
        }
        if summary["peak_rss_mib"] > spec["execution"]["process_rss_ceiling_mib"]:
            raise MemoryError("Standard-library reanalysis exceeded declared memory ceiling")
        if any(sha256(ROOT / name) != value for name, value in sources.items()):
            raise ValueError("Implementation changed during reanalysis")
        if any(sha256(row["path"]) != row["sha256"] for row in audit):
            raise ValueError("Prediction source changed during reanalysis")
        write(output / "summary.json", summary)
        render(summary, output / "REPORT.md")
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "status",
                        "verdict",
                        "primary_range",
                        "primary_at_least_6pct",
                        "completed_policy_cases",
                        "peak_rss_mib",
                        "elapsed_seconds",
                    )
                }
            )
        )
    except Exception as error:
        write(
            output / "failure.json",
            {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute required; this command performs a portfolio reanalysis")
    run(args.output.resolve())
