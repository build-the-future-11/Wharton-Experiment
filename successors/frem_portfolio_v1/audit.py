"""Independent arithmetic verifier of saved holdings against raw forecast outcomes.

Does not import the portfolio engine or regenerate forecasts/positions. Verifies
trade costs, cash, security values, NAV continuity, terminal liquidation and all
comparison ratios from saved evidence. This is not independent model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(output):
    frozen = json.loads((output / "freeze.json").read_text())
    for name, expected in frozen["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    prediction_audit = json.loads((output / "forecast_audit.json").read_text())
    for item in prediction_audit:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("Prediction identity changed")
    config = frozen["specification"]
    sources, maximum, count, observations = {}, 0.0, 0, 0

    def close(actual, expected):
        nonlocal maximum
        delta = abs(actual - expected)
        maximum = max(maximum, delta)
        if not math.isfinite(delta) or delta > 1e-7 * max(1.0, abs(expected)):
            raise ArithmeticError(f"Accounting mismatch {actual} != {expected}")

    def outcomes(lineage, fold):
        key = lineage, fold
        if key in sources:
            return sources[key]
        base = ROOT / config["existing_model_runs"][lineage]
        if lineage == "S4":
            path = base / fold / "11/off.csv"
        elif lineage == "S3":
            path = base / "public" / fold / "11/frem/off.csv"
        else:
            path = base / fold / "frem/predictions.csv"
        values = {}
        for row in csv.DictReader(path.open()):
            # Independent route: target month follows decision month. Adapter
            # uses the month before label maturity; those routes must coincide.
            decision = datetime.fromtimestamp(int(row["time"]) / 1e9, UTC)
            year, month = (
                (decision.year + 1, 1) if decision.month == 12 else (decision.year, decision.month + 1)
            )
            entry = int(datetime(year, month, 1, tzinfo=UTC).timestamp())
            values[entry, row["asset"]] = math.expm1(float(row["actual"]))
        sources[key] = values
        return values

    terminals = {}
    for path in sorted((output / "ledgers").glob("*.json")):
        tokens = path.stem.split("__")
        lineage, fold = tokens[:2]
        bps = int(tokens[3] if len(tokens) == 5 else tokens[-1])
        fee = bps / 10000
        ledger = json.loads(path.read_text())
        realised = outcomes(lineage, fold)
        prior_holdings = dict.fromkeys(ledger["periods"][0]["holdings"], 0.0)
        prior_cash, prior_nav = 100000.0, 100000.0
        total_fee = 0.0
        for row in ledger["periods"]:
            observations += 1
            close(row["nav_start"], prior_nav)
            after_trade = prior_nav - row["fee"]
            bought = {a: row["weights"][a] * after_trade for a in prior_holdings}
            notional = sum(abs(bought[a] - prior_holdings[a]) for a in bought)
            close(row["notional"], notional)
            close(row["fee"], fee * notional)
            cash = prior_cash - sum(bought[a] - prior_holdings[a] for a in bought) - row["fee"]
            close(row["cash"], cash)
            holdings = {a: value * (1 + realised[row["time"], a]) for a, value in bought.items()}
            for asset, value in holdings.items():
                close(row["holdings"][asset], value)
            liquidation = row.get("liquidation_fee", 0.0)
            if "liquidation_fee" in row:
                close(liquidation, fee * sum(holdings.values()))
            nav = cash + sum(holdings.values()) - liquidation
            close(row["nav_end"], nav)
            close(row["net_factor"], nav / prior_nav if prior_nav else 1.0)
            total_fee += row["fee"] + liquidation
            prior_holdings, prior_cash, prior_nav = holdings, cash, nav
        close(ledger["terminal_wealth"], prior_nav)
        close(ledger["fees"], total_fee)
        terminals[path.stem] = prior_nav
        count += 1
    comparisons = list(csv.DictReader((output / "comparisons.csv").open()))
    for row in comparisons:
        key = "__".join(row[k] for k in ("lineage", "fold", "market", "model", "arm", "seed", "cost_bps"))
        control = "__".join(row[k] for k in ("lineage", "fold", "market", "cost_bps"))
        close(float(row["terminal_wealth"]), terminals[key])
        for name in ("equal_weight", "buy_hold", "cash"):
            close(float(row[f"vs_{name}"]), terminals[key] / terminals[control + "__" + name] - 1)
        if row["vs_memory_off"]:
            off = "__".join(
                [
                    row["lineage"],
                    row["fold"],
                    row["market"],
                    row["model"],
                    "off",
                    row["seed"],
                    row["cost_bps"],
                ]
            )
            close(float(row["vs_memory_off"]), terminals[key] / terminals[off] - 1)
    return {
        "status": "PASS_INDEPENDENT_ACCOUNTING_RECONSTRUCTION",
        "ledgers": count,
        "periods": observations,
        "comparisons": len(comparisons),
        "max_absolute_arithmetic_residual": maximum,
        "new_fits": 0,
        "independent_training_reproduction": False,
        "source_files": len(frozen["sources"]),
        "prediction_files": len(prediction_audit),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = audit(args.run.resolve())
    with args.output.open("x") as file:
        json.dump(receipt, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(receipt))
