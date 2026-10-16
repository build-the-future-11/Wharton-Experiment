"""Separate arithmetic reconstruction of saved S5 scores and holdings ledgers."""

import argparse
import json
import math
from pathlib import Path

from successors.frem_portfolio_v1.evidence import sha256, write

from .experiment import EXPERTS, FOLDS, ROOT, SEEDS, SOURCE, read


def audit(out):
    freeze = json.loads((out / "freeze.json").read_text())
    for path, digest in freeze["identities"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Changed frozen file: {path}")
    nll_error, account_error, score_rows, ledger_count = 0.0, 0.0, 0, 0
    for fold in FOLDS:
        for seed in (*SEEDS, "ensemble"):
            forecasts = json.loads((out / fold / str(seed) / "forecasts.json").read_text())
            portfolio = json.loads((out / fold / str(seed) / "portfolio.json").read_text())
            if seed != "ensemble":
                tables = [read(SOURCE / fold / str(seed) / m / "off.csv") for m in EXPERTS]
                for mode, record in forecasts.items():
                    for event in record["trace"]:
                        t = event["time"]
                        w = event["weights"]
                        diag = event["diagnostic"]
                        if abs(sum(w) - 1) > 1e-12 or any(v < 0 for v in w):
                            raise ValueError("Invalid mixture weights")
                        if diag.get("latest_maturity", -1) >= t:
                            raise ValueError("Future episode used")
                        losses = []
                        for time, asset in sorted(tables[0]):
                            if time != t:
                                continue
                            rr = [table[time, asset] for table in tables]
                            density = sum(weight * math.exp(-row["loss"]) for weight, row in zip(w, rr))
                            losses.append(-math.log(density))
                            score_rows += 1
                        nll_error = max(nll_error, abs(sum(losses) / len(losses) - event["nll"]))
            for cost, cell in portfolio.items():
                for mode, ledger in cell["ledgers"].items():
                    source = (
                        "zero" if mode == "volatility_managed" else mode if mode in forecasts else "equal"
                    )
                    periods = forecasts[source]["periods"]
                    holdings = {k: 0.0 for k in periods[0]["returns"]}
                    cash = 100000.0
                    fee_rate = int(cost) / 10000
                    if len(periods) != len(ledger["periods"]):
                        raise ValueError("Accounting period mismatch")
                    for p, row in zip(periods, ledger["periods"]):
                        if (p["time"], p["end"]) != (row["time"], row["end"]):
                            raise ValueError("Accounting horizon mismatch")
                        before = sum(holdings.values()) + cash
                        pre_return = {k: row["holdings"][k] / (1 + p["returns"][k]) for k in holdings}
                        notional = sum(abs(pre_return[k] - holdings[k]) for k in holdings)
                        computed_fee = fee_rate * notional
                        residuals = [
                            before - row["nav_start"],
                            computed_fee - row["fee"],
                            before - computed_fee - sum(pre_return.values()) - row["cash"],
                            sum(row["holdings"].values())
                            + row["cash"]
                            - row.get("liquidation_fee", 0)
                            - row["nav_end"],
                            row["net_factor"] - row["nav_end"] / row["nav_start"],
                        ]
                        account_error = max(account_error, *map(abs, residuals))
                        holdings = row["holdings"]
                        cash = row["cash"]
                    account_error = max(
                        account_error, abs(ledger["terminal_wealth"] - ledger["periods"][-1]["nav_end"])
                    )
                    ledger_count += 1
                candidate = cell["ledgers"]["context"]["terminal_wealth"]
                for mode, comparison in cell["comparisons"].items():
                    expected = candidate / cell["ledgers"][mode]["terminal_wealth"] - 1
                    if abs(expected - comparison["relative_wealth"]) > 1e-12:
                        raise ValueError("Wrong benchmark ratio")
    if nll_error > 1e-10 or account_error > 1e-6:
        raise ValueError(f"Arithmetic mismatch: {nll_error}, {account_error}")
    return {
        "status": "PASS_ARITHMETIC_AND_FROZEN_IDENTITIES",
        "score_rows": score_rows,
        "ledgers": ledger_count,
        "max_nll_error": nll_error,
        "max_accounting_residual": account_error,
        "frozen_files": len(freeze["identities"]),
        "independent_retraining": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    write(args.output, result)
    print(json.dumps(result, indent=2))
