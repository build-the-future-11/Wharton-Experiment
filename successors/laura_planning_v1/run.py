"""Reproduce declared planning sensitivities; no data download, model fit or orders."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from .core import (
    FlexibilityRule,
    allocate_residual,
    annuity_due_reserve,
    bounded_facility_commitment,
    discounted_reserve,
    money_record,
    planning_scenario_grid,
    required_constant_growth,
    strategic_sleeves,
)


def write(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def scenarios(as_of: str) -> dict:
    source = "Declared 2026-10-13 team policy sensitivities; hypothetical rates, not market observations"
    grid = planning_scenario_grid(flexibility=FlexibilityRule(".20"),
                                  known_at="2026-10-13", as_of=as_of, source=source)
    dated_prices = []
    for year in (2027, 2028, 2031, 2033):
        for rate in (Fraction(0), Fraction(1, 50), Fraction(1, 25)):
            factors = {payment_year: (1 + rate) ** -(payment_year - year)
                       for payment_year in range(2033, 2043)}
            cost = discounted_reserve(factors, valuation_year=year,
                                     known_at="2026-10-13", as_of=as_of, source=source)
            dated_prices.append({"valuation_date": f"{year}-01-01", "currency": "USD",
                                 "source_kind": "scenario_assumption", "source": source,
                                 "assumptions_known_at": "2026-10-13", "as_of": as_of,
                                 "flat_discount_rate": str(rate), "reserve_cost": money_record(cost),
                                 "discount_factors": {str(y): str(d) for y, d in factors.items()},
                                 "executable_quote": False, "implementation_costs_assumed": "0"})
    required_growth = [{"reserve_rate_assumption": str(rate),
                        **required_constant_growth(annuity_due_reserve(rate))}
                       for rate in (Fraction(0), Fraction(1, 50), Fraction(1, 25))]
    flexibility = []
    for retained in ("0", ".20", ".30"):
        for costs in (0, 1000, 5000):
            value = allocate_residual(600000, 500000, flexibility=FlexibilityRule(retained),
                                      implementation_costs=costs)
            flexibility.append({"retained_fraction_assumption": retained,
                                "assumptions": "Hypothetical wealth600000,reserve500000,costs as specified",
                                **value.to_record()})
    # All amounts below describe an explicitly hypothetical future balance.
    # 500k nominal operating cash is separate; 40k cash + 60k long-only equity is surplus.
    # Residual equity returns are terminal simple returns over the full two years.
    commitments = []
    for equity_return in (Fraction(-1), Fraction(-2, 5), Fraction(0), Fraction(21, 100), Fraction(1)):
        terminal_surplus = 40000 + 60000 * (1 + equity_return)
        commitments.append({"two_year_surplus_equity_return_assumption": str(equity_return),
                            **bounded_facility_commitment(terminal_surplus, 40000, 80000)})
    breach = bounded_facility_commitment(30000, 40000, 80000)
    return {
        "scope": "Exact conditional planning arithmetic, not a backtest, forecast or fitted distribution",
        "as_of": as_of, "grid": grid, "dated_reserve_price_sensitivities": dated_prices,
        "required_constant_growth_to_fund_reserve_only": required_growth,
        "cost_and_flexibility_sensitivities": flexibility,
        "hypothetical_2031_communication_example": {
            "balance_kind": "hypothetical_future_balance_not_observed",
            "valuation_date": "2031-01-01", "as_of": as_of,
            "starting_wealth": money_record(600000), "nominal_operating_cash_assumed_secured": money_record(500000),
            "surplus_equity": money_record(60000), "secured_surplus_cash": money_record(40000),
            "cash_return_assumption": "0", "implementation_cost_assumption": "0, assumed already budgeted",
            "no_leverage_no_withdrawal_no_rebalancing_into_secured_cash": True,
            "funding_probability": None,
            "conditional_communicated_range": [money_record(32000), money_record(80000)],
            "range_basis": "Algebraic bounds of the proposed capped gift rule, conditional on protected cash and operating reserve remaining intact",
            "scenarios": commitments,
            "protection_breach_counterexample": breach,
        },
        "hypothetical_strategic_target": {
            "valuation_date": "2031-01-01", "source_kind": "scenario_assumption",
            "source": "600k wealth,500k current reserve requirement,zero incremental costs, not Laura observed NAV",
            **strategic_sleeves(600000, 500000, secured_surplus_floor=40000),
        },
        "market_observations_used": 0, "market_fits": 0, "protected_access": False,
        "wins_pnl_used": False, "calibrated_probabilities": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", default="2026-10-13")
    args = parser.parse_args()
    # The fixed future-balance example is a 2026 planning illustration, not a rolling live API.
    if args.as_of != "2026-10-13":
        parser.error("this declared scenario package is dated 2026-10-13; author a new dated package for updates")
    payload = scenarios(args.as_of)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "scenarios.json", payload)
    rows = payload["grid"]["rows"]
    lines = ["# Complete declared sensitivity grid", "",
             "All values USD. Hypothetical constant net growth and reserve discount rates; zero incremental implementation costs. No probabilities. This is not a simulated investment policy or market forecast.", "",
             "| Growth/year | Reserve rate | Wealth at 2033 | Reserve | Operating shortfall | Facility capacity (80%) | Flexibility (20%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        labels = [f"{float(Fraction(row[key])) * 100:g}%" for key in
                  ("annual_net_growth_assumption", "annual_reserve_discount_assumption")]
        amounts = [f"{float(row[key]['usd']):,.2f}" for key in
                   ("wealth", "required_operating_reserve", "operating_shortfall", "facility_capacity", "retained_flexibility")]
        lines.append("| " + " | ".join(labels + amounts) + " |")
    with (args.output / "SENSITIVITY_TABLE.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    files = [Path(__file__), Path(__file__).with_name("core.py"),
             args.output / "scenarios.json", args.output / "SENSITIVITY_TABLE.md"]
    receipt = {"created_at": datetime.now(UTC).isoformat(), "status": "CONDITIONAL_ARITHMETIC_COMPLETE",
               "grid_cells": len(rows), "model_fit_or_backtest": False,
               "source_and_output_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "grid_cells": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
