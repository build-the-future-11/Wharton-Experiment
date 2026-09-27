"""Generate assumption-only financial sensitivity evidence, without market downloads."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from wharton_lab.competition.stress import scenario_inputs, simulate

ROOT = Path(__file__).resolve().parents[1]


def main():
    config_path = ROOT / "configs/competition_stress.json"
    cfg = json.loads(config_path.read_text())
    records = []
    for scenario in cfg["scenarios"]:
        returns, obligations = scenario_inputs(cfg, scenario)
        for policy in cfg["policies"]:
            for cost in cfg["cost_bps"]:
                r = simulate(returns, obligations, policy=policy, cost_bps=cost,
                             initial_wealth=cfg["initial_wealth"],
                             reserve_months=cfg["reserve_months"],
                             planning_inflation=cfg["planning_inflation"])
                r["scenario"] = scenario["id"]
                r["terminal_real_wealth"] = r["terminal_wealth"] / (1 + scenario["annual_inflation"]) ** (cfg["months"] / 12)
                records.append(r)
    source_paths = [config_path, ROOT / "src/wharton_lab/competition/stress.py", Path(__file__).resolve()]
    payload = {"status": "ASSUMPTION_ONLY_NOT_CLIENT_DATA", "confirmatory": False,
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
               "environment": {"python": platform.python_version(), "numpy": np.__version__},
               "config": cfg, "results": records}
    out = ROOT / "runs/competition_stress"; out.mkdir(exist_ok=True)
    (out / "RESULTS.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    rows = [{k: v for k, v in r.items() if k != "trajectory"} for r in records]
    with (out / "SUMMARY.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Financial sensitivity results", "", "**ASSUMPTION ONLY. Not market history, client wealth, a forecast, a probability estimate, or WInS performance.**", "",
             "Generated from `configs/competition_stress.json` by `scripts/run_competition_stress.py`. Units: initial wealth = 1. No deposits, taxes, income, financing or cash yield. Two hypothetical sleeves; monthly rebalancing and pro-rata liquidation. Asset returns and inflation are explicit scenario assumptions. Costs apply to every risky-asset purchase and sale, including forced sales for payments.", "",
             "Primary display: 10 bps. Full sensitivity: 0/5/10/25/50 bps in SUMMARY.csv. These costs are research assumptions, not verified WInS rules. Reserve policy uses a fixed 2.5% planning inflation rate even when realized scenario inflation is higher. Future asset returns never enter policy decisions.", "",
             "| Scenario | Policy | End wealth | End real wealth | Paid / due | Unpaid | First miss | Net drawdown | Fees |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["cost_bps"] == 10:
            lines.append(f'| {r["scenario"]} | {r["policy"]} | {r["terminal_wealth"]:.4f} | {r["terminal_real_wealth"]:.4f} | {r["funded_fraction"]:.2%} | {r["unpaid_at_end"]:.4f} | {r["first_shortfall_month"] or "none"} | {r["max_drawdown_ex_withdrawals"]:.2%} | {r["total_fees"]:.4f} |')
    lines += ["", "## Interpretation limits", "", "A reserve can soften a near-term market shock but cannot make an infeasible spending plan solvent. In high-inflation or large-payment scenarios, compare unpaid obligations before terminal wealth: a portfolio ending with more wealth may simply have paid less. Net drawdown excludes withdrawals; end wealth includes them. The reserve is held as zero-yield cash, so its opportunity cost is explicit. The fixed-cash and liability policies do not isolate cash level from reserve timing; no causal superiority claim follows.", "", "Early and late shocks use the same shock vector and otherwise identical return assumptions. They illustrate sequence-of-returns sensitivity. Five hand-designed paths are not a calibrated return distribution: do not attach confidence intervals or failure probabilities to them. EWMA sizing, M11 MPC, and M12 are not implemented in this calculator. The calculator tests accounting and transparent policy assumptions only.", ""]
    (ROOT / "reports/competition_package/FINANCIAL_STRESS.md").write_text("\n".join(lines))
    print(f"Wrote {len(rows)} scenarios/policies/cost combinations with complete monthly trajectories")


if __name__ == "__main__":
    main()
