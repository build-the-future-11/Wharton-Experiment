# Financial sensitivity results

**ASSUMPTION ONLY. Not market history, client wealth, a forecast, a probability estimate, or WInS performance.**

Generated from `configs/competition_stress.json` by `scripts/run_competition_stress.py`. Units: initial wealth = 1. No deposits, taxes, income, financing or cash yield. Two hypothetical sleeves; monthly rebalancing and pro-rata liquidation. Asset returns and inflation are explicit scenario assumptions. Costs apply to every risky-asset purchase and sale, including forced sales for payments.

Primary display: 10 bps. Full sensitivity: 0/5/10/25/50 bps in SUMMARY.csv. These costs are research assumptions, not verified WInS rules. Reserve policy uses a fixed 2.5% planning inflation rate even when realized scenario inflation is higher. Future asset returns never enter policy decisions.

| Scenario | Policy | End wealth | End real wealth | Paid / due | Unpaid | First miss | Net drawdown | Fees |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| steady_assumption | fully_invested | 1.0275 | 0.8027 | 100.00% | 0.0000 | none | 0.00% | 0.0014 |
| steady_assumption | fixed_cash_20pct | 0.9601 | 0.7501 | 100.00% | 0.0000 | none | 0.00% | 0.0011 |
| steady_assumption | liability_reserve | 1.0187 | 0.7958 | 100.00% | 0.0000 | none | 0.00% | 0.0012 |
| early_joint_crash | fully_invested | 0.6560 | 0.5124 | 100.00% | 0.0000 | none | 27.57% | 0.0015 |
| early_joint_crash | fixed_cash_20pct | 0.6797 | 0.5310 | 100.00% | 0.0000 | none | 22.06% | 0.0011 |
| early_joint_crash | liability_reserve | 0.6561 | 0.5126 | 100.00% | 0.0000 | none | 26.90% | 0.0013 |
| late_joint_crash | fully_invested | 0.7424 | 0.5800 | 100.00% | 0.0000 | none | 27.50% | 0.0014 |
| late_joint_crash | fixed_cash_20pct | 0.7469 | 0.5835 | 100.00% | 0.0000 | none | 22.00% | 0.0011 |
| late_joint_crash | liability_reserve | 0.7368 | 0.5756 | 100.00% | 0.0000 | none | 27.43% | 0.0012 |
| stagnation_high_inflation | fully_invested | 0.2778 | 0.1287 | 100.00% | 0.0000 | none | 0.23% | 0.0017 |
| stagnation_high_inflation | fixed_cash_20pct | 0.2781 | 0.1288 | 100.00% | 0.0000 | none | 0.18% | 0.0014 |
| stagnation_high_inflation | liability_reserve | 0.2779 | 0.1287 | 100.00% | 0.0000 | none | 0.20% | 0.0016 |
| funding_failure | fully_invested | 0.0000 | 0.0000 | 61.47% | 0.5552 | 84 | 19.70% | 0.0019 |
| funding_failure | fixed_cash_20pct | 0.0000 | 0.0000 | 62.89% | 0.5348 | 86 | 16.50% | 0.0015 |
| funding_failure | liability_reserve | 0.0000 | 0.0000 | 63.10% | 0.5318 | 86 | 12.93% | 0.0017 |

## Interpretation limits

A reserve can soften a near-term market shock but cannot make an infeasible spending plan solvent. In high-inflation or large-payment scenarios, compare unpaid obligations before terminal wealth: a portfolio ending with more wealth may simply have paid less. Net drawdown excludes withdrawals; end wealth includes them. The reserve is held as zero-yield cash, so its opportunity cost is explicit. The fixed-cash and liability policies do not isolate cash level from reserve timing; no causal superiority claim follows.

Early and late shocks use the same shock vector and otherwise identical return assumptions. They illustrate sequence-of-returns sensitivity. Five hand-designed paths are not a calibrated return distribution: do not attach confidence intervals or failure probabilities to them. EWMA sizing, M11 MPC, and M12 are not implemented in this calculator. The calculator tests accounting and transparent policy assumptions only.
