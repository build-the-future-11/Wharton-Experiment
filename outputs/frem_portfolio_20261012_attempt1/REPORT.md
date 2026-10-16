# FREM portfolio evidence — 12 October 2026

**REQUESTED_ALL_CASE_6PCT_TARGET_NOT_MET**. The requested 6% minimum is an ending-wealth target. A positive result in one historical path is not a universal floor or a live investment recommendation.

## What was executed

90 source-hashed forecast files; 17396 independently audited rows; 1424 model/seed/window/market/cost cases; 2276 shared-accounting ledgers. No new fits or protected observations.

All models used the fixed 80% equal-weight anchor plus up to 20% positive log-signal sleeve. Unallocated weight stays in zero-return cash. This is a declared unoptimized research policy, not an optimized Laura mandate. Costs, initial cash and final liquidation are shared; drifted holdings are carried between periods. The same-basket equal-weight, buy-and-hold and cash controls are retained.

## Primary: S4 corrected distance versus equal weight

| Window | Seed | Cost per traded dollar | Uplift | Versus memory off | Max drawdown |
|---|---|---:|---:|---:|---:|
| fresh_2023 | 11 | 0 bp | -2.96% | 0.00% | 5.10% |
| fresh_2023 | 11 | 10 bp | -2.91% | 0.00% | 5.11% |
| fresh_2023 | 11 | 25 bp | -2.84% | 0.00% | 5.12% |
| fresh_2023 | 11 | 50 bp | -2.73% | 0.00% | 5.14% |
| fresh_2023 | 29 | 0 bp | -0.21% | 0.00% | 5.83% |
| fresh_2023 | 29 | 10 bp | -0.22% | 0.00% | 5.85% |
| fresh_2023 | 29 | 25 bp | -0.25% | 0.00% | 5.88% |
| fresh_2023 | 29 | 50 bp | -0.29% | 0.00% | 5.93% |
| fresh_2023 | 47 | 0 bp | -0.08% | 0.00% | 6.00% |
| fresh_2023 | 47 | 10 bp | -0.09% | 0.00% | 6.02% |
| fresh_2023 | 47 | 25 bp | -0.10% | 0.00% | 6.04% |
| fresh_2023 | 47 | 50 bp | -0.12% | 0.00% | 6.09% |
| fresh_2023 | ensemble | 0 bp | -0.63% | 0.00% | 5.74% |
| fresh_2023 | ensemble | 10 bp | -0.63% | 0.00% | 5.76% |
| fresh_2023 | ensemble | 25 bp | -0.62% | 0.00% | 5.78% |
| fresh_2023 | ensemble | 50 bp | -0.62% | 0.00% | 5.82% |
| fresh_2024 | 11 | 0 bp | 0.35% | 0.00% | 3.40% |
| fresh_2024 | 11 | 10 bp | 0.35% | 0.00% | 3.50% |
| fresh_2024 | 11 | 25 bp | 0.34% | 0.00% | 3.65% |
| fresh_2024 | 11 | 50 bp | 0.34% | 0.00% | 3.90% |
| fresh_2024 | 29 | 0 bp | 0.24% | 0.00% | 3.39% |
| fresh_2024 | 29 | 10 bp | 0.24% | 0.00% | 3.49% |
| fresh_2024 | 29 | 25 bp | 0.24% | 0.00% | 3.64% |
| fresh_2024 | 29 | 50 bp | 0.23% | 0.00% | 3.89% |
| fresh_2024 | 47 | 0 bp | 0.25% | 0.00% | 3.38% |
| fresh_2024 | 47 | 10 bp | 0.25% | 0.00% | 3.48% |
| fresh_2024 | 47 | 25 bp | 0.24% | 0.00% | 3.63% |
| fresh_2024 | 47 | 50 bp | 0.24% | 0.00% | 3.88% |
| fresh_2024 | ensemble | 0 bp | 0.28% | 0.00% | 3.39% |
| fresh_2024 | ensemble | 10 bp | 0.27% | 0.00% | 3.49% |
| fresh_2024 | ensemble | 25 bp | 0.27% | 0.00% | 3.64% |
| fresh_2024 | ensemble | 50 bp | 0.27% | 0.00% | 3.89% |

Observed range across all 32 primary cases: -2.96% to 0.35%. Cases at least 6%: 0/32. Seed outcomes share the same market history and are not independent financial repetitions.

Intervals in summary.json use predeclared 3- and 6-month paired circular blocks of saved net-wealth increments. These intervals condition on the realized policy path, omit retraining uncertainty, are not simultaneous, and cannot certify future coverage. Short windows remain explicitly insufficient for a block length.

## Across time, currencies and predecessor versions

All six older crisis/recovery windows and four methods are in comparisons.csv. Each listing-market basket is separate; currencies are never combined. Two-stock local baskets are narrow accounting illustrations. S3/S4 forecast source histories are exploratory, retrospective survivor samples with adjusted-price proxies. Changing checkpoints and training chronology prevents a causal attribution of cross-version portfolio differences.

Incomplete declared market/model groups: 20. Each is retained in summary.json; missing prices/assets are not backfilled and incomplete groups do not become passing cases.

## Why an all-scenario 6% floor fails

In a zero-return world with zero costs and identical cash flows, a long-only self-financing strategy and the benchmark both end at their starting wealth. Uplift is exactly zero. No retriever, JEPA loss or backtest can change this accounting identity. Constructed stress counterexamples are in stress.json; they test the claim, not model skill.

## Laura Gao

Use $300,000 at the beginning of 2027 and $150,000 at the beginning of 2028. Before the first 2033 payment, establish a reserve for ten $50,000 beginning-year payments through 2042. The reserve calculation includes the immediate payment (annuity due). At zero growth and zero reserve discount, wealth is $450,000 against $500,000 of obligations: a $50,000 shortfall. WInS profits are not added to these contributions. The scenario grid in laura.json uses assumed rates, not observed yields or calibrated funding probabilities. Facility capacity remains null without an explicit flexibility buffer.

Current research direction and sourced instrument candidates are in LAURA_CURRENT_RESEARCH.md. No certified ticker allocation can be produced without current eligible instruments, holdings, prices, risk/funding policy and admissible forecast inputs. The external quant/ML framework is an explicit dated input contract in engine.external_forecasts; its source has not yet been identified or integrated.

## Verification and reproducibility

The source/config snapshot is in freeze.json, per-file reconstruction in forecast_audit.json, per-period holdings/cash/costs in ledgers/, and every comparison in comparisons.csv. The runner refuses existing output directories. Earlier negative experiments and S4 freeze files are preserved.

These are exploratory reanalyses and engineering tests, not independent training reproduction, point-in-time market certification, proof of profitability, live fills or public submission readiness.
