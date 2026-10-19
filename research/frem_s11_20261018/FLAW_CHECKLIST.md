# FREM-S11 adversarial flaw and repair checklist

Date: 18 October 2026. Scope: corrected pre-data P1 successor only. No archive
has been acquired or parsed, no market fit or policy replay has run, and no
protected outcome has been accessed.

## What is already strong

- P1 is a separately versioned successor; S8 negatives and the unrun S9/S10
  states are not rewritten.
- The primary forecast gate dominates the secondary portfolio diagnostic.
- Two controls, two chronological folds, a fixed 5% practical margin, paired
  uncertainty, zero tuning trials, fixed seeds and one development execution
  are declared before data access.
- Official release vintages make data revision measurable instead of silently
  treating the latest reconstruction as historical truth.
- Source, interpreter, archive, data-admission, consumption and output hashes
  fail closed; every score, weight, allocation and wealth path is retained.
- The generated implementation is causal by construction and includes
  future-perturbation, calendar, missing-value, receipt, accounting and
  reconstruction checks.

## Flaw -> cause -> severity -> repair -> required evidence

| Flaw | Cause | Severity | Repair / present disposition | Required evidence |
|---|---|---:|---|---|
| No qualified execution host | Latest macOS free-plus-speculative memory was only 161.516-219.125 MiB and pageouts increased | P0 blocker | Keep the 512 MiB three-sample rule; do not kill unrelated work or waive it | Fresh source-bound `PASS` receipt with three stable samples, then stable post-run pageouts |
| Original P0 mislabeled LoBM/HiBM and carried target rather than drifted holdings | Translation error before data access | P0 validity defect, repaired prospectively | Preserve P0; execute only P1 with growth/value order and post-return drift | Generated semantic and accounting tests plus P1 protocol hash |
| 2008 source was incorrectly declared CSV | Official archive offers 2008 TXT; 2016/2024 are CSV | P0 data blocker, repaired prospectively | Exact URL allowlist and strict dual-format parser | Successful no-overwrite acquisitions and TXT/CSV admission receipts on a qualified host |
| Online proper-score aggregation is not novel by itself | Proper scoring and online probabilistic forecast aggregation are established | P1 claim risk | Limit novelty to a candidate application/combination until a full overlap review establishes otherwise | Claim-by-claim comparison with [Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437), [Thorey et al. 2017](https://doi.org/10.1002/qj.2940), and [Berrisch and Ziel 2024](https://doi.org/10.1016/j.ijforecast.2024.01.005) |
| Energy score alone may weakly diagnose dependence errors | A scalar proper score can hide which marginals or dependence features failed | P1 interpretation risk | Keep it primary because it was frozen; add only descriptive, predeclared decomposition in a future successor | Calibration/rank diagnostics and transformed scores, without replacing P1's endpoint |
| Four specialists may not be meaningfully diverse | They share the same rolling 120-month outcome support and differ mainly in weights | P1 mechanism risk | Equal and recent controls remain mandatory; retain per-origin weights and scores | Measured expert score correlation, weight entropy and specialist dominance from the one run |
| Stacker is heuristic, not a demonstrated regret-optimal algorithm | Median-scaled exponentially weighted centered losses with shrink/cap has no supplied theorem | P1 theory risk | Make no optimality claim; test only the empirical frozen rule | Formal regret analysis in successor work or empirical comparison with a recognized online aggregator |
| Hyperparameters lack empirical calibration | Zero tuning trials intentionally avoid data-driven search | P1 power risk, not a defect | Keep fixed for P1; accept possible non-support | Valid one-shot result; any calibrated alternative is a new successor |
| Twelve-month block length is asserted, not empirically justified | No pre-data dependence analysis is available | P1 uncertainty risk | Preserve the fixed block to avoid post-outcome choice | Report effective block counts and interval width; future protocol may justify block choice on separate data |
| Only two chronological folds are available | The design prioritizes a bounded, revision-aware development study | P1 precision/generalization risk | Require success in both folds and close as inconclusive if precision is inadequate | Complete paired observations, intervals and fold-specific effects; no dropped months |
| Fama/French portfolios are reconstructed research series, not tradable instruments | Returns aggregate CRSP securities under research construction rules | P0 scope limit | Restrict claims to portfolio-return forecasting | Cite the [official archive contract](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/six_portfolios_archive.html) and [Fama-French construction context](https://www.jufinance.com/mag/fin534_16/Common_risk_factors_Fama_French_JFE1993.pdf); make no deployment claim |
| Revision vintages do not prove real-time constituent availability | Published portfolio series abstract away security-level membership and vendor latency | P0 scope limit | Treat revision stress as robustness only | Exact old/new archive hashes and common-cutoff change counts |
| Economic policy is deliberately stylized | Coarse 25% grid, 50% cap, zero-return cash, 10 bps fee, no taxes/slippage/liability inputs | P1 economic limitation | Keep secondary and non-deployable; fee reduces investable wealth before return and holdings then drift exactly | Exact ledger reconstruction and comparison with cash/equal-assets; no profitability or mandate claim |
| Fee convention was initially ambiguous | Initial code subtracted a fee from wealth but did not bind its effect on holdings | P0 accounting defect, repaired prospectively | P1 applies one-way risky-weight turnover cost before return to investable wealth | Hand-calculated fee/drift tests and all ledger identities |
| Source rights do not imply redistribution rights | Public download availability is not a license grant for repackaging | P0 release limit | Retain URLs/hashes locally; do not publish raw archives without verified permission | Source terms/licensing review before any public artifact |
| A positive development result would still be unconfirmed | P1 has zero confirmation runs and uses development outcomes | P0 claim boundary | Freeze a new unused-data confirmation only after valid support | New protocol, untouched outcomes, exact freeze, independent reproduction |
| Independent reproduction is absent | All current work is same-checkout engineering | P1 verification gap | Do not claim verified science | Clean-host reproduction of source build, data admission, raw analysis and paper numbers |

## Terminal rules

P1 is run-ready only after generated verification, package build, a fresh host
`PASS`, exact archive receipts and a passing data-admission receipt. A valid
negative, mixed or inconclusive one-shot result closes P1. A positive
development result is not final success; it authorizes only a separately frozen
unused-data confirmation. No result authorizes trading, allocation, publication
or a claim that the original FREM memory hypothesis succeeded.
