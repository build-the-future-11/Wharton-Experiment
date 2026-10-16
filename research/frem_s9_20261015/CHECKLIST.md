# Current repair checklist after S9 implementation

2026-10-15. A checked engineering item means the implementation/property was verified, not that future performance improved. Preserve the original [loss checklist](../../docs/FREM_LOSS_REPAIR_CHECKLIST_20261015.md) and [restart audit](../restart_20261015/REPORT.md).

| Item / severity | Failure or uncertainty | Implemented response and how it works | Evidence / remaining work |
|---|---|---|---|
| C01 / P0 | Combined losses hid contributions | Complete saved-ledger asset/fee attribution | Previous 240-ledger audit complete; S8 loss unchanged. |
| C02 / P0 | Weekly forecasts extended over four intervals | Direct four-interval scenarios versus explicit constant weekly repetition; paired historical origins | Implemented and chronology-tested. Market MSE comparison blocked. |
| C03 / P0 | Log moments do not supply finite expected simple returns | Finite observed price/dividend scenarios; no Student-t exponential moment | Mathematical support and accounting tested. Future scenario calibration unestablished. |
| C04 / P1 | Weak experts and complicated routing | Small empirical forecast with global and zero-prediction controls | Implemented diagnostic baseline. Full predecessor expert-selection study remains proposed. |
| C05 / P1 | Stale fitting/calibration may matter | Train-only scales and two predeclared chronological fits | Implemented chronology; no schedule-refitting ablation or claimed drift cure. |
| C06 / P1 | Diagonal risk ignores dependence | Full covariance versus diagonal-only ablation at fixed scenarios | PSD argument, direct variance comparison and grid optimality pass. Economic effect untested. |
| C07 / P1 | Exposure and fees confound gains | Equal assets, 80%-risk equal assets, cash and buy-and-hold; four costs; cash dividends and exact entry/exit | Implemented full 96-ledger matrix; endpoint drawdown scope explicit. No market ledger produced. |
| C08 / P1 | Contextual memory advantage unsupported | Local versus global scenarios with matched horizon/risk arms | Implemented control. This standard neighbor method is not a new FREM mechanism or proof of the original memory claim. |
| C09 / P2 | Unnecessary JEPA/reservoir complexity | Keep predecessors unchanged; use a small deterministic diagnostic | Old ablations preserved. No new component efficacy claim or expanded neural architecture. |
| C10 / P0 | Missing canonical data and misleading freshness | Narrow pre-2017 retrospective admission with original source hashes and disclosed availability assumptions | 3,021 sessions, 608 examples; canonical point-in-time admission and unused confirmation still unresolved. |
| C11 / P0 | Unbounded experimentation and weak inference | Two fits; no search; fixed folds; paired blocks 2/4; 600s/128MiB ceiling; persistent consumption marker | Frozen and tested. Host admission failed at 74.8/89.8/120.9MiB. No empirical precision or positive effect established. |
| C12 / P1 | Incomplete reproduction and release | Specification, executable pipeline, checkpoints, generated report, source archive, hashes and failure receipts | 37 local tests and lint pass. Independent reproduction, full empirical manuscript and research release remain unfinished. |
| C13 / P2 | Client funding and eligibility unverified | Keep conditional Laura planning separate from model validation | No allocation, funding-confidence or current eligibility claim added. |

- [x] Write and record the separate exploratory hypothesis before implementation.
- [x] Implement paired horizon forecasts, risk controls, accounting, checkpointing and experiment outputs.
- [x] Verify the complete constructed engineering matrix and real-data preparation.
- [x] Freeze sources, data identities, environment, methodology and budget.
- [x] Attempt execution and retain the nonzero blocked receipt.
- [ ] Run the frozen real-data matrix after host admission succeeds.
- [ ] Audit the resulting evidence, retain negative/inconclusive outcomes and complete independent verification.
- [ ] Establish a separately admitted untouched confirmation design and any investment success claim.

Tests establish implementation properties. Only the frozen empirical comparison can test whether these changes help. Existing source/availability limitations restrict the claim even if its exploratory results are positive.
