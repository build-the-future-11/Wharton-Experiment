# FREM restart: loss attribution, repairs and readiness

2026-10-15. **Saved-result audit complete. Successor Checkpoint 1 NOT READY. Predictive improvement remains inconclusive; the S8 economic primary remains failed.** The model is not perfect and positive investment research has not been established.

## What is going well

The project has retained the negative studies, exact source freezes and detailed ledgers. S6 completed 24 fits; S7 and S8 completed three base fits each. S7/S8 already repaired holiday continuity, dividend accounting, mixture-complementarity treatment and the earlier hard allocation/gating restrictions. S8's forecast improved its own policy's ending wealth relative to equal forecast averaging, although it still lost to cash and broad benchmarks. These are useful engineering and diagnostic results, not a profitable-system certificate.

This restart implemented a separate streaming auditor and mathematical reference functions without changing frozen model code. The final run checked **240 saved strategy ledgers and 36,720 interval records**, across all 15 forecast variants, four policies and four cost levels. All **49 selected source/data/evidence identities** matched their pre-session hashes. The largest accounting discrepancy was **$8.30e-10**. This new audit covers strategy ledgers; the 12 benchmark ledgers were not freshly re-audited here. Benchmark numbers below come from the preserved S8 summary and its older audit.

## Why the primary lost

The primary S8 four-interval policy at 10 bps per side began with $100,000:

| Component | Realized dollar contribution |
|---|---:|
| SPY | +$6,988.60 |
| TLT | -$5,511.30 |
| GLD | -$1,921.78 |
| Total asset contribution before fees | -$444.48 |
| Trading and final liquidation fees | -$810.24 |
| Net result | **-$1,254.73** |
| Ending wealth | **$98,745.27** |

Unrounded components reconcile exactly within the declared tolerances; individually rounded dollars can differ by a cent. Average risky weights were approximately 27.44% SPY, 16.92% TLT and 19.33% GLD. The carried path returned +0.99% in 2020, +7.35% in 2021 and -8.92% in 2022. The 2022 TLT contribution alone was -$5,694.75. These describe realized exposures, not a claim that the system could have predicted the market path or should now be retuned to avoid it.

Cash ended at $100,000; equal-weight assets ended at $104,927.25. Under the same S8 H4 policy, equal forecast mixing ended at $96,211.62 and S6 memory at $97,134.00. S8's relative forecast improvement therefore helped within that policy, but did not meet the investment objective.

The zero-cost S8 path ended at $99,990.33. Reducing fees alone is not an established cure. The difference between zero-cost and 10-bps outcomes includes changed decisions and compounding, so it is not a pure fee estimate.

The saved log-score improvement over equal forecast mixing is **0.00136750 nats**, versus the unchanged practical margin of **0.01**. The two prespecified descriptive 95% intervals are approximately **[-0.003749, 0.005494]** and **[-0.003186, 0.005203]**. Both include zero and both upper endpoints are below 0.01. These dependent development intervals are not untouched confirmatory evidence. They do not prove that every future model must fail, but they provide no support for calling this version a practical predictive success.

## Repair checklist and actual completion

The full flaw → cause → severity → repair → required-evidence definitions remain in the [13-item checklist](../../docs/FREM_LOSS_REPAIR_CHECKLIST_20261015.md). Status after this restart:

| Item | What was done | What remains before claiming the repair works |
|---|---|---|
| C01: loss attribution, P0 | **Complete:** full saved strategy-ledger reconciliation, per-asset and carried-year attribution, same-policy comparison. | This is diagnosis; causal performance improvements still require a new experiment. |
| C02: forecast horizon, P0 | Exact compounding, availability and serial-covariance contracts implemented and tested. | Horizon-matched expert training/calibration and a valid forward comparison. No model horizon was changed. |
| C03: economic inputs, P0 | Proved the log-return/expected-simple-return distinction; implemented finite-scenario moment and exact fee references. | A specified, calibrated market scenario/return estimator and fixed-policy comparison. |
| C04: expert quality, P1 | Fixed averaging retained as incumbent; prohibited selecting a post-hoc S8 winner. | Prespecified inner-fold expert comparison with all seeds and equal information. |
| C05: stale fit/calibration, P1 | Separated the observed chronology from the unproved drift explanation. | A fixed versus scheduled-refit comparison with matured labels and purged targets. |
| C06: joint risk, P1 | Implemented joint scenario covariance; mathematical PSD proof and correlation counterexamples checked. | Real estimator calibration and a same-forecast comparison with the diagonal policy. |
| C07: exposure/costs, P1 | Attributed exposures and fees; checked entry/exit and cash financing. | New constraint/exposure-matched controls, holding-policy comparison and sourced cash yield. |
| C08: contextual memory, P1 | Kept existing tiny increments and failed practical claim visible; no architecture expansion. | A distinct mechanism and forward benefit over fixed/global/recent-loss controls. |
| C09: JEPA/reservoir components, P2 | No unsupported component was promoted; old ablations preserved. | Specific defect or learning hypothesis, then a bounded compute-matched component test. |
| C10: data/freshness, P0 | Access/readiness inventory completed using existing metadata; protected outcomes kept closed. | Canonical panel is absent; historical availability and unused evaluation admission remain unresolved. |
| C11: discriminating experiment, P0 | Three candidate repair approaches and an explicit finite proposed ceiling documented. | Exact admissible folds, specification, precision calculation, resource admission and freeze. |
| C12: reproduction/release, P1 | New tool passed 40 tests and lint; all audit failures and source versions retained; evidence links recorded. | Independent retraining/review and a full research paper/release remain incomplete. |
| C13: Laura decisions, P2 | Existing conditional planning evidence kept separate. | Verified dated mandate/eligibility inputs and supported funding calibration; no transactions authorized. |

## How the proposed improvements would work

First establish a simple, horizon-correct forecast baseline. Separately test a decision layer using finite joint returns, explicit cash and exact costs. Only after those comparisons should a materially distinct memory mechanism be considered. This separates better forecasting from better decisions and prevents several simultaneous changes from hiding the cause of a result. Standard optimization, covariance and expert-tracking methods have prior art; see [the review](PRIOR_ART_AND_IDEA_REVIEW.md).

The [mathematical contract](MATHEMATICAL_CONTRACT.md) proves the relevant accounting and probability identities and gives their assumptions and counterexamples. It cannot prove future profitability. The [successor feasibility contract](NEXT_EXPERIMENT_CONTRACT.md) proposes at most 12 base fits, one worker/thread, 60 minutes of training and no paid compute, once a valid experiment exists. It is not a freeze or a launched campaign.

## Why implementation stops at the idea gate

The supplied [user pipeline](USER_REQUEST.txt) requires a defensible contribution and feasible discriminating experiment before implementing the next model, and its ultimate rule accepts valid negative evidence. No distinct new mechanism or unused admitted confirmation set has been established. The canonical `data/raw/panel.parquet` and provenance sidecar are absent. Retrospective ETF data and regional factors do not silently satisfy that contract. Previously used training data can still support labeled development; closed evaluation outcomes cannot become fresh confirmation.

Standard resource admission also failed again: 146.67, 197.44 and 146.58 MiB free-plus-speculative memory, below 512 MiB, with rising pageouts. The earlier S6-specific RAM override remains recorded unchanged. Resource availability alone would not fix the scientific/data gaps.

Accordingly, **zero new market fits, zero new policy replays, no new scientific freeze and no protected-outcome access** occurred. No completed study was reopened, no success threshold lowered, and no result was relabeled positive. A successor remains incomplete, not certified impossible. This session completed the bounded diagnostic work that can support the next decision.

## Verification and reproduction

- [Final attribution and receipt](attribution_attempt5/RECEIPT.json): 240 ledgers, 36,720 records, 7.39 seconds, 39.50 MiB process peak RSS. All original artifacts retained.
- [Generated attribution report](attribution_attempt5/REPORT.md) and [machine attribution](attribution_attempt5/ATTRIBUTION.json): every forecast/policy/cost cell, yearly contributions and full-precision amounts.
- [Tests](tests_attempt5.log): 40 passing standard-library tests. [Lint](lint_attempt2.log): passed. These establish bounded engineering properties only.
- Audit attempts 1–3 failed on unhandled retained JSON/ledger formats. Their failure receipts and pre-repair source archives remain. Attempts 4 and 5 pass; the final review added duplicate asset-order rejection and lint repairs. These were audit-tool corrections, not defects invalidating S8 or permission to rerun its science.
- [Checkpoint record](CHECKPOINTS.json), [data inventory](DATA_READINESS.json), [claim map](CLAIMS.json) and [completion receipt](RECEIPT.json) distinguish verified work, historical evidence and unresolved requirements.

From the repository root, run the 40 engineering checks with `rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/restart -p 'test_*.py' -v`. To regenerate saved-result attribution, use `rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 -m successors.frem_audit_v1.audit --root . --manifest research/restart_20261015/before_manifest.json --output <new-output-directory>`. The output directory must not exist. These commands do not train or trade. Same-host reproduction is not independent replication.
