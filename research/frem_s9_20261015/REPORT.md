# FREM-S9 implementation and execution disposition

2026-10-15. **Implemented and tested; first market attempt blocked by host memory admission. Zero new market fits or market-policy results.** The model has not become a demonstrated winning system.

The prior turn completed diagnosis. This turn advanced a separate, executable exploratory study: directly forecast the four-interval decision horizon, retain finite joint price/dividend scenarios, compare cross-asset covariance with diagonal risk, and test context against a global forecast. The scientific purpose is a controlled failure diagnosis using standard methods, not an original-algorithm claim or a reopened S8 result.

## Concrete work completed

- [Specification](SPECIFICATION.md) recorded before implementation, with equations, units, feature chronology, parameters, assumptions, exact folds, comparisons, uncertainty, failure behavior and finite budget.
- [Source package](../../successors/frem_s9/README.md): deterministic nearest-neighbor/global scenarios, JSON checkpoints, exact finite-grid portfolio policy, cash dividend accounting, full matrix runner and generated report.
- Direct horizons and constant-repetition horizons share the same training origins and neighbor selection. Joint/diagonal policies share forecasts, costs, risk aversion and action grid. Global and zero forecasts expose weak context skill.
- Exact entry and price-only liquidation costs; dividends remain cash. All-cash actions, no-reinvestment buy-and-hold and complete control/cost coverage are tested.
- [37 passing tests](tests_attempt2.log), including a complete constructed two-fold/96-ledger workflow, strict chronology, future perturbations, checkpoint equivalence, independent hand-calculated accounting identities, finite-grid optimum, block statistics and blocked/consumed launcher handling. [Lint passed](lint_attempt2.log). The initial lint failure and earlier source/test versions remain retained.
- [Narrow data admission](DATA_ADMISSION.json): 3,021 daily sessions, 626 weekly anchors, 608 examples. Fold A: 243 training examples and 38 four-interval evaluation blocks; fold B: 399 and 51. Preparation took about 0.18 seconds at 28.98 MiB peak RSS; it fitted no model.
- [Development freeze](freeze1.json): 20 source/data/specification identities, exact environment and source archive. This is not a confirmatory freeze and grants no protected-data access.

## Attempted execution

The [first market-attempt receipt](../../outputs/frem_s9_20261015_attempt1/RECEIPT.json) records exit **3**, `BLOCKED_RESOURCE_ADMISSION`, **zero market fits**, 1.57 seconds and 27.53 MiB launcher peak RSS. Available free-plus-speculative host memory was **74.81 / 89.84 / 120.94 MiB**, below the unchanged **512 MiB** floor, with pageouts increasing from 648590 to 648591. These host readings differ from the launcher's own small process footprint.

No scientific fit-start or consumption marker was created. The same unconsumed frozen study can be attempted in a fresh output directory after admission succeeds; a completed/started evaluation cannot be replayed by deleting its consumption receipt. No processes or user applications were closed, resource guards changed, paid compute started or protected outcomes opened.

## What the study can and cannot answer

The exploratory matrix is fixed at two deterministic fits, 64 strategy ledgers and 32 benchmark ledgers, one worker, 600 seconds wall time and 128 MiB process peak RSS. There are no duplicated random-seed fits or search trials. Prediction is assessed through paired four-interval mean-squared error; the original 0.01-nat memory criterion remains unsatisfied and unchanged. The earlier 6% wealth objective remains explicit as a separate gate against equal assets and buy-and-hold, in both folds, while beating cash.

These pre-2017 observations were within predecessor training. They support a labeled retrospective diagnostic, not untouched confirmation. Current-vintage quotes, assumed release times, surviving ETFs, ex-date dividend accrual, finite empirical tails, discrete actions, endpoint-only drawdown and limited dependent histories constrain interpretation. Passing an exploratory criterion would require independent, admitted new evidence before any confirmation or investment-success claim.

The earlier restart gate remains an accurate historical record. This successor resolves the narrow exploratory contribution, specification and data-preparation questions; it does not make the broader canonical dataset or protected confirmation ready. The original S6–S8 conclusions and all 105 selected predecessor/source/artifact identities remain preserved.

## Remaining work

See the [updated 13-item checklist](CHECKLIST.md) and [local adversarial review](REVIEW.md). First run this exact frozen matrix when resource admission passes. Then audit all outcomes, including failures, produce the empirical analysis and obtain independent reproduction/review. A valid negative or inconclusive result closes S9; it does not authorize tuning against its outcomes until it becomes positive. Full empirical paper/release and original-project confirmation remain incomplete.

The exact continuation command is in the source [README](../../successors/frem_s9/README.md). No new user approval is required for that already authorized local, bounded attempt once its prerequisites are met. No background run or monitoring has been scheduled.
