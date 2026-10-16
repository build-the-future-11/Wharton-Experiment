# S5 upgrade backlog — 13 October 2026

The S5-R1 pilot is closed. Items below distinguish implemented work from proposed future changes. Nothing in this backlog authorizes tuning the completed S3/S4/S5 windows until they win. Future scientific changes require a new recorded protocol and bounded budget. No novelty or profitability is claimed.

## Implemented and exercised in the S5-R1 pilot

1. **Contextual expert routing.** Weight complete expert distributions instead of adding mean residual corrections.
2. **Date-level memory.** Pool contemporaneous assets into one dated loss episode; assets are not independent repetitions.
3. **Strict delayed feedback.** An episode becomes retrievable only after every outcome in that date group matures.
4. **Exact mixture likelihood.** Score the actual Student-t mixture rather than substituting a matched Student-t density.
5. **Fixed ensemble control.** Isolate the economic value of routing from the value of combining experts.
6. **Recent-loss control.** Test whether ordinary adaptation explains any improvement.
7. **Age-matched random control.** Match selected date counts in fixed age buckets without using outcomes.
8. **Parametric context control.** Weighted linear regression predicts expert losses with the same dated information.
9. **Evidence-dependent shrinkage.** ESS-based weights pull sparse estimates toward equal averaging.
10. **Bounded memory and recency.** Cap memory at60 dates, nearest set12, recent baseline24, fixed one-year half-life.
11. **Chronological inner selection.** Select only between temperatures0.5 and1 on the second validation half.
12. **Training-only context normalization.** Use first validation half for center/scale, with fixed floors and clipping.
13. **Fit-cutoff verification.** Reject expert training or validation-label overlap with forecast issuance/evaluation.
14. **Complete expert-grid checks.** Reject missing, duplicate or mismatched asset/date/outcome records.
15. **Distribution validation.** Reject nonfinite values, invalid scales, bad degrees of freedom and malformed mixtures.
16. **Deterministic checkpoints.** Serialize pending and matured episodes and restore exact routing state.
17. **All-seed reporting and ensembles.** Retain every optimization seed and use complete ensembles for primary wealth comparisons.
18. **Correct mixture moments.** Include between-expert dispersion in log-return variance for the unchanged allocation rule.
19. **Economic controls.** Compare equal-weight stocks, buy-and-hold, cash, conservative volatility management and each expert.
20. **Trading-cost stresses.** Evaluate10/20/40bps with fees and final liquidation included.
21. **Honest interval availability.** Preserve3/6-month block analyses and insufficient-date flags; no block shortening.
22. **Mechanism falsification suite.** Run null, recurring advantage, abrupt change and delayed feedback across three independent generated paths.
23. **Frozen provenance and resource caps.** Hash111 files; enforce one worker,128MiB and300seconds for the replay.
24. **Arithmetic audit and failure retention.** Reconstruct7128 scores and312 ledgers; retain infrastructure failure, old results and the negative successor disposition.

## Prioritized next design work; not implemented or run

25. **Expert specialization.** Specify genuinely distinct trend, adaptive-volatility, regularized linear and neural experts before any fresh training.
26. **Weak-expert exclusion rule.** Choose inclusion only inside inner development folds; the current mixture diluted the better predictive experts.
27. **Simpler ensemble as incumbent.** Require every proposed mechanism to beat fixed averaging and recent-loss weighting before accepting complexity.
28. **Forecast-to-policy attribution.** Separate selection, exposure, risk scaling, turnover and fee contributions to relative wealth.
29. **Objective-aligned model selection.** Freeze an economic selection objective inside development and retain likelihood as a diagnostic; never choose based on outer wealth.
30. **Crossed forecast/policy experiment.** Hold forecasts fixed while comparing policies, then policy fixed while comparing forecasts, within a small declared matrix.
31. **Risk and covariance baseline.** Add a properly specified common-currency covariance baseline, with no invented diversification benefit.
32. **Turnover-aware allocation.** Test a cost-aware allocation/no-trade rule against the same static rule with identical cash and risk constraints.
33. **Exposure-matched comparisons.** Report common-risk/exposure controls so lower leverage or higher beta cannot masquerade as selection skill.
34. **Fresh-data coverage inventory.** Track all previously inspected dates/assets, historical eligibility, terminal outcomes and source timestamps before defining later evaluation.
35. **Precision-based fold design.** Estimate detectable effects from development date blocks; increase history only if it can resolve the declared economic target.
36. **Fair tuning budgets.** Freeze each expert and router search budget and include all failed trials and deterministic-versus-random fit counts.
37. **Execution-time alignment.** Verify the forecast, label, trade and valuation horizons instead of relying on retrospective monthly coordinates.
38. **Dated-liability evaluation.** Introduce actual contributions/payments into matched wealth paths after mandate inputs are supplied; no funding probabilities without calibration.
39. **Measured low-memory fresh-run path.** Profile one admitted smoke fit, reuse bounded cached features, and set full-run costs from measurements; retain the512MiB guard.

## Conditional; add only when the stated gate is satisfied

40. **External market-state context.** Add dated volatility/trend/breadth context only with documented availability and a predeclared comparison against forecast-only context.
41. **Regime-change forgetting.** Test a reset or forgetting mechanism only if the diagnostic evidence identifies stale expert reliability as the failure mode.
42. **Learned retrieval metric.** Consider only if simple linear context predicts stable expert advantage and a fixed metric demonstrably misses it.
43. **Tail-risk expert or risk-only routing.** Require stable calibration and economically relevant shortfall/drawdown benefits; retain ending-wealth target separately.
44. **Fresh frozen confirmation.** Only after development passes, provenance/precision are sufficient and the unchanged protected boundary is explicitly covered by a complete freeze.
45. **Independent retraining/reproduction.** Repeat the final supported protocol in a clean environment and regenerate raw scores, ledgers and claims; current arithmetic audits do not satisfy this.

## Priority decision

First strengthen the simple expert/policy baselines and diagnose wealth attribution. The current contextual memory mechanism lost to fixed averaging economically in both windows and lost to recent/linear routing predictively in both. Larger memory, deeper keys or more seeds are not justified by that evidence. Fresh heavy training remains resource-blocked; no protected evaluation should be spent on this pilot.

Backlog count:24 implemented,15 prioritized proposals,6 conditional proposals. These45 items are not45 experiments; a new protocol must select a small discriminating subset before execution.
