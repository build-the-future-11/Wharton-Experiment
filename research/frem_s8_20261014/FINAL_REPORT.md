# S8 frozen development result

Prediction practical-support gate: **False**. Economic point-win gate against all four controls: **False**.

S7 and all prior results remain closed. This successor was specified and source-frozen before acquiring 2020–2022 ETF quotes; no 2019 data entered its fitting, features, gate updates or evaluation.

## Why S7 rejected the gains and stayed in cash

The largest 2018 routing gain was approximately 0.00568 nats. Both chronological half means were positive (about 0.0052 and 0.0062), but both were below the required 0.01. The gate followed its protocol; it treated the threshold for a substantial scientific claim as a threshold for permitting any routing. Positive development means alone did not establish a reliable advantage.

The saved 2019 signal candidates had positive gross weekly utility on all 51 dates, but all net scores were negative after the two-way fee estimate. Their implied break-even one-way costs were only 0.15–5.41 bps, below the tested 10 bps. The 80% anchor had the same problem (maximum 5.61 bps). With cash as the initial state and no small-position candidates, cash remained the maximizer. This was a conservative one-week surrogate, not evidence the rally was predictable. No new policy or threshold was tested on 2019.

## Successor and matched comparisons

S8 softly mixes equal expert densities and the contextual mixture using only delayed mature scores. A fixed-share update retains recovery weight for both strategies. The allocation solves a continuous concave log-moment surrogate and commits holdings for four weekly intervals, with a one-interval ablation. It allows cash and partial entry, charges actual self-financing costs, and never imposes a risky anchor.

All forecasts share three newly fitted S6 base models (seeds 11,29,47). Training ends before 2017; scale calibration uses 2017; gate initialization uses 2018. The primary evaluation is one continuous 2020–2022 path after the eight-return warmup. The S6 row is the unchanged S6 memory mechanism under the common data and fit contract, not its old 2017–2018 score.

| Forecast | NLL, lower better | H4 wealth, 10 bps | H1 wealth | S7 cost-aware wealth | S6 signal wealth |
|---|---:|---:|---:|---:|---:|
| s8 | -1.992065074 | $98,745.27 | $97,122.15 | $100,000.00 | $103,300.36 |
| equal | -1.990697574 | $96,211.62 | $96,193.80 | $100,000.00 | $103,196.06 |
| memory | -1.991381286 | $97,134.00 | $95,837.40 | $100,000.00 | $103,199.45 |
| s7 | -1.990697574 | $96,211.62 | $96,193.80 | $100,000.00 | $103,196.06 |
| fixed_half | -1.992003549 | $98,524.79 | $96,904.91 | $100,000.00 | $103,321.07 |
| s8_global | -1.992039547 | $98,758.12 | $97,135.40 | $100,000.00 | $103,296.84 |
| mixture_ungated | -1.992654989 | $100,441.75 | $97,631.89 | $100,000.00 | $103,536.24 |
| mixture_global | -1.992608708 | $100,421.77 | $97,635.57 | $100,000.00 | $103,531.07 |
| recent | -1.990012337 | $97,396.86 | $95,806.72 | $100,000.00 | $103,175.55 |
| zero | -1.960478965 | $100,000.00 | $100,000.00 | $100,000.00 | $104,720.48 |
| ridge | -1.949792427 | $103,436.62 | $122,905.14 | $100,000.00 | $105,317.80 |
| trend | -1.976339794 | $104,658.55 | $95,150.37 | $98,066.44 | $103,322.67 |
| adaptive_volatility | -1.568032885 | $100,000.00 | $100,000.00 | $100,000.00 | $104,720.48 |
| reservoir_jepa | -1.949477723 | $104,666.28 | $100,000.00 | $100,000.00 | $104,948.99 |
| xgboost_hist | -1.943218190 | $92,535.67 | $103,276.39 | $118,342.11 | $106,211.78 |

Asset benchmarks at 10 bps: cash $100,000.00, buy_hold $104,114.71, equal_weight $104,927.25.

## Predeclared uncertainty and mechanism checks

- equal NLL minus S8: 0.001367500; 6-date 95% interval [-0.0037489799298092044, 0.005494434960005422]; 12-date interval [-0.003185569058352858, 0.005202727069774156].
- memory NLL minus S8: 0.000683788; 6-date 95% interval [-0.00552657881461166, 0.009759428069919923]; 12-date interval [-0.005894522605121223, 0.010299795020554564].
- fixed_half NLL minus S8: 0.000061525; 6-date 95% interval [-0.0005105384392622733, 0.0005121854509806051]; 12-date interval [-0.00047258260440154583, 0.00046156698045078453].
- s8_global NLL minus S8: 0.000025527; 6-date 95% interval [2.93197257281521e-06, 4.9744446832835344e-05]; 12-date interval [5.076349465678175e-06, 4.879780175492029e-05].
- S8 H4 relative wealth versus equal_h4: 2.633420%; conditional 6-date interval [-0.00600410241238172, 0.0712784208535308]; 12-date interval [-0.008000229267630174, 0.07153890266874145].
- S8 H4 relative wealth versus s6_memory_h4: 1.658818%; conditional 6-date interval [-0.011018707377444177, 0.0472859574225311]; 12-date interval [-0.012024700483499446, 0.04788128638736808].
- S8 H4 relative wealth versus s8_legacy_cost_aware: -1.254726%; conditional 6-date interval [-0.27522019700622796, 0.30179901868082054]; 12-date interval [-0.25730456577553956, 0.3038925937136579].
- S8 H4 relative wealth versus equal_weight_assets: -5.891676%; conditional 6-date interval [-0.2871764881099783, 0.21441765503504567]; 12-date interval [-0.31090559427231346, 0.24565146618212275].

Soft-update and locality ablations both beaten at the point estimate: **True**. No favorable arm replaces S8 as the primary.

## Economics and year decomposition

S8 H4 at 10 bps: mean risky weight 63.69%, 0 cash-only dates, 38 trade dates, fees $810.24, maximum drawdown 12.26%.

| Origin year | S8 NLL | Fixed-average NLL | S6 memory NLL | S8 H4 net return | Equal-asset net return |
|---|---:|---:|---:|---:|---:|
| 2020 | -1.602678968 | -1.608225295 | -1.598246729 | 0.990% | 19.929% |
| 2021 | -2.560606881 | -2.556091681 | -2.565978196 | 7.351% | 4.671% |
| 2022 | -1.794126473 | -1.789190131 | -1.790943806 | -8.919% | -16.413% |

Year rows decompose the same carried path by forecast origin; they are not reset portfolios or selected subgroup tests. Final liquidation belongs to the last interval. All 0/10/20/40 bps outcomes and every crossed baseline are retained in ALL_BASELINES.csv; seed and year NLLs are in SUMMARY.json and ANNUAL_PREDICTION.csv.

## Evidence limits and preservation

New to this ETF successor lineage does not mean globally unseen: earlier research already used 2020/2022 on other assets. These are current-vintage survivor ETF histories, with assumed availability, ex-date dividend accrual, weekly reinvestment, zero cash yield and no actual fills. The four-week policy assumes persistent log-moment signals and diagonal risk; its surrogate is not a calibrated expected return. Conditional date-block intervals do not simulate refitting or policy reoptimization. Seeds are not independent market histories.

The scientific 0.01-nat threshold was preserved. The study stops after the frozen matrix, regardless of sign. Original 2025+ protected outcomes remain untouched. No funding certification, novelty, canonical confirmation, external publication or independent reproduction is established.

See AUDIT.json for the separate saved-output arithmetic, raw-quote/feature chronology and preservation audit; completion.json and resources.json under the output directory for execution. Earlier failed test collection and calendar fixture checks remain in the research directory with their source snapshots. Every market seed, forecast, checkpoint, gate state, ledger and data acquisition receipt is retained.


## Final audit, freshness and interpretation

The final arithmetic audit passed: **34,560 NLL values, 1,230 soft-gate decisions, 252 ledgers and 2,640 raw labels** reconciled; all **319 predecessor file hashes** remain unchanged. Maximum NLL error was 4.44e-15 and maximum accounting residual 4.37e-11 dollars. The full pre-intake suite passed 142 checks. After the narrowly scoped adapter repair, all 18 S8 checks passed; the 125 S6/S7 checks and their sources remain unchanged. Changed-source Ruff passed. This is same-host engineering verification, not independent retraining.

The original source/protocol freeze preceded the download. Final source freeze2 followed a recorded intake-only repair: an old TLT event dated 2004-12-31 was excluded by timestamp before amount access because it precedes every possible holding. No market fit or score had occurred. The earlier freeze and source archive, failed DNS download, failed construction, test collection/fixture failures, and analysis-import failure remain saved. Only the analysis invocation was corrected after fitting; no frozen source or market output changed.

The repository-wide freshness check found **SPY and GLD in a legacy ETF feature cache beginning in 2020**. Thus the evaluation was newly acquired and excluded from S6/S7 fitting and selection, but **is not globally untouched repository data**. No legacy outcome values were inspected during that metadata check. See FRESHNESS_AUDIT.json. No 2019 targets, features or feedback entered S8; the only 2019 access was diagnosis of already-saved S7 decisions.

Small predictive point gains do not establish useful new memory. Relative to the fixed-half blend, soft updates improved NLL by only 0.0000615 nats, with both intervals spanning zero. Context versus the global soft gate contributed only 0.0000255 nats; the global version actually ended $12.84 higher under the four-week policy. Ungated mixture routing had a better point NLL than S8; it remains a secondary control, not a replacement primary.

The four-week policy escaped the cash state without forced investment, averaging 63.69% risky exposure. It still had weak realized allocation: the zero-cost S8 path ended at $99,990.33, and the 10-bps path at $98,745.27. Their difference is **not a pure fee effect**, since changing cost also changes allocations; nominal fees at 10 bps were $810.24. On the carried path, S8 returned 0.99% in 2020 versus 19.93% for equal-weight assets, then 7.35% versus 4.67% in 2021 and -8.92% versus -16.41% in 2022. The later loss reduction did not recover the missed 2020 gains. These are descriptive decompositions, not a tuned regime rule.

**Terminal disposition:** predictive improvement remains inconclusive and below the frozen practical threshold; the primary economic gate failed. The bounded successor work is closed. A new evaluation or different policy requires a new hypothesis and freeze; neither 2019 nor this completed 2020–2022 run may be retuned to seek a favorable result.

The market worker completed three base fits in 14.19 seconds; peak RSS 272.61 MiB on macOS. One process/thread and the parent two-hour limit were retained under the documented user RAM waiver.

![S8 primary controls and uncertainty](/Volumes/PRO-BLADE/GitHubRepos/Wharton-Experiment/research/frem_s8_20261014/RESULTS_FINAL.png)
