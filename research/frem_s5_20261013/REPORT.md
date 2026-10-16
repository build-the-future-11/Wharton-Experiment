# S5-R1 development result — 13 October 2026

**Verdict: small exploratory benchmark point gains, no robust benchmark win, no evidence that contextual memory adds economic value over fixed expert averaging. The +6% target was not met.**

## Executed scope

Diagnostic audit of retained S3/S4 evidence; frozen S5-R1 specification; 15 targeted runtime tests; 12 generated mechanism cells; six complete retained-public fold/seed cells. The four retained experts are zero, Ridge, FREM and no-JEPA. Five routers and each standalone expert were scored. Seed ensembles were compared with equal-weight stocks, buy-and-hold, cash and a conservative volatility-managed basket at 10, 20 and 40 bps.

This is an explicitly amended lightweight pilot, not completion of the proposed fresh-trained S5 matrix. It reuses two spent public survivor windows and existing experts, instead of training adaptive-volatility/trend experts and a new three-fold matrix. There were zero new base-model fits, no protected access, no current Laura allocation, and no independent retraining. Router temperature selection and online updates did execute.

## Why predecessor memory did not help

- S3 combined correction: 2023 validation gain +0.003504 nats, evaluation gain -0.014116; 2024 validation gain +0.001378, evaluation gain +0.007006. Selection improvements were small and did not consistently transfer.
- Combined retrieval beat matched random by -0.003511 nats in 2023 and only +0.000166 in 2024. It did not establish retrieval-specific relevance.
- About 3.6 effective neighbors and average retrieved ages 127–140 days; these are diagnostic observations, not proof that increasing capacity or shortening age would fix results.
- Correction/residual correlations were only 0.015–0.049 across 2023 seeds and 0.104–0.127 in 2024. Descriptive pooled correlations do not establish a causal explanation.
- In S4, the best Euclidean nonzero candidate still harmed one validation half in every 2023 seed. The best worst-half gains in 2024 were just 0.000504–0.001951 nats, below the frozen0.01 requirement. All18 selections stayed zero; lowering the gate would change the question after observing failure.
- The evidence favors unstable/weak conditional residual relevance over a universally disconnected memory path. A unique causal root cause has not been established.

## Economic results

Relative terminal wealth, seed-ensemble context strategy, 10bps costs; horizons are twelve months in2023 and ten months in2024. Percentages are not annualized.

| Comparator | 2023 | Jan–Oct 2024 |
|---|---:|---:|
| Equal-weight stock basket | +1.09370% | +0.46622% |
| Buy-and-hold | +0.08127% | +0.21353% |
| Fixed equal-weight expert ensemble | -0.05514% | -0.00429% |
| Recent-loss router | -0.00363% | +0.00281% |
| Matched-random router | +0.01065% | +0.01018% |
| Linear context router | +0.00391% | +0.00019% |
| Ridge signal policy | -1.44144% | -0.00749% |

The context strategy ended with $118,041.93 and $123,341.44 respectively from $100,000. Fixed expert averaging ended higher in both windows ($118,107.05 and $123,346.74). The fixed ensemble is a mechanism-off control, distinct from equal-weight stocks.

At40bps the context strategy lost to buy-and-hold by0.31735% and0.23004%. The primary three-month-block intervals against equal-weight stocks were[-0.574%,+2.801%] and[-1.132%,+2.582%]; both include zero. Buy-and-hold intervals also include zero. The six-month sensitivity for2024 remains INSUFFICIENT_DATES. These are descriptive conditional-policy-path bootstrap intervals, not independent retraining or calibrated future coverage.

## Predictive and mechanism evidence

Context improved seed-mean NLL over equal averaging by about0.00101/0.00110 nats, below0.01. Recent-loss and linear routing scored better than context in both windows. The individual no-JEPA expert also scored better than the context mixture in both windows. Higher likelihood did not uniformly imply higher portfolio wealth.

The generated gate passed: null harm0.00000685 nats; recurrence gain versus recent0.94050 and versus random0.06216. However, linear routing outscored contextual retrieval on the generated recurrence, abrupt and delayed cases; recent routing also won on abrupt change. All these results remain visible. Generated signal recovery establishes implementation behavior only.

## Execution, verification and boundaries

Worker elapsed 5.934s; peak RSS 31.766MiB. Separate audit verified 7,128 score rows, 312 ledgers and 111 frozen identities. Maximum NLL discrepancy 4.44e-16; accounting residual 2.91e-11. This is independent arithmetic, not independent retraining.

Fresh-training admission failed: available MiB [279.9375, 318.328125, 382.078125], pageouts [99139, 99139, 99139]. Normal512MiB floor and pageout guard were retained. No inherited slow-mode exception was applied.

Attempt1 failed because the sandbox denied ps-based resource monitoring, before any mechanism/market output. Its receipt and exact source snapshot remain intact. A recorded infrastructure repair used50ms in-process getrusage peak checks and a parent wall timer with unchanged128MiB/300s caps; attempt2 was newly frozen. First lint findings were also retained and repaired before outcome evaluation.

## Disposition

Close S5-R1 without tuning these results. The full fresh-training campaign remains unexecuted; the pilot does not justify enlarging retrieval or spending the protected tail. Proposed follow-up priorities are stronger expert specialization, explicit risk/objective alignment, and data/precision readiness, evaluated under a new protocol. The upgrade backlog is a prioritized research queue, not a promise that every architecture feature will help.

Artifacts: specification.json, diagnostic.json, audit.json, start_decision.json, resource_monitor_repair.json; outputs/frem_s5_20261013_attempt2/{freeze,completion,mechanisms,resource_admission,resource_usage,process}.json.
