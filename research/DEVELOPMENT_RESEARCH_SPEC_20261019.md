# Development research specification: integrity before model claims

Date: 2026-10-19. Phase: **DEVELOPMENT**, with a completed-source audit and execution gated by retained verification receipts. **Novel method status: `NOT_ESTABLISHED`.** This specification contains no asserted run counts, performance estimates, or passing-test claims.

## 1. Research contract and scope

The working scientific source descends from `fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c` in `build-the-future-11/Wharton-Experiment`. It contains twelve repaired model implementations and model-specific synthetic task adapters. The separate public FREM partial prototype remains a separate implementation. Neither the twelve labels nor the prototype name establishes a complete trained Fourier–reservoir–JEPA architecture.

The proposed present contribution is reproducible engineering integrity: identify a concrete defect, preserve its counterexample, implement a bounded correction, and verify the exact resulting behavior. A secondary development study may characterize the existing model components on constructed data. Its outcome may be negative.

The [existing model protocol](../protocol/PORTFOLIO_RESEARCH_V1.md), source/configuration snapshots, and historical evidence remain preserved. This document does not rewrite old results or authorize access to a spent or protected evaluation set. The [session contract](development/integrity_20261019/contract.json) and machine-produced execution receipts govern actual work. An amendment must record the additional development matrix below before its first cell starts; a plan in this document is not an execution receipt.

Related primary sources and their precise relevance are in [the accompanying review](../literature/INTEGRITY_RELATED_WORK_20261019.md). Financial benchmarking, multiple-search accounting, and transaction-aware evaluation are established concerns in that literature; this pass makes no priority claim for them.

## 2. Layer A: falsifiable engineering properties

These are the properties targeted by this session's deterministic fixtures. Implementation presence and test definitions are known; executed outcomes must be established from the retained command records before a property is marked verified.

| ID | Property and falsification condition | Required evidence |
|---|---|---|
| E1 | Every admitted split has a positive training/test extent, preserves the declared purge, and has no train/test intersection within a fold. Invalid size, type, step, or lockbox parameters must raise before slice construction. A single admitted overlap falsifies the property. | Before-fix counterexample and patched `test_split_boundaries.py` results; the exact ordinary and lockbox-only boundaries used. |
| E2 | A selected MPC policy can be passed unchanged into its ledger, meets declared caps/cash constraints, and reproduces its reported objective. Materially infeasible optimizer output is rejected rather than projected into success. | `test_mpc_policy_contract.py` results, injected optimizer output, exact returned weights, reconstructed objective, and representative unchanged finite-bank choices. |
| E3 | An as-of snapshot uses observations with `event_time < decision_time` and `available_time <= decision_time`, and only membership updates effective and available by the decision. Future events or unpublished revisions cannot alter the earlier snapshot. | `test_point_in_time.py` results covering future perturbations, publication delay, revisions, membership, missingness, ordering, and hash admission. |
| E4 | Scalers and fitted forecast parameters use only admitted training information; labels must have matured at each fit. Perturbing future values cannot change forecasts already emitted. | `test_development_demo.py` results, saved training bounds, maturity flags, scaler/parameter states, and replayed predictions. |
| E5 | Cash, holdings, costs, paid obligations, and any unpaid obligation follow the declared conservation equation without implicit funding or repeated payment. | Existing accounting tests plus analytic entry-cost, zero-return, liquidation, exhaustion, and carried-position checks; residuals and units recorded. |

E1 concerns each fold's training/test boundary; the generic splitter deliberately permits successive test windows to overlap when a smaller positive step is requested. The new lockbox-only convention is explicit: such a fold is valid when at least `min_train` samples plus the purge gap precede its held-out segment, even if no ordinary `test_size` window fits.

E2 preserves the existing economic objective and finite policy bank. Its bounded solver correction validates attempted controls first, clips accepted bound roundoff, and repairs the investment total on a capped simplex. It does not claim a global optimum, and an unsuccessful numerical optimizer does not become successful merely because a feasible bank policy exists.

E3 checks declared record clocks and file hashes. A manifest field stating “market” or a licence string is not independent proof of vendor authenticity, usage rights, point-in-time correctness, or corporate-action handling. External data evidence remains a separate admission requirement.

## 3. Defect and limitation ledger

Severity is relative to the claim that could be invalidated. The implementation status below records this document's source-review boundary; final validation status belongs in execution receipts.

| Flaw or gap | Cause | Severity | Repair or containment | Required evidence and present status |
|---|---|---|---|---|
| Oversized lockbox trains on held-out rows | A negative training-slice stop is interpreted relative to the array end; insufficient parameter checks | High: invalid holdout | Validate integer types, positive step, nonnegative purge/lockbox, available training history, and every interval | Split repair and unittest fixtures implemented; execution result must be linked. |
| SLSQP policy is scored after projection but returned before projection | Optimizer-result tolerances differ from ledger tolerances; original weights were retained | High: non-executable reported policy | Canonicalize only accepted solver roundoff under caps/budget, then score exact returned weights | Controller repair and injected-result fixtures implemented; execution result must be linked. The historical finite-bank route is not shown invalid by this defect. |
| ETF cache ignores requested dates and can label unidentified arrays as nonsynthetic | Cache identity omits interval; sidecar labels need not match the data hash; unknown provenance defaults to a permissive flag | High: invalid market provenance and information boundary | Require explicit file/hash/source/clock/corporate-action/missingness/universe declarations on a separate admission path; reject unverifiable evidence | New admission module and fixtures are present. The legacy ETF path is not thereby certified or fully repaired; no real market dataset is admitted by this specification. |
| Earlier repaired evaluator computes serial covariance on reordered blocks and artificial joins | Latest fold emitted first, then error blocks concatenated; omitted rows lose their time gaps | High for p-value claims | Retain true time identifiers, restore chronology, and adopt a justified gap rule before new HAC inference | Open limitation of the preserved earlier evaluator. This development design makes no fresh DM/BH claim using that route. |
| Complete FREM training route and evidence are absent from the admitted source evidence | Model names and separate repositories can be conflated with an implemented, trained composition | High for architecture/novelty claims | Require pinned module/symbol inventory for Fourier features, learned latent prediction, recurrent state/readout, prediction head, and integrated evaluation before making a composition claim | Separate partial prototype acknowledged; complete FREM and its market performance remain `NOT_ESTABLISHED`. |
| Component benefit is confounded with nonlinear capacity or extra features | A richer predictor can beat linear ridge for reasons unrelated to the named component | High for mechanism attribution | Compare with own ablation, record feature/training-budget differences, and require capacity-aware nonlinear controls in a subsequent frozen study | Current task baselines retained. Full matched nonlinear comparison is outside the bounded matrix and remains required for stronger mechanism claims. |
| Market labels and company data lack admissible historical vintages | No recovered evidence here for availability clocks, historical membership, delistings, revisions, adjusted-price policy, or a licensed market panel | High for market generalization | Recover and hash a documented dataset; define targets, release lags, revision policy, survivorship treatment, and execution timing before any market run | Real-market evaluation blocked. Synthetic arrays cannot fill this gap. |

Two analytic witnesses motivate the repairs. `walk_forward_folds(100, min_train=20, test_size=10, purge_gap=5, lockbox_size=98)` formerly formed training slice `[0:-3]` and test slice `[2:100]`, overlapping 95 rows. A successful SLSQP attempt with weights `[0.60000003, 0.40000002]` could satisfy the controller's tolerance yet fail its ledger; complementary two-scenario returns make this attempted policy preferable to all bank candidates, so the witness exercises selection rather than an unused trial.

For the inference limitation, chronological differences `[1,2,3,4,5,6]` become `[5,6,3,4,1,2]` when two-row blocks are concatenated newest first. At lag one, their centered cross-product sums are respectively `8.75` and `4.75`; the corresponding Bartlett HAC variance expressions are `4.375` and `3.708333...`. These are hand-derived witnesses, not reported empirical results. The estimator's ordering requirement is directly documented in [R9](../literature/INTEGRITY_RELATED_WORK_20261019.md#r9-official-hac-implementation-contract).

## 4. Layer B: model development question

**Question:** within each predeclared constructed world and task, does an existing repaired model's full variant improve its own loss over its declared ablation and its information-matched baseline? Which components fail to help, and are any apparent benefits consistent across the fresh development seeds?

This is a characterization of existing implementations, not confirmation of a new financial mechanism. It does not test a jointly trained FREM model, real returns, suitability for a client, or a trading strategy's market alpha.

For model `m`, world `w`, and seed `s`, retain the declared primary aggregate loss `L(m,w,s,variant)` and its baseline loss. Report both paired contrasts:

- Baseline contrast: `D_base = L(full) - L(baseline)`.
- Component contrast: `D_component = L(full) - L(own_ablation)`.

Negative values favor the full variant under that task's loss convention; positive values favor its comparator. A favorable baseline contrast alone does not identify the component's contribution. A favorable component contrast with an unfavorable baseline contrast shows that the component may help an otherwise weaker system. With incomplete pairs, report missingness rather than compute a replacement comparison.

## 5. Bounded development design

| Item | Predeclared design |
|---|---|
| Source | Pinned repaired parent plus separately hashed session repairs; exact source tree recorded before running |
| Phase | `DEVELOPMENT`; no confirmation or market execution |
| Seeds for the additional matrix | `2026101911`, `2026101912`, `2026101913` |
| Optional interface preflight | Distinct seed `2026101910`; its subset recorded before execution; receipts kept separate and never pooled into matrix results |
| Worlds | `null`, `linear_signal`, `regime_shift`, `heavy_tails`, in this order |
| Models and variants | M01–M12; full plus each model's single existing declared ablation |
| Maximum matrix | `12 models × 2 variants × 4 worlds × 3 seeds = 288 cells`; this is an upper bound, not a completion claim |
| Data size | Existing 320-step, five-anonymous-asset generators; training origins 20–235; test origins 240–259 (`n_test=20`); five-step labels when the task requires them |
| Scaling and maturity | Common scale fit on training-era returns; target maturity strictly precedes the first outer test decision; model-specific training bounds retained |
| Compute boundary | One local CPU lane, one numeric thread, no paid compute or Actions dispatch; total child-process wall-time budget 900 seconds across the session, including earlier verification and any preflight |
| Execution order | World order above, seeds ascending, model IDs ascending, then full and declared ablation; changes require a retained amendment before new outcomes |
| Partial-run rule | Preserve completed cells and every failure/timeout; unstarted cells remain unstarted; do not change architecture, seeds, or baseline to fill the table within the deadline |
| Existing engineering demonstration | Its separate contract/seed and outputs remain separate; they are not extra matrix replications |

The synthetic generator has an IID null, a linear autoregressive signal, a switch in autoregressive coefficient and innovation scale at the training boundary, and Student-t heavy-tailed innovations. Exact parameters remain those in the [source builder](../src/wharton_lab/research/causal_portfolio.py) and existing model protocol. These are designed stress environments, not observations of financial market regimes.

The runtime must gate expensive cells on dependency availability, memory, and the remaining time budget. If a required dependency is unavailable or a bound is reached, the appropriate result is a documented blocked or partial matrix. Reading protected outcomes or inventing synthetic market substitutes cannot resolve such a block.

## 6. Preserve the actual task definitions

The following table is derived from the pinned model protocol and `scripts/run_portfolio_research.py`; it is not a cross-model leaderboard.

| Model | Primary task/loss | Declared main baseline | Own ablation and interpretation constraint |
|---|---|---|---|
| M01 | Quantile forecast; mean pinball loss | Ridge prediction plus training-only residual quantiles | Remove regime feature; empirical quantiles retained as a further reference |
| M02 | Future residual-return ranking; negative date-level Spearman correlation | Residual-label ridge ranker | Train raw-return rather than residual labels; ranking target remains explicit |
| M03 | Next-return prediction across assets; MSE | Linear PCA–AR(1) using the same history/factors | Legacy single-snapshot loading target; this is partly a defect comparison, not clean proof of a new nonlinear mechanism |
| M04 | Next-return prediction; MSE | Information-matched full-training ridge | Remove matured episodic memory; retain fixed-backbone output and calibration/backbone row counts |
| M05 | Next-return prediction; MSE | Information-matched ridge | Use linear experts only; a one-batch experiment does not establish birth/merge/prune benefits |
| M06 | Next-return prediction; MSE | Ridge on the same flattened windows | Remove predictive latent loss while retaining other objective terms; preserve internal horizon purge |
| M07 | Future realized covariance; Frobenius error | Trailing 20-step sample covariance | Remove context; global training covariance remains a further reference; implemented task is conditional SPD regression |
| M08 | Next-return node forecasts; mean asset MSE | Information-matched multioutput ridge | Remove graph edges; implementation is supervised with a graph frozen from training history |
| M09 | Next-return prediction; MSE | Information-matched ridge | Fix group weights; random fusion features are not a trained representation |
| M10 | Joint future path distribution; energy score | Conditional ridge residual bootstrap | Remove context; unconditional bootstrap is an additional reference |
| M11 | Hypothetical liability policy; empirical CVaR95 of unpaid amounts | Cash-only policy; equal-weight reference also retained | Require full investment; finite-bank open-loop control under the existing ledger and costs |
| M12 | Next-return prediction; MSE | Information-matched ridge | Remove memory; scenario-to-planner diagnostics are separate from the primary task and are not jointly trained FREM evidence |

M03 refits prequentially from history available at each test origin; most other predictors are frozen across their outer block. M04 partitions its training interval between backbone fitting and residual-memory calibration. M07's realized covariance target has sampling noise and a horizon different from its trailing covariance reference. These differences must be visible in interpretation.

M11 and the M12 planner diagnostic use reset-state, nonoverlapping five-step windows. With `n_test=20`, only four such windows are available per seed. The exact empirical CVaR at level 0.95 then equals the worst of those four observed losses, because the tail mass is only `4 × 0.05 = 0.2` observations. It must not be described as a well-estimated population 95th-percentile tail risk or a live receding-horizon backtest.

## 7. Evidence, analysis, and decision rules

The independent replication unit is the simulation seed within a world. Overlapping targets, dates, assets, forecast samples, and saved files are not additional independent seeds. Report all three paired seed values, their arithmetic mean, spread/range, and direction. If a t interval is emitted, label it a descriptive development approximation with two degrees of freedom and no robustness or discovery implication.

No discovery threshold, best-seed selection, pooled cross-task ranking, or confirmatory p-value is authorized. A loss contrast can support the narrow statement that the full variant performed better or worse on these development fixtures. Three consistently favorable seed contrasts may motivate further study; they do not establish market generalization or statistical discovery. Mixed signs, deterioration, failed cells, or a stronger simple baseline are scientifically admissible outcomes.

Retain a command/configuration record, parent commit, source hashes, package versions, thread limits, every attempted cell, dataset hashes, predictions, targets, per-observation losses, timing, exceptions, and saved model/scaler states where supported. Numerical replay must reconstruct recorded metrics from those artifacts. Source changes after a failure require a new source fingerprint and an explicit amendment; historical receipts remain available.

Promotion to a stronger study requires corrected engineering gates, complete paired observations, an explicit mechanism distinct from known ingredients, capacity-aware nonlinear controls, and an independently specified evaluation boundary. Promotion to market evaluation additionally requires admissible dated data and an executable portfolio protocol with costs, turnover, risk, and benchmark definitions. Those are conditional future requirements, not work claimed complete here.

## 8. Honest completion states

| State | Meaning |
|---|---|
| `ENGINEERING_VERIFIED` | Required deterministic checks pass at a recorded source fingerprint; this says nothing about forecast superiority |
| `DEVELOPMENT_COMPLETE` | Every predeclared matrix cell has a terminal receipt and all intended paired analyses are available; failures remain visible |
| `DEVELOPMENT_PARTIAL` | Budget, dependency, memory, or recorded cell failures leave comparisons incomplete |
| `BLOCKED_MARKET_DATA` | No admitted real-market panel supports a financial performance claim |
| `NOT_ESTABLISHED` | Current state of novelty, complete integrated FREM training, and market-alpha claims |

An engineering fix can be complete while the scientific component hypothesis fails. A bounded development matrix can be useful while market evidence remains blocked. Neither outcome should be rewritten into an unsupported success claim.
