# FREM loss diagnosis and improvement checklist

15 October 2026. Updated status: saved-ledger attribution complete; mathematical references tested; successor implementation and experiment execution remain pending. See the [restart report](../research/restart_20261015/REPORT.md).

The immediate objective is to find useful forecasts and convert them into defensible portfolio decisions. Adaptive memory must then demonstrate an additional benefit. S6, S7 and S8 remain closed with their recorded outcomes. This checklist does not create a new scientific freeze, authorize a compute campaign, or change an existing success threshold.

**What is established**

S8's primary four-week policy ended at **$98,745.27** from $100,000 at 10 bps one-way costs. Equal-weight assets ended at **$104,927.25**. S8 beat equal forecast averaging under the same policy ($96,211.62), but this did not establish economic success. Its prediction gain over equal averaging was **0.0013675 nats**; both prescribed intervals include zero, and the practical threshold remains 0.01 nats. [Result and limitations](../research/frem_s8_20261014/FINAL_REPORT.md), [receipt](../research/frem_s8_20261014/RECEIPT.json).

The zero-cost S8 policy ended at **$99,990.33**. Nominal fees at 10 bps were **$810.24**. Fees therefore do not explain the whole failure. The zero-cost and 10-bps terminal values are not a pure fee decomposition because cost changes also change allocations. On the carried path, S8 gained **0.99% in 2020** versus **19.93%** for equal-weight assets; subsequent relative gains did not erase that deficit. Those observations identify where the shortfall occurred, not a causal explanation or a new rule for detecting rallies.

Evidence labels below mean: **measured** is supported by saved results; **source-confirmed** describes implemented behavior; **hypothesis** is an explanation whose economic effect still needs testing. Priority P0 means a prerequisite or a threat to the main interpretation; P1 means a major candidate improvement; P2 means conditional work after the core case is established. An unchecked item is unfinished work, not a claim that its proposed remedy will succeed.

**Repairs already made and retained**

| Earlier issue | Existing response | Remaining limitation |
|---|---|---|
| S6 omitted holiday intervals and dividends | S7/S8 use continuous common-session weeks and dividend accounting | Current-vintage data, assumed availability, ex-date accrual and ignored payment delay remain limitations. |
| S6 weighted experts by individual losses, losing mixture complementarity | S7 introduced joint mixture-likelihood fitting | Neither contextual weighting nor its incremental investment benefit is established. |
| S7's hard gate chose zero adjustment | S8 introduced delayed soft mixing | S8's gain over a fixed half blend is only 0.0000615 nats, with intervals spanning zero. |
| S6 forced an 80% risky anchor; S7 had coarse candidates | S8 allows continuous partial positions and cash | Allocation quality still fails the primary economic gate. |
| Forecast quality and policy quality were confounded | S7/S8 crossed forecasts with policies and cost levels | Extend that design for new hypotheses; do not present it as a new invention. |

Sources: [S6 result](../research/frem_s6_real_20261014/completed_ram_override/REPORT.md), [S7 diagnosis and result](../research/frem_s7_20261014/REPORT.md), [S8 specification](../research/frem_s8_20261014/SPECIFICATION.md).

**Checklist in execution order**

- [x] **C01 — Explain the existing portfolio loss precisely. P0; saved-ledger attribution complete.**

  **Why it matters:** The headline loss combines market exposure, asset weights, cash, holding decisions and costs. A prediction-score improvement cannot identify which of these helped or hurt.

  **Work:** Reconcile existing saved per-asset contributions, cash exposure, trade fees and terminal liquidation to the published totals. Report the existing yearly decomposition and same-policy forecast comparisons. Use only retained decisions for this diagnosis; do not run a new policy on closed 2019 or 2020–2022 outcomes. A pure fixed-decision cost counterfactual belongs in the separately specified future experiment.

  **How it could help:** It directs subsequent work toward the source of the loss and avoids attempting to repair gross allocation losses solely by lowering fees.

  **Required evidence:** Contribution totals reconcile to the ledger within documented numerical tolerance. Distinguish nominal fees from compounding drag and cost-induced changes to allocations. A decomposition is complete even if it does not identify a single causal defect.

  **Completed evidence:** The [new attribution](../research/restart_20261015/attribution_attempt5/REPORT.md) reconciles all 240 saved strategy ledgers and 36,720 interval records, with maximum dollar discrepancy 8.30e-10. Primary SPY contribution +$6,988.60, TLT -$5,511.30, GLD -$1,921.78 and fees $810.24 explain ending wealth $98,745.27. Origin-year totals carry wealth through the full path. No policy replay or causal improvement is claimed.

- [ ] **C02 — Give every expert an explicit forecast horizon. P0; source-confirmed mismatch risk, benefit unproved.**

  **Why it matters:** The inherited trend expert averages historical daily returns, and adaptive volatility is built from daily returns, while labels span weekly intervals of varying length. S8 then extends weekly moments over four holding intervals. The [expert code](../successors/frem_s6/model.py) and [policy code](../successors/frem_s8/policy.py) make these approximations visible. The earlier S6 diagnosis found that mechanically multiplying volatility by sqrt(5) worsened the saved mixture score.

  **Work:** Specify units and actual start/end times for features, targets, means, scales and allocations. Train/calibrate each expert for its declared return horizon. Compare a directly modeled holding-period forecast with the unchanged weekly approximation in a new study. Handle holiday calendars and variable intervals explicitly; do not assume square-root time scaling without its assumptions.

  **How it could help:** Prevent a daily signal or stale weekly estimate from being interpreted as a calibrated multiweek opportunity.

  **Required evidence:** Hand-verifiable horizon aggregation and maturity tests pass, including varying interval lengths and serial dependence. New chronological validation improves horizon-specific calibration and the predefined endpoint. Mechanical unit consistency alone does not prove predictive advantage.

- [ ] **C03 — Make the investment inputs economically meaningful. P0; source-confirmed surrogate, hypothesis about loss.**

  **Why it matters:** S8 optimizes a log-moment ranking surrogate with a diagonal variance penalty. It is not an exact expected-wealth or expected-utility calculation. Student-t log-return mixtures cannot be turned into finite expected simple returns by exponentiating their mean; exponentiating the full unbounded Student-t tail also lacks a finite expectation. Existing code documents the surrogate rather than claiming otherwise.

  **Work:** Keep the predictive density endpoint separate. Specify a decision model for horizon-matched total simple returns, using explicit finite-moment forecasts or finite joint scenarios with documented tail assumptions. Include actual costs and cash in portfolio wealth. Select/calibrate the decision mapping only within inner chronological development folds. Do not clip historical losses to make an optimizer look good.

  **How it could help:** Allocation would use tested return and downside estimates instead of treating a better density score as evidence of profitable trades.

  **Required evidence:** Scenario probabilities, return support, units and portfolio-wealth arithmetic pass exact small examples. Forward development checks validate the decision inputs and net utility against the unchanged surrogate under common constraints. If only economics improves, claim a policy improvement, not proof of the original memory hypothesis.

- [ ] **C04 — Improve the expert set before adding routing complexity. P1; weak incremental ensemble evidence is measured.**

  **Why it matters:** Memory cannot manufacture useful information absent from its experts. S6's eight families all lost predictively to fixed averaging at the point estimate. Conversely, an expert with poor standalone likelihood can still improve a mixture; blindly pruning it can repeat S6's complementarity failure.

  **Work:** Benchmark the existing zero, ridge, trend, volatility, reservoir and tree experts under identical information and fitting budgets. Measure joint mixture contribution, calibration and error diversity. Use a prespecified inner-fold rule for retaining or excluding experts, with the full fixed ensemble retained as a control. Add a new specialist only when a specific missing capability is identified.

  **How it could help:** Improve the underlying information and avoid dilution without removing useful complementary distributions.

  **Required evidence:** The selected expert set beats the fixed full ensemble on forward development folds, including all declared seeds. Report both density and decision outcomes; a post-hoc winner from S8 is not a validated replacement.

- [ ] **C05 — Test whether stale fitting and calibration matter. P1; chronology measured, causal effect unproved.**

  **Why it matters:** S8 trained before 2017, calibrated scales in 2017 and initialized its gates in 2018, before evaluation in 2020–2022. Distribution drift is plausible, but these dates alone do not prove it caused the loss.

  **Work:** Define a small comparison of fixed versus scheduled expanding-window fitting and calibration. Freeze the update schedule before the new evaluation. Use only labels that have matured before each decision, purge overlapping targets, and give all competing methods the same new information and budget.

  **How it could help:** Refresh return/risk estimates when the old relationship has changed, while preserving a causal forecasting sequence.

  **Required evidence:** Chronology audits pass for every update; calibration and forward performance improve across the predefined folds. Log update costs. Reject the change if refitting adds instability or merely benefits from an unfair information advantage.

- [ ] **C06 — Model joint portfolio risk. P1; diagonal-risk assumption is source-confirmed.**

  **Why it matters:** The current penalty ignores cross-asset covariance. Its impact on this particular loss has not been isolated.

  **Work:** Add a past-only shrinkage covariance baseline or synchronized joint return scenarios. Compare it with the existing diagonal-risk policy using identical forecasts, limits and fees. Inspect downside and concentration as well as ending wealth.

  **How it could help:** Distinguish genuine diversification from several holdings exposed to the same shock.

  **Required evidence:** Covariance matrices are finite and positive semidefinite; known correlated examples produce the expected risk ranking. Forward net utility and downside outcomes meet declared criteria. Better drawdown alone does not satisfy an unchanged ending-wealth target.

- [ ] **C07 — Separate signal skill from exposure and trading decisions. P1; measured underperformance, individual causes unresolved.**

  **Why it matters:** S8 averaged 63.69% risky exposure, with an 80% risky budget and per-asset caps, while the broad equal-weight asset benchmark has different exposure. It also pays to rebalance. These differences can dominate small forecasting gains.

  **Work:** Retain the original cash and asset benchmarks. Add constraint-matched and risk/exposure-matched controls in the future protocol; publish both sets. Compare declared holding/no-trade policies under the same forecast and risk model, preserving self-financing costs and liquidation. Test horizon persistence rather than assuming it. Include an explicit, sourced cash-yield convention and cost sensitivity.

  **How it could help:** Reveal whether gains come from better information, taking more risk, holding more cash, or paying fewer costs; avoid trades whose forecast benefit does not justify their burden.

  **Required evidence:** The proposed policy wins the prespecified net endpoint under common constraints without hiding the broad benchmark result or violating risk limits. No forced risky exposure or cost assumption chosen after seeing a rally.

- [ ] **C08 — Require contextual memory to earn its place. P1; tiny incremental effects measured.**

  **Why it matters:** In S8, context added about 0.0000255 nats relative to the global gate, while the global version ended $12.84 higher economically. Soft updating added only about 0.0000615 nats over the fixed half blend, with uncertainty spanning zero. These results do not justify a larger memory system.

  **Work:** Keep fixed averaging as the incumbent. Test a small declared family containing recent-loss weighting, global mixture weighting, fixed blending and the proposed contextual mechanism. First establish that observable past context predicts differences in expert reliability. Evaluate delayed availability and memory-state leakage explicitly.

  **How it could help:** Use adaptation only when it captures repeatable changes in expert advantage; otherwise retain the simpler forecast.

  **Required evidence:** Incremental out-of-time benefit over the simple controls and the relevant mechanism-off ablations, under the frozen practical threshold. If memory fails, retire it from the selected system and retain the negative research finding.

- [ ] **C09 — Retain JEPA, reservoirs and delayed updates only with component evidence. P2; weak S6 ablation differences measured.**

  **Why it matters:** S6's no-JEPA and immediate-update comparisons were nearly identical to the full model. Architectural presence is not proof of useful learning.

  **Work:** Inspect retained optimization and representation diagnostics. If a specific defect or undertraining hypothesis is identified, specify one bounded, compute-matched component experiment. Retain gradient, anti-collapse and checkpoint checks. Remove a component from a future selected candidate when its contribution is unsupported; preserve the old version.

  **How it could help:** Spend compute on demonstrated capability and reduce unstable or redundant mechanisms.

  **Required evidence:** The component changes the intended learned quantity and improves a declared forward endpoint against a matched control. Larger parameter counts, more epochs or an attractive architecture name do not count as success.

- [ ] **C10 — Establish an honest data and freshness contract. P0; evidence limitation measured.**

  **Why it matters:** The canonical panel is absent. The current three-ETF history is retrospective and current-vintage, with assumed availability and simplified dividend/cash handling. Legacy SPY/GLD feature exposure means the S8 interval is not globally untouched. These limitations restrict what can be concluded; they do not explain away the observed loss.

  **Work:** Inventory prior asset/date access using metadata before choosing new data. Admit a narrowly defined dataset with truthful provenance, calendar, actions and availability. Broader equity claims additionally require historical membership, terminal outcomes and FX. Reserve confirmation data only after verifying that it is unused; do not automatically declare another calendar year fresh. Keep the existing 2025+ protected boundary closed during design.

  **How it could help:** Make a future advantage interpretable and prevent leakage or survivor selection from producing an apparent win.

  **Required evidence:** A passing admission audit, complete source identities and a documented exposure inventory. If no suitable unused retrospective interval exists, propose prospective collection rather than relabeling a spent interval.

- [ ] **C11 — Design a small experiment capable of resolving the question. P0; current uncertainty measured.**

  **Why it matters:** Three seeds on one 153-week path are not three independent market histories. More variants on the same path do not cure weak precision and create selection opportunities.

  **Work:** Use development-only date blocks to estimate uncertainty and a plausible detectable effect. Freeze a finite trial budget, chronological folds, complete seeds, primary comparisons, costs, multiplicity handling and stopping rules. Stage the work: baseline and contracts; one forecast change at fixed policy; one policy change at fixed forecasts; only then memory. Exact budgets and dates remain to be set before execution.

  **How it could help:** Obtain a discriminating answer with fewer confounded changes, rather than an expensive sweep that selects a lucky configuration.

  **Required evidence:** A complete protocol and resource admission before fitting. For the original scientific claim, the prescribed 95% interval must clear the existing 0.01-nat practical margin against the named control with all required seeds. Economic success needs its own frozen net-benefit, uncertainty and risk criteria. The earlier 6% uplift target, if retained, must remain explicit; a smaller gain cannot be renamed a 6% success.

- [ ] **C12 — Reproduce and release the supported conclusion. P1; independent retraining is absent.**

  **Why it matters:** Arithmetic and source-hash audits verify bounded properties. They do not establish independent training reproducibility, novelty or deployment readiness.

  **Work:** Reproduce the selected protocol in a clean environment, regenerate scores and ledgers, independently review the scientific claim and prior-art boundary, and build paper tables directly from artifacts. Keep the user-facing current state distinct from dated blocked attempts without deleting history.

  **How it could help:** Turn a promising result into a checkable research finding or complete an honest negative/inconclusive study.

  **Required evidence:** Matching result generation within declared tolerances, complete claim-to-artifact links and an independent review. A valid negative result can complete the research; it cannot be labeled predictive or investment success.

- [ ] **C13 — Validate Laura-specific decisions separately. P2; planning implementation exists, empirical funding support does not.**

  **Why it matters:** Conditional cash-flow calculations do not certify investment returns or a funding probability. Current instrument eligibility remains unverified in the project record.

  **Work:** Resolve dated official rules, approved instruments, mandate constraints and transaction inputs. Apply the validated decision model to dated contributions, payments and protected reserves. Calibrate any funding probability on appropriate evidence rather than treating a scenario grid as confidence.

  **How it could help:** Evaluate success against the client's liabilities and constraints rather than an isolated backtest return.

  **Required evidence:** Verified mandate inputs, correct liability accounting and supported risk/funding calibration. No portfolio weights, trade orders or external submissions follow from this checklist.

**Proposed delivery sequence and stopping rules**

1. Deliver C01's read-only attribution, C02/C03's mathematical contracts, and C10/C11's data/experiment feasibility. Stop before fitting if the problem cannot be evaluated honestly or precisely enough.
2. Implement only the selected successor components in a separate version; preserve frozen S6/S7/S8 code. Verify horizon, return, risk, chronology and ledger invariants before an admitted bounded run.
3. Test forecasting and policy changes separately against the simple incumbent. A failed change is retained and rejected; it is not bundled into a larger model until something wins.
4. Test memory and architectural additions only after the underlying forecast/policy case is credible. Freeze the final candidate before protected confirmation.
5. Reproduce, audit and write the actual supported, mixed, negative or inconclusive conclusion. There is no promised return improvement or guarantee that the memory hypothesis will survive.

The next deliverable is a falsifiable successor specification and a bounded execution contract, not an assumed S9 success. No new model fit, policy replay, protected-outcome inspection or scientific test was performed to produce this checklist.
