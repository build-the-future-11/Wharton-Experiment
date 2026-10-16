# S10 revision 2 adversarial design review

Date: 15 October 2026. Review by a separate assistant agent within the same session and repository. This is internal adversarial review, not external independent verification, independent reproduction, numerical validation, or evidence of market performance.

Scope: read-only review of revision 1 `MODEL_SPECIFICATION.md`, `EXPERIMENT_PROTOCOL.md`, `design.json`, `REVIEW.md`, `IMPLEMENTATION_PLAN.md`, `FAILURE_REPAIR_MAP.md`, and project instructions. No market source values, protected outcomes, fits, policy replay, scientific tests, or predecessor changes were used. This file is the reviewer's sole write. Parent-selected revision 2 resolutions are distinguished below from additional recommendations.

## Verdict and change taxonomy

Revision 1 is a substantial research proposal but not a literal implementation contract: developers could make materially different timing, statistical, resource, and checkpoint choices while following its prose. The following additions close that ambiguity; they do not establish that the model is optimal, profitable, novel, calibrated, or scientifically answerable.

- **C — contract clarification:** makes an intended behavior exact without changing its stated scientific meaning. Still record the final bytes in revision 2 before implementation.
- **D — design correction/change:** resolves a contradictory or genuinely unspecified scientific choice. Preserve revision 1 and record why revision 2 differs. No existing evaluation may inherit a new interpretation.
- **E — engineering requirement:** fixes implementation behavior without altering the frozen model, subject to an equivalence check.

The parent has selected the opening-quote clock, dividend event order, frozen A2 distribution, staged precision dependency, exact solver/grid/bootstrap contracts, a concrete precision rule, terminal market interruption policy and diagnostic units in `MODEL_CLARIFICATIONS.md` CL-01 through CL-12. Those decisions are resolved at design level; they remain unimplemented and untested. This reviewer read those clauses after the initial audit and found no new blocking mathematical contradiction. Suggestions here never override those normative clauses. A receipt listing a finding as reviewed is not itself proof its chosen contract was implemented.

## 1 Timing, ownership, and fold construction

### F01 — pre-open knowledge contradicts open-valued holdings (P0, D; parent selection recorded)

**Cause:** revision 1 decides at 12:00 UTC, but current risky dollars, cap feasibility, turnover, and the exact self-financing objective require the later open prices. Assuming ideal target weights does not resolve which prices were known when the policy ranked actions.

**Selected revision 2 correction:** use the scheduled 09:30 America/New_York opening quote as an assumed observation; decision occurs one microsecond later and ideal fill one further microsecond later at that same quote. Convert the exchange-local clock to UTC with a recorded timezone database. The quote is strictly earlier than the decision. Label availability remains midnight UTC following the endpoint session, so observing today's quote does not permit today's matured target to update forecast state immediately.

**Required disclosure/evidence:** this is a zero-latency research convention using assumed quote timing, not verified historical tick provenance or an executable fill. Valuation, caps, turnover and cash equation all use the same observed opening vector. A later change to pre-open, next-open, slippage, or delayed fills is a new decision contract, not a code cleanup. Fixtures must cross US daylight-saving transitions and a same-day label endpoint.

### F02 — correct dividend interval needs explicit event ownership (P0, C; parent selection recorded)

Revision 1's interval `origin < ex_date <= endpoint` is correct for shares bought at origin open and sold at endpoint open under the stated ex-date-credit convention. Do not reverse its endpoints.

Process an ex-date dividend against the shares carried into that session, before any same-open sale or purchase. Existing shares sold at that opening receive the credit; newly purchased shares receive none for that date. A dividend on the first entry date therefore contributes zero to newly opened positions, while an endpoint dividend belongs to the closing interval. Each event has a stable source/action identity, and dividend credits are exactly once.

The same order must apply to ordinary days, rebalance days, A4 forced liquidation, buy-and-hold entry, and final liquidation. The target's accumulated dividend, the per-event ledger, and an explicit share-count reference must agree. Ex-date credit remains an approximation to spendable cash; it is not a payment-date receivable model.

### F03 — calendar and fold endpoints must be integer-indexed (P0, C)

Build one admitted sequence of complete ISO-week common-session anchors `a[0],...,a[m]`. Never create the sequence by dropping bad observations until the data happens to become common. Validate each instrument's required session support first. Unsupported missing sessions or corporate actions fail admission.

For fold boundary `[s,e)`, select the first anchor index `k0` with decision time in that interval and all feature/library/warm-up requirements satisfied. Its eligibility must be determined without reading the realized return of that evaluation block. Subsequent decision indices are exactly `k0+4j`, and a block exists only if `k0+4j+4` exists and its endpoint session is strictly before `e`. Do not count four *remaining eligible* anchors after skipping an invalid interior origin; an invalid continuous segment fails the fold. Freeze the admitted anchor-index list, exact event timestamps, first/last decisions, terminal liquidation, unused terminal sessions and exclusion reasons.

Fold A and B reset dollars to $100,000 independently. Their forecast state is chronological, and B may contain matured A information. They are not independent replications. Specify whether the B snapshot is reused from the single main chronology or reconstructed byte-identically; do not accidentally process A twice when building B.

### F04 — delayed queues need an ordering and uniqueness key (P0, C/E)

At a decision timestamp, collect labels with `availability < decision`, sorted by `(availability, origin, stable_record_id)`. Reject duplicate label/forecast identities. A/B coefficients must refer to the original issued expert weights, original atoms, original fixed score scales and original forecast hash. Store pending forecasts across their complete maturity delay. Do not regenerate their experts from today's library.

Use deterministic ordering before any bounded-window eviction. For equal maturity times, a stable order prevents different surviving 104-record queues. Changing future values, future source order, or unavailable labels must leave earlier states and decisions unchanged. Hashes must cover canonical values and order rather than filesystem enumeration order.

## 2 Preparation and online-control contracts

### F05 — precision and implementation previously depended on each other (P0, D; parent selection recorded)

Revision 1 requests precision before an answerable implementation, yet candidate/control precision requires a functioning chronological model. Selected order: static metadata/source admission, tested minimal precision kernel, one bounded admitted pre-2010 preparation, full empirical system, engineering verification and final freeze. The minimal kernel is actual specified model code with constructed numerical/chronology tests; it is not a market performance demonstration or placeholder.

The precision stage has its own saved input boundary, version, output namespace, consumption/attempt record, resource accounting and disposition. Its output cannot select a favorable neighbor count, score transform, threshold, dates or policy. A data insufficiency or precision failure is retained. Full market evaluation remains unauthorized in this design-only session.

### F06 — burn-in membership and insufficient history (P0, C)

Define the burn-in population as all feature-complete origins whose H4 labels have availability strictly before 2008-01-01 UTC, using the same admitted calendar and dividend target. It is not the later 520-example scenario window. Require 64 distinct origins after all fixed timestamp rules, and record count, min/max origin, maturity range and transform hashes. Population SD uses divisor `n`, not `n-1`; features with SD below `1e-12` use 1, score SD uses `max(SD,1e-4)`.

The first issued prequential forecast on/after 2008-01-01 still requires 64 matured library examples and 13 available weekly returns. If either is unavailable, admission reports the exact first eligible issue and reason. Do not reduce a minimum to make the source work. The exact minimum and resulting count must be frozen before the preparation computes scores.

**Remaining precision limitation:** CL-09 selects at least 24 four-interval pilot blocks in 2008-2009. Those two years contain only approximately 26 such blocks, so this necessarily includes early feedback cold starts. Requiring 24 fully warmed pilot blocks would be incompatible with the fixed two-year window: accumulating 33 matured weekly forecast records plus H4 delay consumes roughly 37 weeks. Record the exact pilot anchor, every cold-start/global-fallback flag, and the count of structural zero paired differences. The whole-window pilot is a disclosed heuristic; it must not be described as a fully warmed, stationary sample representative of later evaluation state. Do not fix that limitation by silently starting feedback before 2008 or opening outer-fold outcomes.

### F07 — no-feedback A2 needs a frozen predictive law (P1, D; parent selection recorded)

Freezing a library and mixture parameters while recomputing query-dependent C probabilities is still an adaptive forecast. Parent selection is to freeze atoms and the final predictive probabilities at each fold's first decision after normal causal preparation. A2 scores later observations but does not feed them into its frozen predictive law. Each fold receives its own causal snapshot; A2 B may be prepared from pre-B history but cannot use B outcomes. Preserve the first frozen forecast identity for every A2 decision.

This is a precise no-update-distribution control. It changes both access to newly matured exemplars and contextual query response; interpretation must not pretend it isolates only one of those operations.

### F08 — ablation dependencies and nearest-neighbor thresholds (P1, C)

- A1 changes atoms and their proper-score cross terms; rebuild its complete issued-forecast and matured-score chronology from the same causal start. Its library admission still requires the full H4 label maturity. Fixed pre-2008 scales stay fixed unless a separate change is declared.
- A6 changes library capacity and therefore every affected expert forecast and A/B record. Rebuilding only the last decision is invalid.
- A7/A8 use `k=16/64` for both expert and local routing, require at least `k+1` leave-one-out contexts, and use `n_eff>=8/32`. Keep the blend pseudocount 64 and maximum blend .5, regularization .05, recency half-life 52, and 104-record feedback cap unchanged. At 64 neighbors, the oldest eligible population still needs at least 65 contexts.
- A3 can share exact U/R/C issued coefficients with main chronology because it only changes the final blend. A4/A5 share exactly the main forecasts; their ES differences must be exactly zero by identity, not merely statistically insignificant.
- Every arm's initial-state and forecast hashes identify whether sharing is mathematically valid. Shared information is not a reason to overwrite another arm's adaptive state.

## 3 Solvers, diagnostics, and numerical certificates

### F09 — exact regularized active-face solve (P0, C/E)

For aggregated coefficients `A,B`, penalty `lambda=.05` and center `m` (uniform for global; alpha_G for local), write

    H = -B + 2*lambda*I
    a = A - 2*lambda*m
    f(alpha) = .5*alpha' H alpha + a' alpha + lambda*m'm.

For every nonempty expert subset S, solve the bordered system

    [H_SS  1] [alpha_S] = [-a_S]
    [1'    0] [nu     ]   [ 1  ]

and set other weights to zero. Check primal feasibility; gradient `g=H alpha+a`; active stationarity `g_S+nu=0`; inactive dual feasibility `g_notS+nu>=0`; and complementary slackness. Use the declared scale-aware tolerances. The full H need not be positive semidefinite: convexity is required on the simplex tangent space. A generic full-space PSD check would falsely reject legitimate energy-score matrices.

Record coefficient scales, all feasible faces, residuals, selected face and objective gap. Positive L2 gives a unique mathematical solution on the simplex; lexicographic tie handling is only a numerical consistency convention. Do not clip an infeasible unconstrained solution into a simplex. A dense grid is an approximate reference, not an exact proof; supplement it with independent tiny analytic cases and KKT verification. Deliberately infeasible/nonfinite inputs must fail before an economic decision.

### F10 — inverse-volatility projection and minimum-variance ties (P1, C/E)

Use the population covariance of U's total simple returns, with each marginal SD floored at `1e-4`. Normalize inverse SD to total .8, then project Euclideanly onto `0<=w_i<=.4, sum(w)=.8`. The projection is `w_i=clip(v_i-tau,0,.4)` with tau chosen so the sum is .8. Specify deterministic sorted-breakpoint solution or bounded bisection, sum/box/KKT tolerances, and retained failures. The answer is not generally normalized clipping.

Minimum variance uses the stated shrinkage matrix and only grid targets with sum exactly .8. Its tie handling must use current holdings valued at the same quote and actual turnover under the relevant fee-root solution. Each benchmark has its own holdings and therefore its own turnover tie, even when forecasts are shared.

### F11 — action count, tolerance ties, and cap drift (P1, C/E)

Construct weights from integer triples `u_i in {0,...,8}` and `sum(u)<=16`, then `w_i=u_i/20`. There are 609 grid actions, plus at most one distinct exact-hold action. The sum-.8 minimum-variance subset contains 45 targets. These are exact combinatorial counts; revision 1's 730 bound was conservative, not a scientific defect.

Compute the objective maximum once. Define the tied set by `J_max-J<=1e-10`; do not implement a pairwise approximate comparator, which can be nontransitive. If eligible exact hold is in that set, choose it. Otherwise choose minimum turnover, then lexicographic weights using a separately declared dollar-turnover comparison tolerance. If the hold weights duplicate a grid action, deduplicate by action semantics and preserve the zero-trade hold identity.

Caps are evaluated at the observed opening quote after the chosen dividend event order, not yesterday's weights. If drift makes hold infeasible, record that fact and choose a feasible rebalance. Decisions never promise continuous intraday caps. Floating feasibility tolerance cannot permit meaningful excess exposure.

### F12 — robustness includes zero-probability atoms (P1, C)

The declared ambiguity set permits `q_other` on any library atom, including atoms assigned probability zero by the current expert mixture. Therefore `min_i U(F_i)` ranges over the entire shared library, not just nonzero-probability scenarios. Otherwise contextual forecasts silently shrink the uncertainty set and gain a policy advantage beyond the stated mixing mechanism. The zero-return control has its declared singleton library; disclose this difference.

### F13 — utility and diagnostic endpoints need units (P1, D/C; CL-12 resolved)

Revision 1 said realized CRRA utility without selecting terminal versus per-block utility. The choice matters. Terminal `U(W_T/W_0)` is monotone in wealth and cannot rank two policies differently from wealth. CL-12 now selects the mean of `U(L_{k+1}/L_k)` over common four-interval blocks, compared for robust/nominal under the same forecast/cost; it remains path-dependent descriptive evidence. Terminal utility is a separate report field rather than an independent risk claim.

CL-12 selects marginal CRPS as the asset average of the exact one-dimensional energy score on `r_a/b_a`, and squared error as the asset average of `(E[r_a]/b_a-r_observed,a/b_a)^2`, with raw-unit diagnostics separate. Loss calibration uses `r_a<0`, with equality counted nonloss, and separately an equal-weight unlevered basket loss. It fixes Brier scores and ten equal-width probability bins, including the final endpoint 1, with sparse-bin counts retained. Portfolio-loss calibration is a different endpoint requiring the saved chosen action and cannot be substituted. The stated central quantile pairs and inclusive coverage rules are also now exact.

## 4 Inference and precision

### F14 — bootstrap algorithm, families, and result labels (P0, C/D)

For each block length L and fold f containing N_f complete blocks, each draw samples `ceil(N_f/L)` uniform integer starts in `[0,N_f)`, expands each into L circular consecutive indices, concatenates and truncates to N_f. Reuse that exact index tensor for all aligned methods, costs, policies, assets, scores and endpoint increments. Preserve original fold counts and pool by total sampled dates. Record generator algorithm/version, seed, tensor hash and draw count; a seed alone does not identify an algorithm across libraries.

H1 recomputes the ratio from resampled sums, never averages datewise ratios. Undefined denominators fail the endpoint under a fixed numeric definition. Compute the original point estimate and the quantiles with one explicitly defined linear interpolation convention. Both controls and both L families must satisfy the stated gate; never select a block family after inspection.

**Selected CL-08 resolution:** all eight ablations at both L values form one 16-test Holm family. Preserve A4/A5 structural-zero p=1. Center each comparison by its original pooled date-weighted mean, resample folds separately, and use `(1+exceedances)/(5000+1)`. Use stable ablation-ID/block-length tie order and cumulative-max adjusted p-values as CL-08 specifies. State whether a zero observed effect and degenerate null lead to p=1; they must never create a favorable significance label. No analyst may revert to separate eight-test families to obtain a more favorable finding.

H1 and H2 should retain separate endpoint labels. Passing one does not imply the other. The all-required intersection can define overall promising development; no unspecified union claim should inherit the two-comparison Bonferroni label. Adaptive, historical, current-vintage ETF-path bootstrap intervals remain descriptive and assumption-dependent.

### F15 — economic increments must telescope without actual fictitious exits (P0, C)

Define liquidatable NAV at each common decision boundary immediately before the rebalance, after required dividends and marks:

    L_k = cash_k + (1-c)*sum(quoted_price_k*shares_k).

Use initial pre-entry cash `L_0=100000`; use final actual post-liquidation wealth for `L_T`. The first block contains its initial entry fee; subsequent blocks contain their actual start-boundary rebalance fees. Store boundary values before trade so the interval log ratio includes the trade that starts it. Exact log increments `ell_k=log(L_{k+1}/L_k)` telescope to `log(W_T/100000)` without booking hypothetical exit fees to cash at intermediate boundaries.

Paired control increments must use those same physical dates. Resampled increments are conditional diagnostics of saved paths, not a new economically feasible holdings replay. Verify the endpoint identity independently. Daily-close drawdown includes the initial capital anchor and final liquidation value as specifically labeled additional marks; do not call those additional observations daily closes or silently omit terminal fees.

### F16 — precision needs a deterministic rule and correct relative units (P0, D)

The phrase "realistically distinguishable" is not an executable admission rule. Specify an inequality before the calibration stage, a minimum complete calibration-block count, and what insufficient count does. No calibration number exists yet and none should be invented.

For a relative endpoint `R=1-mean(X)/mean(Y)`, an absolute ES standard error cannot be compared directly with .05. A first-order influence value is

    psi_i = ((1-R)*Y_i-X_i)/mean(Y).

Use either that relative influence series or the full bootstrap ratio. An explicit projection can scale a calibration standard error by `sqrt(N_cal/N_eval)` under a disclosed stationarity approximation. Compute each control and L family, and use the worst required projected uncertainty. The factor `1.96+.84` describes an approximate separation from a specified null only after defining an alternative; it does not establish 80% power to exceed a 5% margin just because a standard error is small.

**Selected CL-09 resolution:** use sample SD of 5000 bootstrapped pilot relative improvements; project it by `sqrt(n/N)`; require `(1.96+.84)*s_projected <= .10-.05` for all two-control/two-block-length combinations, with at least 24 complete pilot blocks. The prespecified .10 design alternative is an assumption, not an expected S10 result. Degenerate denominators, zero estimated variability, no contextual/global difference, insufficient counts and nonfinite estimates are unidentifiable rather than passing. This is now deterministic at design level; actual precision remains unmeasured, with the cold-start limitation in F06. It does not establish joint power for the full H1/H2 intersection, each-fold positivity, or economic performance.

## 5 Execution, checkpoints, and resource dimensions

### F17 — scientific interruption versus engineering restart (P0, D/E; parent selection recorded)

Revision 1 simultaneously requires crash/restart equivalence and forbids automatic retry after consumption. **Selected CL-10 resolution:** actual market runs have no resume command. Admission failures before scientific start leave the freeze unconsumed. First start atomically creates the consumption record. Every interruption after consumption is terminal for that attempt; checkpoints support audit and constructed engineering replay only. A scientific continuation protocol is deliberately absent from revision 2. It cannot be invented during recovery. Corrected scientific execution requires documented review and a justified separately recorded protocol under the original validity-defect rules, never a deleted marker or reset budget.

Checkpoint engineering still needs transactional writes: uniquely named temporary artifact, flushed contents, manifest/checksum, atomic commit within the same filesystem, then committed cursor. Exactly-once identities prevent duplicate events in constructed replay checks. Partial artifacts remain quarantined with retained failure receipts. A complete output is accepted only after its manifest agrees with committed event counts. These quality requirements do not authorize replaying a consumed market campaign.

### F18 — process budgets must include the real full dimensions (P1, C/E)

The 520x520 float64 distance matrix alone is 2,163,200 bytes, approximately 2.063 MiB. Temporary broadcast arrays, all-arm copies, stored pending forecasts, action matrices, statistics draws, parser buffers and numerical libraries add to that. A full 609x520 action-by-scenario float64 utility matrix is approximately 2.416 MiB; a 3-coordinate expanded tensor triples that. These arithmetic sizes are not measured peak RSS. CL-11 additionally includes the owned supervisor/worker process tree in a conservative aggregate 256 MiB RSS ceiling; profiling must measure that bound rather than only the numerical worker.

Stream origins and arms rather than storing all duplicate historical atoms for every forecast in memory. Preserve reconstruction identities on disk. Reuse common exact distances only when their data/transform identities match; A1/A6/A7/A8 can invalidate different cached terms. For leave-one-out OOD quantiles, compute exact k-th distances using deterministic rank selection, not an approximate neighbor index without a new equivalence contract.

Freeze counts for unique issue origins, actual per-arm updates, score constructions, active-face solves, scenario rebuilds, policy decisions, bootstrap work and total wall clock. "Two folds" is not two fits. State whether prequential history is shared or recomputed; duplicate work counts against the same budget. One numerical thread must include BLAS/OpenMP settings before import. Resource admission is performed before heavy imports, and actual preparation/campaign runtime is retained even when an attempt fails terminally.

### F19 — source admission must separate identity from access (P0, C/E)

The retained raw files can contain observations outside the selected historical window. Hashing file bytes is not authorization to expose later values. The adapter must validate metadata and admit a bounded view without logging or returning excluded outcomes. Manifest records source version, extraction predicate, exact admitted rows/identities, unsupported events and assumed availability conventions. A positive hash check is neither point-in-time provenance nor a data-license determination.

No protected-data fixture may use a real protected value. Future-perturbation tests use generated values and engineered timestamps. Do not claim official timestamps from a convention such as midnight availability or the opening-quote clock.

## 6 Closure evidence required before implementation and execution

| Reviewed design area | Design disposition | Remaining evidence boundary |
|---|---|---|
| Clock, dividends, calendar, frozen A2 and neighbor sensitivities | Resolved prospectively by CL-01 through CL-05 | Implemented chronology, admitted source/calendar, engineered adversarial fixtures |
| Grid, projection, active-face solve, shared bootstrap and multiplicity | Resolved prospectively by CL-06 through CL-08 | Independent numerical references, actual residual/certificate checks, statistical implementation tests |
| Precision dependency and decision rule | Resolved prospectively by CL-09; pilot cold-start weakness disclosed | Real bounded pre-2010 preparation, actual counts and honest precision disposition |
| Consumed-run interruption | Resolved prospectively by CL-10: no empirical resume | Tested rejection/atomic-consumption behavior and retained failures |
| Resource/source/failure scope | Resolved prospectively by CL-11 | Actual admission, measured aggregate process-tree RSS/runtime, source integrity |
| Utility, calibration, intervals and robust atom set | Resolved prospectively by CL-12 | Implemented metric/accounting equivalence and actual diagnostic results |
| Novelty, market signal, calibration, independent reproduction and untouched confirmation | Not established; design cannot close these | Separate literature/evidence review, authorized experiments, independent work and an admitted unused evaluation |

The parent should integrate each selected resolution into machine-readable revision 2 contracts and a traceability map, leaving revision 1 intact. A finding is design-closed only when the relevant field/equation/event order is explicit; runtime-closed only when actual implementation and retained verification establish it.

Before full scientific execution, require: exact source/calendar/admission manifest; bounded precision result under the frozen rule; complete chronology and distribution implementation; independently calculated numerical/accounting fixtures; checkpoint interruption/equivalence and corruption rejection; complete matrix identities; both statistics families; measured resource feasibility; and a new source/data/config/environment freeze. This review supplies no PASS for those future gates.

Even after all engineering gates pass, the null remains credible: U, equal mixing, or global stacking may be sufficient; contextual routing may add variance; historical support may miss the important shocks; robust utility may favor cash; and the ETF sample may be too small to distinguish the desired effect. Code quality makes those possibilities observable and auditable. It cannot remove them by design.

## Final integration review of the assembled blueprint

A read-only second pass through the index, CL-01–14, requirements and operating sequence found three additional details within the existing solver/precision/operations findings. These have been resolved prospectively, without executing a model:

- WP06 now implements and tests the necessary subset of the same production bootstrap, atomic artifact, one-use consumption, cancellation and resource modules later completed in WP10-WP12. Its preflight requires the reviewed precision declaration and test evidence, not its own unproduced precision result. Full-campaign preflight still requires the passing result. Requirement ownership includes WP06 for those subsets.
- CL-09 now binds pilot bootstrap SeedSequence1902 children [(PILOT,2),(PILOT,4)], separate from campaign streams, and defines projection count N=N_A+N_B from frozen prospective block indices.
- CL-14 explicitly recovers nu_original=s*nu_scaled before original-scale KKT checks. The positive objective scaling preserves the constrained optimum.

The reviewer found the fixed denominator rejection and initial/final drawdown anchors coherent. This is internal design review, not mathematical proof of the whole pipeline, executed tests or independent empirical reproduction. Runtime evidence for every resolution remains pending.
