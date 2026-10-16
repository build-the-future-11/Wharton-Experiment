# S10 revision 2 normative model clarifications

15 October 2026. These clauses supersede only the identified ambiguities in revision 1. Revision 1, its receipt and scientific predecessors remain byte-preserved. No S10 model, market result or scientific freeze exists. Changes below are prospective design choices recorded before implementation and outcomes.

Authority: original protected-data and evidence rules remain highest. Within S10, this file's numbered clauses override conflicting revision-1 prose; otherwise the original model equations and proposed protocol apply. Architecture, quality and runtime documents implement this combined contract. A contradiction is a design defect to resolve, not a choice for an implementer to guess.

## CL-01 Decision information and ideal execution

Revision 1 combined a 12:00 UTC pre-open decision with portfolio values based on a later unknown opening quote. Revision 2 changes the declared research information clock:

1. For a retained session date, construct the assumed regular-session open at 09:30 America/New_York, converted to UTC with the locked timezone database.
2. At that timestamp, expose the same opening quote vector to all policies for holdings valuation. Process the entering-share dividend entitlement phase before any trades.
3. Decision time is quote time plus one microsecond; ideal fill time is quote time plus two microseconds at the identical quote.
4. The microseconds encode ordered research events, not observed latency. Actual quote publication, slippage, asynchronous fills and market impact remain unmodeled.
5. Forecast features still use only labels with their specified availability strictly before the decision. The label ending at this open becomes available at midnight UTC after this session; it cannot enter the current forecast. The current open is used for policy valuation, not added as an unplanned forecast feature.

The current [NYSE hours page](https://www.nyse.com/trade/hours-calendars), checked 15 October 2026, lists a 09:30 Eastern core open. It does not certify historical vendor publication times, each instrument's opening auction timing, or every historical session calendar. Those remain source-admission issues. The ideal same-quote decision/fill assumption is a versioned scientific convention, not an engineering-equivalent correction or a live trading claim.

## CL-02 Dividends and event ownership

Keep the revision-1 target interval origin < ex_date <= endpoint. Shares bought at the origin's ex-date open do not receive that day's distribution. Shares held into the endpoint ex-date retain entitlement even if sold at that open.

At each event date: mark entering shares to the observed opening price; determine dividend cash from entering shares; credit the declared ex-date cash convention exactly once; then value pre-trade NAV, choose action and trade. New shares bought at this open receive no same-date dividend. At a fold's first entry there are no entering shares. At a final liquidation, credit entering-share entitlement before sale.

Historical per-origin forecast atoms use the same convention, without reinvestment. Daily closes are marks only and do not trigger an extra credit. Stable event identity is (source identity, asset, ex-date, source-event ordinal); duplicate identity is an error. This cash-on-ex-date model is not payment-date settlement.

## CL-03 Calendar and fold indices

Require identical observed session-date sequences across the three sources. Do not intersect them to hide missing observations. The earliest retained session in each ISO week forms the anchor. Calendar/source anomalies fail admission; a declared holiday is not inferred from a zero quote.

For each fold, form eligible anchor indices whose decision date lies in [fold_start, fold_end_exclusive). Select the first such index i0 with complete required feature and library history. Publish any initial exclusions with timestamp-only reasons. Evaluate indices i0, i0+4, i0+8, ... only when endpoint i+4 exists and its date is strictly less than fold_end_exclusive. No later outcome may change i0 or skip a selected block. Incomplete trailing blocks are reported as unused sessions, not imputed.

Missing price/dividend/calendar integrity at a scheduled block fails the campaign. It does not silently skip that block. There are no parameter-based redefinitions of folds after preflight. Freeze the exact date/index manifest before evaluation.

## CL-04 Frozen-state ablation

A2 saves the primary candidate's complete issued joint atom array and final scenario probabilities at the first evaluation decision of each fold. It returns that same distribution at every later fold decision. It does not recompute contextual neighbors from changing query features. Scoring continues against realized outcomes, but cannot change that distribution. All other policies still receive their own current holdings and quotes.

This is an intentionally static-forecast comparator. More selective variants, such as frozen libraries with changing queries, are separate experiments and are not part of the 192-ledger design.

## CL-05 Neighbor sensitivities

For arbitrary declared k, expert-C lookup requires at least k candidates and the leave-one-out reference quantile requires at least k+1. Local routing requires k selected matured records and k+1 total eligible contexts for its reference quantile. Otherwise use the specified global fallback.

Primary k=32 uses concentration floor 16; A7 k=16 uses floor 8; A8 k=64 uses floor 32. Keep the fixed pseudocount 64, blend cap .5, penalty .05, recency half-life 52 and 95th-percentile rule unchanged across these arms. Tie-breaking uses origin then stable record identity. Distances are squared Euclidean throughout; do not compare squared and unsquared values.

## CL-06 Exact grid and benchmark projection

Represent weights as integer units u_a in 0..8, with w_a=u_a/20. The main grid is sum(u)<=16 and contains exactly 609 distinct points. Add the exact current-hold action only if feasible and not an existing identical grid action, for at most 610 candidates. Retain its special tie preference even if its vector matches a grid point.

Minimum-variance benchmark actions have sum(u)=16 and contain exactly 45 points. These counts were verified by integer enumeration during design; no market model was executed.

For inverse-volatility target z normalized to sum .8, compute the unique Euclidean projection onto 0<=w_a<=.4 and sum(w)=.8:

    w_a = min(.4, max(0, z_a - tau)).

Find tau by monotone bisection bracketed by min(z)-.4 and max(z), at most 100 iterations, to simplex residual <=1e-12 and projection KKT residual <=1e-10. Validate the result; do not silently normalize after clipping. Independently verify against enumeration of active lower/free/upper sets in constructed cases.

For optimization ties, define the best objective first; collect candidates within its stated tolerance; prefer feasible hold, otherwise lowest actual dollar turnover, then lexicographic weights. Do not use sequential pairwise approximate comparisons whose result changes with iteration order.

## CL-07 Stacking solve

For the selected average or locally weighted score coefficients A and B, anchor c and lambda=.05, define H=-B+2lambda I and a=A-2lambda c. On each nonempty active set S, solve

    [ H_SS  1 ] [alpha_S] = [-a_S]
    [ 1'    0 ] [nu     ]   [  1 ]

with inactive alpha=0. Check primal feasibility, active stationarity, and inactive reduced gradients (H alpha+a)_i+nu >= -tolerance. Choose the feasible minimum after objective and residual checks. The simplex tangent restriction is the relevant convexity condition; H need not be positive definite on the full unconstrained space. Do not reject a correct solver merely because a full-space Cholesky fails.

The coefficient tolerances in revision 1 remain in force. Require an explicit conditioned-solve check, retain residuals, and fail if no candidate can be certified. Numerical code must compare projected tangent curvature or the bordered system appropriate to the constraints.

## CL-08 Shared bootstrap and multiplicity

Generate exactly one reproducible index tensor for each block length and fold, shape (5000, N_fold). For each draw choose uniform circular-block start indices, concatenate blocks, and truncate to exactly N_fold. Reuse those indices for every forecast, policy and cost comparison. Never resample assets or methods independently.

Use SeedSequence(1901).spawn(4), associating children in the exact order [(A,2),(B,2),(A,4),(B,4)] with Generator(PCG64(child)). For each draw sample ceil(N_fold/block_length) int64 uniform start indices in [0,N_fold), concatenate the consecutive modular blocks, then truncate to N_fold. The generator/library version is locked. Adding a report must not change the original streams. Store the index arrays and their identities.

Primary H1/H2 requirements remain unchanged: both controls and both block families must pass; no favorable block selection. The secondary eight-ablation ES tests at two block lengths form one declared 16-test Holm family. Keep policy-only zero differences at p=1. Sort equal p-values by ablation ID then block length. Holm-adjusted p-values use the cumulative maximum of (m-rank+1)*p_sorted, capped at one; report every raw and adjusted value.

This clarifies an underspecified revision-1 secondary family. It does not change the primary practical margins, the 192-ledger matrix or the original FREM threshold. Bootstrap inference remains descriptive under the declared dependence assumptions.

## CL-09 Precision gate without a circular prerequisite

The executable precision estimator needs tested model code. Therefore the order is:

    metadata and static source admission
    -> minimal model/scoring/chronology kernel with constructed verification
    -> separately admitted pre-2010 precision preparation
    -> complete experiment implementation and profiling
    -> source/data/environment freeze and full run readiness.

Implementation of the minimal kernel is permitted as engineering work after the empirical question/specification review; precision is a gate to the full market campaign, not an impossible prerequisite to writing the code that measures it. This session remains planning only.

The future preparation uses only origins on/after 2008-01-01 whose four-interval labels mature before 2010-01-01. Begin at the first eligible anchor, then score every fourth anchor using the same frozen algorithms. Require at least 24 complete pilot blocks. Include cold-start global/local fallbacks and exact zero-difference dates; publish their counts. This is not a demand for 24 fully warmed blocks, and the roughly two-year pilot may poorly represent mature adaptive state. No protected or outer-fold outcome is used to make this decision.

The pilot has its own RNG schedule: SeedSequence(1902).spawn(2), children assigned in the exact order [(PILOT,2),(PILOT,4)], each using Generator(PCG64(child)). Use CL-08's per-draw int64 circular-block start generation, concatenation and truncation with pilot count n. Save both 5000-by-n index arrays and share them between the two pilot controls. Pilot streams never advance or replace the four full-campaign streams.

For each of two controls and two block lengths, estimate s_p as the sample standard deviation (divisor 4999) of the 5000 bootstrapped relative ES improvements on the pilot. Use timestamp-only prospective pooled evaluation count N=N_A+N_B and pilot count n to form s_projected=s_p*sqrt(n/N). N_A and N_B are the exact planned eligible four-interval block counts from CL-03, not weekly-origin counts or effective independent sample sizes. The design alternative is a 10% relative improvement, against the unchanged S10 5% practical margin. The prespecified planning condition is

    (1.96 + 0.84) * s_projected <= .10 - .05

for all four combinations. This is a stationarity-based approximate precision heuristic, not validated joint power for the two folds/economic gates. Publish the assumptions and all values. Degenerate score denominator, fewer than 24 blocks, zero estimated score-difference variability, nonfinite output or no contextual/global difference yields PRECISION_UNIDENTIFIABLE, not an automatic pass.

Passing this condition does not establish signal or expected positive results. Failing it yields PRECISION_INADEQUATE; no threshold reduction, window selection or extra trial follows automatically. No pilot observed point gain selects parameters. There is no claimed economic power calculation from this rule.

Before any real precision preparation, implement and test the required subset of the production bootstrap, immutable artifact, exclusive consumption, process supervision, cancellation and resource modules. This belongs to WP06 and is extended by WP10-WP12 later; no duplicate or unguarded precision runner is allowed. Precision preflight requires the reviewed precision declaration and its kernel/guard test receipts, but does not require its own as-yet-unproduced precision result. Full-campaign preflight requires a passing precision result. The two operations have distinct consumption identities.

## CL-10 Checkpoints and consumed attempts

Actual market runs in revision 2 have no resume command. Once scientific processing begins and the persistent consumption marker is written, an interruption or failed attempt is terminal for that attempt. Checkpoints support audit and engineering replay-equivalence tests, not automatic reopening of outcomes.

Preflight failures remain unconsumed and can be retried with a fresh output directory after the failed prerequisite is repaired. A scientifically started run requires a documented review and, where justified by a real validity defect, a separately recorded corrected protocol; no marker deletion or hidden restart is allowed. A future same-run continuation policy would require its own predeclared design, including cumulative budgets and exact recovery rules.

## CL-11 Budgets, source scope and failure classification

Revision-1 limits remain upper ceilings: 600 seconds for separate precision preparation, 3600 seconds for the full campaign, one compute worker/thread, 256 MiB process RSS and the 512 MiB host admission floor. Revision 2 additionally treats 256 MiB as a conservative aggregate RSS ceiling for the owned supervisor/worker process tree. This stricter accounting is a prospective engineering budget choice, not a fresh measured admission result.

Hashing the original source bytes is allowed for identity verification. The reader exposes only approved date-filtered records to modeling, statistics and logs. No protected or post-cutoff outcome is summarized to choose the window. Source decoding does not justify downstream use of out-of-scope values.

Model insufficiency fallback is allowed only at the exact specified sparse/OOD gates. Parser corruption, leakage, numerical failure, missing controls or exhausted budgets are failed runs, not successful cash forecasts. An economic policy's valid cash choice is recorded separately from any run failure.

## CL-12 Diagnostic units, utility and calibration

At common four-interval boundaries before the next rebalance, record liquidatable NAV L_k = cash + (1-c)*risky_value after entering-share dividend credit. These hypothetical liquidation charges are valuation deductions only; they are never debited from cash. Define L_0=100000 before first entry and L_last as the actual final-liquidated NAV. Boundary log increments log(L_(k+1)/L_k) telescope exactly to log(L_last/100000). Keep actual daily marked NAV and hypothetical liquidatable NAV as distinct fields.

H3's descriptive utility endpoint is the equal-date mean of U(L_(k+1)/L_k), with gamma=4. Show each fold and pooled date weighting. This is a one-block utility proxy using a fixed liquidation valuation convention, not lifetime expected utility. Report terminal U(L_last/100000) separately; because it is monotone in terminal wealth, it cannot independently validate a risk improvement.

For each empirical forecast, marginal CRPS uses the ES scalar formula in fixed standardized total-return units z_a=r_a/b_a. Average the three asset CRPS values within a date before date aggregation. MSE similarly averages squared standardized errors of the predictive mean. Also report raw per-asset simple-return CRPS/MSE with clear units; never combine raw and standardized columns. Joint ES and variogram use the existing fixed standardized vectors.

Loss calibration events are r_a<0 for each individual asset, and (r_1+r_2+r_3)/3<0 for a separate equal-weight unlevered basket. Forecast probabilities are the scenario mass satisfying the exact event; equality at zero is non-loss. Report mean squared probability error (Brier), mean forecast probability, observed frequency and counts. Use ten fixed bins [0,.1),...,[.8,.9),[.9,1]; report bins with fewer than ten observations as sparse, retain all data, and do not adapt the bins to outcomes. Basket results remain separate from the three-asset average. These probabilities are not policy-loss, funding or recession probabilities.

Central intervals use quantile pairs (.25,.75), (.10,.90), (.025,.975), with revision-1 discrete quantile convention and inclusive endpoints for empirical coverage. State tie counts and widths; no bin/interval choice is tuned after evaluation.

The robust utility minimum ranges over every atom in the declared forecast support, including atoms with zero nominal mixture weight, because the contaminating distribution may allocate mass there. The zero-return forecast has its declared single atom. No unseen shock is silently added to that support.

## CL-13 Package and verification ownership

Revision 2's canonical future package is successors/frem_s10, source is src/frem_s10 within it, and its sole test source is successors/frem_s10/tests. This supersedes the revision-1 tentative tests/s10 location. Root legacy tests and dependencies remain unchanged. No duplicate discovery adapter or implicit root-suite integration is required; CI and reproduction commands name the isolated package/test path explicitly.

The detailed nested module layout in ARCHITECTURE_AND_INTERFACES.md replaces the initial flat filename sketch. This is a planned engineering organization change; numerical behavior and the study matrix remain controlled by the normative scientific clauses.

## CL-14 Remaining numerical and event boundaries

The relative ES endpoint is the ratio of date-weighted means, `1 - mean(candidate)/mean(control)`, including inside each bootstrap draw. A control mean <=1e-12 in the fixed standardized score units is numerically undefined. Apply that same fixed rule to observed, pilot and resampled endpoints; never discard an undefined draw or switch to mean datewise ratios. An undefined required endpoint prevents its scientific claim; undefined pilot endpoints yield PRECISION_UNIDENTIFIABLE.

Action feasibility uses absolute weight tolerance 1e-12. Record any within-tolerance residual; this is not a license to clip a materially infeasible holding. After collecting the objective-tied set using the revision-1 absolute utility tolerance 1e-10, select hold if present. Otherwise find the minimum actual dollar turnover over that whole set, collect turnovers within 1e-10*max(1,V) of that minimum, and choose lexicographic weights. Do not chain pairwise approximate comparisons.

For each active-face bordered system in CL-07, divide H and a by s=max(1,max(abs(H)),max(abs(a))) before forming the bordered solve, retaining the unit equality row/column. Require finite two-norm condition number <=1e12, then validate residuals in the original coefficient scale, recovering nu_original=s*nu_scaled. A singular or excessive-condition system is an explicit numerical failure, not a silently skipped face. This conditioning bound supplements the original feasibility, symmetry, curvature and KKT bounds; it is not a scientific hyperparameter search.

Daily-close drawdown uses the ordered actual daily-close NAV marks, plus the initial 100000 capital anchor before entry and actual final-liquidation NAV after final exit. Label the two anchors separately; neither is invented as a daily close. Compute the running high-water mark and the largest proportional fall from it. Include final fees exactly once in that final mark. Intraday extrema remain unobserved.

The mutable stream rejects every already-applied event ID and every cursor regression, including a byte-identical duplicate. Engineering replay restores a checkpoint and continues strictly after its saved cursor; it does not replay applied events into live state. Reading/auditing an immutable saved artifact twice is harmless and does not advance the stream. No market recovery behavior is introduced.
