# FREM S10 experiment protocol

15 October 2026. Proposed development protocol, not an execution authorization or scientific freeze. No S10 market value has been inspected or generated to choose these settings.

## 1 Scope and data admission

Use the existing retained SPY/TLT/GLD source identities referenced by S9's freeze, filtered to observations before 2017. Do not open new 2017+ data or consume the canonical 2025+ boundary. The historical data is current-vintage, already exposed through predecessor training and therefore retrospective development. The exact three source hashes are carried in design.json; source contents remain unchanged.

Before implementation is called run-ready, independently verify the calendar, source hashes, splits/dividends, warm-up capacity, timestamp conventions, units, historical access inventory and permissible local use. S9 admission is relevant prior evidence, not an S10-specific PASS. The pre-2008 transform and 2008 onward prequential warm-up change requirements and must be checked.

Outer folds:

| Fold | Scaler data | Feedback warm-up | Evaluation decisions |
|---|---|---|---|
| A | Labels mature before 2008-01-01 | Origins from 2008-01-01; only matured records used | 2010-01-01 through 2012-12-31 |
| B | Same immutable pre-2008 transform | Origins from 2008-01-01; all chronological matured past allowed | 2013-01-01 through 2016-12-31 |

Evaluation begins at each fold's first eligible weekly anchor and advances four anchors at a time. Keep only blocks with endpoint before the next fold boundary. Both folds start their wealth ledgers at $100,000 cash. Forecast state at B may use past A labels, as declared online learning; the folds are not independent replications. No S10 fold outcome may change parameters, model identity, date ranges, costs, primary choices or the feedback algorithm.

The S9 record contains 38 and 51 blocks under its own preparation. Treat those as indicative counts, not verified S10 counts. Require >=24 complete evaluation blocks per fold; publish all timestamp-based exclusions and terminal unused sessions. Exclusions cannot depend on returns. If a source defect or missing session makes the continuous ledger unsupported, fail admission rather than splice away the interval.

## 2 Primary hypotheses and thresholds

H1, distribution mechanism: contextual stacking improves the joint holding-period predictive distribution over both global stacking and equal expert averaging.

For each control c, define R_c = 1 - mean(ES_S10)/mean(ES_c), pooling decision blocks with equal weight per date. Reject a nonpositive or numerically degenerate denominator as an undefined relative endpoint; do not replace it after inspection. A promising development finding requires:

- Positive point improvement in each fold against each control.
- The lower one-sided 97.5% bootstrap bound for pooled R_c is strictly above 0.05 against both controls.
- The condition holds for both predefined block lengths.

The two 97.5% bounds use Bonferroni protection for the two primary comparisons. Serial bootstrap validity remains assumption-dependent; development intervals are descriptive. This is a new 5% energy-score margin, not a conversion of the original 0.01-nat memory hypothesis.

H2, economic translation: at 10 bps one-way fees, the fixed primary contextual/robust policy exceeds both broad equal-asset rebalancing and buy-and-hold terminal wealth by at least 6% in each fold and exceeds cash in each fold. Report matched 80%-risk controls alongside those broad benchmarks. Additionally require:

- Positive lower one-sided 97.5% paired-block bound for mean log-return advantage against both broad controls, for both block lengths.
- Daily-close maximum drawdown no greater than the 80%-risk equal-asset benchmark in either fold.
- No accounting, constraint or source-integrity violation.

This preserves the original 6% relative ending-wealth objective and adds an explicit risk condition; it does not redefine it as annualized return or density quality. Block inference describes the retained path under assumptions; it cannot establish a sampling distribution for every possible adaptive-policy market path. A descriptive economic point pass remains development evidence.

H3, policy robustness: the robust policy improves realized CRRA utility over nominal under identical forecasts. This is secondary; robust may lose by becoming overconservative. Report terminal wealth, drawdown and cash exposure even when utility improves. Do not select nominal as the primary after seeing that it wins.

The original FREM-P0 density hypothesis remains unevaluated by S10: empirical atoms do not yield continuous NLL. No original 0.01-nat success label, SOTA claim or funding guarantee follows from passing H1 or H2.

## 3 Core matrix and strong controls

Eight forecast identities:

1. U, unconditional joint empirical distribution.
2. R, recent joint empirical distribution.
3. C, contextual neighbor distribution.
4. Equal mixture of U/R/C.
5. Global regularized mixture G.
6. Contextual shrunk mixture S10.
7. Individual-score weighting: softmax(-m_e), where m_e is the recency-weighted mean standalone ES over the last 26 matured issued forecasts, half-life 52 weeks, temperature one. Before 26 records use equal mixture. This preserves the deliberately simpler individual-expert-loss control.
8. Zero-return control, one atom g=(1,1,1), d=(0,0,0), explicitly deterministic and uncalibrated.

Cross every forecast with nominal/robust policies, 0/10/20/40 bps costs, and both folds: 8 x 2 x 4 x 2 = **128 strategy ledgers**.

Six benchmark identities, at all costs and both folds: 6 x 4 x 2 = **48 benchmark ledgers**.

| Benchmark | Exact behavior |
|---|---|
| Cash | Zero-yield cash throughout |
| Equal assets 100% | Target 1/3 in each ETF at every four-interval decision |
| Equal assets 80% | Target 0.8/3 in each ETF, remaining cash |
| Buy and hold 100% | Equal-dollar entry once, dividends remain cash, final liquidation only |
| Inverse volatility 80% | Use past U scenario total-return standard deviations with floor 1e-4; normalize inverse vol to total .8; Euclidean projection onto w>=0, each<=.4, sum=.8 |
| Minimum variance 80% | Minimize w' Sigma_shrink w over the same .05 grid with sum=.8 and each<=.4, Sigma_shrink=.5 Sigma_U+.5 diag(Sigma_U); tie-break lower turnover then lexicographic w |

The explicit 50% covariance shrinkage is a fixed benchmark, not a claim to implement Ledoit-Wolf. All weights use only matured information. Broad 100%-risky benchmarks deliberately retain different exposure; the constrained controls supplement them rather than replace them. All charge actual entry/rebalance/final-liquidation fees consistently.

Zero and simple controls remain in every report. Include average risky exposure, risk contributions and compute for all methods. Expert forecasts are shared; give all mixtures the same information and update dates. Do not add redundant random seeds to deterministic methods.

## 4 Eight fixed ablations

Each ablation uses the contextual forecast and robust policy except for its one stated change, primary cost 10 bps, both folds: **16 additional ledgers**. Total requested development output is **192 ledgers**.

| ID | Change | Question |
|---|---|---|
| A1 | Replace direct four-interval atoms with constant repetition of the one-interval atoms from the same origins: g^4 and d(1+g+g^2+g^3); rebuild all issued-score records causally for that arm | Does direct horizon construction matter? |
| A2 | Freeze library and all mixture state at each fold start; scoring continues but cannot update forecasts | Does the declared online schedule help? |
| A3 | Use alpha_L directly when eligible; keep alpha_G fallback | Does shrinkage prevent contextual overfitting? |
| A4 | Sell everything at every four-interval endpoint before the next decision | Does carried ownership reduce turnover without sacrificing useful adaptation? |
| A5 | Exclude the special exact-hold action while retaining the grid | Does explicitly offering hold matter? |
| A6 | Scenario window 260 rather than 520 | Does the fixed longer history help or hurt? |
| A7 | Neighbor counts 16 rather than 32 for both expert and local routing; eligibility floor 8, otherwise same rules | Small fixed neighbor sensitivity |
| A8 | Neighbor counts 64 rather than 32 for both expert and local routing; eligibility floor 32, otherwise same rules | Large fixed neighbor sensitivity |

A1 still requires the direct four-interval label maturity for admission, even though it uses one-interval inputs; it receives no timing advantage. A2 must prepare a separate causal initial state for each fold. A3 cannot use future losses to decide whether local is eligible. A4 charges actual repeated liquidation, not merely a counterfactual fee reserve. A5 may sometimes reproduce hold if holdings exactly coincide with a grid point; report this.

The ablation family is reported in full. Secondary centered-bootstrap tests use Holm adjustment over all eight pooled ES comparisons. For each ablation define delta_t = ES_ablation,t - ES_primary,t, so positive favors the full primary. Subtract the pooled original mean from every date delta before paired fold-wise resampling under the null. The one-sided p-value is (1 + number of centered bootstrap means >= observed mean)/5001. Sort all eight p-values and apply the standard step-down Holm thresholds .05/(8-rank+1), stopping at the first failure. A4/A5 do not alter forecasts and their exact-zero ES difference is retained with p=1 rather than tested as invented forecast variation. Economic ablation analyses are explicitly descriptive. No best ablation is promoted into the primary result.

## 5 Statistics and precision feasibility

Use 5,000 circular moving-block bootstrap draws, seed 1901, block lengths 2 and 4 complete four-interval decisions (roughly eight/sixteen weeks). Resample each fold separately, retaining all assets, methods, costs and policies on each sampled date. Pool by original fold observation count. Compute 2.5/97.5 percentiles using linear interpolation. Both prescribed block families are shown; neither may be selected because it is favorable.

Primary economic block quantities are actual liquidated-endpoint log wealth increments from the retained path. Do not treat randomly concatenated holdings states as a physically executable replay. Sum-of-log-increment inference is a conditional descriptive diagnostic. Report exact original-path final wealth separately.

Before admitting the full study, evaluate precision using only pre-2010 prequential calibration diagnostics in a separately named, retained preparation stage. Estimate paired-score dispersion for the fixed candidate and controls with the same date-block scheme. A rough detectable-effect calculation using 1.96+0.84 times a development block-standard-error scale is a planning heuristic, not a power theorem or proof of enough independent regimes. Compute projected width at the proposed evaluation counts, with explicit stationarity assumptions.

If the 5% margin is not realistically distinguishable, report PRECISION_INADEQUATE. Do not lower the threshold, alter evaluation dates, add trials or sample protected data to make it pass. A redesign or an explicitly descriptive diagnostic requires a recorded new protocol revision before execution. This design does not claim the small ETF history can resolve H1.

Seeds apply to bootstrap and generated engineering fixtures only. Learned statistical units are dates; the two chronological folds and three assets are not independent experimental replications.

## 6 Bounded work and failure rules

Current authorization: design mapping only. Proposed later ceiling, requiring readiness and an execution directive:

- One worker, one numerical thread, no paid compute.
- At most 2 fold campaigns; all eight declared forecast arms and eight fixed ablations.
- At most 1,000 weekly issue/update origins per arm across both folds, including warm-up; actual timestamp count must be frozen after admission. Stop if exceeded rather than truncate.
- At most 192 market ledgers; no extra architecture or hyperparameter sweep.
- At most 60 minutes combined preparation/forecasting/decision/analysis wall time; at most 256 MiB process peak RSS.
- Standard >=512 MiB free-plus-speculative host admission with non-rising pageouts. Historical S6 waivers do not silently carry into this new protocol.
- A single bounded pre-2010 precision preparation of <=10 minutes, <=256 MiB RSS, one worker, charged separately and reported. This is not performed in the design session.
- No automatic retry after consumption, no response-driven extra data, and no enlargement because the result is negative.

The forecasting procedure has many small deterministic updates; do not misleadingly report it as only two model fits. Count and time every mixture solve, library rebuild, calibration origin and policy decision. Share immutable forecasts across policy/cost arms and cache exact pair-distance terms. Benchmark actual resource use before freezing the full run; proposed ceilings are not measured feasibility.

Generated chronology/accounting/known-signal cases are engineering tests. They may establish that the implementation detects an injected known mechanism, but cannot establish real market predictability. Leakage, source corruption, invalid arithmetic and incomplete control families produce nonzero failed receipts and an INVALID disposition pending a recorded review.

## 7 Terminal disposition and model selection

- H1 pass, H2 fail: distribution improvement only; no successful investment system.
- H1 fail, H2 pass: narrow economic result; no supported contextual distribution mechanism.
- Both pass: promising bounded retrospective development, still requiring independent verification and untouched confirmation.
- Either inconclusive: say which precision/identification limitation prevents a conclusion.
- Both fail with valid execution: close the frozen S10 version as negative/unsupported within scope; retain the complete evidence.

If context has no useful increment, global stacking or even U/equal mixing remains a candidate for a separately frozen simpler system. Selecting it afterward does not make S10's primary hypothesis pass.

No untouched confirmation interval is currently admitted. Before confirmation, audit exposure metadata, select real available data without reading protected outcomes, freeze the complete implementation/data/protocol/environment, and determine feasible precision. A prospective window starts after an actual future freeze; no historical start date is invented. No future collection or monitoring is scheduled by this design.
