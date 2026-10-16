# FREM-S9: horizon and joint-risk diagnostic

Version 1, 2026-10-15. **Exploratory retrospective methodology study; no new algorithm or investment success claimed.** This is a separate successor, not a reopened S7/S8 evaluation. The original memory hypothesis, 0.01-nat criterion and earlier 6% wealth objective remain unchanged.

## Question, contribution and idea gate

Previous finding: S8 improved its density point estimate but its economic primary lost money. Remaining problem: a scaled short-horizon forecast and diagonal risk may misrepresent a four-week decision. Hypothesis: directly observed four-interval joint scenarios improve four-interval return prediction over constant repetition of one-interval scenarios, and joint risk changes the decisions beneficially when the forecasts are held fixed. Falsifiers are unchanged or worse prediction and net utility in the paired comparisons below.

The contribution is an explicit, reproducible diagnostic separating horizon, contextual conditioning and cross-asset risk on the project's existing training-era data. It is an empirical method comparison, not a claim of original nearest-neighbor forecasting or a universal explanation of S8's losses. It cannot causally establish what caused the closed S8 result on another path.

The earlier restart gate combined full canonical confirmation with exploratory feasibility. This study addresses the latter using available data with a narrower claim, while canonical admission and unused confirmation remain unresolved. Checkpoint 1 is satisfied **for this bounded exploratory question only**: mechanism, controls, dates, source limitations, fixed budget and falsifiers are explicit. It does not certify the broader project IDEA READY or confirmation ready.

## Prior art and reasons the proposal may fail

Direct forecasting is established and does not uniformly beat shorter-horizon alternatives. [Marcellino, Stock and Watson (2006)](https://www.princeton.edu/~mwatson/papers/Marcellino_Stock_Watson_JoE_2006.pdf) describe the robustness/estimation-variance tradeoff and find iterated forecasting often stronger in their macroeconomic setting. Our scaled scenario baseline is explicitly **constant repetition**, not their recursive AR method. Their results supply a counterargument, not evidence for ETF gains.

Conditional empirical optimization is established in [Bertsimas and Kallus, From Predictive to Prescriptive Analytics](https://arxiv.org/abs/1402.5481), first submitted in 2014. Its general framework motivates contextual empirical scenarios; its theoretical guarantees are not asserted here. [Boyd et al. (2017)](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) separate portfolio decisions from the unresolved quality of their input forecasts. These three sources were consulted on 2026-10-15; paper versions/dates are not retrieval dates.

Major objections: limited dependent history; fixed surviving ETFs; high-dimensional neighbor distances; unstable conditioning; no full tail support; current-vintage prices; assumed observation availability; simplified dividends and fills; discrete actions; sparse rebalancing; possible superiority of unconditional forecasts. Remedy: fixed global controls, identical neighbor identities across horizons, paired dates, complete cost reporting, explicit claim boundaries, exact accounting, no test-based tuning, and a terminal negative/inconclusive option. Independent reproduction and generalization are separate unmet requirements.

## Data and chronology

Use only the existing retained SPY/TLT/GLD source files in `data/s6_real_etf_20261014`, bound by their original acquisition hashes. They contain 2005–2018 current-vintage historical prices retrieved on 2026-10-13. Build this study solely from dates before 2017. Do not read any 2019+ source or protected outcome. Canonical panel admission remains absent; this adapter admits a narrower retrospective contract only. Source access does not establish redistribution rights; no raw-data release is authorized.

Require identical daily asset calendars, positive finite open/close prices, increasing timestamps, no splits, valid nonnegative dividends and matching identities. Reject rather than intersect away calendar failures. First common open in each ISO week defines an anchor. A modeled decision occurs at 12:00 UTC on the anchor date, before the recorded open. Interval outcomes are assumed available at midnight UTC after the ending open date. These are conservative research conventions, not observed vendor publication times or proof of attainable fills.

For each asset and interval [s,e], retain price gross g=P_e/P_s and dividend yield d=sum(D with s<ex-date<=e)/P_s. Dividends are accrued on ex-date as cash, with no reinvestment and zero cash yield; actual payment delay/taxes/spreads are unmodeled. Price and dividend components stay separate so liquidation fees are never charged on dividend cash. Endogenous defaults are outside this positive-price ETF contract.

Features at decision i use the last 13 weekly total log returns whose outcome-availability timestamp is strictly earlier than decision i. Per asset: last return, sum of last four, sum of last thirteen, and population standard deviation of thirteen. Asset-major SPY/TLT/GLD order gives 12 inputs. This deliberately excludes the return ending at the current unobserved open. There is no future imputation.

Every training exemplar stores x in R^12 and paired (g,d) in R^3 x R^3 for horizons one and four. Both horizon arms use the same origin rows and require the four-interval outcome to be available before fitting. Four-interval outcomes are computed directly from the endpoint prices and total cash dividends, not by reinvesting weekly dividends.

Two fixed folds:

| Fold | Exemplars allowed | Evaluation decisions |
|---|---|---|
| A | Outcomes available strictly before 2010-01-01 | 2010-01-01 through 2012-12-31 |
| B | Outcomes available strictly before 2013-01-01 | 2013-01-01 through 2016-12-31 |

These periods were within predecessor training and are not globally unseen. Fold B may use earlier fold A history because it is a separately specified chronological fit; no S9 result selects its parameters. Evaluation decisions advance four anchors at a time from the first eligible anchor; retain only complete four-interval blocks ending before the fold boundary. Report unused terminal calendar days. Require at least 64 training exemplars and 24 evaluation blocks per fold. Do not inspect outcomes to change these dates.

## Complete deterministic model

Fit feature means and population scales on training x only; a scale below 1e-12 becomes 1. Standardize training/query vectors with those fixed values. Local forecasts use the 32 closest training exemplars under squared Euclidean distance, ties broken by chronological origin. Global forecasts use every training exemplar uniformly. No outcome values choose neighbors, parameters or stopping. Store scaler, training identities and scenario components in a JSON checkpoint; verify a reload gives identical forecasts.

Four variants: `h1_local`, `h4_local`, `h1_global`, `h4_global`. Both local horizons share the exact same 32 selected origins. For h4 use the observed four-interval (g,d). For h1 constant repetition use g^4 and d*(1+g+g^2+g^3), an explicit counterfactual assumption of repeated weekly price and proportional dividend behavior. All scenario probabilities are uniform and finite. Reject overflow or invalid support; never clip observed losses. This is not Student-t exponentiation and does not estimate a density NLL.

Forecast expected total simple return is mean(g+d-1). Training is deterministic scaler/memory construction, two fits total. Repeating seeds would produce duplicates, so no fake three-seed independence is reported. No neural gradients, JEPA, reservoir or online updates are part of this diagnostic. They remain unchanged in the predecessor.

## Exact policy and controls

Use a declared finite action set w_i in {0, 1/12, 2/12, 3/12, 4/12}, sum(w)<=0.8, long-only. Lexicographic order breaks objective ties; cash is included. Risk aversion is fixed at 4. Each four-interval block starts in cash, buys once and fully liquidates at its end. This deliberately makes the decision model and realized accounting identical; it is not S8's continuous carry policy. No selected result is described as a direct replacement for S8.

With proportional fee c, entry NAV fraction q/V=1/(1+c*sum(w)). Under scenario s, terminal wealth factor is

F_s(w) = [1-sum(w)+sum_i w_i*((1-c)*g_si+d_si)]/[1+c*sum(w)].

This follows from exact cash-funded entry, cash dividends, price-only liquidation costs and zero cash yield. The support g>0,d>=0 and sum(w)<=0.8 gives strictly positive F. Optimize mean(F-1) - 4 Var(F). `joint` uses the full covariance of a_si=(1-c)*g_si+d_si-1; `diagonal` drops only off-diagonal covariance. All other inputs, actions and fees are identical. Enumeration gives the exact optimum over the declared finite set, not over continuous portfolios. Compare selected objective with cash and every alternative; reject nonfinite quantities.

Costs are 0/10/20/40 bps one-way; primary cost is 10. Retain all four forecasts x two risk modes x four costs x two folds = 64 strategy ledgers. Controls at each cost/fold: cash; monthly equal assets (100% risky); monthly equal assets with 80% risky budget; equal-asset buy-and-hold with dividends retained as cash and only entry/final exit fees. These add 32 benchmark ledgers. No borrowing or live orders.

## Endpoints, uncertainty and terminal decisions

Primary diagnostic: paired difference in date-level MSE of expected four-interval total simple returns, scaled-local minus direct-local, averaged over all three assets. Report relative MSE improvement and each fold separately. A promising exploratory horizon effect requires positive point improvement in both folds, at least 5% pooled relative improvement, and both prescribed descriptive intervals entirely above zero. This new MSE criterion is not a replacement for the original 0.01-nat memory criterion.

Use 2,000 circular moving-block bootstrap draws per interval, seed 1901, blocks of 2 and 4 complete four-interval decisions, resampled separately within each fold. Combine fold sums weighted by their original observation counts. Percentiles use linear interpolation. Dates, not assets or repeated deterministic fits, are resampling units. These are descriptive retrospective intervals; no confirmatory p-value or publication guarantee is implied. Failure of the exploratory criteria is labeled INCONCLUSIVE_OR_UNSUPPORTED, with negative point effects explicit.

Secondary comparisons retain the full crossed table: joint versus diagonal at fixed forecasts; local versus global at fixed horizon; costs and broad benchmarks. Report endpoint drawdown at observed four-interval marks, not unobserved intraperiod maximum drawdown; fees, risky exposure and per-block returns. The 6% relative ending-wealth objective versus both equal-asset rebalancing and buy-and-hold remains a separate investment gate, required in both folds at 10 bps along with beating cash. Even passing this exploratory gate would not constitute confirmation. Do not choose a secondary winner as the primary. No learning curves/OOD claims are made without additional predefined experiments.

## Execution, failure handling and budget

One worker/thread, two deterministic model fits, no hyperparameter trials, 600 seconds wall time, 128 MiB process peak RSS; retain the standard 512 MiB free-plus-speculative host admission floor and non-rising pageouts. No paid compute. Admission precedes market fitting. A blocked execution exits nonzero with a retained receipt. The historical S6 RAM override is not changed.

Require dataset/split validation, numerical tests, checkpoint equivalence, ledger reconciliation, strict JSON, source/data identities and a complete generated-artifact check. Engineering examples use constructed inputs and are labeled tests, never market results. Freeze this specification, implementation, data identities and environment before one market attempt. Retain incomplete output and failures; never silently resume or replace results. A valid completed diagnostic is terminal for this version. Confirmation requires a separate feasible protocol and untouched admitted evidence.

## Proof and implementation scope

Finite scenario covariance is PSD by sum of squared projections; the diagonal covariance is PSD because each diagonal variance is nonnegative. The exact terminal-factor formula proves self-financing and correct fee/dividend treatment under the stated assumptions. Enumeration proves finite-grid optimality only. Train-only scaling and strict outcome maturity make the declared chronology auditable. None proves the empirical distribution is calibrated or guarantees future gains.

Runtime and memory complexity: O(nd) fitting/storage, O(nd+n log n) neighbor selection, O(sd^2) moments with three assets, and O(K*3^2) action enumeration after moments. Here n is training exemplars, d=12 inputs, s<=n scenarios and K is the finite action count. No synthetic market result, original-algorithm novelty, historical data availability, canonical admission, independent reproduction or complete research release is claimed.
