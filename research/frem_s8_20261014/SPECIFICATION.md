# FREM S8: soft routing and a cost horizon aligned with holding

Written before acquiring the 2020–2022 evaluation quotes. This is an additive, bounded development successor; S7 and 2019 remain closed. No guarantee of improvement or novelty is claimed.

## Diagnosis and falsifiable changes

S7 required a practical 0.01-nat gain in each 2018 half to permit any adjustment. All nine seed/strength cases had positive half means, but the largest means were only about 0.0052 and 0.0062. That was a valid frozen gate, not an implementation error. Selection and the threshold for a scientific claim serve different purposes.

S7 compared a weekly utility surrogate with twice the immediate turnover cost. Its candidate set jumped from cash to at least 80 percent risky exposure. Its saved 2019 inputs never justified that fee under that surrogate. The diagnosis uses retained decisions only; no new forecast, policy or parameter is tested on 2019.

S8 replaces binary acceptance with delayed probabilistic mixing of equal expert averaging and contextual mixture routing. A two-strategy probability vector starts at (0.5,0.5). On each newly available complete date, multiply probabilities by exp(-asset-mean NLL), normalize, and mix 1/52 back toward the initial prior. Prediction is a convex combination of the two density forecasts. Apply updates only when availability is strictly before issue; the current outcome cannot affect its own forecast. The 1/52 share encodes a fixed annual adaptation scale for weekly dates; it is not selected on outcomes. The old 0.01-nat practical threshold remains the reporting threshold, with no binary deployment gate.

The successor policy solves, for H committed weekly intervals:

    max_w H * sum_i(mu_i*w_i - 4*v_i*w_i^2)
          - c*sum_i(abs(w_i-h_i)) - c*sum_i(w_i)
    subject to 0 <= w_i <= 1/3 and sum_i(w_i) <= 0.8.

Here mu and v are log-return mixture moments, h is current risky weight, c is one-way cost. This is a concave ranking surrogate, not expected simple return or an exact utility integral. Student-t log returns lack finite exponential moments. The last cost term reserves estimated exit cost for all alternatives including hold. The optimum is computed by the separable soft-threshold solution and a nonnegative budget multiplier. H=4 is primary; H=1 isolates the holding-horizon change. The holdings are committed between scheduled reviews, with no intermediate rebalancing; H shrinks at the end of the available evaluation calendar. Actual one-way costs and final liquidation are charged by the self-financing ledger. No compulsory risky anchor. Nonpositive predicted opportunities may still leave cash. Holding coefficients constant for four weeks, diagonal risk, and ignoring within-block drift in optimization are explicit approximations that may fail.

## Frozen experiment

- Assets: existing retrospective SPY, TLT, GLD basket; current-vintage survivor sample. No broader market or mandate claim.
- Train: 2005–2016, all label/future-feature availability purged before 2017.
- Scale calibration: outcomes available within 2017. Prequential gate initialization and original S7 strength selection: 2018. No 2019 quotes, features, targets, selection or feedback in S8 execution.
- Fresh evaluation: January 2020 through December 2022, fetched only after code/protocol freeze. Start after eight fresh daily returns exist; then retain every continuous first-session weekly interval with both endpoints before 2023. Do not bridge the missing 2019 year. This window is fresh to this ETF successor lineage, not universally unseen history: older research already examined 2020/2022 on other assets.
- Calendar: actual 09:30 America/New_York session opens, daily-close features conservatively available at next UTC midnight, outcome availability four hours after end open. Availability is assumed, not vendor-vintage-certified. Quote/action timestamps and acquisition HTTP timestamps retained. Dividends accrue on ex-date and reinvest at the weekly endpoint; payment delay/taxes ignored. Cash return is zero.
- Three base fits, seeds 11/29/47, unchanged S6 full model defaults with XGBoost. All forecasts share each seed's experts and identical online information.
- Forecasts: S8; equal; S6 memory; S6 recent; original S7 gate; ungated contextual mixture; global mixture; S8 global (locality ablation); fixed half equal/contextual blend (soft-update ablation); six individual experts. No base-model search or post-result selection.
- Cross every forecast with S6 signal, S7 cost-aware, S8 H=4, S8 H=1; one-way costs 0/10/20/40 bps. Also cash, equal-weight assets, equal-weight buy-and-hold at each cost.
- Primary prediction comparisons: S8 versus fixed equal averaging and S6 memory. Both must have >0.01-nat lower bounds for both fixed 6- and 12-date circular block intervals (2000 resamples, seed 1901). No block selection. Report all other comparisons descriptively. Complete seed density ensemble primary; date/asset-balanced mean-of-seed scores also reported. Seeds are initialization repeats, not independent market histories.
- Primary economics: S8 H=4 at 10bps versus (i) equal forecast H=4, (ii) S6 memory H=4, (iii) S8 S7-cost-aware, and (iv) equal-weight assets. All four terminal wealth differences must be positive to claim a point win; conditional block intervals on net log increments at blocks 6 and 12 are reported, never treated as an independent portfolio simulation or calibrated future guarantee. Report each calendar year without subgroup selection, fees, turnover, exposure and drawdown.
- A successful mechanism claim additionally requires beating fixed-half and global ablations; otherwise attribute any gain narrowly.
- Budget: exactly three base fits, one process/thread, 7200-second outer deadline. Continue the recorded user RAM waiver; preserve the original guarded runners. No cloud charges or unrelated-process termination.
- Stop after this matrix. Preserve every failure, checkpoint, seed, comparison, acquisition and analysis. No fresh-data tuning or repeat after a valid negative result. Original 2025+ protected boundary stays untouched. This is development, not canonical confirmation.

## Prior-art boundary and risks

This uses established expert aggregation and constrained cost-aware allocation, not a demonstrated new algorithm. Herbster and Warmuth, Tracking the Best Expert (Machine Learning 32, 1998): https://mwarmuth.bitbucket.io/pubs/J39.pdf . Boyd et al., Multi-Period Trading via Convex Optimization (2017): https://arxiv.org/abs/1705.00109 . Primary sources checked on 2026-10-14. Their results do not establish financial forecast accuracy for this implementation. This bounded work tests the diagnosed decision mechanisms, without a novelty or general no-regret claim for the full contextual system.

Potential failures: weak experts, a soft blend lagging regime shifts, informative gains below the practical threshold, narrow-density experts dominating mixture scores without better returns, conservative means still producing cash, correlated risks omitted by diagonal approximation, stale four-week forecasts, and development-period selection bias. The fixed controls and predeclared years expose these limits; none authorizes retuning the closed evaluation.
