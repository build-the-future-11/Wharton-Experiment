# Evaluation specification

**Prospective only. No scientific score, runtime-test pass, model training result or portfolio return has been produced in this authoring session.** The procedures below are implemented for later deliberate execution.

## 1. Primary estimand and information boundary

The primary comparison is the candidate model against the same model with memory disabled under the frozen data, split, feature, budget and seed protocol. Let `loss` be predictive negative log likelihood. For decision date `d`, asset `a` and seed `s`, define

\[
\Delta_{d,a,s}=\operatorname{NLL}_{\mathrm{control},d,a,s}
                    -\operatorname{NLL}_{\mathrm{candidate},d,a,s}.
\]

Average over the complete declared seed family for each matched `(time, asset_id)` observation, then within each asset and UTC date, then over assets equally within that date, then over dates equally. Positive mean difference favors the candidate. Random initializations are repeated computations on the same market history; they are not independent market histories. This weighting is explicit so that adding more securities on one day does not silently make that day dominate the primary result.

If an asset has multiple decision observations in one UTC date, average those observations within the asset/date first, then average assets equally. This prevents different sampling frequencies from changing cross-asset weights. A declared daily panel ordinarily has one such observation per asset/date.

The execution label begins at the first executable observation strictly after the decision, and ends `H` executable intervals later. Labels mature when their required outcome information is available. Evaluators consume saved prediction tables and never have an interface to `future_x`. Training may access future targets only inside the training partition after the maturity/purge boundary; frozen prediction may not use them.

## 2. Splits and held-out data

`chronological_split` operates on unique decision timestamps, preserving same-time asset groups. It first excludes the protected tail and any earlier decision whose outcome matures in that tail. Fractions set the nominal development boundaries. An explicit number of unique decision timestamps is skipped at the beginning of validation and test. Training labels must mature **strictly before** the first validation decision; validation labels must mature strictly before the first test decision. Equality is excluded.

`walk_forward_splits` uses expanding training windows with separate validation and test blocks. Test blocks never overlap. The protected tail is absent from fold construction. Prior observed market history may appear in a later context; unknown future outcomes may not. Time splits are not randomized over asset rows, and validation is not another name for the final test.

Use validation for hyperparameters, loss-controller policies and calibration as explicitly defined in the frozen protocol. Development test results can falsify a design; once inspected they cannot be relabeled as untouched confirmation. A final protected evaluation requires the exact source fingerprint, protocol/configuration freeze, complete baselines, seed family and failure accounting. Low-data subsets retain whole dates and use a predeclared earliest or most-recent selection rule.

## 3. Forecast table and distributional metrics

Required columns are `time`, `label_time`, `asset_id`, `prediction`, `scale`, `actual`; predictions also save `student_df`. Matrix cells save `model` and `seed` identities. Asset and seed IDs are finite nonnegative integers; missing keys cannot silently disappear in grouping. Entry/exit timestamps must be supplied together and are required for portfolio accounting. Times are UTC nanoseconds or explicit timezone-qualified strings. `prediction` and `actual` are cumulative log returns. `scale` must be positive. NaN actual outcomes are legitimate for live-style inference but are rejected by the scoring evaluator.

The conditional density is Student-t with location `mu`, scale `sigma` and degrees of freedom `nu > 2`. Its negative log likelihood is

\[
\log\sigma + \tfrac12\log(\nu\pi)
+\log\Gamma(\nu/2)-\log\Gamma((\nu+1)/2)
+\tfrac{\nu+1}{2}\log\left(1+\frac{(y-\mu)^2}{\nu\sigma^2}\right).
\]

Every saved per-row `student_df` takes precedence over a scalar fallback. A mismatched Gaussian score would evaluate a different predictive distribution. The standard deviation is `sigma*sqrt(nu/(nu-2))`; scale and standard deviation are not interchangeable. [SciPy's Student-t reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) documents the location-scale convention used by the implementation.

`score_predictions` returns every original observation plus:

| Metric | Role |
|---|---|
| NLL | Primary proper distributional score; lower is better. |
| CRPS | Secondary proper score, approximated by deterministic 96-node quantile quadrature. |
| Brier score for positive return | Proper binary event score using the Student-t probability of positive cumulative return. |
| Interval coverage and interval score | Predeclared 50%, 80% and 95% intervals; reports coverage with width and a proper interval score. |
| PIT histogram | Calibration diagnostic; no false independent-observation significance test is attached. |
| MSE, MAE and directional accuracy | Supporting point-prediction summaries; not substitutes for calibrated uncertainty or portfolio returns. |
| Reliability bins and ECE | Diagnostics; ECE is explicitly not a proper scoring rule. |

`evaluate_predictions` also reports cross-sectional Spearman rank association after averaging repeated observations within each asset/date. A date with fewer than three assets, constant forecasts or constant outcomes has an undefined rank correlation. The report preserves the number of undefined dates and returns no average if every date is undefined; it never substitutes zero or supplies an independent-observation significance test. Any mean is explicitly conditional on defined dates.

The proper-score motivation follows Gneiting and Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation* (2007), [author-hosted article](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). The implementation's numerical CRPS approximation needs later convergence verification, especially near the low end of allowed degrees of freedom. None was run here.

The evaluator never deletes nonfinite forecasts or mismatched labels to improve a score. A failed run remains failed. Evaluating multiple seeds together without explicit matched aggregation is rejected. Aggregate NLL, CRPS, point scores, Brier scores, coverage, width and interval scores use the asset/date weighting above. Reliability bins and PIT histograms pool individual observations and are labeled as such. Auxiliary portfolio costs do not change the definition of predictive NLL on the target return distribution.

## 4. Dependence-aware comparisons

`paired_date_test` outer-joins candidate and control by the declared observation keys and rejects duplicates or missing pairs. The reporting layer additionally verifies identical actual outcomes, label/execution times, any saved source/protocol identifiers, and the entire configured seed family. `average_seeds` rejects incomplete observation-by-seed groups and nonfinite scores. The reporting layer first requires every configured seed directory and every planned model/seed cell, so an entirely absent seed cannot vanish from that check.

`date_block_bootstrap` first forms date means, then samples contiguous circular blocks of dates with replacement. All same-date asset outcomes stay together. The percentile interval describes the mean score difference. The reported two-sided test uses centered bootstrap differences and a finite-resample correction. The seed controls reproducibility, not the number of independent histories. The resampler requires at least two blocks of observed dates; a tiny sample cannot be turned into persuasive evidence by requesting more bootstrap draws.

Freeze block length before test inspection. It must be at least the label-overlap bound and may need to be longer for market serial dependence. `minimum_overlap_block_length` supplies a conservative lower bound based on decision-date groups up to label maturity. That helper is not an automatic optimal bandwidth estimator. Label overlap, delayed publication and calendar differences can make `H` alone too small.

Block methods are motivated by dependent observations, not a claim that financial returns are stationary in every period. Künsch (1989), [*The Jackknife and the Bootstrap for General Stationary Observations*](https://doi.org/10.1214/aos/1176347265), is an original block-bootstrap reference. Structural breaks can still undermine inference. Report predeclared stress periods and cross-period heterogeneity; do not select the most favorable block length or regime after seeing held-out returns.

`compare_models` reads the exact `research/protocol.json` content hash saved with the run, using the archived `run/protocol.json` when present. Confirmatory analysis also checks that its scoring and reporting source files match their frozen hashes. Confidence level, practical threshold, number of resamples and bootstrap seed cannot be changed after the freeze. A different block length must belong to the declared sensitivity family; it is labeled sensitivity and cannot replace the prescribed primary block. Every block length must still satisfy the actual overlap bound. A prelisted block that is too short is rejected rather than silently enlarged.

`compare_family` loads all seven prescribed mechanism comparisons and applies `holm_adjust` to their complete two-sided p-value family. It implements the step-down familywise procedure of Holm (1979), [original article](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf). The command refuses a partial family. Report raw and adjusted p-values, effect sizes and confidence intervals. Its intervals are marginal intervals, not simultaneous family intervals. Holm controls the declared seven-comparison family at the chosen block length; it does not authorize selection among favorable sensitivity block lengths. Only the prescribed primary block is eligible for confirmatory secondary decisions. A statistically nonzero predictive gain is not automatically economically useful; the frozen minimum-effect threshold and costs determine that separate judgment.

After deliberate execution of a complete matrix, the interfaces are:

```bash
frem compare --run runs/COMPLETED_RUN --candidate frem --control no_memory
frem compare-family --run runs/COMPLETED_RUN
```

Python callers may specify `block_length`, `n_resamples`, `confidence` and `bootstrap_seed`; `compare_models` additionally accepts `threshold`. Omitted values use frozen protocol defaults for confirmation and saved configuration defaults where supplied in development. Contradictory confirmatory arguments raise an error. Analysis artifacts include prediction-file hashes and settings in their identity; different inputs or settings create distinct filenames and retain earlier evidence.

## 5. Portfolio accounting is a separate evaluation

An `H`-period forecast at every daily timestamp overlaps its neighbors. Compounding every label as a separate daily gain double-counts holdings and risk. `build_execution_ledger` accepts explicit long-only decision weights and rejects overlapping entry/exit intervals. All assets in an interval must share an executable entry and exit; asynchronous cross-market allocations require a synchronized holdings ledger before they can be evaluated together.

The narrow built-in ledger fully liquidates each interval. Weights sum to at most one; the remainder is cash. It does not hide turnover by pretending identical holdings persisted across round trips. Cash returns are supplied as an explicit interval series or are transparently assumed zero. This is research accounting, not a funded account or a broker integration.

With fractions `w_i`, risky asset gross returns `G_i`, cash gross return `G_c`, entry bps `b_in`, and exit bps `b_out`, the implementation computes

\[
\alpha=\frac{1}{1+10^{-4}b_{\rm in}\sum_iw_i},\qquad c_{\rm in}=1-\alpha,
\]
\[
W_{\rm net}=\alpha\left[\sum_i w_iG_i+
 (1-\sum_iw_i)G_c\right]
 -10^{-4}b_{\rm out}\alpha\sum_iw_iG_i.
\]

The factor `alpha` solves the entry self-financing identity for weights expressed as fractions of post-fee wealth: retained wealth plus fees equals initial wealth. The last term charges liquidation costs against the terminal risky notional. Turnover in initial-wealth units is `alpha*sum(w_i) + alpha*sum(w_i*G_i)`. Report gross and net returns, cost assumptions, turnover, cash exposure and drawdown. Liquidity, market/sector caps and mandate feasibility are enforced by the portfolio layer before proposed weights are supplied; this ledger is not a substitute for those checks.

`evaluate_portfolio` always checks nonoverlap and valid execution timing. It reports unannualized outcomes by default. Annualized Sharpe, volatility and geometric return require explicit `periods_per_year` and `regular_periods=True`, which is a caller attestation to the protocol cadence. Irregular or selectively skipped trades must not be annualized as 252 daily observations. Serial dependence and finite-sample uncertainty remain even after correct bookkeeping.

A Student-t law for **log** returns has no finite positive exponential moment. Therefore `exp(mu)-1` is a median simple return, not an expected simple return, and a lognormal mean adjustment is invalid. Portfolio decisions may use documented log-return utility, robust quantile criteria or explicitly truncated scenarios. They must not call the nonexistent untruncated simple-return mean an estimate.

## 6. Required reporting and stopping decisions

Persist the source fingerprint, configuration and protocol IDs, random seed, model variant, training/validation boundaries, last known label times, budgets, complete metrics and every failure. The experiment registry is the authority for which runs were attempted. `preserve_failure` writes a unique append-only failure artifact rather than an empty success record.

`render_report` refuses an incomplete matrix, checks the full seed/observation grid and identical outcome/execution evidence across all models, and only then writes a ranking table and figure. Its NLL weighting agrees with the comparison and evaluation functions. Seed standard deviation describes optimizer variability, not a confidence interval over independent financial histories. Existing completed reports are preserved; request a new destination for another rendering. Wealth charts require an explicit validated nonoverlapping ledger and never derive a trading path by compounding overlapping forecast labels.

A failed integrity check, missing baseline, omitted seed, insufficient date blocks, unstable training result or unusable portfolio ledger is a reportable outcome. It is not permission to discard the run, modify the protected split or announce a winning model. If the candidate does not exceed the frozen threshold against the memory-off control, the central hypothesis is unsupported under that protocol. A negative result should guide the simpler fallback model and the next independent experiment.
