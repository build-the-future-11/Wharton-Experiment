# Upgrade and evidence checklist

Checked boxes mean implemented, not numerically validated. This is a bounded, reviewable upgrade inventory; adding every conceivable technique would not guarantee quality.

## Implemented in this candidate

- [x] Preserve all S13 and older protocols, failures, receipts and outcomes in place.
- [x] Contractive multiscale reservoir; spectral features; causal n-gram keys.
- [x] Fixed capacity, training-residual-guided neuron recruitment with checkpointed statistics.
- [x] Timestamp-mature same-asset FIFO engrams; vectorized exact wrap behavior.
- [x] Personalized graph retrieval over valid neighbors; removal control.
- [x] Two-dimensional anisotropic stochastic diffusion with positive damping/noise and exact conditional Gaussian transitions.
- [x] Masked multi-offset EMA JEPA with covariance/variance and cosine regularization.
- [x] Multiobjective capped expert routing; heavy-tail mixture NLL and auxiliary risk losses.
- [x] AdamW, clipping, deterministic seeding, cosine schedule, early stopping, CUDA/AMP checks, exact epoch resume.
- [x] Zero/ridge/XGBoost/GRU/hybrid baselines and19 total ablation arms.
- [x] Strict OHLCV ingestion, raw hashes, hostile-value rejection, fixed calendars/cohort and honest proxy labels.
- [x] Publication/vintage as-of join with no future backfill.
- [x] Purged chronological splits, maximum-target maturity, train-only robust scales.
- [x] Three expanding historical folds, three seeds,20-observation labels and171 declared cells.
- [x] Raw predictive laws, independent SciPy score/moment reconstruction, paired block inference and family-wide multiplicity correction.
- [x] Capped allocation, training covariance, exact proportional-cost accounting, holdings drift, liquidation, complete-period checks and cost stress.
- [x] Independent allocation and drawdown readback.
- [x] Self-contained hash-bound GPU notebook; private Kaggle metadata; immutable logs/manifests/checkpoints and total timeout.
- [x] Source and installed-wheel dependency-free tests, static checks and release integrity verification are executable workflows.

## Additional 0.1.2 implementation checks

- [x] Reject decision times at/after the first target observation, and keep the protected boundary fixed.
- [x] Reject extra/missing CSV fields, invalid open interest, derived overflow and nonfinite JSON exponents.
- [x] Exact integer-microsecond timestamps; valid/stable admission samples at all three observations.
- [x] Bind nested manifests; reject symlinks/special files; verify prior resume time before charging it.
- [x] Keep graph memory arithmetic in float32 under AMP; mask ineligible values before moments.
- [x] Stable centered mixture variance; vectorized compounded labels; avoid unused metric synchronizations.
- [x] Record every complete, failed and not-started cell without reducing the 171-cell denominator.
- [x] Verify balanced retained calendars, control execution contracts and complete suite inference.
- [x] Stream downloaded ZIP integrity against a trusted local release and installed source inventory.
- [x] Regression tests for all these contracts; numerical tests still require target execution.

## Additional 0.1.3 input-to-result readback

- [x] Retain exact input bytes and supplier manifests before fitting; reject changed inputs and overwrites.
- [x] Verify every fold as the exact UTC chronological prefix of the retained source panel.
- [x] Reconstruct training/test labels, calendars, splits and train-only preprocessing from retained inputs.
- [x] Reconcile every cell source/data/configuration/runtime binding.
- [x] Reject broadcast-shaped prediction vectors, substituted model policies and altered inference settings.
- [x] Add eight dependency-free tests and ten generated numerical readback tests; numerical execution remains required.

## Additional 0.1.5 retrieval provenance

- [x] Retain full-arm test retrieval traces: mature episode decision/maturity times, asset, slot, similarity, age, values, base/graph/final weights, coverage and memory moments.
- [x] Reconstruct saved trace moments and reject unmatured, cross-asset, unknown-origin, out-of-bank or non-normalized retrieval evidence during readback.
- [x] Build versioned wheel, release ZIP, notebook and Kaggle metadata paths that refuse to overwrite prior candidates.
- [x] Keep the package export version synchronized with the wheel and accept both bare RunConfig files and declared suite-plan wrappers in the read-only `plan` CLI.
- [ ] Execute the new trace contracts in the exact numerical runtime and CUDA qualification; static checks alone do not establish trace correctness.

## Additional 0.1.6 GRU interface repair

- [x] Preserve Kaggle Version 3 (`360505332`) as a failed partial qualification: host admission passed, 104 tests passed, 14/19 arms completed, GRU failed, five arms did not start, and market intake did not start.
- [x] Keep S14 retrieval tracing out of the shared non-trace prediction call used by controls such as GRU.
- [x] Add a numerical regression test that exercises `predict` with the GRU control and finite returned laws.
- [x] Preserve the frozen thresholds, folds, arms, seeds, cohort rules, and budgets.
- [x] Execute the corrected 0.1.6 package in the exact CUDA runtime: 105 tests and all 19 generated arms passed with independent verification before intake.

## Additional 0.1.7 ETF taxonomy repair

- [x] Preserve Kaggle Version 4 (`360508080`) as a failed pre-market run: generated CUDA qualification passed, intake failed on two `spy.us.txt` basename matches, and zero historical cells started.
- [x] Resolve only the source archive's declared `Data/ETFs/<ticker>.us.txt` branch for the fixed ETF cohort, while recording all same-name matches.
- [x] Fail closed if zero or multiple canonical ETF paths are mounted; never select by observed returns or downstream metrics.
- [x] Add dependency-free regression tests for the stock/ETF basename collision and duplicate canonical mounts.
- [ ] Execute the separately frozen 0.1.7 package in the exact CUDA runtime; Version 4's evidence does not transfer to changed source.

## Evidence gates still requiring execution

- [ ] Full numerical suite on exact package, including SPDE matrix exponential/moments, graph causality, gradients and epoch resume.
- [ ] CUDA and AMP test success; exact GPU package runtime qualification.
- [x] Authenticated private Kaggle upload, saved Version 3 source identity, and launch/failure receipt for 0.1.5.
- [x] Authenticated private Kaggle upload, exact saved notebook identity, and terminal pre-market failure receipt for corrected 0.1.6.
- [ ] Authenticated private Kaggle upload, saved notebook identity, and terminal run receipt for corrected 0.1.7.
- [ ] Actual selected-dataset intake and all171 benchmark cells, without dropped failures.
- [ ] Download and independently verify the saved Kaggle results bundle.
- [ ] Demonstrated predictive improvement, efficiency or generalization. These are outcomes, not implementation properties.

## Further research / data work, not silently implemented or run

- [ ] Ragged whole-market panel with verified listing/delisting identity and point-in-time eligibility.
- [ ] True corporate-action/terminal-return reconciliation and executable price/spread/impact data.
- [ ] SEC filings with publication timestamps and macro real-time vintages.
- [ ] Separate Jane Street weighted regression task and access admission.
- [ ] Timestamped news/sentiment mapping; labels alone are insufficient.
- [ ] LightGBM, transformer/DLinear, learned reservoir, risk-sensitive allocation and joint multivariate tail extensions as separately scoped comparisons.
- [ ] Real missingness/OOD stress, broader horizons and data-fraction studies with a finite frozen budget.
- [ ] Empirical block-length adequacy, uncertainty calibration, genuinely untouched confirmation and independent reproduction.

No assertion of perfect code, guaranteed profitability, calibrated crash probabilities, physical PDE identification or zero remaining defects is supported.


## 21 October 2026 — Version 5 evidence

- [x] Preserve exact Version 5 terminal failure: 107 collected, 105 passed, 2 failed, 0 skipped; qualification and historical cells did not start.
- [x] Reproduce both fixture failures on the exact 0.1.7 wheel; correct only the fixture's directory contract, retaining every assertion.
- [x] All 59 dependency-free tests pass with warnings as errors after correction; scoped static checks pass.
- [x] Verify local versioned source/wheel/archive/notebook and retained saved-code identity; keep every frozen 0.1.7 artifact.
- [ ] Download and integrity-verify Version 5 outputs; Chrome blocked the normal archive action, so visible-log evidence is not represented as downloaded bundle verification.
- [ ] Build a new immutable corrected test payload and pass all 107 tests plus 19 generated CUDA arms before the unchanged 171-cell historical suite.

See `research/frem_s14_version5_failure_20261021/FAILURE_RECEIPT.json`. No GPU passes transfer to the corrected payload and no scientific result is claimed.

## 0.1.9 source-calendar successor

- [x] Preserve 0.1.8 Version 6 as a valid pre-market source-admission failure after 107 tests and 19/19 generated qualification arms.
- [x] Cancel the accidentally queued identical Version 7 before execution; do not interpret it as a repeat.
- [x] Freeze SPY/IWM/DIA/TLT/IEF/GLD/SLV before historical model outcomes and exclude QQQ solely from the source-calendar audit.
- [x] Allowlist exact source hashes and retain exact calendar equality; do not intersect, delete rows, forward fill or impute.
- [ ] Build and locally verify immutable 0.1.9 artifacts.
- [ ] Execute the exact 0.1.9 target tests and all 19 generated CUDA qualification arms.
- [ ] Execute and independently reconstruct every one of the unchanged 171 historical cells.


## 21 October 2026 — Version 6 qualified; historical source calendars incompatible

Private Kaggle Version 6 (`360684950`, package 0.1.8) passed all 107 warning-strict
tests in 27.75 seconds, completed 19/19 generated CUDA arms and passed its in-run
independent 222-file semantic verifier. Saved notebook cells match the local
candidate; GPU T4 x2, Internet off, fixed source and private visibility are verified.
The run failed after 105.6 seconds during strict historical intake, before any fit.

All eight independently downloaded public source files match the executed hashes.
QQQ has 2736 selected dates; all seven other ETFs have 2735 and lack 2011-02-17.
No dates or prices were removed, filled or fabricated. The frozen equality rule
cannot admit these exact bytes. Historical completion remains 0/171; S14 remains
NOT_EVALUATED, with no predictive or economic baseline advantage established.

The 308 run files plus Kaggle log are retained. Anchored integrity verification
passes for 18 payload members, 15 installed modules and the qualification manifest.
The local ZIP was reconstructed from API-downloaded files. Local numerical
readback was refused by the unchanged memory/pageout gate. The temporary API token
was revoked and its local secret removed; other tokens remain intact.

Added a separate read-only calendar preflight with three passing regression tests;
it does not modify or qualify the frozen GPU payload. Read
`research/frem_s14_kaggle_0_1_8_20261021/FAILURE_RECEIPT.json` and `FLAW_CHECKLIST.md`.
Do not launch Version 6 again or silently weaken its intake. Recover compatible
source observations or specify a separately versioned missing-data study before
fitting. The broader research/paper completion request is not yet fulfilled.


### Current frontier reconciled across research chats

The original eight-ETF Version 6 failure above is historical. The separate
seven-ETF 0.1.9 study (private Kaggle Version 8) is also historical and terminal:
its archived output was integrity-verified, but independent semantic readback
failed. Its saved-source identity is recorded in
`research/frem_s14_calendar_successor_20261021/SAVED_IDENTITY.json`; the failure
decision is `research/decisions/frem_s14_version8_semantic_readback_failure_20261021.json`.
QQQ was excluded prospectively on source-calendar grounds; retained histories
were not intersected or filled. No Version 8 historical metric is valid evidence.
Do not relaunch or alter Version 8. The 0.2.0 successor below is a separate
prospective candidate; baseline comparison and final research release remain
pending its independently verified evidence.

## 0.2.0 semantic-reproducibility successor

- [x] Preserve Version 8 as an invalid execution rather than relaxing its
  numeric tolerance or inferring a performance result from retained metrics.
- [x] Freeze a new score-artifact contract: persisted target-runtime NLL, Brier
  and coverage vectors feed ordered scalar aggregation, while an independent
  scorer still validates every score vector within its declared law tolerance.
- [x] Replace NumPy-dependent train-only scaler reductions with deterministic
  sorted scalar medians and `math.fsum` standard deviation.
- [x] Reserve a 4000-second per-fold ceiling for in-run semantic readback under
  the unchanged notebook-wide 13500-second cap.
- [x] Build and hash-bind the new 0.2.0 release and private Kaggle notebook.
- [ ] Pass target warning-strict tests, 19 generated qualification arms and both
  semantic verifiers before opening the fixed historical-proxy suite.
- [ ] Complete and independently verify the 171-cell suite without excluding
  failures; classify rather than optimize for a favorable outcome.
