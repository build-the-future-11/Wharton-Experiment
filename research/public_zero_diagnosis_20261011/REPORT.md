# Why zero won and why S2 memory switched off

This is a descriptive audit of retained public S2 predictions, not new training, parameter selection, causal proof, or protected evaluation. Every prediction hash was checked against its original receipt and every Student-t NLL reconstructed. All 24 prediction cells (12 variants x 2 seeds) were retained. The zero predictor is deterministic; its two copies are not independent replications.

## Where zero won

S2 evaluated December 2007 through June 2009, 23 surviving stocks, 19 monthly outcomes, seeds 11 and 29. Zero had the lowest aggregate NLL across all 12 variants, in both seeds. Its aggregate NLL was -0.709526 versus frozen S2 -0.603182. This is likelihood superiority, not profitability. The later six-window campaign contained only FREM, no-memory, Ridge and XGBoost: zero was NOT evaluated there.

Using equal-asset monthly scores averaged across the two seeds, zero beat every other variant in 12/19 months: January-March 2008, June-July 2008, September-December 2008, January-February 2009, and June 2009. This is a descriptive winner count, not a significance test. It did not dominate every month; full monthly scores follow.

## Measured sources of the score gap

Frozen S2 predicted +1.277% and +1.535% mean monthly log return for seeds 11 and 29; realized mean was -1.525%. Zero's mean was 0. It therefore avoided some positive directional error during this crisis sample. Zero also had lower RMSE (0.114149 vs 0.115471/0.116451) and MAE (0.087101 vs 0.088359/0.088904).

Uncertainty was more consequential for NLL. Nominal 95% intervals covered only 83.52%/83.30% of outcomes for S2, versus 89.47% for zero. Both undercovered. Zero's scale was 0.070497; average S2 scale 0.064428/0.062951. An all-orders three-field decomposition of the saved distributions attributes 0.067295 nats (63.28%) of the 0.106343 gap to scale, 0.031316 (29.45%) to location and 0.007732 (7.27%) to degrees of freedom. These are descriptive parameter substitutions, NOT deployed forecasts or evidence that replacing a field will generalize.

Source audit: zero uses location=0, df=5 and a scale estimated only from training-outcome RMS, with the Student-t variance conversion. It is a probabilistic zero-mean baseline, not a distribution with zero uncertainty. No evaluation-derived scale fitting was found. S2 uses a neural heteroskedastic distribution plus its declared validation-only scale grid. The procedures differ; this is not a matched-calibration causal experiment.

## Why memory was off

The memory was not empty or disconnected: 366 episodes matured, 238 were evicted, and 128 remained. Every evaluation row retrieved four neighbors. With same-asset retrieval there were just 5-6 retained candidates per stock. The retained labels were already 394-547 days old at the FIRST evaluation decision. Frozen evaluation does not refresh them.

The declared validation search correctly rejected nonzero corrections. At weight 0.25, even the best scale choice worsened validation NLL by 0.009561/0.012113 nats. At weight 0.5 the penalties were 0.034334/0.039807; at weight 1 they were 0.127417/0.139537. Both selected weight 0, scale multiplier 1. This establishes the immediate cause of zero memory contribution: every offered nonzero correction was worse on validation. It does not establish the population-optimal weight is exactly zero.

Sparse, old residuals are plausible upstream causes, not experimentally isolated ones. Source inspection also shows that age enters normalized neighbor weights: a common increase in all record ages cancels in softmax. The half-life changes relative weighting but does not itself attenuate the total correction as the entire bank ages. That matches current code; introducing absolute-age shrinkage would be a new scientific mechanism, not a bug fix. S2 deliberately disables the legacy utility/controller; blaming those components for this S2 outcome would be incorrect.

## Safe changes made

Added a reproducible standard-library diagnosis command with receipt hash checks, exact outcome pairing, NLL reconstruction, complete model/seed checks, monthly winner tables and descriptive parameter decomposition. Added future-fit diagnostics reporting the validation penalty by correction weight, retrieval counts, per-asset occupancy and memory ages. These expose the failure mode without changing forecasts, calibration grids, retention, training, seeds or thresholds.

Four focused unit tests and changed-file Ruff pass. Parsed the integration source; the full scientific/runtime suite and a new neural fit were not rerun in this diagnostic session. The previous 194-test receipt remains historical. Existing results are untouched; no claim that model performance has been fixed.

## Fresh evidence needed before behavior changes

No demonstrated implementation defect warrants forcing a new model behavior from these outcomes. Test any proposed calibration, per-asset capacity, age shrinkage, or delayed-online update as a separately declared successor on a genuinely fresh sample. Predeclare one-factor comparisons, null and recurring-signal controls, complete seeds, zero and historical-mean baselines, and identical calibration/information budgets. Do not add zero retrospectively to the spent six-window study and call it prospective. Do not search more gate weights, scales or epochs on this completed crisis period. No such experiments were run here.

## Monthly NLL (lower is better)

| Month | Zero | Frozen S2 | Best variant |
|---|---:|---:|---|
| 2007-12 | -1.3359 | -1.3941 | frozen_residual |
| 2008-01 | -0.2585 | 0.4049 | zero |
| 2008-02 | -1.1770 | -1.0997 | zero |
| 2008-03 | -0.9900 | -0.8639 | zero |
| 2008-04 | -0.6958 | -0.9123 | legacy_frem |
| 2008-05 | -1.2566 | -1.3271 | standardized_no_memory |
| 2008-06 | -0.7390 | -0.2673 | zero |
| 2008-07 | -0.7614 | -0.5585 | zero |
| 2008-08 | -1.3826 | -1.5281 | gru |
| 2008-09 | -0.2163 | -0.1781 | zero |
| 2008-10 | 0.4113 | 0.7655 | zero |
| 2008-11 | -0.1378 | 0.4056 | zero |
| 2008-12 | -0.8421 | -0.4088 | zero |
| 2009-01 | -0.6610 | -0.3457 | zero |
| 2009-02 | -1.0004 | -0.7995 | zero |
| 2009-03 | -0.5010 | -0.7825 | frozen_residual |
| 2009-04 | -0.0806 | -0.5537 | patch_transformer |
| 2009-05 | -0.6885 | -0.9493 | legacy_no_memory |
| 2009-06 | -1.1678 | -1.0679 | zero |
