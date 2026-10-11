# Checkpoint 3 — prospective methodology

Status: **methodology fully specified; empirical gate blocked by missing admitted
data and team-only WInS rules.** No protected boundary exists and no market
experiment is authorized.

## Scientific question and unit

The primary estimand is the paired change in mean date-level Student-t negative
log likelihood: `NLL(control) - NLL(full)`. The statistical unit is a UTC
decision date, not an asset row. The practical threshold is 0.01
nat/observation. All assets eligible on a date remain grouped for splitting,
resampling and inference.

## Required data admission

The panel must include immutable raw-file hashes, provider/license, retrieval
time, observation time, availability time, exchange calendar, point-in-time
universe membership, delistings, corporate actions, dividends, split factors,
local prices, FX to USD, volume/liquidity fields, and missingness reasons. Total
returns must be reconstructible without treating yield changes as bond returns.
Every transformation must be fit on the training prefix. A manifest validator
must reject absent provenance, duplicate keys, future availability, current-
membership backfill and unexplained revisions.

The official WInS approved universe and trading rules are a separate input from
the scientific market panel. They are required before a competition portfolio
can be called compliant.

## Chronological split policy

1. Sort unique decision dates.
2. Reserve the final contiguous interval as protected exactly once, only after
   data validation and freeze. Its start date remains blank until data arrives.
3. Use earlier dates for development with an expanding-window train and later
   contiguous validation interval.
4. Apply an embargo at least equal to the maximum target horizon between every
   train/validation/protected boundary.
5. Construct low-data regimes by taking the earliest 10%, 25% and 50% of the
   otherwise eligible training dates; never subsample asset rows within dates.
6. Fit normalization, feature selection, imputation, engrams and hyperparameters
   separately inside each training prefix.

The actual dates, row counts, feature availability and SHA-256 hashes must enter
the freeze artifact; placeholders are not sufficient.

## Model family

Primary candidate: `rejepa_full`.

Primary control: `rejepa_no_world_no_engram`, retaining the same encoder/fusion
shape and output tasks while using deterministic context in place of imagined
worlds and zero retrieval.

Mechanism controls: world-off, engram-off, both-off, one interpreter,
Fourier-key-off and a joint return/risk head. The implementation exposes the
corresponding feature flags. A widened both-off control must be chosen during
development to fall within ±5% trainable parameters of the full model; a
separate run caps wall time to the full model's observed budget.

Baselines: zero/historical mean, ridge, elastic net, gradient-boosted trees,
echo-state ridge, GRU, DLinear and PatchTST-style patch transformer. Every
baseline receives the same feature fields, dates and horizon. Broken baseline
cells invalidate—not strengthen—the comparison.

## Tuning and budgets

Development seeds are 11, 29 and 47. Confirmatory seeds are 11, 29, 47, 71 and
97. Model selection minimizes validation date-level NLL subject to no material
calibration degradation. Search spaces, maximum trials and early-stopping rules
must be serialized before launching. Development ceiling is 72 cells/12 hours;
the confirmatory template ceiling is 225 cells/24 hours. These are ceilings, not
authority to run every preset.

Record per cell: source hashes, code commit or tree hash, configuration hash,
seed, exact dates/row counts, parameter count, training steps, wall time, peak
RSS/accelerator memory, stop reason, prediction hash and terminal status.

## Endpoints

Primary: mean date-level Student-t NLL.

Mandatory guardrails: PIT/interval calibration, coverage by horizon and country,
and no catastrophic directional-Brier degradation.

Secondary forecasting endpoints: CRPS when implemented, MAE, date-level rank
correlation, directional Brier score and expected-shortfall calibration.

Secondary decision endpoints: net-of-cost certainty equivalent, maximum
drawdown, turnover, reserve funding probability and coverage of the 2031
facility-contribution interval. Decision endpoints cannot rescue a failed
forecasting primary endpoint.

## Statistical analysis

Average asset losses within each date, pair candidate and control by the exact
date, and bootstrap contiguous date blocks. Primary block length is 20 dates;
5 and 60 are sensitivity analyses. Use 5,000 resamples and a two-sided 95%
percentile interval. Apply Holm correction across the predeclared mechanism
control family. Report every seed and low-data fraction, then the declared
aggregate—never only the favorable regime.

Classification of the primary result:

- supported: complete valid lower interval endpoint exceeds 0.01 and calibration
  is not materially worse;
- negative: complete valid upper endpoint is at or below zero;
- mixed: a component/regime improves but the primary rule fails;
- inconclusive: valid interval overlaps the practical threshold;
- invalid: any integrity, matching, completeness or provenance gate fails.

## Robustness and OOD checks

- costs at 0, 1×, 2× and 3× the declared schedule;
- country/currency leave-one-group-out diagnostics;
- volatility and correlation-regime slices defined on trailing data only;
- thin-history and high-missingness slices;
- alternative 5/20/60-date bootstrap blocks;
- five independent latent-world draw sets using frozen seeds;
- true deep-ensemble sensitivity, distinct from within-model world samples;
- reservoir contraction, world diversity, gate entropy, engram utilization and
  head-gradient cosine diagnostics;
- no-memory, shuffled-mature-memory and utility-neutral memory controls.

## Failure handling

OOM, timeout, non-finite loss, missing prediction rows, unavailable baseline,
zero-variance metric, provenance defect or post-freeze mutation receives an
explicit failed/invalid receipt. No seed may be silently dropped. A genuine
implementation defect requires a recorded explanation and a new corrected
freeze; an unfavorable valid result never does.

## Freeze prerequisites

Checkpoint 3 may formally pass only after all of the following exist:

- admitted data manifest and validator receipt;
- actual chronological split dates and embargo;
- official WInS universe/rules snapshot if competition claims are in scope;
- fixed feature schema, horizons, cost schedule and eligibility policy;
- exact search spaces and cell budget after resource admission;
- protected-boundary access control;
- signed configuration, source and protocol hashes.

Until then, the methodology is implementable but not frozen.
