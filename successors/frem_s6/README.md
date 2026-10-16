# FREM-S6: reservoir, delayed engram and predictive latent successor

S6 is an implemented, numerically tested research successor. **It is not a trained
market model or an established improvement over S5.** The final session report is
[research/frem_s6_20261013/REPORT.md](../../research/frem_s6_20261013/REPORT.md).
S3/S4/S5 source, results and negative conclusions remain unchanged.

## Actual implementation

- `kernel.py`: frozen training normalization; three contractive reservoir
  branches with different leak rates; cross-branch products and random Fourier
  features; separate bounded pending/mature date memory; strict feedback clocks;
  delayed quantized gradient ripples with quantization error feedback.
- `model.py`: trainable tanh encoder, linear latent predictor and stopped EMA
  target; supervised regression plus predictive-latent and variance penalties;
  zero, ridge, trend, adaptive-volatility and reservoir/JEPA experts; an optional
  actual XGBoost histogram-tree expert; exact Student-t mixture likelihood;
  calibration only after fitting; memory/equal/recent routing; checkpoints.
- `policy.py`: common-currency simple-return arithmetic, available-history
  covariance shrinkage, multivariate Student-t/jump scenarios, a bounded
  log-mixture-to-scenario bridge, exact fee accounting and multi-period
  long-only/cash/turnover-constrained portfolio candidate search with CVaR and
  terminal-shortfall penalties.
- `data.py`: an explicit development window contract. Sidecar admission and
  protected-date checks precede numeric outcome processing. Dated feature,
  future-target, calibration and evaluation checks enforce causal availability,
  complete date/asset grids and synchronized nonoverlapping outcomes.
- `experiment.py`: a fixed 24-fit development matrix, raw distributions and
  scores, whole-date delayed feedback, complete seed mixtures, common-policy
  fee/liquidation ledgers at 10/20/40 bps and prescribed 3/6-date block diagnostics.
- `run.py`: current resource admission before scientific imports, source/data
  identity freeze, current integrity evidence, exact protocol/config/date binding,
  exclusive artifacts, failure retention and one worker/thread.
- `check.py`: separately bounded integrity checks (128 MiB, 60 seconds), with
  source identities and actual measured resources.

The XGBoost expert uses 32 depth-two histogram trees and one worker. It never
silently falls back to another estimator. Its optional runtime branch was not
exercised during this low-memory session. There is no full PyTorch vision JEPA,
autonomous topology search, integer-only inference, distributed upscaling or
novel stochastic-calculus theorem hidden behind these names.

## Precise interpretations and limitations

“Rebound ripples” means a finite scheduled distribution of clipped gradients
over future update steps, with actual quantization and retained rounding error.
Immediate/unquantized updates are an explicit control. The final update norm
cap can discard additional magnitude; standard error-feedback convergence
results cannot simply be claimed for this optimizer.

“Mathematical proofing and exampling” means the
[mathematical contracts](MATHEMATICAL_CONTRACT.md), exact-reference cases,
finite-difference derivatives and adversarial counterexamples. These establish
specific invariants, not future returns, probability calibration or universal
correctness. “Network divergence” is fixed interacting branches, not autonomous
addition of useful neurons. “Dual” was interpreted as forecasting plus portfolio
research; the user's alternative definition has not been supplied.

Full/half/zero context changes the visible sequence input. Zero context still has
training-derived target moments, fitted parameters and, for the memory arm,
strictly matured historical losses. It is not literally zero information. The
fixed equal router supplies the control without outcome-adaptive routing.

Student-t **log returns have no finite exponential expectation**. The policy
bridge requires declared finite log bounds and a correlation matrix. It samples
a common expert regime and correlated t shocks, clips logs to those bounds,
then converts to simple returns. That is a changed, explicitly conditional
scenario distribution; it is not a calibrated estimate of expected returns.
The standalone simple-return jump sampler also states its flooring, dependence
and jump assumptions. Neither sampler knows future prices.

Portfolio planning optimizes only a finite candidate set and makes one initial
trade over the supplied multi-period paths. It marks terminal holdings; the
separate realized research ledger includes terminal liquidation. The planner
currently penalizes a terminal target, not Laura's complete dated liability
schedule. Source case cashflows remain in the existing mandate/portfolio layer.
There is no market manipulation, broker integration or current allocation.

The model adapter requires finite log targets, and therefore refuses default
observations that need a point-mass model. Policy accounting handles exact total
loss, but that does not establish full global data/model support. Provenance
strings and a matching hash are necessary records, not independent verification
that an external provider's assertions are true.

## Verification and execution

From the repository root, create a **new** output directory for every attempt:

```sh
rtk proxy .venv/bin/python -m successors.frem_s6.check --output research/s6_new_integrity
```

`run` requires `--execute`, `--output`, `--integrity`, `--dataset` and `--protocol`
for an admitted fit. Without complete inputs it writes a blocked admission
receipt before scientific imports. The dataset is a JSON `s6-windows-v1` bundle
with `partitions` containing `train`, `calibration` and `evaluation`; all timestamps
are UTC seconds. Its sibling `.manifest.json` uses `s6-manifest-v1`, an exact
SHA256 digest, outcome boundary and actual source/rights/point-in-time records.
`data.py` defines every required field and validation rule.

The separate frozen development protocol must bind all 24 exact configurations,
the three seeds, actual fold dates, a source-backed spent-date inventory with no
overlap, and a development precision analysis requiring at least 24 evaluation
dates with 3/6-date blocks. The `primary_endpoint` is
`mean_date_level_mixture_nll`; `economic_policy` is `shared_fixed_80_20_signal`.
All widths/contexts/controls are retained; outer outcomes never choose a winner.
The unexecuted matrix is 3 seeds × (3 contexts × 2 widths + 2 full-context controls).
Simple experts are fitted inside each cell; enabling XGBoost adds 24 tree fits
and must be frozen in every cell configuration. The fixed cap is 7,200 seconds
and 768 MiB sampled worker RSS, subject to the normal 512 MiB admission floor.
Native calls remain cooperative with RSS monitoring; parent wall timeout is hard.

No full development protocol or admitted S6 window dataset was fabricated in
this session. The launcher has no confirmatory mode and preserves the existing
2025 protected boundary. Existing equity survivor panels, factor portfolios and
synthetic fixtures are not interchangeable with this admission contract.

Read [prior art and data/Laura limits](PRIOR_ART_AND_LIMITS.md) before making any
claim of scientific novelty, global coverage, causation, speed or investability.
