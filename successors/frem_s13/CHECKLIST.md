# S13 upgrade and acceptance checklist

`[x]` means implemented source, not a passed runtime or a financial finding.
`[ ]` marks evidence still required. This is the finite S13 implementation scope;
“every possible enhancement” and “zero flaws” are not verifiable specifications.
Earlier model families, frozen protocols, failed runs and all prior evidence
remain unchanged. Existing historical models are still available in their own
packages; S13 does not inherit their test receipts or scientific results.

## Architecture and mathematics

- [x] Multiscale fixed reservoir with three leak rates (`model.Encoder`).
- [x] Largest-singular-value normalization and explicit contraction bound.
- [x] Masked, preallocated, bounded neuron recruitment with a training-only trigger.
- [x] Fixed-full-capacity and no-growth controls; mask preserved in checkpoints.
- [x] Causal nonzero Fourier energy bands over trailing windows.
- [x] Trailing discrete return ngrams in fixed retrieval coordinates.
- [x] Explicit no-spectral, no-ngram and no-reservoir controls.
- [x] Bounded engram ring; strict maturity and same-asset retrieval masks.
- [x] Recency-weighted top-k cosine retrieval; finite empty-bank behavior.
- [x] Memory keys independent of learned encoder/EMA drift.
- [x] No-memory control with identical neural tensor dimensions.
- [x] Learned context encoder and future-latent predictor.
- [x] EMA target encoder, no target gradients, synchronized reservoir mask.
- [x] Variance-floor and off-diagonal covariance representation regularization.
- [x] No-JEPA control.
- [x] Separate return, downside-probability and lower-tail expert routers.
- [x] Router simplex/cap constraints, balance regularization and uniform-router control.
- [x] Proper Student-t mixture NLL with logsumexp and finite variance.
- [x] Correct within-plus-between mixture variance and stochastic scenario API.
- [x] Brier and pinball auxiliary losses; no-risk control.
- [x] Independent CDF/PIT/coverage calculation from the issued predictive law.

## Training, baselines and operational code

- [x] Chronological complete-date minibatches and lazy context gathering.
- [x] CPU and explicit CUDA paths; unavailable CUDA fails without fallback.
- [x] Optional CUDA AMP; Fourier/recurrent/likelihood arithmetic kept in float32.
- [x] AdamW, cosine schedule, gradient clipping and nonfinite-gradient rejection.
- [x] Validation early stopping and exact best-epoch state restoration.
- [x] Deterministic seeds, reference thread counts and explicit deterministic-algorithm mode.
- [x] Immutable epoch model/optimizer/scheduler/scaler/RNG/preprocessing checkpoints.
- [x] Restricted checkpoint deserialization, content hashes and identity binding.
- [x] Same-contract single-cell epoch resume with retained failed attempts and spent wall time.
- [x] Zero, ridge, GRU and native XGBoost controls; no missing-baseline substitution.
- [x] XGBoost hist/CUDA support, validation early stopping, persisted native model.
- [x] Validation-only XGBoost residual scale and neural/tree distribution blend.
- [x] Retained hybrid validation laws, fixed weight grid and deterministic tie handling.
- [x] Single declared campaign deadline and cooperative per-batch/tree-iteration checks.
- [x] Process RSS, cell wall time, parameter counts and CUDA peak allocation reporting.
- [x] Failures retain a traceback, completed-cell denominator and file inventory.
- [x] Standalone package, CLI, dependency metadata and deterministic release builder.
- [x] Kaggle notebook: exact ZIP verification, isolated wheel install, dependency check,
  target tests, generated qualification, optional attached-data development, readback.

## Data, finance and scientific evaluation

- [x] Strict balanced CSV adapter with timezone-aware observation/availability dates.
- [x] Real source/rights/universe declarations required without fabricated examples.
- [x] Whole-file checksum and protected-prefix filter before numerical value parsing.
- [x] No implicit missing-data fill, future imputation, yield-as-return substitution,
  duplicate assets or current-membership backfill.
- [x] Compounded simple-total-return horizon labels with explicit maturity timestamps.
- [x] Purged chronological train/validation/development-tail partitions and embargo.
- [x] Training-only robust normalization and target scaling.
- [x] Training-only covariance shrinkage and PSD checks.
- [x] Long-only capped allocation with cash remainder and converged proximal solver.
- [x] Exact transaction-cost fixed point, post-return holdings and terminal liquidation.
- [x] Nonoverlapping holdings intervals, terminal losses, cash and equal-weight controls.
- [x] Fixed 1×/2×/4× cost stress ledgers.
- [x] Date-averaged NLL, MAE, mixture Brier, 90% coverage and rank IC.
- [x] Complete-seed-first aggregation and paired circular date-block inference.
- [x] Holm multiplicity adjustment and insufficient-block failure classification.
- [x] Independent law, summary, calendar, holdings, cash-flow and bootstrap reconstruction.
- [x] Exact result inventory verification including extra-file and symlink rejection.
- [x] Development-only runner; no protected-confirmation or trading command.

## Acceptance evidence

- [x] Standard-library configuration, resource, integrity and failure-path tests executed.
- [x] Source/test lint and syntax/format checks are included in the verification receipt.
- [x] Numerical regression tests authored: split leakage, labels, protected filtering,
  contraction, growth, memory maturity/overflow, gradients, EMA, mixture score,
  reference optimizer, fee arithmetic, bootstrap and exact epoch resume.
- [ ] Numerical tests executed on an admitted host.
- [ ] CUDA/AMP tests executed on a compatible NVIDIA target.
- [ ] All generated qualification cells completed and independently reverified.
- [ ] Exact built wheel tested end to end on Kaggle Linux.
- [ ] Actual supplier data/rights/availability/universe audit completed.
- [ ] Declared real-data development matrix completed with every failure retained.
- [ ] Protected protocol frozen and untouched confirmatory study completed.
- [ ] Independent scientific reproduction and any performance claim supported.

## Explicit exclusions and why

PCGrad/Pareto claims, learned utility retention, cross-asset graph memory, latent
world rollouts, online validation-label adaptation, DDP/multi-GPU, conformal
coverage guarantees under arbitrary time dependence, optimal trading policies,
live brokerage, automatic public downloads and global point-in-time universe
reconstruction are not silently asserted as S13 implementations. Several have
historical implementations in earlier packages; adopting them would change this
new mechanism or data contract and requires its own specification and ablation.
No finite checklist certifies all mathematically possible upgrades. This
exclusion list is a scope boundary, not pseudocode or placeholder functionality.

## Concrete residual flaws and required evidence

| Issue | Cause / severity | Repair or acceptance requirement |
| --- | --- | --- |
| Numerical execution unverified | Local RAM admission fails; release blocker | Run authored tests and generated qualification on admitted hardware; fix and version any failures |
| GPU compatibility unmeasured | Local machine has no qualified CUDA receipt; release blocker | Exact Kaggle wheel/runtime/driver test, no CPU substitution |
| Finite-width/growth effectiveness unknown | Larger effective capacity could explain gains; scientific blocker | Full/no-growth/fixed-full comparison with complete seeds and measured compute |
| Auxiliary risk heads differ from return CDF | Multiobjective heads are not one joint calibrated law; medium | Score the predictive mixture independently; calibrate future hypotheses prospectively |
| Student simple-return tails extend below -1 | Parametric approximation violates extreme return support; high for tail use | Do not treat sampled extremes as prices/capital; a bounded-support successor needs a separate test |
| Fixed FIFO memory forgets rare regimes | Bounded storage/recency bias; scientific uncertainty | Report memory-off/control results; do not imply utility-optimal retention |
| Offline training and fixed bank | No continual adaptation in evaluation; scope limitation | Do not claim online learning; a new delayed-update study needs its own protocol |
| Balanced universe can be selected retrospectively | Provider metadata cannot prove survivor-free data; high | Source audit and explicit restricted-universe conclusions |
| Fractional splits are exploratory | They do not establish untouched protection; high | Dataset/date-specific freeze before confirmatory access |
| Common validation used for calibration and stopping | Development model selection can overfit; medium | Evaluate fixed choices on a new untouched protocol; preserve development labels |
| Hybrid has extra compute | It trains both neural and tree models; medium | Report real resource cost; no equal-compute or efficiency claim |
| Notebook packaging is not target execution | Runtime differences remain unknown; release blocker | Retain actual target tests, generated outputs and downloaded independent verification |
