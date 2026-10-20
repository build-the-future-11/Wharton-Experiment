# S13 prospective model and experiment specification

Authored 19 October 2026 before S13 market access. Status: implementation
candidate; no numerical verification, target qualification or scientific freeze.
Predecessor S12 is unchanged and remains separately governed.

## Question and falsifier

Under a fixed observed panel, does delayed episodic retrieval combined with
bounded reservoir growth improve date-averaged out-of-time predictive log score
over the same JEPA model without memory, without growth, with full fixed
capacity, and over the strongest registered simpler baseline? A central failure
is no practical improvement over memory-off and a stronger simple baseline,
or disappearance of the effect under the capacity control. A future claim
requires a predeclared 0.01-nat margin, multiple complete seeds, dependence-aware
uncertainty and an untouched confirmatory protocol. Generated tests cannot
answer this question. No novelty or positive-result claim is made here.

## Operators and dimensions

Input X has shape B×L×F, scaled using training-only median and MAD. Nonfinite
inputs are rejected by the dataset adapter, and normalized features are clipped
to [-20,20]. Per-channel last value, mean, standard deviation and K nonzero
Fourier energy bands form a fixed key. Energy is `log(1 + mean(|rFFT|²))` with
orthonormal FFT normalization over the trailing window. The last n standardized
return signs, scaled by sqrt(ngram_bonus), add discrete sequence information.
This is a deterministic key of width F(3+K)+n. It uses no learned coordinates,
so training/EMA drift cannot invalidate retained keys.

For each leak α in (0,1], reservoir state obeys

`s[j] = M ((1-α)s[j-1] + α tanh(W s[j-1] + U x[j]))`.

The diagonal mask M activates a prefix of neurons. Fixed W is rescaled by its
largest singular value to ||W||₂=ρ<1. Because tanh and the mask are 1-Lipschitz,
the recurrence contracts by at most (1-α)+αρ<1 for identical input. This is a
state-stability bound, not a generalization or profitability theorem. Every
context window starts at zero. Concatenated bank states and fixed features
feed Linear→LayerNorm→SiLU→Linear to a d-dimensional latent z.

Training-only neurogenesis enables `growth_step` preallocated units every
`growth_every` epochs if the previous training epoch's standardized MSE exceeds
the fixed threshold. Capacity cannot exceed R. Matrix dimensions and optimizer
parameter identities never change; all initial matrices are seeded and fixed.
No validation or test statistic triggers growth. Masks are copied to the EMA
target. The no-growth arm starts at the same initial capacity; the fixed-full
arm starts with R active units, controlling for eventual reservoir capacity.
This is bounded unit recruitment, not a biological-neurogenesis claim.

## Episodic retrieval

The FIFO ring stores at most C fixed keys, standardized realized horizon
returns, asset IDs and maturity timestamps. Keys are L2-normalized. Only records
with maturity strictly earlier than the query decision and the same asset are
eligible. The top-k logits equal cosine similarity/temperature minus
ln(2)×age_days/half_life. Masked softmax produces a mean, standard deviation
and occupancy fraction. Empty retrieval returns zero. Memory storage is O(CD),
and dense query work is O(BCD). Reservoir work is O(BLKR²).

Training batches contain complete dates in chronological order. Before a batch,
only training labels already mature before its earliest decision may enter the
bank. This conservatively excludes within-batch arrivals; batch size is part of
the model contract. The bank is reset each epoch, then completed with training
records before validation. Validation/evaluation labels never enter it. The
fixed trained model and training-only bank are held constant during evaluation.
This is not a claim of prequential online retraining or utility-learned retention.

## JEPA and routed predictive law

The target encoder follows EMA θ_target←βθ_target+(1-β)θ_online. It receives a
trailing window shifted h steps forward, exclusively during training. Its
gradient is disabled. A learned predictor maps current z to the future target
latent. The representation objective combines squared prediction error, a
per-dimension standard-deviation floor and squared off-diagonal covariance.
Variance/covariance regularization is an adaptation of VICReg; it is not a
guarantee against every form of representation collapse.

E MLP experts consume z plus the three retrieval summaries. Each predicts a
location μ (with retrieved mean as residual context), positive softplus scale σ,
loss-event probability and lower-tail quantile. Three separate softmax routers
weight return likelihood, loss probability and tail quantile. If c is the
configured maximum weight, `w = a softmax(logits) + (1-a)/E` with
`a=(c-1/E)/(1-1/E)` enforces w≤c, positivity, and sum(w)=1.

The return law is a mixture of Student-t(ν, μ, σ), ν>2. Exact NLL uses logsumexp.
The standard deviation includes within- and between-component variance. The
training objective is NLL + λ_jepa JEPA + λ_risk(Brier + 0.05 pinball) plus a
mean-router balance penalty. Separate risk heads are auxiliary predictions,
not asserted to equal the mixture CDF/quantile. Evaluation Brier, PIT and 90%
coverage are reconstructed from the actual mixture CDF. Scenario generation
samples mixture components and Student noise; samples are in model-standardized
units and require the saved target scale. A Student approximation to simple
returns has support below -1; do not use unbounded samples as executable prices
or treat its extreme tails as calibrated capital requirements.

## Controls and XGBoost

Registered arms: full, no_memory, no_jepa, no_neurogenesis, no_reservoir,
no_spectral, uniform_router, no_risk, zero, ridge, xgboost, hybrid_xgboost,
fixed_full_capacity, no_ngram, gru. All receive the same chronological samples
and seeds. Ridge and XGBoost see last/mean/std/change summaries from the same
context; they are information-matched, not equal-capacity reproductions.
GRU uses the same latent width and Student head. Actual parameters, wall time,
process lifetime RSS and peak CUDA allocation are recorded. The hybrid consumes
both a neural and tree fit; it has a larger compute budget and earns no
compute-efficiency claim from a better score.

XGBoost uses `hist`, the explicitly requested device, depth 4, η=.05, L2=5,
one thread, fixed seed and validation early stopping. No silent substitute is
used. A Student residual scale is calibrated on validation. The hybrid combines
the full neural and XGBoost laws, selecting neural weight from
{0,.25,.5,.75,1} by validation NLL with a fixed first-grid-point tie rule. It
never fits a residual tree against evaluation labels. Validation outputs and
selection scores are retained. The comparison against XGBoost remains explicit.

## Research portfolio and inference

Training horizon returns estimate covariance, shrunk 10% toward its diagonal
with positive jitter. Allocation minimizes `-μ'w + γ/2 w'Σw + fee||w-w_prev||₁`
subject to 0≤w≤cap and sum(w)≤1; cash is the remainder. A proximal-gradient
solver uses a scalar multiplier for the simplex constraint and fails if the
declared convergence tolerance is not met. This objective uses a first-order
turnover-cost approximation; accounting uses the exact fee equation.

After choosing target w, solve `q + fee Σ|q w - previous| = 1`. Then NAV grows
by `q(1+w'r)`, and holdings drift by `w_i(1+r_i)/(1+w'r)`. Terminal liquidation
is charged. Only nonoverlapping horizon intervals are retained. Cash return is
explicitly zero; no borrowing, slippage, tax, FX or eligibility model is implied.
Every cell runs the declared 1×/2×/4× fee stresses; cash and capped equal-weight
controls are preserved. Forecasting scores use all dates, while accounting uses
the nonoverlapping subset.

For each comparison, average the complete declared algorithmic seeds within
arm, aggregate contemporaneous assets per date, then use paired circular date
blocks. Confidence intervals use 2.5/97.5 percentiles, two-sided p-values use
centered bootstrap deviations with a plus-one correction, and the registered
comparison family uses Holm adjustment. Fewer than two declared blocks gives
INSUFFICIENT_DATES; block length is never silently reduced. Seeds are not
independent market histories. Reported intervals are development evidence and
do not prove a confirmatory claim or independence across overlapping horizons.

## Sources checked 19 October 2026

- VICReg, Bardes et al., submitted 11 May 2021, revised 28 January 2022:
  https://arxiv.org/abs/2105.04906. Used for variance/covariance regularization,
  not to assert finance performance.
- XGBoost official GPU guide, current page retrieved 19 October 2026:
  https://xgboost.readthedocs.io/en/stable/gpu/index.html. `device=cuda` with
  `tree_method=hist`; compatibility depends on the installed build and driver.
- PyTorch official serialization API, retrieved 19 October 2026:
  https://docs.pytorch.org/docs/stable/generated/torch.load.
  Restricted `weights_only=True` loading is used explicitly.
- Gradient Surgery for Multi-Task Learning, Yu et al., 2020:
  https://arxiv.org/abs/2001.06782. Reviewed as an alternative; PCGrad is not
  included because it changes the declared optimization and needs a separate
  controlled comparison. Multiobjective routing here uses an explicit weighted
  loss, not a claim to implement gradient surgery or Pareto optimality.

No existing financial study, profitability benchmark or independent replication
is claimed by these sources. Stronger mathematical guarantees, optimal routing,
universal enhancements, and zero-defect certification are outside the evidence.
