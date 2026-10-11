# RE-JEPA: prospective model specification

Document state: **complete pre-experiment specification; no empirical results.**

## Abstract

RE-JEPA is a causal joint-embedding predictive model proposed for probabilistic
international return forecasting when admissible histories are short. A fixed
contractive reservoir and trailing Fourier bank encode context. A stochastic
transition produces multiple latent paths, learned experts interpret their
distribution, and a context-only gate combines the interpretations. A separate
episodic store retrieves immutable Fourier-keyed records only after their
outcomes have matured. Growth and capital-protection heads predict a Student-t
return distribution and downside probability/magnitude. The primary experiment
asks whether the two proposed mechanisms jointly improve date-level NLL over a
matched mechanism-off backbone under predeclared low-data regimes. This paper is
prospective: it specifies the algorithm, objectives, controls, failure rules and
client boundary before any admitted market experiment. Retrieval, online memory
refresh and residual-aware memory calibration are established prior art; the
candidate contribution is the falsifiable coupled mechanism and its
observability-safe financial evaluation, not those ingredients individually.

## 1. Variables and chronology

At decision time `t`, `X_t ∈ R^(L×F)` contains only values with
`availability_time <= t`. `Y_t ∈ R^(H×F)` is the later target window and is used
only in training after it becomes available. `r_t^real ∈ R` is the declared
base-currency total return over the forecast horizon. A sample is indivisible at
the date level: every eligible asset on a date remains in the same split.

No imputation may use a value or membership learned after `t`. Features are fit
and normalized on the training prefix only. Validation follows training and the
protected interval follows validation, with embargo at least the maximum label
horizon.

## 2. Context encoder

The fixed reservoir state evolves as

`h_i = (1-a)h_(i-1) + a tanh(W_in x_i + W_rec h_(i-1))`,

where `||W_rec||_2 = ρ < 1` and `a ∈ (0,1]`. Both matrices are sampled from the
declared initialization seed and stored as non-trainable buffers.

For every feature, a real FFT is computed on the trailing observed window only.
The non-DC power spectrum is partitioned into `B` contiguous frequency bands;
log mean energy in each band forms `s_t ∈ R^(FB)`. The context encoder is

`z_t = E_θ([x_t, h_t, s_t]) ∈ R^D`.

The default configuration is `reservoir_dim=32`, `latent_dim=24`, `B=4`,
`ρ=0.85`, and `a=0.35`.

## 3. EMA target and JEPA objective

The target latent is `z*_t = E_ξ(Y_t)`, with stop-gradient. Target parameters
follow `ξ <- τξ + (1-τ)θ`, `τ=0.99`, after every optimizer step. At inference,
`Y_t` and `E_ξ` are absent. The latent term is smooth L1 between the fused
prediction and `z*_t`. A batch standard-deviation floor penalizes collapsed
context representations.

## 4. Latent playground

From `z_t`, the shared stochastic transition produces `K=8` paths of length
`Q=5`:

`u_(k,q+1) = LN(u_(k,q) + μ_φ(u_(k,q)) + softplus(σ_φ(u_(k,q))) ε_(k,q))`,

where `ε ~ N(0,I)` and scale is capped at 1 for numerical stability. The path
summary concatenates the all-step mean, endpoint mean and endpoint standard
deviation. Each of `M=3` experts maps that summary to an interpretation `v_m`.
A gate `g=softmax(G z_t)`, which never sees the future outcome, yields
`v_t = Σ_m g_m v_m`. A pairwise endpoint-distance floor discourages identical
worlds. These paths are latent Monte Carlo samples, not asserted economic
scenarios.

## 5. Delayed-credit spectral engrams

The query is `q_t = P([z_t,s_t])`. Each stored record is

`e_j=(id_j,k_j,v_j,d_j,o_j,c_j,u_j)`

for key, value, decision time, outcome-available time, recorded time and finite
utility. It is readable at `t` only if `o_j <= t`; creation requires `d_j < o_j
<= c_j`. Keys and values are detached immutable tensors. Retrieval selects the
eight largest cosine-similarity-plus-log-utility scores and returns their
softmax-weighted values. Gradients reach the current query and downstream fusion
but never rewrite a stored episode.

## 6. Fusion and prediction heads

`p_t = F_ψ([z_t,v_t,e_t])`.

The growth head emits location `μ_t` and scale
`b_t=softplus(b'_t)+10^-4` for a Student-t distribution with fixed `ν=5`. The
protection head emits `π_t=sigmoid(l_t)`, the probability of a return below the
declared downside threshold, `m_t=softplus(m'_t)` for downside magnitude,
`v_t=softplus(v'_t)+10^-4` for realized-volatility supervision, and
`d_t=softplus(d'_t)` for drawdown exposure. The joint-head control emits all six
scalar outputs from one final head. The reported lower tail quantile is
`μ_t + b_t F^-1_tν(0.05)`; a Normal critical value is not used for a Student-t
forecast.

The training objective is

`L = λ_z L_smoothL1(p,z*) + λ_g NLL_tν(r;μ,b)
   + λ_p [BCE(π,1[r<c]) + smoothL1(m,max(c-r,0))
          + smoothL1(v,|r|) + smoothL1(d,max(-r,0))]
   + L_diversity + L_collapse`,

with default `λ_z=λ_g=λ_p=1`. All choices are tuned on development dates only.

## 7. Training algorithm

1. Validate finite, shape-correct, chronology-admitted batches.
2. Seed Python, NumPy, Torch and every latent-world generator.
3. For each chronological development batch, pass its declared forecast horizon
   into the horizon embedding and run the forward pass using only engrams
   matured by its `as_of` time.
4. Backpropagate the composite loss, fail on non-finite values, clip global
   gradient norm at 1, take AdamW step, then update the EMA target.
5. Evaluate validation batches without gradients. Aggregate loss by sample count
   rather than equally weighting unequal batches. Validation selects the epoch
   only; it never supplies a gradient or creates an engram.
6. Restore the lowest validation-loss state and serialize model configuration,
   weights, metadata, optional memory and SHA-256 manifest atomically.

Default optimizer values are learning rate `3e-4`, weight decay `1e-4`, maximum
100 epochs, patience 15 and minimum improvement `1e-5`.

## 8. Frozen inference algorithm

Load and hash-verify the checkpoint. Set evaluation and inference-only modes.
Use the frozen scenario seed assigned to the cell, retrieve only mature records,
and emit the two growth-head scalars, four protection-head scalars, derived
lower-tail quantile, and diagnostic worlds/gates/record IDs.
The implementation snapshots every state tensor and raises if inference changes
one. There is no test-time fitting, engram creation, threshold selection or
choice among seeds.

## 9. Computational complexity

For batch size `N`, context length `L`, reservoir width `R`, feature count `F`,
latent width `D`, worlds `K`, world horizon `Q`, interpreters `M`, memory size
`J`, and retrieved neighbors `k`, the dominant forward costs are approximately
`O(NL(RF+R²))` for the dense reservoir, `O(NFL log L)` for FFT features,
`O(NKQD²)` for worlds, `O(NMD²)` for interpreters and `O(NJD + NkD)` for exact
memory search. Memory is `O(JD + NKQD)`. The reservoir implementation is dense
and intentionally small; an optimized sparse form is future work.

## 10. Numerical and safety constraints

- Reservoir spectral norm is below one and verified after construction.
- Student-t degrees of freedom exceed two; scale has a `10^-4` floor.
- Transition scale is softplus-positive and capped at one.
- Every external tensor and loss must be finite; non-finite gradients fail.
- Checkpoints fail closed on any hash mismatch or missing file.
- Engram exports reject duplicate identifiers, invalid maturity, excess capacity
  and non-finite values.
- The protected checkpoint is read-only and its state hash is checked before and
  after inference.

## 11. Allocation and Laura Gao boundary

The model produces forecasts; it does not issue trades. A separate constrained
research allocator may combine expected return and downside estimates subject
to point-in-time eligibility, country/sector/currency/liquidity limits,
transaction costs and a reserve sleeve. The client layer uses the case cash
flows—$300,000 at the beginning of 2027, $150,000 at the beginning of 2028, no
other cash flow before 2033, then ten $50,000 beginning-of-year payments in
2033–2042—and establishes an operating reserve before the first 2033 payment.
WInS gains/losses are excluded from the long-term projection. The model does not
estimate facility cost, tax or Taiwan legal requirements.

The scenario allocator uses the empirical Rockafellar--Uryasev formulation. For
scenario losses `ell_s(w)=-R_s w`, it introduces `eta` and nonnegative `xi_s`
with `xi_s >= ell_s(w)-eta` and minimizes
`-mu'w + gamma[eta + sum_s xi_s/((1-alpha)S)]` plus linear turnover and
uncertainty penalties. Cash, security, sector, country, currency and maximum-loss
proxy limits are linear constraints solved by HiGHS. Every returned solution is
checked again against the declared limits.

## 12. Contemporary retrieval controls and novelty boundary

RAF and RAFT already establish retrieval-augmented time-series forecasting.
FreshCast further establishes that a streaming memory and validation-calibrated
residual alignment can be more important than a static bank. Consequently,
RE-JEPA must compare against causal RAFT-style retrieval and a FreshCast-style
fresh residual correction before attributing benefit to matured utility
retention. A win over no-memory alone cannot support a memory novelty claim.

## 13. Engineering dataset and run boundary

The repository includes a deterministic two-regime, multi-asset generator with
heterogeneous loadings, seasonality, synthetic FX and explicit decision/outcome
times. It exists only to exercise data validation, purged chronological splitting,
training, checkpoint hashing and frozen inference. Its mandatory label is
`SYNTHETIC_ENGINEERING_FIXTURE_NOT_MARKET_EVIDENCE`; its scores cannot support a
forecast, finance, novelty or competition claim.

## 14. Falsifiability and prohibited claims

The primary threshold, controls and decision rule are in
`research/protocol.json`. Passing unit tests proves implementation consistency,
not forecast skill. Until an admitted panel is frozen and the complete protected
matrix succeeds, claims of novelty, calibration, low-data advantage, portfolio
outperformance and competition readiness remain prohibited.
