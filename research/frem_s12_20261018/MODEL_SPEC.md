# FREM-S12 P0 mathematical model specification

## Inputs and chronology

At forecast origin `t`, the implementation receives only returns
`x_0, ..., x_(t-1)` and proper-score feedback whose outcomes have matured before
`t`. Each `x_t` is a finite six-dimensional monthly return vector. Every
transformation is fitted from the supplied prefix.

## Fourier-reservoir state

For rolling train-only median `m_t` and MAD scale `s_t`, define
`z_t = (x_t - m_t) / max(s_t, 1e-6)`. Fixed seeded random Fourier features are

`phi_t = sqrt(2 / d_phi) cos(W z_t + b)`.

The fixed recurrent reservoir is

`h_t = tanh(rho R h_(t-1) + gamma U phi_t)`,

where `R` is rescaled to the frozen spectral radius. No reservoir parameter is
trained. The state used to forecast `x_t` is computed only through `x_(t-1)`.

## Engram scenario laws

Each matured pair `(h_i, x_(i+1))` can enter a bounded library. P0 issues five
finite joint laws from the latest admitted library:

1. short uniform, last 24 outcomes;
2. medium exponentially weighted, half-life 24 months over 60 outcomes;
3. long uniform, last 120 outcomes;
4. context, 24 nearest robust-scaled reservoir states within 120 outcomes;
5. tail/regime, sign and volatility matched with 10% uniform shrinkage.

Every law retains atoms and probabilities. No marginal recombination may break
the observed joint outcomes.

## Router

The router feature vector concatenates the current reservoir state, exponentially
weighted matured expert regret relative to equal mixing, expert-score dispersion
and current state volatility. A one-hidden-layer tanh network produces logits:

`a_t = tanh(A q_t + c)` and `p_t = softmax(B a_t + d)`.

Weights are constrained differentiably:

`w_t = alpha p_t + (1 - alpha) / K`,

with `alpha = (cap - 1/K)/(1 - 1/K)`, guaranteeing `max(w_t) <= cap`.
P0 uses `cap=0.60`. Before 24 matured feedback rows, weights are equal.

For expert laws `F_i`, outcome `y`,

`ES(sum_i w_i F_i, y) = sum_i w_i A_i(y) - 0.5 sum_(i,j) w_i w_j B_ij`,

where `A_i = E||X_i-y||` and `B_ij = E||X_i-X_j||`. This gives the exact
weight gradient `A - B w`. The router is updated only after `y` matures using
Adam plus frozen entropy concentration penalty. Gradient clipping and finite
checks are mandatory. Router/optimizer/RNG/feedback state is checkpointed.

## Mandatory forecast laws

- routed S12 finite mixture;
- router-off equal expert mixture;
- S11 delayed capped score stacker over the same five experts;
- recent empirical law;
- shrinkage Gaussian finite law;
- shrinkage Student-t finite law;
- ridge VAR(1) residual-bootstrap finite law;
- parameter-matched causal autoregressive MLP residual-bootstrap law.

The MLP uses the last twelve flattened monthly returns, one tanh hidden layer,
online Adam squared-error training after maturity and a joint residual library.
It receives the same information and update opportunities as the candidate.

## Scores and statistics

Energy score is primary. Variogram score with order `p=0.5` and equal unordered
dimension-pair weights is the dependence guard. Both are calculated directly
from finite laws. The paired primary interval is a circular 12-month block
bootstrap with 10,000 draws and seed 1912; 6/24 blocks are sensitivity only.
The fixed matrix aggregates forecast origins equally within fold, folds equally
within family, then families equally.

## Search chronology

The 24 predeclared configurations are scored only on Fold A validation
(1998-1999), using seeds 11 and 29 across the three families. The resulting 144
cells select one configuration by equally aggregated mean energy score, with
trial ID as the deterministic tie break. The source-current selection receipt
is mandatory for every fixed-matrix cell. No later validation or test outcome
can enter selection.

## Failure behavior

Nonfinite inputs, probabilities, states, gradients, parameters or scores fail
the cell. Missing months, identity mismatch, existing outputs, incomplete seed
families and unowned cells fail closed. A valid negative result is terminal for
P0; only a genuine validity defect can create a prospectively corrected freeze.

