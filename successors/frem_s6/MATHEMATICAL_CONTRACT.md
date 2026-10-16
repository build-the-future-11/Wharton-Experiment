# S6 mathematical contracts and executable examples

This document states properties of the actual algorithms in `kernel.py`. The
proofs use real arithmetic; tests check representative float64 behavior, hostile
inputs, date boundaries and serialization. Neither proofs nor tests establish
prediction quality, market causality, novelty, calibrated uncertainty or profit.

## 1. Train-only normalization

`TrainOnlyScaler.fit` accepts nonempty rows with finite numeric time IDs, rejects
any row later than `train_end`, and then refuses a second fit. It computes each
feature mean and population standard deviation from the accepted array only.
Every intermediate axis between the row and feature axes is pooled. A standard
deviation below `scale_floor` becomes one; transformed coordinates are clipped
to `[-clip, clip]`. Division by the largest training magnitude before computing
moments reduces overflow risk without changing the intended moments.

Consequently, changing a held-out value cannot change these statistics when the
caller does not include that value in the training array. This does **not** prove
the supplied historical feature windows were released on time. The data adapter
must enforce release times, fitting target maturity, grouping and fold boundaries.

## 2. Contractive reservoir branches

For branch `b`, let `a_b` be its leak, `W_b` its recurrent matrix, and `U_b` its
input projection. The implemented recursion is

\[
h'_{b}=(1-a_b)h_b+a_b\tanh(W_bh_b+U_b\operatorname{clip}(x)+c_b),
\qquad 0<a_b\leq1,\quad\|W_b\|_2\leq\rho<1.
\]

Initialization multiplies an orthogonal matrix by `rho`. Checkpoint loading
verifies the spectral bound; generated or restored states start inside the cube.
Because `tanh` maps each real coordinate into `[-1,1]`, the update is a convex
combination of two points in that cube. Therefore the cube is invariant. This
includes `a_b=1`; `rho=0` removes recurrent dependence and is valid.

For states `h` and `g` driven by the same input, the coordinate derivative of
`tanh` has absolute value at most one. Thus

\[
\|h'_b-g'_b\|_2
\leq(1-a_b)\|h_b-g_b\|_2+a_b\|W_b(h_b-g_b)\|_2
\leq(1-a_b+a_b\rho)\|h_b-g_b\|_2.
\]

The contraction factor is strictly below one for every configured branch.
This is forgetting of initial state under identical input, not proof that
different financial histories become distinguishable or predictable.

Each input window starts from zero. Its encoded features contain the bounded
last input, final and temporal-average branch states, coordinatewise products
between distinct branches, and random Fourier features of the final states.
All non-Fourier coordinates have magnitude at most one. Each Fourier coordinate
has magnitude at most `sqrt(2/rff_dim)` when `rff_dim > 0`. Setting `rff_dim=0`
removes that feature group. Cross-branch products are readout features and do not
feed into the recurrence, so they do not change its contraction proof.

`context=k` processes only the final `k` observed steps. `context=0` and
`context="zero"` return an exact all-zero vector, including the Fourier group.
This makes the encoder input independent; a downstream head can still use its
training-derived intercept. Full and zero context are not claims of equal
economic information or equal predictive performance.

## 3. Dated engram kernel and bounded pending storage

Each episode records key `k_i`, value `v_i`, origin `o_i`, maturity `m_i`, and
availability `a_i`, with `o_i < m_i <= a_i`. Numeric time IDs use a shared unit.
An episode may be retrieved at time `t` only if

\[
o_i<t,\quad m_i<t,\quad a_i<t.
\]

The strict inequalities deliberately exclude releases at the same timestamp.
Queries advance monotonically. New inserts whose availability precedes the
existing query clock are rejected; they would revise information that should
already have been available to an earlier issued query.

Pending episodes occupy a separate queue capped by `max_pending`; insertion
fails atomically when that queue is full. At a query, newly eligible episodes
join mature storage. The latest `capacity` episode origins are retained, with
availability and insertion order as deterministic tie breakers. A future pending
episode therefore cannot evict an eligible episode or otherwise change a query
before its release. Total episode storage is bounded by
`capacity + max_pending`. This is streaming inference: past queries after clock
advancement require an earlier saved checkpoint and cannot be requested against
the later mutable state.

For retained eligible episodes, the log weight is

\[
\ell_i=-\frac{\|q-k_i\|_2^2}{2\sigma^2}-\lambda(t-a_i),
\quad\sigma>0,\ \lambda\geq0,
\quad w_i=\frac{\exp(\ell_i-\max_j\ell_j)}
{\sum_j\exp(\ell_j-\max_j\ell_j)}.
\]

The output is `sum_i w_i v_i`; unavailable episodes appear in neither sum.
Weights are nonnegative and sum to one, so every value coordinate remains
between the corresponding minimum and maximum eligible value. If `n` episodes
are eligible, the reported effective count is `1/sum_i w_i^2`, which lies in
`[1,n]` by nonnegativity and Cauchy–Schwarz. Identical weights attain `n`; one
concentrated weight attains one. Empty retrieval returns zero output, support
zero and effective count zero, so it is distinguishable from a measured zero.
Extremely large distances that leave no finite log weight fail explicitly.

This is an information-timing property. Similarity weights alone do not identify
causal treatment effects or turn correlation into causation.

## 4. Quantized delayed gradient ripples

For an observed gradient `g_s`, clipping gives

\[
\bar g_s=g_s\min(1,G/\|g_s\|_2),\qquad\|\bar g_s\|_2\leq G,
\]

with the zero vector left unchanged. `ripple_steps=R>=1` normalized weights are
`p_j=d^j/sum_{k=0}^{R-1} d^k`, for `0<=d<=1`. At `d=0`, the first weight is
one and all later weights are zero; at `d=1`, the weights are uniform. Event
`p_j g_s` matures at `s+delay_steps+j`. Thus the total scheduled contribution
from one observation is exactly its clipped gradient in real arithmetic.

On an `apply` with matured events, let `a_t` be their sum, `r_{t-1}` the stored
quantization error, and `q>0` the quantization step. The updater computes

\[
u_t=a_t+r_{t-1},\qquad Q_t=q\,\operatorname{round}(u_t/q),
\qquad r_t=u_t-Q_t.
\]

Nearest rounding implies `|r_t[j]|<=q/2` for every coordinate, including ties
under the implementation's ties-to-even rule. Rearranging and summing gives

\[
\sum_{t=1}^{T}Q_t+r_T=\sum_{t=1}^{T}a_t+r_0.
\]

This is the error-feedback telescoping identity. It concerns matured clipped
gradients and quantization, not the original unclipped gradients. With `q=0`,
the implementation takes `Q_t=u_t` and sets the residual to zero; no tiny proxy
quantization is substituted. `delay_steps=0` supports the immediate control;
with `R=1` and `q=0`, this is ordinary clipped gradient descent.

The actual parameter update is

\[
\theta_t=\theta_{t-1}-\operatorname{clip}_{\|\cdot\|_2\leq B}(\eta Q_t).
\]

Therefore each exact parameter increment has norm at most `B`. Clipping loss is
not added to the quantization residual: doing so would invalidate its `q/2`
bound. Only when this final norm clip is inactive does the telescoping identity
also describe learning-rate-scaled parameter displacements. Floating-point
rounding can introduce additional arithmetic error, especially when parameter
magnitude dwarfs update magnitude.

Operations must be chronological and `apply` may occur at most once per step.
Skipping forward consumes every matured queued contribution. The pending queue
is hard bounded and overflow is rejected without modifying updater state. A
finite training budget may finish with pending updates; a trainer must retain
and report that state rather than silently extending its gradient-step budget.

## 5. Verification scope

`tests/s6/test_kernel.py` contains executable examples for the arguments above,
including train/held-out contamination rejection, zero-context input invariance,
contraction under shared inputs, exact checkpoint feature replay, same-time
release exclusion, the future-eviction attack with capacity one, out-of-order
release handling, quantization telescoping, and immediate unquantized updates.
It also rejects malformed, nonfinite, over-capacity and temporally inconsistent
inputs and checkpoints. Test execution and its pass/fail receipt must be reported
separately; the existence of this source file is not an execution result.
