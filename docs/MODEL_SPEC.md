# FREM model and learning specification

**Version:** 0.1, prospective implementation, 10 October 2026. **Evidence status:** no training, model inference, simulations, backtests, or runtime tests were performed during authoring. Source parsing is not empirical validation. This document describes executable code, the intended experiment, and its limitations.

FREM expands to **Causal Fourier–Reservoir Engram Model**. The research question is whether delayed predictive utility can improve a bounded memory of recurring market contexts, under point-in-time information and a fixed resource budget. The combination of Fourier features, a reservoir, temporal JEPA, or boosted trees is not itself a defensible novelty claim. The proposed contribution is a particular causal spectral/engram retention mechanism and its falsification protocol.

## 1. What each proposed ingredient actually means

| Requested idea | Implemented mechanism | Claim boundary |
|---|---|---|
| JEPA investment prediction | A trainable temporal context encoder, a future-window EMA teacher, and a latent prediction auxiliary objective; a separate supervised return head | JEPA supplies a representation-learning objective. It does not make an uninformative series predictable. This is a temporal adaptation, not a reproduction of I-JEPA. |
| Reservoir computing with little context | A fixed, leaky recurrent state with a sufficient Euclidean contraction bound and a trainable readout/fusion network | Fewer trained recurrent weights is a resource advantage to measure. Good few-shot performance is a hypothesis, not a guarantee. |
| Fourier anomaly memory | An ordinary trailing-window real DFT; train-reference spectral surprise; learned frequency-band weights; episodic retention with novelty and delayed utility | The transform remains a DFT. An unusual spectrum is not automatically an anomaly, a fraud event, or an investment opportunity. |
| N-grams / engrams | Five-bin symbols of a specified standardized feature, length-n motifs, fixed retrieval keys, and a bounded store of matured outcomes and forecast residuals | This is not the static hashed language lookup mechanism of the Engram paper. Quantization boundaries and the source feature must be registered. |
| XGBoost | A real boosted-tree baseline and an optional residual predictor fitted on a dedicated later calibration interval | XGBoost is not a graph-search or graph-traversal algorithm. The optional hybrid uses additional calibration information, which must be counted in comparisons. |
| Choosing how to learn | Ordinary backpropagation, a bounded expert-weight update from matured errors, and validation-only schedules for learning rate and JEPA weight | This is an auditable adaptive controller. It does not learn a new differentiation rule, implement general meta-learning, or reproduce continual/constructive backpropagation. |
| Spatial and temporal selection | Prefix-causal input-channel/time gates; optional externally constructed point-in-time peer/market factors | Channel gating alone is not a graph model. Cross-asset context comes from the data layer, which joins only information available at each decision. |
| Stochastic mathematics / physics | A finite-variance Student-t conditional return likelihood with drift/scale parameterization; contraction mathematics for the reservoir | No conservation law is assumed for prices. This version is not a PINN, a solved market SDE, or a causal economic simulator. Stochastic calculus does not automatically reduce computation. |

## 2. Observation and information contract

Each example contains a context $X_{i,t}\in\mathbb{R}^{L\times F}$ for asset $i$ at decision time $t$. The model forecasts the cumulative base-currency log return $Y_{i,t}$ over $H$ **executable return intervals**, with entry strictly after the decision. The data layer supplies an outcome availability time $a_{i,t}$, which can be later than the economic exit time.

Every forecast is a function only of the information available at that decision. A memory record is eligible only if its outcome availability satisfies $a_{j,s}<t$. Equality is deliberately excluded. A quote, publication, or close recorded at the decision timestamp cannot be assumed to have arrived first without explicit event ordering.

The `WindowData` contract is `x[N,L,F]`, `future_x[N,H,F]`, `y[N]`, `time[N]`, `label_time[N]`, `asset_id[N]`, ordered feature/asset names, and optional executable `entry_time`/`exit_time`. Rows are sorted by `(time, asset_id)`. Same-time rows are processed as one group.

`Trainer.predict` accesses `x` to construct predictions. It does not read, preprocess, or forward `future_x`. `y` is copied to a reporting-only `actual` column; in explicitly enabled online replay, it is queued as hidden feedback and used only after maturity. Unknown inference actuals may be `NaN`. Latest unlabeled windows with unknown executable maturity are restricted to frozen prediction.

The robust scaler is fitted on training contexts only. Training medians impute missing values and training IQRs scale features. Missingness masks remain binary when supplied by the data layer. Validation/test data never fit the scaler or spectral reference. Overlapping training contexts repeat some observations in these fitted statistics; this implementation makes no claim that those repetitions are independent samples.

## 3. Fixed reservoir and contraction result

For each supplied window, reset $h_0=0$ and update

\[
h_\ell=(1-\alpha)h_{\ell-1}+\alpha\tanh(W h_{\ell-1}+U x_\ell+b),
\qquad 0<\alpha\le1.
\]

`FixedReservoir` initializes fixed seeded matrices and rescales $W$ using its largest singular value so that $\lVert W\rVert_2=\rho<1$ up to floating-point error. All reservoir matrices are buffers, not trained parameters. Because coordinatewise tanh is 1-Lipschitz, two states driven by the same next input satisfy

\[
\lVert h_\ell-\widetilde h_\ell\rVert_2
\le(1-\alpha+\alpha\rho)
\lVert h_{\ell-1}-\widetilde h_{\ell-1}\rVert_2.
\]

The factor is strictly below one. This establishes fading dependence on the initial state under this sufficient condition. It does **not** establish that market information has been preserved, that long dependencies can be learned from short windows, or that the reservoir will beat a linear model. A spectral-radius-only scaling would not provide this same one-step norm bound; the implementation uses the operator norm deliberately. See Jaeger's corrected technical report [1].

FREM concatenates the final reservoir state with the temporal latent and spectral features. Every train/inference window uses the same reset rule; hidden states are not carried across examples or across assets. `reservoir_ridge` supplies a separate fixed-reservoir control with a ridge readout.

## 4. Trailing spectrum and train-reference surprise

For each input feature, center the observed trailing context, multiply by a Hann window, and compute

\[
\widehat X_k=L^{-1/2}\sum_{\ell=0}^{L-1}
w_\ell(X_\ell-\overline X)e^{-2\pi i k\ell/L}.
\]

Nonoverlapping frequency bands average squared magnitudes and real/imaginary coefficients. The raw descriptor comprises `log1p(band_power)`, mean real coefficients, and mean imaginary coefficients. The model learns a shared per-band gate $g_b=2\sigma(\beta_b)$, bounded between zero and two. This can suppress or emphasize frequencies through ordinary gradients.

Training contexts alone define reference mean $m_{bf}$ and variance $v_{bf}$ of log band power. Spectral surprise is

\[
s_{bf}(X)=\min\left(20,
\frac{|\log(1+P_{bf}(X))-m_{bf}|}{\max(\sqrt{v_{bf}},10^{-3})}\right).
\]

The model receives gated spectral descriptors and gated surprise values. The scalar novelty score is the mean surprise. These reference statistics remain frozen during default and online inference. They are not an expanding test-set fit.

The raw DFT descriptor, not the learned band weights or surprise reference, forms part of the memory key. Consequently, gradients changing the spectral gate do not invalidate old keys. A trailing window may show an unusual frequency pattern simply because volatility, reporting intervals, or liquidity changed. Attribution requires downstream analysis; the code does not infer the cause.

The DFT frequency axis is cycles per observation slot. It is a calendar-time frequency only for equally spaced observations; holidays, missing bars, mixed calendars, and varying reporting cadence must not be interpreted as a regular physical time grid. The data features include elapsed time, but this does not turn the DFT into an irregular-time spectral estimator.

## 5. Temporal encoder, channel selection, and JEPA objective

At each observed position, a feature gate uses the current input and the prefix mean. A width-three time convolution uses left padding only. Neither gate depends on positions beyond its own prefix. Their sigmoid outputs rescale inputs by a factor in $(0,4)$ when multiplied together. A GRU and learned pooling over the fully observed context produce a layer-normalized temporal latent $z_t$.

The EMA teacher has the same temporal architecture and encodes `future_x` **only during fitting**. It is always in evaluation mode, its parameters have `requires_grad=False`, and its output is detached. After an optimizer step,

\[
\bar\theta\leftarrow m\bar\theta+(1-m)\theta.
\]

A predictor maps the context latent to a normalized future latent. The auxiliary objective is normalized latent MSE, supplemented by VICReg-style context variance and off-diagonal covariance penalties [2,3]. For a batch of at least two examples, the code penalizes standard deviations below 0.5 and mean squared off-diagonal covariance. For a singleton batch, empirical variance/covariance penalties are explicitly skipped and the skipped sample count is logged. The trainer accumulates complete chronological time groups until its nominal row batch size is reached, which normally supplies more than one example.

These penalties mitigate a known representational-collapse failure mode. They do not prove that collapse cannot occur or that the representation is useful. Planned diagnostics include latent variance, variance/covariance penalties, context gate entropy, predictive score, and JEPA loss. A low latent prediction error alone is insufficient evidence of forecasting value.

Only the temporal encoder participates in this future-latent objective. The fixed reservoir and spectral branches enter the supervised return predictor through fusion. There is no hidden path from the future teacher output into the current prediction head.

## 6. Fixed-key motifs and bounded episodic memory

The selected standardized feature is bucketed at `[-1, -0.25, 0.25, 1]` into five symbols. The last `memory_ngram` symbols form a motif. Its base-five integer code supports optional exact-match filtering, while its one-hot representation contributes to the retrieval key. With the default data features, index zero is `log_return`; for a custom feature order, explicitly set `ngram_feature_index`.

The complete key concatenates separately normalized blocks:

1. Latest raw standardized feature vector.
2. Trailing mean standardized feature vector.
3. Final fixed-reservoir state, when enabled.
4. Raw spectral descriptor, when enabled.
5. Quantized motif representation, when enabled.

The concatenation is normalized and detached. Learned temporal latents, learned spectral band gates, and the supervised readout do not enter the key. The scaler is fixed after fitting. This removes **key representation drift** under parameter updates. It does not eliminate distribution shift or drift in the stored residual target.

A matured episode stores its key, asset, decision and availability times, motif code, realized return, prediction-time base residual, spectral novelty, utility, and credited-use count. The default retrieval requires the same asset. Cross-asset retrieval is an explicit option, separate from point-in-time cross-asset context features. With an unseen asset and same-asset retrieval, memory is empty and the ordinary predictor remains available.

Eligible records are ranked by cosine similarity, with stable identifier tie breaks. A bounded softmax uses similarity, age decay, and learned utility. Retrieval weighting stays identical across eviction policies so the utility/FIFO/random comparison isolates retention. The read-only retrieval summary has seven fields: residual mean, return mean, residual standard deviation, mean similarity, log neighbor count, mean utility, and normalized age. The neural head uses these summaries to gate an additive residual correction.

For selected memory IDs, delayed credit compares squared error of the base predictor against the memory-adjusted predictor:

\[
\Delta_t=\operatorname{clip}\left(
\frac{(Y_t-\mu_t^{base})^2-(Y_t-\mu_t^{mem})^2}
{\sigma_t^2+(Y_t-\mu_t^{base})^2+10^{-12}},-1,1\right).
\]

Each selected record receives a weighted exponential utility update only after $a_t<t_{next}$. A record evicted before its feedback arrives receives no retroactive credit. This is an explicit bounded credit assignment heuristic, not a causal attribution theorem.

With a full memory, the utility policy evicts the smallest score

\[
\tanh(u_j)+0.1\tanh(s_j)+0.03\log(1+c_j)
-\log(2)\frac{\operatorname{age}_j}{\text{half-life}}.
\]

The optional surprise threshold can reject low-surprise writes while still processing their feedback. Its default is zero, so ordinary episodes are retained as candidates too. A positive threshold requires the spectral branch. Mandatory controls include FIFO and uniform random eviction among current candidates. **Uniform eviction is not uniform sampling of all historical events.** Its purpose is to remove utility-based retention under the same capacity constraint.

Both stored episodes and pending outcomes have explicit capacity bounds. Pending-capacity exhaustion fails clearly rather than silently dropping unknown labels. The clock cannot move backwards, and retrieval behind the current utility clock is rejected. Historical evaluation after online adaptation requires restoring an earlier checkpoint.

During offline fitting, each epoch starts with empty episodic memory and resets the expert feedback logit, avoiding a false claim that repeated epochs represent independent market histories. Stored residuals refer to the predictor that produced them during the training trajectory. Old residuals therefore can become stale when the readout changes, even though their keys remain stable. This limitation must be measured with no-memory and retention controls; it is not hidden by refitting on test outcomes.

## 7. Probabilistic returns and the meaning of scale

The fused context produces drift $d_t$, positive step scale $q_t$, and degrees of freedom $\nu_t$:

\[
\mu_t^{base}=H d_t,\quad
\sigma_t=\sqrt H\operatorname{softplus}(q_t)+\epsilon,
\quad \nu_t=2.1+\operatorname{softplus}(v_t).
\]

The memory gate lies in $[0,1]$ after multiplication by an availability indicator and the controller's expert weight. The final location is the base location plus the gated retrieved residual mean. The stochastic objective is the negative log likelihood of

\[
Y_t\mid\mathcal F_t\sim t_{\nu_t}(\mu_t,\sigma_t).
\]

Here `scale` means **Student-t scale**, not standard deviation. Conditional variance is $\sigma_t^2\nu_t/(\nu_t-2)$. The positive degrees-of-freedom offset makes that variance finite. Scaling by $H$ and $\sqrt H$ is a discrete-time parameterization inspired by drift/diffusion notation; the model does not estimate a continuous-time law or assume that independent increments are literally true. Training can compensate through its context-dependent outputs, and the calibration of that choice must be tested across horizons.

An untruncated Student-t distribution over log return has no finite positive exponential moment. Therefore `expm1(prediction)` is a transform of conditional log-return location, **not** an expected arithmetic return or expected future price. Scenario clipping/truncation, if used for portfolio decisions, changes the distribution and must be recorded. Portfolio claims need empirical net-of-cost evaluation, not an invalid exponential-moment calculation.

With `use_stochastic=False`, the head trains by MSE, and its scale is a constant fitted from training-trajectory residual RMS; `student_df=1e6` represents a practically Gaussian reporting density. This is a fixed-scale ablation, not a guarantee of calibration. The Gaussian/Student-t scale conventions must remain explicit in evaluation.

## 8. Bounded adaptation and optional boosted-tree residuals

The adaptive controller has two different information sources:

- **Matured prediction feedback:** a bounded log-odds update compares the base expert with a full retrieved-residual expert. It changes the memory multiplier only after outcome availability.
- **Tuning validation:** a plateau rule reduces the optimizer learning rate and the JEPA auxiliary weight within registered bounds. It never selects on the protected evaluation interval.

Online mode adapts the first part only. It takes no neural gradient steps and does not refit the scaler or spectral reference. The ordinary stochastic/JEPA losses still use ordinary backpropagation. The terminology “constructive backpropagation” should not be used for this implementation; continual-backpropagation literature uses different mechanisms, including replacement of underused units [4].

`residual_xgboost=True` requires a dedicated chronological calibration set after neural tuning. The residual tree receives current trailing summaries, the frozen neural location, and the neural scale; its target is realized return minus the frozen neural location. It uses the real XGBoost implementation [5], with an explicit installation error if the optional dependency is missing. At inference, its correction is clipped and multiplied by a preregistered fixed weight.

The hybrid has no end-to-end differentiable tree/neural training. Its neural uncertainty scale is retained, so its total predictive calibration is unproven. Memory utility is credited to the neural memory component before the tree correction, avoiding confounding a memory record's credit with a separate residual learner. Report the extra calibration interval, tree fit, clipping, and fixed ensemble weight as part of the model's information/resource budget.

## 9. Training lifecycle and public APIs

```python
from frem.model import ModelConfig
from frem.training import Trainer, TrainingConfig

trainer = Trainer(ModelConfig(input_dim=number_of_features), TrainingConfig())
fit_report = trainer.fit(train_windows, validation_windows)
predictions = trainer.predict(test_windows)  # frozen by default
trainer.save("outputs/run_id/checkpoint")
restored = Trainer.load("outputs/run_id/checkpoint", device="cpu")
```

This is usage documentation, not a command executed during delivery. `context_length` and `horizon` must match the supplied data. The project CLI infers these dimensions from the registered configuration/window builder. `fit(train, validation=None, calibration=None)` permits exactly one split per trainer instance. A failed/partial fit is not silently reused to fit a different split.

The trainer checks that every training label matures before tuning starts, and that tuning outcomes mature before optional residual calibration. Forecast decisions must follow **all** information used to choose the fitted state, including validation labels. When online state has advanced further, its watermark also constrains future frozen calls. These boundaries prevent retrospective forecasts from accidentally using a model selected with later evidence.

Same-time rows are predicted together before feedback is queued. Complete time groups are accumulated for an optimizer step; the configured batch size is a target row count, not permission to split a simultaneous cross-section. Non-finite losses or gradients fail the run. The best completed tuning checkpoint is retained; actual executed gradient steps, completed/attempted epochs, stop reason, parameter counts, latent diagnostics, gate entropy, and memory statistics are reported.

Checkpoint selection and the validation controller use date-balanced negative log likelihood: average Student-t losses within each UTC-date/asset group, average equally over assets present in that UTC date, then average equally across UTC dates. This matches primary evaluation even when markets decide at different times of day or an asset has multiple observations within one date. The optimizer's training objective remains a row mean. History and fit artifacts record the weighting and validation row, UTC-date, and distinct decision-timestamp counts.

Wall-clock limits are checked between time groups, reference-fitting batches, and epochs. They are **soft process limits**: one native estimator call or tensor operation cannot be forcibly interrupted by this Python API. Use an external process timeout for a hard wall limit. When validation was requested, a wall-limited run with no completed validation raises an incomplete-run error; it cannot produce a selected research checkpoint merely because some weights exist.

Checkpoint directories contain tensor-only model/optimizer files, JSON configuration/scaler/controller/memory state, and optional native XGBoost state. RNG states are preserved. Loads use `torch.load(..., weights_only=True)` for tensor files. The optimizer is preserved for audit or an explicitly designed continuation workflow; `fit` does not offer an implicit resume/refit mode. Keep a separate frozen checkpoint before starting any online evaluation.

`observe_outcomes` accepts explicit live feedback columns `time`, `asset_id`, `actual`, and `available_at`. It requires known pending predictions and respects publication order. It does not connect to a market feed, broker, or order API. The latest-window data helper produces unknown execution/maturity and therefore stays in frozen mode until the caller supplies an actual online event schedule.

## 10. Baselines, ablations, and experiment interpretation

`make_baseline(name, seed=0, **kwargs)` returns a fitted/predictable baseline with the same prediction columns. Available names are `zero`, `ridge`, `reservoir_ridge`, `xgboost`, `gru`, `dlinear`, and `patch_transformer`. The latter two are compact architecture controls, not claimed replications of DLinear/PatchTST benchmark results. The patch baseline mixes input channels and uses a small causal token encoder; it is not the original channel-independent PatchTST recipe [6,7].

When enough training dates exist, baseline uncertainty scale uses an internal chronological tail with purged earlier labels and an earlier-only scaler. The point model is then refitted on the full training interval. The two fits are included in its reported budget. With very short training histories, the code labels its conservative training-target scale fallback explicitly. All baselines output `student_df=5` and Student-t scale. Sklearn checkpoints use local joblib artifacts; loading them requires `trust_joblib=True` because pickle deserialization needs explicit trust.

Required mechanism-off comparisons include memory, reservoir, spectral features, motifs, JEPA, feature/time gates, controller, and stochastic scale. Memory-off keeps its unused gate module allocated but passes no memory signal. Consequently, allocated parameter count can match while **active gradient-bearing** parameter count differs. Reservoir/spectral removal also changes the fusion input width. These are scientific mechanism ablations, not automatically parameter- or FLOP-matched controls. Register matched-width/compute controls separately when making an efficiency or capacity claim.

The strongest direct retention comparison is utility versus FIFO versus uniform eviction at the same capacity, key dimensions, neighbor count, input information, training schedule, and seed. The entire unselected model/seed matrix, failures, and budgets must be preserved. Seed repeats do not create additional independent market histories.

Low-data experiments should use predeclared chronological training fractions. Unseen-asset experiments should remove those assets from training **and** tuning; with same-asset memory, this tests fallback to the shared non-memory predictor. Regime and missing-data perturbations should be defined before accessing the protected evaluation. Memory occupancy, retrieval frequency, and spectra are mechanistic diagnostics; they are not performance results.

## 11. Prior art and novelty boundary

Temporal JEPA and reservoir/Fourier combinations already have close precedents. TS-JEPA studies temporal joint embeddings [8]; a more recent paper explicitly examines whether JEPA pretraining benefits forecasting and reports dependence on the backbone/setting [9]. Frequency-domain reservoir work and multiscale random-Fourier reservoir work further narrow the scope for a combination-only novelty claim [10,11]. Engram's static language-oriented memory is relevant motivation but a different memory mechanism [12].

The candidate contribution must survive a direct equal-capacity retention test: does **delayed predictive utility for a causal spectral episodic residual store** add value over simpler retention and the same no-memory predictor? It may fail because motifs do not recur, outcomes are effectively unpredictable, utility estimates are too noisy, the drift in stored residuals dominates, or implementation and transaction costs erase any gain. A negative result with a sound protocol is a valid outcome. The code and prospective paper must not be presented as evidence that the mechanism works.

## 12. Acceptance checks for the later execution session

### Computational scope

With reservoir width $R$, temporal width $U$, key width $D$, memory capacity $C$, and pending bound $P$, a dense reservoir costs $O(L(R^2+RF))$ per context; the temporal GRU costs $O(L(U^2+UF))$, before input-gate costs. The DFT costs $O(FL\log L)$. Retrieval scans the bounded store and sorts candidates, costing $O(CD+C\log C)$, with episode storage $O(CD)$ and pending storage $O(PD)$. These are algorithmic counts, not measured timings.

The current trainer computes fixed reservoir/spectral keys before retrieval and recomputes those branches in `forward`; that factor-of-two specialist work is intentional simple code, not an efficiency achievement. It does not implement Engram's constant-time hashed lookup, an approximate-nearest-neighbor index, or FRESCO's frequency-domain recurrent update. Datasets/windows are materialized in memory, so very large panels need an explicitly designed streaming/window-cache extension. Evaluate actual peak memory, wall time, and equal-budget comparisons before making efficiency claims.

### Authored checks

The included authored tests cover operator-norm contraction; causal gates; spectral reference restrictions; key stability after parameter changes; target-gradient isolation and EMA; each ablation's output shape; singleton regularizer handling; same-time grouping; train/validation outcome purging; frozen forecasts that do not read future targets; frozen memory immutability; online prefix invariance; model/baseline checkpoint round trips; memory maturity, capacity, retention and utility; and controller timing.

These tests were not run during the original authoring session. They now pass within the retained 163-test engineering suite. No admitted-dataset compatibility, calibrated uncertainty, investment return, or empirical performance is claimed from that engineering result.

## Primary references

1. Herbert Jaeger, corrected technical report, *The “echo state” approach to analysing and training recurrent neural networks*. [Author-hosted report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf). The sufficient contraction condition informs this implementation; the proof above is stated for its explicit norm-bounded leaky update.
2. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023. [Primary paper](https://arxiv.org/abs/2301.08243).
3. Bardes, Ponce and LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. [Primary paper](https://arxiv.org/abs/2105.04906).
4. Dohare et al., *Maintaining Plasticity in Deep Continual Learning*. [Primary paper](https://arxiv.org/abs/2306.13812). Consult the actual algorithm before using its name for another update rule.
5. Chen and Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016. [Primary paper](https://arxiv.org/abs/1603.02754).
6. Zeng et al., *Are Transformers Effective for Time Series Forecasting?* [DLinear primary paper](https://arxiv.org/abs/2205.13504).
7. Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. [PatchTST primary paper](https://arxiv.org/abs/2211.14730).
8. *Joint Embeddings Go Temporal*. [TS-JEPA primary preprint](https://arxiv.org/abs/2509.25449).
9. *Does Joint-Embedding Predictive Architecture Pretraining Help Time Series Forecasting?* [Primary preprint](https://arxiv.org/abs/2609.31680).
10. *Frequency Domain Reservoir Computing*. [FRESCO primary preprint](https://arxiv.org/abs/2606.24969).
11. *Reservoir Computing via Multi-Scale Random Fourier Features for Forecasting Fast-Slow Dynamical Systems*. [Primary preprint](https://arxiv.org/abs/2511.14775).
12. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*. [Engram primary paper](https://arxiv.org/abs/2601.07372).
