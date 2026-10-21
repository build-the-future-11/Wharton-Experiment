# S14 prospective mechanism and evaluation specification

## Architecture and mathematics

1. **Reservoir.** Fixed multiscale leaky tanh states have recurrent spectral norm bounded by rho<1. The state map bound is 1-leak+leak*rho<1, including coordinate projection by the active mask. All input projections are batched across time; memory population skips unnecessary recurrence. This bound concerns state contraction, not financial stability or predictive accuracy.
2. **Neurogenesis.** Tensor capacity is fixed. At registered epochs, training-only normalized prediction MSE triggers recruitment. Inactive neurons are ranked by absolute sample correlation of candidate responses with training residuals. Counts, sums, squares and cross-products are checkpoint buffers. Ties use stable index order. Neither validation nor test labels update this state. No-growth and fixed-full-capacity controls remain.
3. **Engram.** Fixed statistical, Fourier and n-gram coordinates prevent learned-key drift. Vectorized FIFO wrap retains exactly the same records as scalar sequential insertion, including writes larger than capacity. Queries require same asset and strictly earlier outcome maturity. Top-k cosine/age weights undergo one convex personalized random-walk step over the valid selected neighbors; invalid pairs are zeroed and renormalized. No evaluation labels are appended. `no_graph` removes diffusion and `no_memory` removes the full retrieval pathway.
4. **Dimensional stochastic field.** The latent vector is a periodic rows-by-columns lattice (default4x8), in dimensionless coordinates. Let Lr,Lc be positive graph Laplacians and A=Dr*Lr+Dc*Lc+kappa*I, with strictly positive learned coefficients. The conditional model is dZ=(-AZ+f(z0))dt+sigma*dW, where bounded forcing f=tanh(linear(z0)) stays constant during a transition. In the orthonormal Laplacian basis each mode has mean exp(-a*dt)z+(1-exp(-a*dt))/a*f and variance sigma^2*(1-exp(-2a*dt))/(2a). `expm1` avoids cancellation. This is exact for that conditional finite-dimensional linear Gaussian model; it is not a convergence proof for a nonlinear continuum PDE or a physical financial law. Dense diagonalizing transforms cost O(latent^2), reasonable at this configured width. Latent NLL includes every mode and the Gaussian normalization. Stochastic samples use the full correlated covariance. Prediction uses the conditional mean and mean variance, avoiding inference Monte Carlo noise. `no_spde` uses the original MLP transition.
5. **JEPA.** Student forecast loss is combined with latent prediction at offsets floor(h/2) and h (duplicates removed). A random contiguous span in the online context is replaced by a learned feature mask token during training. EMA stop-gradient targets consume shifted trailing contexts at the registered offsets; purging uses maximum offset h. Target contexts overlap past history by design: this is representation prediction, not raw reconstruction of a fully unseen block. Mode-wise Gaussian latent NLL, cosine agreement, variance floors and covariance penalties regularize unmasked/masked online latents. `no_multihorizon`, `no_mask`, `no_jepa` separate the contributions. EMA follows optimizer steps and target buffers match the online fixed dynamics.
6. **Routing and law.** Three capped-simplex gates separately optimize mixture likelihood, downside Brier and tail pinball losses; balance regularization discourages single-expert collapse. Finite-variance Student mixture scales use softplus. Mixture moments include between-component variation. The Student law can assign mass below -100%; it is a scoring model, not a physically supported simple-return simulator. Return scenarios are not used as a solvency guarantee.

## Fixed development family

19 arms: full, no_memory, no_jepa, no_neurogenesis, no_reservoir, no_spectral, uniform_router, no_risk, zero, ridge, xgboost, hybrid_xgboost, fixed_full_capacity, no_ngram, gru, no_spde, no_multihorizon, no_graph, no_mask. Three seeds11/29/47. No automatic search. Hybrid blending is selected on validation only and has explicitly greater compute.

Version 0.1.9 uses the prospectively frozen source-calendar successor cohort SPY/IWM/DIA/TLT/IEF/GLD/SLV, start2007-01-01, with prefixes ending before2011/2015/2018. QQQ is excluded solely because the pre-market 0.1.8 audit proved its exact provider calendar differs by 2011-02-17; no historical model cell ran before this freeze. All seven exact source hashes are allowlisted and their calendars must match exactly, with no intersection selection, row deletion, forward fill, imputation or asset substitution. Nine causal historical features. Targets compound20 following observations; train60%, validation20%, evaluation remaining20%, with20-observation embargo and strict outcome-maturity purging. Scalers and target units are training-only. The suite is171 cells, not171 independent histories. Forecast evaluation outcomes across folds must not overlap, including label maturity. Per-fold2800 seconds, notebook13500 seconds inclusive of tests/qualification. Run limits are ceilings, not promised completion times.

Independent SciPy reconstruction validates Student likelihoods, moments, scores, allocation targets, wealth arithmetic and terminal liquidation. Date-level paired block bootstrap averages declared seeds first, uses20-date blocks and2000 repetitions; all fold/arm comparisons receive an additional family-wide Holm adjustment. The practical NLL difference is0.01 nats. Historical proxy results remain exploratory irrespective of p-values. Missing or failed cells cannot be dropped.

Ledgers use long-only capped allocations, training covariance shrinkage, convex mean-variance/turnover optimization, exact proportional-fee self financing, nonoverlapping holding periods, drift and liquidation, with1x/2x/4x costs and cash/equal-weight controls. Execution at model boundaries is idealized; no spreads, impact, fills, borrow, FX, taxes or observed cash yield are claimed. Adjusted-close proxy ledgers do not establish tradable returns.

## Sources checked 2026-10-19

- I-JEPA motivation (our financial mechanism is an adaptation, not the published image implementation): https://arxiv.org/abs/2301.08243
- VICReg variance/covariance regularization: https://arxiv.org/abs/2105.04906
- GPU XGBoost device configuration: https://xgboost.readthedocs.io/en/stable/gpu/index.html
- Stock dataset card, CC0 assertion and dividend/split-adjusted price description: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/
- SEC APIs and live-update behavior: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

The finite-grid Gaussian transition is derived above and tested against an independent SciPy matrix exponential and Monte Carlo moments. No unverified theorem or general convergence guarantee is asserted.

## 0.1.2 validity repairs, unchanged scientific family

Mixture variance is evaluated as sum_k w_k [s_k^2*nu/(nu-2) + (mu_k-mu_bar)^2],
avoiding cancellation from E[X^2]-E[X]^2. Graph retrieval disables autocast locally and
casts its inputs to float32, so masked logits remain representable with a populated bank.
All three routing probability vectors are checked in independent readback. Fixed-bar labels
are rejected when the feature-availability decision occurs at/after the first target observation.
Read-only strided-window compounding retains the same product target, including terminal loss.

Every declared cell is retained as COMPLETE, FAILED or NOT_STARTED. Outer inventories bind
nested inventories, and suite readback verifies complete seed/fold membership and family-wide
Holm arithmetic. Qualification admission checks source, runtime, hardware, configuration and
complete numerical readback before market access. Resume consumes verified finite prior wall
time; hardware is included in checkpoint binding. None of these repairs changes the scientific
family, hypothesis, practical margin, bootstrap block, seeds or compute ceiling.

Implementation references checked 2026-10-19:
- PyTorch nested autocast-disabled regions: https://docs.pytorch.org/docs/2.14/amp.html
- NumPy read-only sliding windows: https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html


### 0.1.3 prospective evidence verification

Retain an immutable byte-for-byte input CSV and supplier manifest within each market campaign. The suite also retains its source panel and verifies each fold against the exact UTC prefix. Independent readback reconstructs Samples from that retained input (or the declared generated fixture), then compares preprocessing, purged splits, all training-risk labels, test outcomes, calendars and cell bindings. Raw scalar predictions must be vectors without broadcasting. Model ledgers must retain the model policy; inference metadata and the practical margin are verified in full. These checks change acceptance of inconsistent artifacts, not the architecture, 19 arms, three seeds, three folds, numerical thresholds or compute ceilings. Earlier candidates and outcomes remain preserved.
