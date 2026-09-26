# M01 — Regime-Conditioned Distributional Gradient Boosting

## Objective

For features \(x_t \in \mathbb{R}^d\) and target \(y_t\), estimate conditional quantiles
\(\hat{q}_\tau(x_t)\) for \(\tau \in \{0.05, 0.25, 0.50, 0.75, 0.95\}\).

## Regime (causal)

Past return series \(r_s\) for \(s < t\) defines trailing volatility
\(\sigma_t = \mathrm{Std}(r_{t-W:t-1})\) and lagged sign \(s_t = \mathrm{sign}(r_{t-1})\).

Hard regime:
\[
g_t \in \{0,1,2,3\} \text{ from } (\sigma_t \lessgtr \sigma^\*) \times (s_t \lessgtr 0).
\]

Soft regime: logistic weights \(w_t \in \Delta^4\) over the same partition.

**Binding:** no future smoothing; \(\sigma_t\) uses only \(r_{<t}\).

## Estimator

Per quantile \(\tau\), gradient boosting with loss `quantile` (HistGradientBoostingRegressor when available).

Optional regime-specific models \(\hat{q}_{\tau}^{(g)}\) mixed:
\[
\hat{q}_\tau(x_t) = \sum_g \pi_t(g)\, \hat{q}_\tau^{(g)}(x_t).
\]

## Non-crossing

Post-hoc per row: sort \(\hat{q}_{\tau_1} \le \cdots \le \hat{q}_{\tau_K}\) (documented alternative: isotonic on \((\tau, \hat{q})\)).

## Metrics

Average pinball loss:
\[
\mathcal{L} = \frac{1}{K}\sum_k \mathbb{E}\left[ \rho_{\tau_k}(y - \hat{q}_{\tau_k}) \right],
\quad \rho_\tau(u) = u(\tau - \mathbb{1}_{u<0}).
\]

## Ablations

| Flag | Effect |
|------|--------|
| `use_regimes` | Regime-mixed vs global GBM only |
| `soft_vs_hard` | Soft weights vs hard \(g_t\) |
| `distributional_vs_point` | Full quantile matrix vs median only |
| `calibrate` | Add median residual offset per \(\tau\) |
