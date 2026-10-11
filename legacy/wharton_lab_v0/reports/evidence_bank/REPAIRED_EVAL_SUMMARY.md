# Repaired-pipeline exploratory evaluation

**Generation:** repaired pipeline, exploratory, post-lockbox. **Not confirmatory**; the lockbox is spent and is not reopened.

- Label: 1-step daily return of asset 0; causal features lag1, mean_lag5, vol_lag20, xs_mean_lag1 (rows < t only).
- Folds: expanding window, 8 × 100 disjoint test rows, min train 300.
- ETF: `data/raw/etf/SPY_AGG_GLD_VNQ_EFA_returns.npy` (1691 rows, sha256 `6c234590d0b4c28901c9919044141e0617ae73b592b37bd510728386e26cfd29`); test folds overlapping lockbox rows [800, 960] dropped.
- Synthetic: SIGNAL world, seeds [11, 23, 47, 89, 131] (lockbox seed excluded), 1200 steps.
- Primary contrast (pre-declared): `ridge_std` vs `mean`.
- Git HEAD at run: `2b146663bd2efca373d3b5f93ee3f2a4018ff8ed`.

OOS R² = 1 − MSE(challenger)/MSE(benchmark); positive = challenger better. DM = Diebold–Mariano (Newey–West); q = Benjamini–Hochberg over the challenger family.

## etf_track_a
Excluded folds (lockbox overlap): [{'fold': 7, 'rows': [891, 990]}]

| challenger | benchmark | n | MSE chall. | MSE bench. | OOS R² | DM p | BH q | fold Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_legacy | zero | 700 | 9.5005e-06 | 9.5026e-06 | +0.0002 | 0.592 | 0.748 | 0.375 |
| ridge_legacy | mean | 700 | 9.5005e-06 | 9.5173e-06 | +0.0018 | 0.127 | 0.234 | 0.297 |
| ridge_legacy | persistence_1step | 700 | 9.5005e-06 | 1.8338e-05 | +0.4819 | 7.19e-24 | 2.16e-23 | 0.0156 |
| ridge_legacy | static_last | 700 | 9.5005e-06 | 2.0090e-05 | +0.5271 | 4.02e-17 | 1.13e-16 | 0.0469 |
| ridge_std | zero | 700 | 9.4967e-06 | 9.5026e-06 | +0.0006 | 0.925 | 0.925 | 0.938 |
| ridge_std | mean | 700 | 9.4967e-06 | 9.5173e-06 | +0.0022 | 0.742 | 0.81 | 0.812 |
| ridge_std | persistence_1step | 700 | 9.4967e-06 | 1.8338e-05 | +0.4821 | 4.89e-24 | 1.57e-23 | 0.0156 |
| ridge_std | static_last | 700 | 9.4967e-06 | 2.0090e-05 | +0.5273 | 8.18e-17 | 2.18e-16 | 0.0469 |

## synthetic_signal_s11

| challenger | benchmark | n | MSE chall. | MSE bench. | OOS R² | DM p | BH q | fold Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_legacy | zero | 800 | 2.3740e-02 | 2.3632e-02 | -0.0046 | 0.217 | 0.342 | 0.25 |
| ridge_legacy | mean | 800 | 2.3740e-02 | 2.3663e-02 | -0.0032 | 0.317 | 0.461 | 0.547 |
| ridge_legacy | persistence_1step | 800 | 2.3740e-02 | 4.7275e-02 | +0.4978 | 4.87e-32 | 2.6e-31 | 0.00781 |
| ridge_legacy | static_last | 800 | 2.3740e-02 | 5.8143e-02 | +0.5917 | 2.74e-13 | 6.16e-13 | 0.00781 |
| ridge_std | zero | 800 | 2.3811e-02 | 2.3632e-02 | -0.0076 | 0.163 | 0.279 | 0.195 |
| ridge_std | mean | 800 | 2.3811e-02 | 2.3663e-02 | -0.0062 | 0.221 | 0.342 | 0.461 |
| ridge_std | persistence_1step | 800 | 2.3811e-02 | 4.7275e-02 | +0.4963 | 1.18e-31 | 5.67e-31 | 0.00781 |
| ridge_std | static_last | 800 | 2.3811e-02 | 5.8143e-02 | +0.5905 | 2.82e-13 | 6.16e-13 | 0.00781 |
| ridge_legacy_on_leaky_features | zero | 800 | 1.0269e-04 | 2.3632e-02 | +0.9957 | 7.83e-83 | — | 0.00781 |
| ridge_legacy_on_leaky_features | mean | 800 | 1.0269e-04 | 2.3663e-02 | +0.9957 | 8.56e-83 | — | 0.00781 |
| ridge_legacy_on_leaky_features | persistence_1step | 800 | 1.0269e-04 | 4.7275e-02 | +0.9978 | 1.02e-65 | — | 0.00781 |
| ridge_legacy_on_leaky_features | static_last | 800 | 1.0269e-04 | 5.8143e-02 | +0.9982 | 4.72e-32 | — | 0.00781 |

## synthetic_signal_s23

| challenger | benchmark | n | MSE chall. | MSE bench. | OOS R² | DM p | BH q | fold Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_legacy | zero | 800 | 2.2443e-02 | 2.2515e-02 | +0.0032 | 0.577 | 0.748 | 0.641 |
| ridge_legacy | mean | 800 | 2.2443e-02 | 2.2501e-02 | +0.0026 | 0.642 | 0.753 | 0.641 |
| ridge_legacy | persistence_1step | 800 | 2.2443e-02 | 4.1986e-02 | +0.4655 | 1.84e-35 | 1.47e-34 | 0.00781 |
| ridge_legacy | static_last | 800 | 2.2443e-02 | 5.4967e-02 | +0.5917 | 1.92e-31 | 8.39e-31 | 0.00781 |
| ridge_std | zero | 800 | 2.2470e-02 | 2.2515e-02 | +0.0020 | 0.787 | 0.839 | 1 |
| ridge_std | mean | 800 | 2.2470e-02 | 2.2501e-02 | +0.0014 | 0.848 | 0.884 | 0.945 |
| ridge_std | persistence_1step | 800 | 2.2470e-02 | 4.1986e-02 | +0.4648 | 1.69e-35 | 1.47e-34 | 0.00781 |
| ridge_std | static_last | 800 | 2.2470e-02 | 5.4967e-02 | +0.5912 | 2.61e-31 | 1.04e-30 | 0.00781 |
| ridge_legacy_on_leaky_features | zero | 800 | 9.8719e-05 | 2.2515e-02 | +0.9956 | 2.35e-89 | — | 0.00781 |
| ridge_legacy_on_leaky_features | mean | 800 | 9.8719e-05 | 2.2501e-02 | +0.9956 | 9.36e-90 | — | 0.00781 |
| ridge_legacy_on_leaky_features | persistence_1step | 800 | 9.8719e-05 | 4.1986e-02 | +0.9976 | 6.33e-70 | — | 0.00781 |
| ridge_legacy_on_leaky_features | static_last | 800 | 9.8719e-05 | 5.4967e-02 | +0.9982 | 2.86e-72 | — | 0.00781 |

## synthetic_signal_s47

| challenger | benchmark | n | MSE chall. | MSE bench. | OOS R² | DM p | BH q | fold Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_legacy | zero | 800 | 2.0720e-02 | 2.0682e-02 | -0.0018 | 0.42 | 0.56 | 0.312 |
| ridge_legacy | mean | 800 | 2.0720e-02 | 2.0686e-02 | -0.0017 | 0.256 | 0.384 | 0.312 |
| ridge_legacy | persistence_1step | 800 | 2.0720e-02 | 4.1824e-02 | +0.5046 | 5e-33 | 3.43e-32 | 0.00781 |
| ridge_legacy | static_last | 800 | 2.0720e-02 | 4.6561e-02 | +0.5550 | 1.4e-14 | 3.35e-14 | 0.0234 |
| ridge_std | zero | 800 | 2.0791e-02 | 2.0682e-02 | -0.0053 | 0.153 | 0.271 | 0.109 |
| ridge_std | mean | 800 | 2.0791e-02 | 2.0686e-02 | -0.0051 | 0.117 | 0.224 | 0.195 |
| ridge_std | persistence_1step | 800 | 2.0791e-02 | 4.1824e-02 | +0.5029 | 1.08e-32 | 6.49e-32 | 0.00781 |
| ridge_std | static_last | 800 | 2.0791e-02 | 4.6561e-02 | +0.5535 | 1.39e-14 | 3.35e-14 | 0.0391 |
| ridge_legacy_on_leaky_features | zero | 800 | 1.0500e-04 | 2.0682e-02 | +0.9949 | 1.43e-84 | — | 0.00781 |
| ridge_legacy_on_leaky_features | mean | 800 | 1.0500e-04 | 2.0686e-02 | +0.9949 | 7.88e-84 | — | 0.00781 |
| ridge_legacy_on_leaky_features | persistence_1step | 800 | 1.0500e-04 | 4.1824e-02 | +0.9975 | 7.41e-63 | — | 0.00781 |
| ridge_legacy_on_leaky_features | static_last | 800 | 1.0500e-04 | 4.6561e-02 | +0.9977 | 1.54e-38 | — | 0.00781 |

## synthetic_signal_s89

| challenger | benchmark | n | MSE chall. | MSE bench. | OOS R² | DM p | BH q | fold Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_legacy | zero | 800 | 2.3101e-02 | 2.3025e-02 | -0.0033 | 0.403 | 0.557 | 0.844 |
| ridge_legacy | mean | 800 | 2.3101e-02 | 2.3065e-02 | -0.0016 | 0.685 | 0.765 | 0.945 |
| ridge_legacy | persistence_1step | 800 | 2.3101e-02 | 4.5999e-02 | +0.4978 | 5.42e-26 | 2e-25 | 0.00781 |
| ridge_legacy | static_last | 800 | 2.3101e-02 | 6.8572e-02 | +0.6631 | 3.62e-36 | 5.78e-35 | 0.00781 |
| ridge_std | zero | 800 | 2.3117e-02 | 2.3025e-02 | -0.0040 | 0.406 | 0.557 | 1 |
| ridge_std | mean | 800 | 2.3117e-02 | 2.3065e-02 | -0.0023 | 0.621 | 0.753 | 1 |
| ridge_std | persistence_1step | 800 | 2.3117e-02 | 4.5999e-02 | +0.4974 | 5.93e-26 | 2.03e-25 | 0.00781 |
| ridge_std | static_last | 800 | 2.3117e-02 | 6.8572e-02 | +0.6629 | 5.19e-36 | 6.23e-35 | 0.00781 |
| ridge_legacy_on_leaky_features | zero | 800 | 9.2379e-05 | 2.3025e-02 | +0.9960 | 7e-82 | — | 0.00781 |
| ridge_legacy_on_leaky_features | mean | 800 | 9.2379e-05 | 2.3065e-02 | +0.9960 | 4.61e-82 | — | 0.00781 |
| ridge_legacy_on_leaky_features | persistence_1step | 800 | 9.2379e-05 | 4.5999e-02 | +0.9980 | 6.02e-50 | — | 0.00781 |
| ridge_legacy_on_leaky_features | static_last | 800 | 9.2379e-05 | 6.8572e-02 | +0.9987 | 5.36e-73 | — | 0.00781 |

## synthetic_signal_s131

| challenger | benchmark | n | MSE chall. | MSE bench. | OOS R² | DM p | BH q | fold Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_legacy | zero | 800 | 2.1474e-02 | 2.1512e-02 | +0.0018 | 0.643 | 0.753 | 0.742 |
| ridge_legacy | mean | 800 | 2.1474e-02 | 2.1565e-02 | +0.0042 | 0.21 | 0.342 | 0.312 |
| ridge_legacy | persistence_1step | 800 | 2.1474e-02 | 4.2977e-02 | +0.5003 | 1.9e-40 | 9.11e-39 | 0.00781 |
| ridge_legacy | static_last | 800 | 2.1474e-02 | 3.4286e-02 | +0.3737 | 1.08e-11 | 2.25e-11 | 0.00781 |
| ridge_std | zero | 800 | 2.1525e-02 | 2.1512e-02 | -0.0006 | 0.904 | 0.923 | 0.742 |
| ridge_std | mean | 800 | 2.1525e-02 | 2.1565e-02 | +0.0019 | 0.681 | 0.765 | 0.844 |
| ridge_std | persistence_1step | 800 | 2.1525e-02 | 4.2977e-02 | +0.4992 | 5.22e-40 | 1.25e-38 | 0.00781 |
| ridge_std | static_last | 800 | 2.1525e-02 | 3.4286e-02 | +0.3722 | 1.56e-11 | 3.13e-11 | 0.00781 |
| ridge_legacy_on_leaky_features | zero | 800 | 1.0211e-04 | 2.1512e-02 | +0.9953 | 9.21e-74 | — | 0.00781 |
| ridge_legacy_on_leaky_features | mean | 800 | 1.0211e-04 | 2.1565e-02 | +0.9953 | 3.99e-74 | — | 0.00781 |
| ridge_legacy_on_leaky_features | persistence_1step | 800 | 1.0211e-04 | 4.2977e-02 | +0.9976 | 1.37e-73 | — | 0.00781 |
| ridge_legacy_on_leaky_features | static_last | 800 | 1.0211e-04 | 3.4286e-02 | +0.9970 | 7.39e-57 | — | 0.00781 |

`ridge_legacy_on_leaky_features` re-fits the frozen B_RIDGE on the legacy Track C features `[latent_t, latent_{t-1}]` over the same test rows. It is a **leakage demonstration**, not a forecast, and is excluded from the BH family.
