# M06 — FI-JEPA (Multi-Horizon Financial-Informed JEPA)

## Architecture
- **Context encoder** (GRU + projection)
- **Target encoder** (EMA, stop-grad) updated only from training split
- **Horizon predictors** for latents at horizons **1, 5, 20**
- **Aux heads** for return and volatility
- **VICReg-style** anti-collapse on context latents

## Training constraints
- Hold out trailing `holdout_future_frac` of samples as future split.
- **No EMA update** on target encoder from held-out future batches (predictor loss only).

## I/O
- `X`: `(n, seq_len, input_dim)` or flattened `(n, seq_len * input_dim)`
- `predict` returns aux return head (point forecast proxy)
