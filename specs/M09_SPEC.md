# M09 — Regime-Robust Multi-Timescale Representation

## Objective
Learn market representations that remain useful under heterogeneous **causal** training regimes.

## Architecture
- **Fast stream**: short-horizon returns and volatility features (columns 0…`fast_dim-1` of `X`, plus implied vol from return column).
- **Slow stream**: trailing market-only state (rolling mean return + trailing vol). No fabricated macro series.
- **Fusion module**: gated combination of fast/slow hidden states (`FusionModule`).

## Training
- Regime groups from `causal_regime_ids` on train-period returns only.
- **GroupDRO-style** loop: reweight samples by exponentiated group losses (`groupdro_weights`).

## I/O
- `fit(X, y)` → point forecast model.
- `predict(X)` → `float` vector.

## Metrics
- MSE, worst-group MSE on held-out regimes.

## Files
- `src/wharton_lab/models/m09/model.py`
- `tests/unit/test_m09.py`
