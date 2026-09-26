# M05 — Q-APEN (Budgeted Quantized Adaptive Expert Ensemble)

## Overview
Sparse top-k routing over distinct **linear** and **Gaussian** experts with **disagreement-aware** aggregation and **budgeted quantized** expert memory. Expert **create / merge / prune** uses **matured losses only**; immature losses sit in a pending queue.

## Interfaces
- `fit(X, y)` — `X` `(n, d)`, `y` `(n,)`
- `predict(X)` — disagreement-weighted ensemble forecast
- `memory_nbytes()` — packed int8 codebook + zlib residuals (not float rounding)
- `pending_maturity_count()` — losses awaiting maturity

## Lifecycle
1. Each step queues per-expert losses with `mature_at_step = step + delay`.
2. Lifecycle mutations run only after `min_matured_before_lifecycle` matured records exist.
3. **Create** when mean batch loss exceeds threshold and `max_experts` not reached.
4. **Merge** experts with cosine similarity ≥ threshold.
5. **Prune** worst matured-loss fraction when at capacity.

## Quantized memory
Per-expert weights: int8 codebook, uint8 indices, float16 residuals (zlib-compressed). `used_bytes()` sums physical buffer sizes.
