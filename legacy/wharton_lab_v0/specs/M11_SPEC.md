# M11 — Liability-Conditioned Distributional MPC

## Objective
Receding-horizon portfolio allocation minimizing **CVaR funding shortfall** under scenario returns.

## State & inputs
- Holdings, cash, liquidity via `PortfolioState`.
- **Liability schedule** (required): per-step obligations. Missing schedule → `BlockedClientError` (no invented client wealth).
- Scenario tensor `X`: `(n_scenarios, horizon, n_assets)`.

## Constraints
- Long-only weights, cap `max_weight`, fully invested per step.
- **Non-anticipativity**: one shared weight vector per time step (first action applies to all scenarios).

## Solver
- `scipy.optimize.minimize` SLSQP (QP-lite fallback behavior via smooth objective).

## Failure modes
- Infeasible optimization → defensive all-in-first-asset weights.
- Insolvent cash+holdings → `BlockedClientError`.

## Files
- `src/wharton_lab/models/m11/model.py`
- `tests/unit/test_m11.py`
