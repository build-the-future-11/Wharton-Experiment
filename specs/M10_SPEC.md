# M10 — Conditional Flow-Matching Market Scenario Model

## Objective
Generate joint multi-asset, multi-step return paths conditioned on market context.

## Flow matching
- Bridge: \(X_\tau = (1-\tau) Z + \tau Y\)
- Target velocity: \(v = Y - Z\)
- Learn linear conditional field \(v_\theta(x, c)\) via ridge regression on bridge samples.
- **Sampling**: Euler ODE integration \(\tau: 0 \to 1\) with `n_ode_steps`.

## Options
- **Student-t** base noise when `student_t_df` is set (heavy tails).

## Metrics
- `energy_score` helper (`wharton_lab.evaluation.metrics` / `m10/energy.py`).

## I/O
- `fit(X, y)` with `y` shaped `(n, n_assets * path_steps)`.
- `sample_paths(X, n_paths)` → `(n_paths, path_steps, n_assets)`.

## Files
- `src/wharton_lab/models/m10/model.py`
- `tests/unit/test_m10.py`
