# Evaluation protocol

`PurgedWalkForwardSplitter` constructs expanding chronological training,
validation and test intervals with explicit embargo gaps. Model selection uses
validation only. The final protected interval is defined once after data
admission and is not exposed during development.

Every model receives identical observations, dates, seeds and tuning budget.
Required comparisons include simple historical/linear, momentum, mean-reversion,
equal-weight, inverse-volatility/minimum-variance, JEPA/reservoir/Fourier/engram/
simulation controls, and the full model. Missing or failed baseline cells remain
failed receipts and invalidate their paired comparison.

Forecast evaluation includes NLL, calibration/coverage, MAE, Brier score and
rank behavior. Portfolio evaluation includes net return, Sharpe, Sortino,
maximum drawdown, CVaR, turnover, costs and funding reliability. Paired moving
date-block intervals preserve temporal dependence. All seeds, low-data regimes,
cost sensitivities and mechanism ablations must be reported.

Experiment cells are bound in a hash-verified immutable manifest. The runner
refuses unfrozen manifests and writes a receipt for every completed or failed
cell. Engineering fixtures may be synthetic but cannot support financial claims.
