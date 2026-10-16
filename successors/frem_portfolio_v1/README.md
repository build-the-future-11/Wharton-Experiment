# FREM portfolio evidence layer

An isolated standard-library package connecting retained FREM/S3/S4 forecasts to
one common accounting system. It does not train a new model, change a completed
experiment, access a protected tail, or place orders. The dated specification
precedes the portfolio results. The user's target is **6%–36% higher ending
wealth**, not an annual return promise.

The saved research policy uses an 80% equal-weight anchor plus up to 20% positive
risk-adjusted log-return signal. It is deliberately fixed before the reanalysis;
the result is not an optimized or reviewed client allocation. No leverage is
allowed. The same-basket equal-weight, buy-and-hold and cash controls share entry
costs and terminal liquidation. Student-t log variance is computed from scale and
degrees of freedom; it is not exponentiated into a nonexistent moment-generating
function. Seed ensembling includes predictive and between-seed uncertainty.

Self-financing accounting solves `A + c Σ|w_i A − h_i| = NAV` for post-fee wealth
`A`. Positions drift with realized returns. Missing quotes fail, full losses stay
in the ledger, and separate local-currency markets are never combined. The source
forecasts have conservative blind-month research timing, not verified fills.

Run from the repository root, always choosing a fresh output directory:

```bash
rtk proxy .venv/bin/python -m unittest discover -s successors/frem_portfolio_v1/tests -v
rtk proxy .venv/bin/python -m successors.frem_portfolio_v1.run --output outputs/frem_portfolio_NEW --execute
rtk proxy .venv/bin/python -m successors.frem_portfolio_v1.audit --run outputs/frem_portfolio_NEW --output research/frem_portfolio_audit_NEW.json
```

`engine.external_forecasts` accepts a future quant/ML framework's explicit JSON
payload: `source_sha256`, timezone-qualified `trained_through`,
`max_input_available_at`, `generated_at`, `currency`, `horizon`, and an exact
`forecasts` list containing `asset`, `location`, Student-t `scale` and `df`.
It rejects future inputs, mismatched horizons/currencies and incomplete assets.
The parser validates the contract, not the truthfulness of source attestations.
No external framework source has yet been uniquely identified or integrated.

`engine.laura_path` respects the actual two contributions and beginning-year
operating payments. Unknown funding confidence and flexibility remain unknown.
Scenario rates are assumptions, not yields or calibrated probabilities.

The 12-stock USD S4 sample is the primary portfolio comparison; older two-stock
local-market baskets are supplemental. All earlier public observations remain
development evidence. A flat-return, zero-cost path refutes an across-all-futures
positive floor. Neither passing mechanics nor isolated historical wins establishes
profitability or a competition-approved recommendation.
