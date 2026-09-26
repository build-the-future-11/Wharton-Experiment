# Lockbox summary

Opened once after freeze. Do not re-open without amendment.

| Dataset | Selected (B_RIDGE) MSE | Persistence MSE | Beats persistence? |
|---|---:|---:|---|
| synthetic seed=997 | 9.89328e-05 | 0.0281491 | True |
| ETF offset=800 | 2.14083e-05 | 2.11771e-05 | False |

- H1 (forecast beats persistence): **SUPPORTED** on synthetic; **UNSUPPORTED** on ETF lockbox (ridge slightly worse).
- H2 (EWMA cov finite): **SUPPORTED** (Frobenius finite on both).
- Selection: B_RIDGE (simplicity tie-break vs M04 within 1% on val folds 0–2).
