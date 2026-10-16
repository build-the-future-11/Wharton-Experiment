# Closed S8 descriptive diagnosis

Descriptive reconstruction of closed development evidence; zero fits, zero policy replays, no selection, no protected access.

Verified all 252 ledgers; maximum dollar accounting residual 1.35e-09.

The primary's gross P&L at its saved 10-bps positions was $-444.48; fees including liquidation were $810.24; net P&L was $-1,254.73. Rebate those fees to cash while fixing the dollar positions and ending wealth is $99,555.52.

The separately chosen zero-cost positions end at $99,990.33. The zero-minus-charged wealth gap is $1,245.06: $810.24 direct fees plus $434.82 gross P&L difference between the saved position paths. The latter includes endogenous allocation and sizing effects.

| Asset | Primary gross P&L |
|---|---:|
| SPY | $6,988.60 |
| TLT | $-5,511.30 |
| GLD | $-1,921.78 |

| Year | S8 return | Equal-weight return | S8 risky exposure |
|---|---:|---:|---:|
| 2020 | 0.99% | 19.93% | 65.71% |
| 2021 | 7.35% | 4.67% | 69.59% |
| 2022 | -8.92% | -16.41% | 55.71% |

Annual returns use the exact saved carried path and common dates, grouping by interval start year. Dollar contributions use different carried NAVs and therefore do not isolate a causal allocation effect.

| Forecast | Mean bias | MSE | Mean rank correlation | 80% coverage | 95% coverage |
|---|---:|---:|---:|---:|---:|
| s8 | 0.000551 | 0.000748 | -0.03921 | 51.42% | 77.56% |
| equal | 0.000555 | 0.000748 | -0.04248 | 51.20% | 77.56% |
| memory | 0.000594 | 0.000748 | -0.04901 | 51.63% | 77.56% |
| fixed_half | 0.000551 | 0.000748 | -0.03921 | 51.42% | 77.56% |
| trend | -0.000081 | 0.000780 | -0.02287 | 54.47% | 78.43% |
| adaptive_volatility | -0.000143 | 0.000744 | None | 40.52% | 64.05% |
| ridge | 0.001225 | 0.000744 | 0.127450 | 51.42% | 77.12% |
| zero | -0.000143 | 0.000744 | None | 51.63% | 77.12% |

All 15 forecasts, all three seeds, ensemble, individual assets and each year are retained in forecast_diagnostics.csv. These are descriptive metrics with no newly selected winner or recalibration.

Weekly trading-session counts: {4: 25, 5: 128}. H4 trading-session spans: {19: 23, 20: 14, 18: 1, 5: 1}. Saved target is a variable-duration weekly open-to-open dividend-inclusive log return. Source trend uses the mean of eight daily close returns, and adaptive volatility uses their weighted RMS. H4 multiplies the current weekly surrogate by up to four; no direct four-interval mean or joint path is saved. This is a verified specification mismatch, not an isolated causal loss explanation.

Seed 11: effective kernel support mean 59.9877 of 60 dates (minimum 59.9256); largest normalized date weight maximum 0.0176; weighted mean age 292.7 days.
Seed 29: effective kernel support mean 59.9858 of 60 dates (minimum 59.9212); largest normalized date weight maximum 0.0176; weighted mean age 292.7 days.
Seed 47: effective kernel support mean 59.9846 of 60 dates (minimum 59.9134); largest normalized date weight maximum 0.0176; weighted mean age 292.7 days.

Near-uniform kernel weights are evidence about weak localization, not proof that context cannot help. Saved bank records allow support and age measurement; no new shuffled-key policy is evaluated.

- All analyses are retrospective descriptions of the closed 2020-2022 development path. No causal attribution or future improvement is established.
- Seeds are initialization repeats. Rank correlations use three assets and share the same 153 market intervals.
- Absolute forecast/realized magnitude measures signal scale, not attainable trading profit. Log means are not expected simple returns.
- PIT coverage is marginal per saved density. It does not validate joint risk, temporal calibration or funding probability.
- Exposure is measured from saved positions; original full-exposure passive benchmark is retained. No new exposure-matched policy is run.
- Retrieval ESS is kernel concentration only. Regime labels, shuffled-key and age-matched counterfactuals are unavailable in this closed matrix.
- Zero cash yield, source dividend assumptions, simplified fills and the survivor ETF universe are inherited unchanged.
