# Multiplicity-corrected comparisons (legacy full profile)

**Generation:** legacy lean matrix — leakage-contaminated on both tracks (D-050, D-051); folds degenerate and ETF seeds identical (D-052). Not evidence of forecasting skill.

Paired Wilcoxon on MSE(challenger) − MSE(persistence) after collapsing exact duplicate cells; comparisons with fewer than 6 unique pairs are untestable. Benjamini–Hochberg over testable comparisons. Significant = q ≤ 0.05 **and** mean delta < 0.

| dataset | challenger | raw cells | unique pairs | mean ΔMSE | testable | p | BH q | sig better |
|---|---|---:|---:|---:|:---:|---:|---:|:---:|
| etf_track_a | B_RIDGE | 20 | 1 | -5.362e-07 | N | — | — | N |
| synthetic_track_c | B_RIDGE | 20 | 5 | -4.481e-02 | N | — | — | N |
| etf_track_a | M03 | 20 | 1 | -5.050e-07 | N | — | — | N |
| synthetic_track_c | M03 | 20 | 5 | -2.308e-02 | N | — | — | N |
| etf_track_a | M04 | 20 | 1 | -5.334e-07 | N | — | — | N |
| synthetic_track_c | M04 | 20 | 5 | -4.481e-02 | N | — | — | N |
| etf_track_a | M05 | 20 | 1 | 7.253e-05 | N | — | — | N |
| synthetic_track_c | M05 | 20 | 5 | -3.565e-02 | N | — | — | N |
| etf_track_a | M09 | 20 | 1 | -5.371e-07 | N | — | — | N |
| synthetic_track_c | M09 | 20 | 5 | -3.019e-02 | N | — | — | N |

Testable: **0** / 10. Significant better after BH: **0**.

Lockbox H1 is excluded from this family (already opened once; see LOCKBOX_SUMMARY).
