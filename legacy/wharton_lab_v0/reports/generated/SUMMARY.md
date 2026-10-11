# Experiment summary

Runs indexed (manifest only): 1181
Historical orphan receipts excluded: 1916 (see `HISTORICAL_ORPHANS.json`)

**Generation:** legacy lean matrix — leakage-contaminated on both tracks, degenerate folds, identical ETF seeds (protocol/DECISIONS.md D-050..D-052). `n` counts cells, not independent evaluations.

## Status distinctions

- Implemented ≠ trained ≠ validated.
- Forecasting improvement ≠ investment outperformance.
- Simulated funding success ≠ guarantee.
- Pilot/smoke ≠ FULL completion.

## Scoreboards (default variant means)

### Profile `full`

| model | metric | n | mean |
|---|---|---:|---:|
| B_EWMA_COV | frobenius_cov_error | 40 | 0.004712509630777986 |
| B_PERSISTENCE | mse | 40 | 0.022457302472075866 |
| B_RIDGE | mse | 40 | 5.3653311643289804e-05 |
| M01 | pinball_loss | 40 | 0.007151414793446441 |
| M02 | spearman_ic | 40 | 0.07359403075985992 |
| M03 | mse | 40 | 0.010917798342281692 |
| M04 | mse | 40 | 5.356022666555808e-05 |
| M05 | mse | 40 | 0.004667913658010777 |
| M06 | latent_mse | 40 | 0.5127006191156578 |
| M07 | cov_forecast_mse | 40 | 0.17322364675334603 |
| M08 | graph_latent_mse | 40 | 0.016192998883565467 |
| M09 | mse | 40 | 0.007360504436607196 |
| M10 | energy_score_proxy_mse | 40 | 0.00027493470083486037 |
| M11 | funding_shortfall | 40 | 0.0 |
| M12 | market_mse | 40 | 0.00012996851715410345 |

### Profile `overnight`

| model | metric | n | mean |
|---|---|---:|---:|
| B_EWMA_COV | frobenius_cov_error | 24 | 0.014547499955921079 |
| B_PERSISTENCE | mse | 24 | 0.02052432017597359 |
| B_RIDGE | mse | 24 | 4.7872698740291374e-05 |
| M01 | pinball_loss | 24 | 0.006886200938051138 |
| M02 | spearman_ic | 24 | -0.02421035347504179 |
| M03 | mse | 24 | 0.00982647429620324 |
| M04 | mse | 24 | 4.8276323780290736e-05 |
| M05 | mse | 24 | 0.004301741975359606 |
| M06 | latent_mse | 24 | 0.4529732404157769 |
| M07 | cov_forecast_mse | 24 | 0.17744104762287358 |
| M08 | graph_latent_mse | 24 | 0.014491586825643115 |
| M09 | mse | 24 | 0.006418936835919121 |
| M10 | energy_score_proxy_mse | 24 | 0.0002975926840573714 |
| M11 | funding_shortfall | 24 | 0.0 |
| M12 | market_mse | 24 | 0.00011995552965202347 |

### Profile `pilot`

| model | metric | n | mean |
|---|---|---:|---:|
| B_EWMA_COV | frobenius_cov_error | 12 | 0.01984268341427121 |
| B_PERSISTENCE | mse | 12 | 0.02977700023601874 |
| B_RIDGE | mse | 12 | 5.5508608118025274e-05 |
| M01 | pinball_loss | 12 | 0.009218625693733521 |
| M02 | spearman_ic | 12 | 0.14462802069980535 |
| M03 | mse | 12 | 0.013421990617052525 |
| M04 | mse | 12 | 5.642788430429873e-05 |
| M05 | mse | 12 | 0.004908634313120561 |
| M06 | latent_mse | 12 | 0.6062780666847928 |
| M07 | cov_forecast_mse | 12 | 0.16684122775294857 |
| M08 | graph_latent_mse | 12 | 0.02090856784030633 |
| M09 | mse | 12 | 0.009307420231826713 |
| M10 | energy_score_proxy_mse | 12 | 0.00027040176851805163 |
| M11 | funding_shortfall | 12 | 0.0 |
| M12 | market_mse | 12 | 8.843845508405368e-05 |

### Profile `smoke`

| model | metric | n | mean |
|---|---|---:|---:|
| B_EWMA_COV | frobenius_cov_error | 1 | 0.012507159542685489 |
| B_PERSISTENCE | mse | 1 | 0.02367852028349004 |
| B_RIDGE | mse | 1 | 0.00019101252144849756 |
| M01 | pinball_loss | 2 | 0.010527972295031328 |
| M02 | spearman_ic | 2 | 0.08524254821741672 |
| M03 | mse | 2 | 0.008201687068876817 |
| M04 | mse | 2 | 0.0001731536807475445 |
| M05 | mse | 2 | 0.003641425048318112 |
| M06 | latent_mse | 2 | 0.3908595590460664 |
| M07 | cov_forecast_mse | 2 | 0.20081541754373422 |
| M08 | graph_latent_mse | 2 | 0.0222182887931259 |
| M09 | mse | 2 | 0.007272307919011214 |
| M10 | energy_score_proxy_mse | 2 | 0.0006498999533335024 |
| M11 | funding_shortfall | 2 | 0.0 |
| M12 | market_mse | 2 | 0.0001248795050377569 |

## Receipt index (first 300)

- `full_B_EWMA_COV_default_etf_track_a_f0_s11_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s131_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s23_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s47_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s89_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f1_s11_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f1_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f1_s131_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f1_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f1_s23_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f1_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f1_s47_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f1_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f1_s89_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f1_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f2_s11_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f2_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f2_s131_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f2_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f2_s23_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f2_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f2_s47_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f2_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f2_s89_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f2_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f3_s11_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f3_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f3_s131_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f3_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f3_s23_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f3_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f3_s47_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f3_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f3_s89_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f3_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s11_h20` B_EWMA_COV: frobenius_cov_error=0.010425750401099955 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s131_h20` B_EWMA_COV: frobenius_cov_error=0.0017848793190406787 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s23_h20` B_EWMA_COV: frobenius_cov_error=0.029835616150463574 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s47_h20` B_EWMA_COV: frobenius_cov_error=0.002081246113642104 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s89_h20` B_EWMA_COV: frobenius_cov_error=0.002333477309541608 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f1_s11_h20` B_EWMA_COV: frobenius_cov_error=0.010425750401099955 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f1_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f1_s131_h20` B_EWMA_COV: frobenius_cov_error=0.0017848793190406787 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f1_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f1_s23_h20` B_EWMA_COV: frobenius_cov_error=0.029835616150463574 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f1_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f1_s47_h20` B_EWMA_COV: frobenius_cov_error=0.002081246113642104 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f1_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f1_s89_h20` B_EWMA_COV: frobenius_cov_error=0.002333477309541608 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f1_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f2_s11_h20` B_EWMA_COV: frobenius_cov_error=0.010425750401099955 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f2_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f2_s131_h20` B_EWMA_COV: frobenius_cov_error=0.0017848793190406787 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f2_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f2_s23_h20` B_EWMA_COV: frobenius_cov_error=0.029835616150463574 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f2_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f2_s47_h20` B_EWMA_COV: frobenius_cov_error=0.002081246113642104 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f2_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f2_s89_h20` B_EWMA_COV: frobenius_cov_error=0.002333477309541608 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f2_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f3_s11_h20` B_EWMA_COV: frobenius_cov_error=0.010425750401099955 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f3_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f3_s131_h20` B_EWMA_COV: frobenius_cov_error=0.0017848793190406787 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f3_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f3_s23_h20` B_EWMA_COV: frobenius_cov_error=0.029835616150463574 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f3_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f3_s47_h20` B_EWMA_COV: frobenius_cov_error=0.002081246113642104 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f3_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f3_s89_h20` B_EWMA_COV: frobenius_cov_error=0.002333477309541608 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f3_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s11_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s131_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s23_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s47_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s89_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f1_s11_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f1_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f1_s131_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f1_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f1_s23_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f1_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f1_s47_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f1_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f1_s89_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f1_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f2_s11_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f2_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f2_s131_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f2_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f2_s23_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f2_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f2_s47_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f2_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f2_s89_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f2_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f3_s11_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f3_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f3_s131_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f3_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f3_s23_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f3_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f3_s47_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f3_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f3_s89_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f3_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s11_h20` B_PERSISTENCE: mse=0.028411751731747568 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s131_h20` B_PERSISTENCE: mse=0.034211082718330045 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s23_h20` B_PERSISTENCE: mse=0.09697051689075716 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s47_h20` B_PERSISTENCE: mse=0.0331735594989237 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s89_h20` B_PERSISTENCE: mse=0.0317938861920524 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f1_s11_h20` B_PERSISTENCE: mse=0.028411751731747568 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f1_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f1_s131_h20` B_PERSISTENCE: mse=0.034211082718330045 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f1_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f1_s23_h20` B_PERSISTENCE: mse=0.09697051689075716 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f1_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f1_s47_h20` B_PERSISTENCE: mse=0.0331735594989237 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f1_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f1_s89_h20` B_PERSISTENCE: mse=0.0317938861920524 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f1_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f2_s11_h20` B_PERSISTENCE: mse=0.028411751731747568 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f2_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f2_s131_h20` B_PERSISTENCE: mse=0.034211082718330045 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f2_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f2_s23_h20` B_PERSISTENCE: mse=0.09697051689075716 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f2_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f2_s47_h20` B_PERSISTENCE: mse=0.0331735594989237 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f2_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f2_s89_h20` B_PERSISTENCE: mse=0.0317938861920524 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f2_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f3_s11_h20` B_PERSISTENCE: mse=0.028411751731747568 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f3_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f3_s131_h20` B_PERSISTENCE: mse=0.034211082718330045 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f3_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f3_s23_h20` B_PERSISTENCE: mse=0.09697051689075716 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f3_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f3_s47_h20` B_PERSISTENCE: mse=0.0331735594989237 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f3_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f3_s89_h20` B_PERSISTENCE: mse=0.0317938861920524 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f3_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s11_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s131_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s23_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s47_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s89_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f1_s11_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f1_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f1_s131_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f1_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f1_s23_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f1_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f1_s47_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f1_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f1_s89_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f1_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f2_s11_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f2_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f2_s131_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f2_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f2_s23_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f2_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f2_s47_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f2_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f2_s89_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f2_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f3_s11_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f3_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f3_s131_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f3_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f3_s23_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f3_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f3_s47_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f3_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f3_s89_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f3_s89_h20)
- `full_B_RIDGE_default_synthetic_track_c_f0_s11_h20` B_RIDGE: mse=0.00011295494154411911 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f0_s11_h20)
- `full_B_RIDGE_default_synthetic_track_c_f0_s131_h20` B_RIDGE: mse=9.77965062967873e-05 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f0_s131_h20)
- `full_B_RIDGE_default_synthetic_track_c_f0_s23_h20` B_RIDGE: mse=0.00011047775704001824 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f0_s23_h20)
- `full_B_RIDGE_default_synthetic_track_c_f0_s47_h20` B_RIDGE: mse=0.00010239433280734263 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f0_s47_h20)
- `full_B_RIDGE_default_synthetic_track_c_f0_s89_h20` B_RIDGE: mse=0.0001033627883718241 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f0_s89_h20)
- `full_B_RIDGE_default_synthetic_track_c_f1_s11_h20` B_RIDGE: mse=0.00011295494154411911 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f1_s11_h20)
- `full_B_RIDGE_default_synthetic_track_c_f1_s131_h20` B_RIDGE: mse=9.77965062967873e-05 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f1_s131_h20)
- `full_B_RIDGE_default_synthetic_track_c_f1_s23_h20` B_RIDGE: mse=0.00011047775704001824 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f1_s23_h20)
- `full_B_RIDGE_default_synthetic_track_c_f1_s47_h20` B_RIDGE: mse=0.00010239433280734263 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f1_s47_h20)
- `full_B_RIDGE_default_synthetic_track_c_f1_s89_h20` B_RIDGE: mse=0.0001033627883718241 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f1_s89_h20)
- `full_B_RIDGE_default_synthetic_track_c_f2_s11_h20` B_RIDGE: mse=0.00011295494154411911 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f2_s11_h20)
- `full_B_RIDGE_default_synthetic_track_c_f2_s131_h20` B_RIDGE: mse=9.77965062967873e-05 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f2_s131_h20)
- `full_B_RIDGE_default_synthetic_track_c_f2_s23_h20` B_RIDGE: mse=0.00011047775704001824 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f2_s23_h20)
- `full_B_RIDGE_default_synthetic_track_c_f2_s47_h20` B_RIDGE: mse=0.00010239433280734263 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f2_s47_h20)
- `full_B_RIDGE_default_synthetic_track_c_f2_s89_h20` B_RIDGE: mse=0.0001033627883718241 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f2_s89_h20)
- `full_B_RIDGE_default_synthetic_track_c_f3_s11_h20` B_RIDGE: mse=0.00011295494154411911 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f3_s11_h20)
- `full_B_RIDGE_default_synthetic_track_c_f3_s131_h20` B_RIDGE: mse=9.77965062967873e-05 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f3_s131_h20)
- `full_B_RIDGE_default_synthetic_track_c_f3_s23_h20` B_RIDGE: mse=0.00011047775704001824 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f3_s23_h20)
- `full_B_RIDGE_default_synthetic_track_c_f3_s47_h20` B_RIDGE: mse=0.00010239433280734263 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f3_s47_h20)
- `full_B_RIDGE_default_synthetic_track_c_f3_s89_h20` B_RIDGE: mse=0.0001033627883718241 (evidence_run_id=full_B_RIDGE_default_synthetic_track_c_f3_s89_h20)
- `full_M01_default_etf_track_a_f0_s11_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f0_s11_h20)
- `full_M01_default_etf_track_a_f0_s131_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f0_s131_h20)
- `full_M01_default_etf_track_a_f0_s23_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f0_s23_h20)
- `full_M01_default_etf_track_a_f0_s47_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f0_s47_h20)
- `full_M01_default_etf_track_a_f0_s89_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f0_s89_h20)
- `full_M01_default_etf_track_a_f1_s11_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f1_s11_h20)
- `full_M01_default_etf_track_a_f1_s131_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f1_s131_h20)
- `full_M01_default_etf_track_a_f1_s23_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f1_s23_h20)
- `full_M01_default_etf_track_a_f1_s47_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f1_s47_h20)
- `full_M01_default_etf_track_a_f1_s89_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f1_s89_h20)
- `full_M01_default_etf_track_a_f2_s11_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f2_s11_h20)
- `full_M01_default_etf_track_a_f2_s131_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f2_s131_h20)
- `full_M01_default_etf_track_a_f2_s23_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f2_s23_h20)
- `full_M01_default_etf_track_a_f2_s47_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f2_s47_h20)
- `full_M01_default_etf_track_a_f2_s89_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f2_s89_h20)
- `full_M01_default_etf_track_a_f3_s11_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f3_s11_h20)
- `full_M01_default_etf_track_a_f3_s131_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f3_s131_h20)
- `full_M01_default_etf_track_a_f3_s23_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f3_s23_h20)
- `full_M01_default_etf_track_a_f3_s47_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f3_s47_h20)
- `full_M01_default_etf_track_a_f3_s89_h20` M01: pinball_loss=0.0004754377546547347 (evidence_run_id=full_M01_default_etf_track_a_f3_s89_h20)
- `full_M01_default_synthetic_track_c_f0_s11_h20` M01: pinball_loss=0.015869009872380692 (evidence_run_id=full_M01_default_synthetic_track_c_f0_s11_h20)
- `full_M01_default_synthetic_track_c_f0_s131_h20` M01: pinball_loss=0.013991169561506072 (evidence_run_id=full_M01_default_synthetic_track_c_f0_s131_h20)
- `full_M01_default_synthetic_track_c_f0_s23_h20` M01: pinball_loss=0.013104957365002515 (evidence_run_id=full_M01_default_synthetic_track_c_f0_s23_h20)
- `full_M01_default_synthetic_track_c_f0_s47_h20` M01: pinball_loss=0.011343562532305066 (evidence_run_id=full_M01_default_synthetic_track_c_f0_s47_h20)
- `full_M01_default_synthetic_track_c_f0_s89_h20` M01: pinball_loss=0.014828259829996393 (evidence_run_id=full_M01_default_synthetic_track_c_f0_s89_h20)
- `full_M01_default_synthetic_track_c_f1_s11_h20` M01: pinball_loss=0.015869009872380692 (evidence_run_id=full_M01_default_synthetic_track_c_f1_s11_h20)
- `full_M01_default_synthetic_track_c_f1_s131_h20` M01: pinball_loss=0.013991169561506072 (evidence_run_id=full_M01_default_synthetic_track_c_f1_s131_h20)
- `full_M01_default_synthetic_track_c_f1_s23_h20` M01: pinball_loss=0.013104957365002515 (evidence_run_id=full_M01_default_synthetic_track_c_f1_s23_h20)
- `full_M01_default_synthetic_track_c_f1_s47_h20` M01: pinball_loss=0.011343562532305066 (evidence_run_id=full_M01_default_synthetic_track_c_f1_s47_h20)
- `full_M01_default_synthetic_track_c_f1_s89_h20` M01: pinball_loss=0.014828259829996393 (evidence_run_id=full_M01_default_synthetic_track_c_f1_s89_h20)
- `full_M01_default_synthetic_track_c_f2_s11_h20` M01: pinball_loss=0.015869009872380692 (evidence_run_id=full_M01_default_synthetic_track_c_f2_s11_h20)
- `full_M01_default_synthetic_track_c_f2_s131_h20` M01: pinball_loss=0.013991169561506072 (evidence_run_id=full_M01_default_synthetic_track_c_f2_s131_h20)
- `full_M01_default_synthetic_track_c_f2_s23_h20` M01: pinball_loss=0.013104957365002515 (evidence_run_id=full_M01_default_synthetic_track_c_f2_s23_h20)
- `full_M01_default_synthetic_track_c_f2_s47_h20` M01: pinball_loss=0.011343562532305066 (evidence_run_id=full_M01_default_synthetic_track_c_f2_s47_h20)
- `full_M01_default_synthetic_track_c_f2_s89_h20` M01: pinball_loss=0.014828259829996393 (evidence_run_id=full_M01_default_synthetic_track_c_f2_s89_h20)
- `full_M01_default_synthetic_track_c_f3_s11_h20` M01: pinball_loss=0.015869009872380692 (evidence_run_id=full_M01_default_synthetic_track_c_f3_s11_h20)
- `full_M01_default_synthetic_track_c_f3_s131_h20` M01: pinball_loss=0.013991169561506072 (evidence_run_id=full_M01_default_synthetic_track_c_f3_s131_h20)
- `full_M01_default_synthetic_track_c_f3_s23_h20` M01: pinball_loss=0.013104957365002515 (evidence_run_id=full_M01_default_synthetic_track_c_f3_s23_h20)
- `full_M01_default_synthetic_track_c_f3_s47_h20` M01: pinball_loss=0.011343562532305066 (evidence_run_id=full_M01_default_synthetic_track_c_f3_s47_h20)
- `full_M01_default_synthetic_track_c_f3_s89_h20` M01: pinball_loss=0.014828259829996393 (evidence_run_id=full_M01_default_synthetic_track_c_f3_s89_h20)
- `full_M02_default_etf_track_a_f0_s11_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f0_s11_h20)
- `full_M02_default_etf_track_a_f0_s131_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f0_s131_h20)
- `full_M02_default_etf_track_a_f0_s23_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f0_s23_h20)
- `full_M02_default_etf_track_a_f0_s47_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f0_s47_h20)
- `full_M02_default_etf_track_a_f0_s89_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f0_s89_h20)
- `full_M02_default_etf_track_a_f1_s11_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f1_s11_h20)
- `full_M02_default_etf_track_a_f1_s131_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f1_s131_h20)
- `full_M02_default_etf_track_a_f1_s23_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f1_s23_h20)
- `full_M02_default_etf_track_a_f1_s47_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f1_s47_h20)
- `full_M02_default_etf_track_a_f1_s89_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f1_s89_h20)
- `full_M02_default_etf_track_a_f2_s11_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f2_s11_h20)
- `full_M02_default_etf_track_a_f2_s131_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f2_s131_h20)
- `full_M02_default_etf_track_a_f2_s23_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f2_s23_h20)
- `full_M02_default_etf_track_a_f2_s47_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f2_s47_h20)
- `full_M02_default_etf_track_a_f2_s89_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f2_s89_h20)
- `full_M02_default_etf_track_a_f3_s11_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f3_s11_h20)
- `full_M02_default_etf_track_a_f3_s131_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f3_s131_h20)
- `full_M02_default_etf_track_a_f3_s23_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f3_s23_h20)
- `full_M02_default_etf_track_a_f3_s47_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f3_s47_h20)
- `full_M02_default_etf_track_a_f3_s89_h20` M02: spearman_ic=0.20074463225216996 (evidence_run_id=full_M02_default_etf_track_a_f3_s89_h20)
- `full_M02_default_synthetic_track_c_f0_s11_h20` M02: spearman_ic=-0.2269575148469621 (evidence_run_id=full_M02_default_synthetic_track_c_f0_s11_h20)
- `full_M02_default_synthetic_track_c_f0_s131_h20` M02: spearman_ic=-0.06999390893863255 (evidence_run_id=full_M02_default_synthetic_track_c_f0_s131_h20)
- `full_M02_default_synthetic_track_c_f0_s23_h20` M02: spearman_ic=-0.17077051926298156 (evidence_run_id=full_M02_default_synthetic_track_c_f0_s23_h20)
- `full_M02_default_synthetic_track_c_f0_s47_h20` M02: spearman_ic=-0.13489112227805694 (evidence_run_id=full_M02_default_synthetic_track_c_f0_s47_h20)
- `full_M02_default_synthetic_track_c_f0_s89_h20` M02: spearman_ic=0.33483021166438254 (evidence_run_id=full_M02_default_synthetic_track_c_f0_s89_h20)
- `full_M02_default_synthetic_track_c_f1_s11_h20` M02: spearman_ic=-0.2269575148469621 (evidence_run_id=full_M02_default_synthetic_track_c_f1_s11_h20)
- `full_M02_default_synthetic_track_c_f1_s131_h20` M02: spearman_ic=-0.06999390893863255 (evidence_run_id=full_M02_default_synthetic_track_c_f1_s131_h20)
- `full_M02_default_synthetic_track_c_f1_s23_h20` M02: spearman_ic=-0.17077051926298156 (evidence_run_id=full_M02_default_synthetic_track_c_f1_s23_h20)
- `full_M02_default_synthetic_track_c_f1_s47_h20` M02: spearman_ic=-0.13489112227805694 (evidence_run_id=full_M02_default_synthetic_track_c_f1_s47_h20)
- `full_M02_default_synthetic_track_c_f1_s89_h20` M02: spearman_ic=0.33483021166438254 (evidence_run_id=full_M02_default_synthetic_track_c_f1_s89_h20)
- `full_M02_default_synthetic_track_c_f2_s11_h20` M02: spearman_ic=-0.2269575148469621 (evidence_run_id=full_M02_default_synthetic_track_c_f2_s11_h20)
- `full_M02_default_synthetic_track_c_f2_s131_h20` M02: spearman_ic=-0.06999390893863255 (evidence_run_id=full_M02_default_synthetic_track_c_f2_s131_h20)
- `full_M02_default_synthetic_track_c_f2_s23_h20` M02: spearman_ic=-0.17077051926298156 (evidence_run_id=full_M02_default_synthetic_track_c_f2_s23_h20)
- `full_M02_default_synthetic_track_c_f2_s47_h20` M02: spearman_ic=-0.13489112227805694 (evidence_run_id=full_M02_default_synthetic_track_c_f2_s47_h20)
- `full_M02_default_synthetic_track_c_f2_s89_h20` M02: spearman_ic=0.33483021166438254 (evidence_run_id=full_M02_default_synthetic_track_c_f2_s89_h20)
- `full_M02_default_synthetic_track_c_f3_s11_h20` M02: spearman_ic=-0.2269575148469621 (evidence_run_id=full_M02_default_synthetic_track_c_f3_s11_h20)
- `full_M02_default_synthetic_track_c_f3_s131_h20` M02: spearman_ic=-0.06999390893863255 (evidence_run_id=full_M02_default_synthetic_track_c_f3_s131_h20)
- `full_M02_default_synthetic_track_c_f3_s23_h20` M02: spearman_ic=-0.17077051926298156 (evidence_run_id=full_M02_default_synthetic_track_c_f3_s23_h20)
- `full_M02_default_synthetic_track_c_f3_s47_h20` M02: spearman_ic=-0.13489112227805694 (evidence_run_id=full_M02_default_synthetic_track_c_f3_s47_h20)
- `full_M02_default_synthetic_track_c_f3_s89_h20` M02: spearman_ic=0.33483021166438254 (evidence_run_id=full_M02_default_synthetic_track_c_f3_s89_h20)
- `full_M03_default_etf_track_a_f0_s11_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f0_s11_h20)
- `full_M03_default_etf_track_a_f0_s131_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f0_s131_h20)
- `full_M03_default_etf_track_a_f0_s23_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f0_s23_h20)
- `full_M03_default_etf_track_a_f0_s47_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f0_s47_h20)
- `full_M03_default_etf_track_a_f0_s89_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f0_s89_h20)
- `full_M03_default_etf_track_a_f1_s11_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f1_s11_h20)
- `full_M03_default_etf_track_a_f1_s131_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f1_s131_h20)
- `full_M03_default_etf_track_a_f1_s23_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f1_s23_h20)
- `full_M03_default_etf_track_a_f1_s47_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f1_s47_h20)
- `full_M03_default_etf_track_a_f1_s89_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f1_s89_h20)
- `full_M03_default_etf_track_a_f2_s11_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f2_s11_h20)
- `full_M03_default_etf_track_a_f2_s131_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f2_s131_h20)
- `full_M03_default_etf_track_a_f2_s23_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f2_s23_h20)
- `full_M03_default_etf_track_a_f2_s47_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f2_s47_h20)
- `full_M03_default_etf_track_a_f2_s89_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f2_s89_h20)
- `full_M03_default_etf_track_a_f3_s11_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f3_s11_h20)
- `full_M03_default_etf_track_a_f3_s131_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f3_s131_h20)
- `full_M03_default_etf_track_a_f3_s23_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f3_s23_h20)
- `full_M03_default_etf_track_a_f3_s47_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f3_s47_h20)
- `full_M03_default_etf_track_a_f3_s89_h20` M03: mse=1.9405754202707637e-06 (evidence_run_id=full_M03_default_etf_track_a_f3_s89_h20)
- `full_M03_default_synthetic_track_c_f0_s11_h20` M03: mse=0.025859190969785627 (evidence_run_id=full_M03_default_synthetic_track_c_f0_s11_h20)
- `full_M03_default_synthetic_track_c_f0_s131_h20` M03: mse=0.025902794956017328 (evidence_run_id=full_M03_default_synthetic_track_c_f0_s131_h20)
- `full_M03_default_synthetic_track_c_f0_s23_h20` M03: mse=0.01947699068836563 (evidence_run_id=full_M03_default_synthetic_track_c_f0_s23_h20)
- `full_M03_default_synthetic_track_c_f0_s47_h20` M03: mse=0.017851966718898604 (evidence_run_id=full_M03_default_synthetic_track_c_f0_s47_h20)
- `full_M03_default_synthetic_track_c_f0_s89_h20` M03: mse=0.02007733721264837 (evidence_run_id=full_M03_default_synthetic_track_c_f0_s89_h20)
- `full_M03_default_synthetic_track_c_f1_s11_h20` M03: mse=0.025859190969785627 (evidence_run_id=full_M03_default_synthetic_track_c_f1_s11_h20)
- `full_M03_default_synthetic_track_c_f1_s131_h20` M03: mse=0.025902794956017328 (evidence_run_id=full_M03_default_synthetic_track_c_f1_s131_h20)
- `full_M03_default_synthetic_track_c_f1_s23_h20` M03: mse=0.01947699068836563 (evidence_run_id=full_M03_default_synthetic_track_c_f1_s23_h20)
- `full_M03_default_synthetic_track_c_f1_s47_h20` M03: mse=0.017851966718898604 (evidence_run_id=full_M03_default_synthetic_track_c_f1_s47_h20)
- `full_M03_default_synthetic_track_c_f1_s89_h20` M03: mse=0.02007733721264837 (evidence_run_id=full_M03_default_synthetic_track_c_f1_s89_h20)
- `full_M03_default_synthetic_track_c_f2_s11_h20` M03: mse=0.025859190969785627 (evidence_run_id=full_M03_default_synthetic_track_c_f2_s11_h20)
- `full_M03_default_synthetic_track_c_f2_s131_h20` M03: mse=0.025902794956017328 (evidence_run_id=full_M03_default_synthetic_track_c_f2_s131_h20)
- `full_M03_default_synthetic_track_c_f2_s23_h20` M03: mse=0.01947699068836563 (evidence_run_id=full_M03_default_synthetic_track_c_f2_s23_h20)
- `full_M03_default_synthetic_track_c_f2_s47_h20` M03: mse=0.017851966718898604 (evidence_run_id=full_M03_default_synthetic_track_c_f2_s47_h20)
- `full_M03_default_synthetic_track_c_f2_s89_h20` M03: mse=0.02007733721264837 (evidence_run_id=full_M03_default_synthetic_track_c_f2_s89_h20)
- `full_M03_default_synthetic_track_c_f3_s11_h20` M03: mse=0.025859190969785627 (evidence_run_id=full_M03_default_synthetic_track_c_f3_s11_h20)
- `full_M03_default_synthetic_track_c_f3_s131_h20` M03: mse=0.025902794956017328 (evidence_run_id=full_M03_default_synthetic_track_c_f3_s131_h20)
- `full_M03_default_synthetic_track_c_f3_s23_h20` M03: mse=0.01947699068836563 (evidence_run_id=full_M03_default_synthetic_track_c_f3_s23_h20)
- `full_M03_default_synthetic_track_c_f3_s47_h20` M03: mse=0.017851966718898604 (evidence_run_id=full_M03_default_synthetic_track_c_f3_s47_h20)
- `full_M03_default_synthetic_track_c_f3_s89_h20` M03: mse=0.02007733721264837 (evidence_run_id=full_M03_default_synthetic_track_c_f3_s89_h20)
- `full_M04_default_etf_track_a_f0_s11_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f0_s11_h20)
- `full_M04_default_etf_track_a_f0_s131_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f0_s131_h20)
- `full_M04_default_etf_track_a_f0_s23_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f0_s23_h20)
- `full_M04_default_etf_track_a_f0_s47_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f0_s47_h20)
- `full_M04_default_etf_track_a_f0_s89_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f0_s89_h20)
- `full_M04_default_etf_track_a_f1_s11_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f1_s11_h20)
- `full_M04_default_etf_track_a_f1_s131_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f1_s131_h20)
- `full_M04_default_etf_track_a_f1_s23_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f1_s23_h20)
- `full_M04_default_etf_track_a_f1_s47_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f1_s47_h20)
- `full_M04_default_etf_track_a_f1_s89_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f1_s89_h20)
- `full_M04_default_etf_track_a_f2_s11_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f2_s11_h20)
- `full_M04_default_etf_track_a_f2_s131_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f2_s131_h20)
- `full_M04_default_etf_track_a_f2_s23_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f2_s23_h20)
- `full_M04_default_etf_track_a_f2_s47_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f2_s47_h20)
- `full_M04_default_etf_track_a_f2_s89_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f2_s89_h20)
- `full_M04_default_etf_track_a_f3_s11_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f3_s11_h20)
- `full_M04_default_etf_track_a_f3_s131_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f3_s131_h20)
- `full_M04_default_etf_track_a_f3_s23_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f3_s23_h20)
- `full_M04_default_etf_track_a_f3_s47_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f3_s47_h20)
- `full_M04_default_etf_track_a_f3_s89_h20` M04: mse=1.9121109004934395e-06 (evidence_run_id=full_M04_default_etf_track_a_f3_s89_h20)
- `full_M04_default_synthetic_track_c_f0_s11_h20` M04: mse=0.00011298931309468322 (evidence_run_id=full_M04_default_synthetic_track_c_f0_s11_h20)
- `full_M04_default_synthetic_track_c_f0_s131_h20` M04: mse=9.725994423670881e-05 (evidence_run_id=full_M04_default_synthetic_track_c_f0_s131_h20)
- `full_M04_default_synthetic_track_c_f0_s23_h20` M04: mse=0.00011143921821723804 (evidence_run_id=full_M04_default_synthetic_track_c_f0_s23_h20)
- `full_M04_default_synthetic_track_c_f0_s47_h20` M04: mse=0.00010129600070133894 (evidence_run_id=full_M04_default_synthetic_track_c_f0_s47_h20)
- `full_M04_default_synthetic_track_c_f0_s89_h20` M04: mse=0.00010305723590314457 (evidence_run_id=full_M04_default_synthetic_track_c_f0_s89_h20)
- `full_M04_default_synthetic_track_c_f1_s11_h20` M04: mse=0.00011298931309468322 (evidence_run_id=full_M04_default_synthetic_track_c_f1_s11_h20)
- `full_M04_default_synthetic_track_c_f1_s131_h20` M04: mse=9.725994423670881e-05 (evidence_run_id=full_M04_default_synthetic_track_c_f1_s131_h20)
- `full_M04_default_synthetic_track_c_f1_s23_h20` M04: mse=0.00011143921821723804 (evidence_run_id=full_M04_default_synthetic_track_c_f1_s23_h20)
- `full_M04_default_synthetic_track_c_f1_s47_h20` M04: mse=0.00010129600070133894 (evidence_run_id=full_M04_default_synthetic_track_c_f1_s47_h20)
- `full_M04_default_synthetic_track_c_f1_s89_h20` M04: mse=0.00010305723590314457 (evidence_run_id=full_M04_default_synthetic_track_c_f1_s89_h20)
- `full_M04_default_synthetic_track_c_f2_s11_h20` M04: mse=0.00011298931309468322 (evidence_run_id=full_M04_default_synthetic_track_c_f2_s11_h20)
- `full_M04_default_synthetic_track_c_f2_s131_h20` M04: mse=9.725994423670881e-05 (evidence_run_id=full_M04_default_synthetic_track_c_f2_s131_h20)
- `full_M04_default_synthetic_track_c_f2_s23_h20` M04: mse=0.00011143921821723804 (evidence_run_id=full_M04_default_synthetic_track_c_f2_s23_h20)
- `full_M04_default_synthetic_track_c_f2_s47_h20` M04: mse=0.00010129600070133894 (evidence_run_id=full_M04_default_synthetic_track_c_f2_s47_h20)
- `full_M04_default_synthetic_track_c_f2_s89_h20` M04: mse=0.00010305723590314457 (evidence_run_id=full_M04_default_synthetic_track_c_f2_s89_h20)
- `full_M04_default_synthetic_track_c_f3_s11_h20` M04: mse=0.00011298931309468322 (evidence_run_id=full_M04_default_synthetic_track_c_f3_s11_h20)
- `full_M04_default_synthetic_track_c_f3_s131_h20` M04: mse=9.725994423670881e-05 (evidence_run_id=full_M04_default_synthetic_track_c_f3_s131_h20)
- `full_M04_default_synthetic_track_c_f3_s23_h20` M04: mse=0.00011143921821723804 (evidence_run_id=full_M04_default_synthetic_track_c_f3_s23_h20)
- `full_M04_default_synthetic_track_c_f3_s47_h20` M04: mse=0.00010129600070133894 (evidence_run_id=full_M04_default_synthetic_track_c_f3_s47_h20)
- `full_M04_default_synthetic_track_c_f3_s89_h20` M04: mse=0.00010305723590314457 (evidence_run_id=full_M04_default_synthetic_track_c_f3_s89_h20)
- `full_M05_default_etf_track_a_f0_s11_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f0_s11_h20)
- `full_M05_default_etf_track_a_f0_s131_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f0_s131_h20)
- `full_M05_default_etf_track_a_f0_s23_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f0_s23_h20)
- `full_M05_default_etf_track_a_f0_s47_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f0_s47_h20)
- `full_M05_default_etf_track_a_f0_s89_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f0_s89_h20)
- `full_M05_default_etf_track_a_f1_s11_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f1_s11_h20)
- `full_M05_default_etf_track_a_f1_s131_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f1_s131_h20)
- `full_M05_default_etf_track_a_f1_s23_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f1_s23_h20)
- `full_M05_default_etf_track_a_f1_s47_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f1_s47_h20)
- `full_M05_default_etf_track_a_f1_s89_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f1_s89_h20)
- `full_M05_default_etf_track_a_f2_s11_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f2_s11_h20)
- `full_M05_default_etf_track_a_f2_s131_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f2_s131_h20)
- `full_M05_default_etf_track_a_f2_s23_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f2_s23_h20)
- `full_M05_default_etf_track_a_f2_s47_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f2_s47_h20)
- `full_M05_default_etf_track_a_f2_s89_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f2_s89_h20)
- `full_M05_default_etf_track_a_f3_s11_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f3_s11_h20)
- `full_M05_default_etf_track_a_f3_s131_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f3_s131_h20)
- `full_M05_default_etf_track_a_f3_s23_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f3_s23_h20)
- `full_M05_default_etf_track_a_f3_s47_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f3_s47_h20)
- `full_M05_default_etf_track_a_f3_s89_h20` M05: mse=7.49791520780647e-05 (evidence_run_id=full_M05_default_etf_track_a_f3_s89_h20)
