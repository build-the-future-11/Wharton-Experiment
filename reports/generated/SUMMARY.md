# Experiment summary

Runs indexed: 3099

## Status distinctions

- Implemented ≠ trained ≠ validated.
- Forecasting improvement ≠ investment outperformance.
- Simulated funding success ≠ guarantee.
- Pilot/smoke ≠ FULL completion.

## Scoreboards (default variant means)

### Profile `full`

| model | metric | n | mean |
|---|---|---:|---:|
| B_EWMA_COV | frobenius_cov_error | 120 | 0.008159086739621473 |
| B_PERSISTENCE | mse | 120 | 0.016344243391025557 |
| B_RIDGE | mse | 120 | 5.319319986611754e-05 |
| M01 | pinball_loss | 120 | 0.004489156016188424 |
| M02 | spearman_ic | 121 | 0.03225997894048862 |
| M03 | mse | 121 | 0.010822519751901509 |
| M04 | mse | 121 | 5.2716483146460897e-05 |
| M05 | mse | 121 | 0.004031857046290984 |
| M06 | latent_mse | 120 | 0.5088780163499629 |
| M07 | cov_forecast_mse | 120 | 0.15614400960980104 |
| M08 | graph_latent_mse | 121 | 0.017505530604331705 |
| M09 | mse | 120 | 0.05684547028671962 |
| M10 | energy_score_proxy_mse | 120 | 0.00025943232498238706 |
| M11 | funding_shortfall | 120 | 0.0 |
| M12 | market_mse | 120 | 0.00014182328956382405 |

### Profile `overnight`

| model | metric | n | mean |
|---|---|---:|---:|
| B_EWMA_COV | frobenius_cov_error | 64 | 0.014764014548625807 |
| B_PERSISTENCE | mse | 64 | 0.025913054617600878 |
| B_RIDGE | mse | 64 | 4.899661078821762e-05 |
| M01 | pinball_loss | 64 | 0.004874886104923336 |
| M02 | spearman_ic | 64 | 0.020715860174539927 |
| M03 | mse | 64 | 0.00960555232452882 |
| M04 | mse | 64 | 4.932061956400419e-05 |
| M05 | mse | 64 | 0.0038761173780750857 |
| M06 | latent_mse | 64 | 0.47197239995309903 |
| M07 | cov_forecast_mse | 64 | 0.1598344211785119 |
| M08 | graph_latent_mse | 64 | 0.016423448324502467 |
| M09 | mse | 64 | 0.007439849626869044 |
| M10 | energy_score_proxy_mse | 64 | 0.0002973528745838094 |
| M11 | funding_shortfall | 64 | 0.0 |
| M12 | market_mse | 64 | 0.00015164818872080872 |

### Profile `pilot`

| model | metric | n | mean |
|---|---|---:|---:|
| B_EWMA_COV | frobenius_cov_error | 18 | 0.018461494748849828 |
| B_PERSISTENCE | mse | 18 | 0.0267552603712973 |
| B_RIDGE | mse | 19 | 5.236593571521396e-05 |
| M01 | pinball_loss | 20 | 0.008754362031327949 |
| M02 | spearman_ic | 20 | 0.1094199300151339 |
| M03 | mse | 19 | 0.012842679671496127 |
| M04 | mse | 19 | 5.313759377746553e-05 |
| M05 | mse | 19 | 0.004878130880465921 |
| M06 | latent_mse | 23 | 0.6293448979663431 |
| M07 | cov_forecast_mse | 19 | 0.15971675119143838 |
| M08 | graph_latent_mse | 19 | 0.02031327306574249 |
| M09 | mse | 19 | 0.008952518948431692 |
| M10 | energy_score_proxy_mse | 19 | 0.000279907523260805 |
| M11 | funding_shortfall | 20 | 0.0 |
| M12 | market_mse | 20 | 0.00010368640803888654 |

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

- `dbg_M03` M03: mse=4.920465137463038e-06 (evidence_run_id=dbg_M03)
- `dbg_M04` M04: mse=4.928398694200138e-06 (evidence_run_id=dbg_M04)
- `dbg_M05` M05: mse=7.358961859468335e-05 (evidence_run_id=dbg_M05)
- `dbg_m08` M08: graph_latent_mse=0.03675184525033635 (evidence_run_id=dbg_m08)
- `debug_m02` M02: spearman_ic=0.02283822118966368 (evidence_run_id=debug_m02)
- `full_B_EWMA_COV_default_etf_track_a_f0_s11_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s131_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s23_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s47_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f0_s89_h20` B_EWMA_COV: frobenius_cov_error=0.00013282540279838922 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f0_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f10_s11_h20` B_EWMA_COV: frobenius_cov_error=3.6420087564031594e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f10_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f10_s131_h20` B_EWMA_COV: frobenius_cov_error=3.642038743724217e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f10_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f10_s23_h20` B_EWMA_COV: frobenius_cov_error=3.642025504821984e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f10_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f10_s47_h20` B_EWMA_COV: frobenius_cov_error=3.642034549297235e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f10_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f10_s89_h20` B_EWMA_COV: frobenius_cov_error=3.642019529786953e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f10_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f11_s11_h20` B_EWMA_COV: frobenius_cov_error=3.64199835590823e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f11_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f11_s131_h20` B_EWMA_COV: frobenius_cov_error=3.642013723590077e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f11_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f11_s23_h20` B_EWMA_COV: frobenius_cov_error=3.64200433641444e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f11_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f11_s47_h20` B_EWMA_COV: frobenius_cov_error=3.6419367460536376e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f11_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f11_s89_h20` B_EWMA_COV: frobenius_cov_error=3.642028037116337e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f11_s89_h20)
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
- `full_B_EWMA_COV_default_etf_track_a_f4_s11_h20` B_EWMA_COV: frobenius_cov_error=3.641961644447792e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f4_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f4_s131_h20` B_EWMA_COV: frobenius_cov_error=3.64204781712995e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f4_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f4_s23_h20` B_EWMA_COV: frobenius_cov_error=3.6420331763333745e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f4_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f4_s47_h20` B_EWMA_COV: frobenius_cov_error=3.6420154126670824e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f4_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f4_s89_h20` B_EWMA_COV: frobenius_cov_error=3.642042369719647e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f4_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f5_s11_h20` B_EWMA_COV: frobenius_cov_error=3.6420055802374e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f5_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f5_s131_h20` B_EWMA_COV: frobenius_cov_error=3.641976738823221e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f5_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f5_s23_h20` B_EWMA_COV: frobenius_cov_error=3.642007477019743e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f5_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f5_s47_h20` B_EWMA_COV: frobenius_cov_error=3.641958949558554e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f5_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f5_s89_h20` B_EWMA_COV: frobenius_cov_error=3.6419999144538127e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f5_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f6_s11_h20` B_EWMA_COV: frobenius_cov_error=3.642007500864067e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f6_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f6_s131_h20` B_EWMA_COV: frobenius_cov_error=3.641973280140626e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f6_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f6_s23_h20` B_EWMA_COV: frobenius_cov_error=3.6419950740657776e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f6_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f6_s47_h20` B_EWMA_COV: frobenius_cov_error=3.6421234870212894e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f6_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f6_s89_h20` B_EWMA_COV: frobenius_cov_error=3.6419872417164496e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f6_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f7_s11_h20` B_EWMA_COV: frobenius_cov_error=3.6419940096245836e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f7_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f7_s131_h20` B_EWMA_COV: frobenius_cov_error=3.642004443129653e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f7_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f7_s23_h20` B_EWMA_COV: frobenius_cov_error=3.641987725124238e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f7_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f7_s47_h20` B_EWMA_COV: frobenius_cov_error=3.642009961937167e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f7_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f7_s89_h20` B_EWMA_COV: frobenius_cov_error=3.6420154270430307e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f7_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f8_s11_h20` B_EWMA_COV: frobenius_cov_error=3.642032936711494e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f8_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f8_s131_h20` B_EWMA_COV: frobenius_cov_error=3.641981637832803e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f8_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f8_s23_h20` B_EWMA_COV: frobenius_cov_error=3.642048391646962e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f8_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f8_s47_h20` B_EWMA_COV: frobenius_cov_error=3.642080884793512e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f8_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f8_s89_h20` B_EWMA_COV: frobenius_cov_error=3.6419656965385255e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f8_s89_h20)
- `full_B_EWMA_COV_default_etf_track_a_f9_s11_h20` B_EWMA_COV: frobenius_cov_error=3.6420450971313685e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f9_s11_h20)
- `full_B_EWMA_COV_default_etf_track_a_f9_s131_h20` B_EWMA_COV: frobenius_cov_error=3.6419746300334544e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f9_s131_h20)
- `full_B_EWMA_COV_default_etf_track_a_f9_s23_h20` B_EWMA_COV: frobenius_cov_error=3.64198522820551e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f9_s23_h20)
- `full_B_EWMA_COV_default_etf_track_a_f9_s47_h20` B_EWMA_COV: frobenius_cov_error=3.6420070919864525e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f9_s47_h20)
- `full_B_EWMA_COV_default_etf_track_a_f9_s89_h20` B_EWMA_COV: frobenius_cov_error=3.642028437033815e-05 (evidence_run_id=full_B_EWMA_COV_default_etf_track_a_f9_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s11_h20` B_EWMA_COV: frobenius_cov_error=0.010425750401099955 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s131_h20` B_EWMA_COV: frobenius_cov_error=0.0017848793190406787 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s23_h20` B_EWMA_COV: frobenius_cov_error=0.029835616150463574 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s47_h20` B_EWMA_COV: frobenius_cov_error=0.002081246113642104 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f0_s89_h20` B_EWMA_COV: frobenius_cov_error=0.002333477309541608 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f0_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f10_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f10_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f10_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f10_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f10_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f10_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f10_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f10_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f10_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f10_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f11_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f11_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f11_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f11_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f11_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f11_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f11_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f11_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f11_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f11_s89_h20)
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
- `full_B_EWMA_COV_default_synthetic_track_c_f4_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f4_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f4_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f4_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f4_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f4_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f4_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f4_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f4_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f4_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f5_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f5_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f5_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f5_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f5_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f5_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f5_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f5_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f5_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f5_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f6_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f6_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f6_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f6_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f6_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f6_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f6_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f6_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f6_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f6_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f7_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f7_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f7_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f7_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f7_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f7_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f7_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f7_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f7_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f7_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f8_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f8_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f8_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f8_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f8_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f8_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f8_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f8_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f8_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f8_s89_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f9_s11_h20` B_EWMA_COV: frobenius_cov_error=0.03388384902171979 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f9_s11_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f9_s131_h20` B_EWMA_COV: frobenius_cov_error=0.027947409775335746 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f9_s131_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f9_s23_h20` B_EWMA_COV: frobenius_cov_error=0.0020874011458745195 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f9_s23_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f9_s47_h20` B_EWMA_COV: frobenius_cov_error=0.0030724500074880215 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f9_s47_h20)
- `full_B_EWMA_COV_default_synthetic_track_c_f9_s89_h20` B_EWMA_COV: frobenius_cov_error=0.03165054248308147 (evidence_run_id=full_B_EWMA_COV_default_synthetic_track_c_f9_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s11_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s131_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s23_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s47_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f0_s89_h20` B_PERSISTENCE: mse=2.4455377895599293e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f0_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f10_s11_h20` B_PERSISTENCE: mse=5.071125309565574e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f10_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f10_s131_h20` B_PERSISTENCE: mse=5.070727469618545e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f10_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f10_s23_h20` B_PERSISTENCE: mse=5.070816771102327e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f10_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f10_s47_h20` B_PERSISTENCE: mse=5.071178544539657e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f10_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f10_s89_h20` B_PERSISTENCE: mse=5.071187398774874e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f10_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f11_s11_h20` B_PERSISTENCE: mse=5.0712182494596e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f11_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f11_s131_h20` B_PERSISTENCE: mse=5.07129883263585e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f11_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f11_s23_h20` B_PERSISTENCE: mse=5.070893297914901e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f11_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f11_s47_h20` B_PERSISTENCE: mse=5.070893297914901e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f11_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f11_s89_h20` B_PERSISTENCE: mse=5.071177239722464e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f11_s89_h20)
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
- `full_B_PERSISTENCE_default_etf_track_a_f4_s11_h20` B_PERSISTENCE: mse=5.070535619229269e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f4_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f4_s131_h20` B_PERSISTENCE: mse=5.071154691796112e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f4_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f4_s23_h20` B_PERSISTENCE: mse=5.071238841606761e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f4_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f4_s47_h20` B_PERSISTENCE: mse=5.071218970459717e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f4_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f4_s89_h20` B_PERSISTENCE: mse=5.071453209155073e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f4_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f5_s11_h20` B_PERSISTENCE: mse=5.071160589515872e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f5_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f5_s131_h20` B_PERSISTENCE: mse=5.070923526240521e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f5_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f5_s23_h20` B_PERSISTENCE: mse=5.070991879679348e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f5_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f5_s47_h20` B_PERSISTENCE: mse=5.070876426281203e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f5_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f5_s89_h20` B_PERSISTENCE: mse=5.0712351525354945e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f5_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f6_s11_h20` B_PERSISTENCE: mse=5.07100324999942e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f6_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f6_s131_h20` B_PERSISTENCE: mse=5.07084873488775e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f6_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f6_s23_h20` B_PERSISTENCE: mse=5.0712122966929335e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f6_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f6_s47_h20` B_PERSISTENCE: mse=5.07084873488775e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f6_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f6_s89_h20` B_PERSISTENCE: mse=5.070579319778519e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f6_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f7_s11_h20` B_PERSISTENCE: mse=5.071222441671074e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f7_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f7_s131_h20` B_PERSISTENCE: mse=5.0712301314268936e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f7_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f7_s23_h20` B_PERSISTENCE: mse=5.0710173246984405e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f7_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f7_s47_h20` B_PERSISTENCE: mse=5.0713509519728494e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f7_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f7_s89_h20` B_PERSISTENCE: mse=5.071020835952556e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f7_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f8_s11_h20` B_PERSISTENCE: mse=5.070711764890124e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f8_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f8_s131_h20` B_PERSISTENCE: mse=5.070976107894872e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f8_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f8_s23_h20` B_PERSISTENCE: mse=5.07156291160667e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f8_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f8_s47_h20` B_PERSISTENCE: mse=5.071190830437539e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f8_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f8_s89_h20` B_PERSISTENCE: mse=5.071295070060954e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f8_s89_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f9_s11_h20` B_PERSISTENCE: mse=5.070664564406774e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f9_s11_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f9_s131_h20` B_PERSISTENCE: mse=5.070918872837138e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f9_s131_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f9_s23_h20` B_PERSISTENCE: mse=5.071044212851382e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f9_s23_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f9_s47_h20` B_PERSISTENCE: mse=5.0708378506117286e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f9_s47_h20)
- `full_B_PERSISTENCE_default_etf_track_a_f9_s89_h20` B_PERSISTENCE: mse=5.0708378506117286e-06 (evidence_run_id=full_B_PERSISTENCE_default_etf_track_a_f9_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s11_h20` B_PERSISTENCE: mse=0.028411751731747568 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s131_h20` B_PERSISTENCE: mse=0.034211082718330045 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s23_h20` B_PERSISTENCE: mse=0.09697051689075716 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s47_h20` B_PERSISTENCE: mse=0.0331735594989237 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f0_s89_h20` B_PERSISTENCE: mse=0.0317938861920524 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f0_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f10_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f10_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f10_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f10_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f10_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f10_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f10_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f10_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f10_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f10_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f11_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f11_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f11_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f11_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f11_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f11_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f11_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f11_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f11_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f11_s89_h20)
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
- `full_B_PERSISTENCE_default_synthetic_track_c_f4_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f4_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f4_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f4_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f4_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f4_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f4_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f4_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f4_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f4_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f5_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f5_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f5_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f5_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f5_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f5_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f5_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f5_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f5_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f5_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f6_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f6_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f6_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f6_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f6_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f6_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f6_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f6_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f6_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f6_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f7_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f7_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f7_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f7_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f7_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f7_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f7_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f7_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f7_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f7_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f8_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f8_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f8_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f8_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f8_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f8_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f8_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f8_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f8_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f8_s89_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f9_s11_h20` B_PERSISTENCE: mse=0.026514767059419457 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f9_s11_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f9_s131_h20` B_PERSISTENCE: mse=0.02428230094540822 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f9_s131_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f9_s23_h20` B_PERSISTENCE: mse=0.02501247142386814 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f9_s23_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f9_s47_h20` B_PERSISTENCE: mse=0.030324620624112907 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f9_s47_h20)
- `full_B_PERSISTENCE_default_synthetic_track_c_f9_s89_h20` B_PERSISTENCE: mse=0.02671762324227333 (evidence_run_id=full_B_PERSISTENCE_default_synthetic_track_c_f9_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s11_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s131_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s23_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s47_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f0_s89_h20` B_RIDGE: mse=1.909358074561327e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f0_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f10_s11_h20` B_RIDGE: mse=4.920235249229206e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f10_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f10_s131_h20` B_RIDGE: mse=4.920248581614878e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f10_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f10_s23_h20` B_RIDGE: mse=4.920105800460092e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f10_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f10_s47_h20` B_RIDGE: mse=4.920296430538785e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f10_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f10_s89_h20` B_RIDGE: mse=4.92030151077126e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f10_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f11_s11_h20` B_RIDGE: mse=4.920109693797901e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f11_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f11_s131_h20` B_RIDGE: mse=4.920149643684858e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f11_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f11_s23_h20` B_RIDGE: mse=4.9202212426477275e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f11_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f11_s47_h20` B_RIDGE: mse=4.9200831329307816e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f11_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f11_s89_h20` B_RIDGE: mse=4.920198252222792e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f11_s89_h20)
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
- `full_B_RIDGE_default_etf_track_a_f4_s11_h20` B_RIDGE: mse=4.9201980353420705e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f4_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f4_s131_h20` B_RIDGE: mse=4.919995320279841e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f4_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f4_s23_h20` B_RIDGE: mse=4.920331682841211e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f4_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f4_s47_h20` B_RIDGE: mse=4.920177849157888e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f4_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f4_s89_h20` B_RIDGE: mse=4.920177849157888e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f4_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f5_s11_h20` B_RIDGE: mse=4.92023379145804e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f5_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f5_s131_h20` B_RIDGE: mse=4.920107782410918e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f5_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f5_s23_h20` B_RIDGE: mse=4.920271745335682e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f5_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f5_s47_h20` B_RIDGE: mse=4.920200864356417e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f5_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f5_s89_h20` B_RIDGE: mse=4.920246249170338e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f5_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f6_s11_h20` B_RIDGE: mse=4.92033811446926e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f6_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f6_s131_h20` B_RIDGE: mse=4.920131262946192e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f6_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f6_s23_h20` B_RIDGE: mse=4.920238247851895e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f6_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f6_s47_h20` B_RIDGE: mse=4.920238247851895e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f6_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f6_s89_h20` B_RIDGE: mse=4.920154085919035e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f6_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f7_s11_h20` B_RIDGE: mse=4.920381301241223e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f7_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f7_s131_h20` B_RIDGE: mse=4.920209774740268e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f7_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f7_s23_h20` B_RIDGE: mse=4.920231933238657e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f7_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f7_s47_h20` B_RIDGE: mse=4.920188856488354e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f7_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f7_s89_h20` B_RIDGE: mse=4.920188856488354e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f7_s89_h20)
- `full_B_RIDGE_default_etf_track_a_f8_s11_h20` B_RIDGE: mse=4.920208421521973e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f8_s11_h20)
- `full_B_RIDGE_default_etf_track_a_f8_s131_h20` B_RIDGE: mse=4.920068123685912e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f8_s131_h20)
- `full_B_RIDGE_default_etf_track_a_f8_s23_h20` B_RIDGE: mse=4.920207804995989e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f8_s23_h20)
- `full_B_RIDGE_default_etf_track_a_f8_s47_h20` B_RIDGE: mse=4.920215233598673e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f8_s47_h20)
- `full_B_RIDGE_default_etf_track_a_f8_s89_h20` B_RIDGE: mse=4.9204162362765795e-06 (evidence_run_id=full_B_RIDGE_default_etf_track_a_f8_s89_h20)
