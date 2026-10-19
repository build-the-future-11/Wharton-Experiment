# Bounded synthetic integrity study: M01–M12

**Date:** 19 October 2026. **Evidence class:** development engineering and constructed model characterization. **Source parent:** `fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c`. **Branch:** `research/wharton-integrity-20261019`.

## Question and conclusion

The bounded question was: **After repairing demonstrable split, policy-boundary, and optional-import defects, can the existing twelve model adapters produce finite, chronologically aligned, independently scored synthetic results, and what do their fixed baseline and ablation comparisons reveal?** A separate three-asset demonstration tested point-in-time admission, nested forecasting, and portfolio accounting together.

The engineering evidence supports these bounded checks: 288 matrix cells completed and passed artifact/score verification; a declared 24-cell subset replayed exactly; 156 retained secondary-control comparisons were verified. Model effectiveness remains mixed. Several gains against a named conditional comparator reverse against an unconditional control already saved by the runner. No model is promoted. Market effectiveness, architectural novelty, and completion of a production financial pipeline remain unestablished.

These M01–M12 results do not validate the separate FREM proposal.

## Defect witnesses and repairs

Three reproducible witnesses motivated source changes:

- **Split boundaries:** with 100 observations, purge five, and lockbox 98, the previous negative training stop produced actual training rows `[0:97]` and test rows `[2:100]`. Rows 2–96 overlapped. The repaired function rejects that invalid configuration before constructing folds, validates sizes/steps, and supports valid lockbox-only boundaries. This witness concerns an invalid configuration; it is not a claim that every historical split leaked. See [split regressions](../../tests/unit/test_split_boundaries.py).
- **Returned policy feasibility:** injected optimizer weights `[0.60000003, 0.40000002]` exceeded the unit budget despite optimizer tolerance. Simply normalizing `[0.6, 0.39999995, 0]` would breach a 0.6 asset cap. Repairs remove only permitted numerical tolerance noise, enforce the exact returned policy constraints, and score the returned allocation. Material infeasibility retains the verified finite-policy fallback. The witnesses inject optimizer results without running optimization; see [policy regressions](../../tests/unit/test_mpc_policy_contract.py).
- **Optional dependencies:** accounting and linear-model imports previously traversed Torch-dependent exports. A subprocess deliberately blocking Torch reproduced the failure. Lazy model exports now preserve those supported paths; see [import regression](../../tests/unit/test_optional_model_imports.py).

A new opt-in [point-in-time module](../../src/wharton_lab/data/point_in_time.py) admits explicit files with exact-byte hashes and declared provenance, retains observation vintages, and rejects unavailable, missing, conflicting, or stale required values. It does not silently recover an unknown legacy cache. Metadata admission validates supplied declarations; it does not independently certify a provider, licence, or source truth.

The repaired integrity suite recorded 58 passing tests plus 70 passing subtests; the model/tensor suite recorded 44 passes, three small numerical-gradient checks passed, the demonstration/journal suite recorded nine passes, and eight standard-library Kaggle driver contracts passed. The latter cover memory denial, failed children, timeouts, and incomplete archive publication; they do not establish remote execution. These gates are not exhaustive proofs. The original before-fix failures and the original demonstration recorder failure remain retained. Incremental attempt/model journals now preserve completed work when a later fit fails.

## Design, information sets, and units

The [fixed runner](../../scripts/run_portfolio_research.py) used twelve models, four worlds, three fresh matrix seeds (`2026101911`–`2026101913`), and full plus one predeclared ablation: 288 cells. A separate seed `2026101910` provided a 24-cell execution preflight. The matrix replay used only `null`, seed `2026101911`, and both variants. Computation used one CPU lane and one numerical thread, with explicit time/resource gates. No matrix hyperparameters were selected from these results.

Each [synthetic world](../../src/wharton_lab/research/causal_portfolio.py) contains five anonymous assets and 320 generated sessions. The cutoff is 240; training origins 20–235 keep five-session training targets mature before evaluation. Twenty test origins are scored. Returns are normalized using pre-cutoff RMS only. Features use earlier returns. M03 explicitly refits on expanding observed history at each origin; other adapters follow their recorded fixed-training procedures. The matrix has no confirmation or market holdout.

| Model / implemented adapter | Primary loss and target | Named comparator | Ablation |
|---|---|---|---|
| M01 regime quantile boosting | Mean pinball, normalized scalar return | Ridge calibration quantiles | No regimes |
| M02 residual ranker | Negative Spearman, five-asset residual ranks | Residual Ridge ranker | Raw training labels |
| M03 nonlinear factor loadings | MSE, five normalized returns | PCA/AR1 linear loadings | Single-snapshot loading target |
| M04 episodic residual correction | MSE, normalized scalar return | Information-matched Ridge | No episodic memory |
| M05 mixed expert prototype | MSE, normalized scalar return | Information-matched Ridge | Linear experts only |
| M06 predictive latent prototype | MSE, normalized scalar return | Ridge on the same windows | Predictive loss off |
| M07 conditional SPD covariance | Frobenius error, future realized covariance | Trailing-20 sample covariance | No context |
| M08 supervised graph forecasting | MSE, five normalized returns | Ridge on the same windows | No edges |
| M09 fixed fusion / weighted Ridge | MSE, normalized scalar return | Information-matched Ridge | Fixed group weights |
| M10 affine conditional flow | Energy score, normalized 25-dimensional path | Conditional Ridge residual bootstrap | No context |
| M11 finite policy bank | CVaR95, unpaid artificial currency | Cash only | Fully invested |
| M12 composed forecasting stack | MSE, normalized scalar return | Information-matched Ridge | No memory |

The independent unit is a simulation seed within one world. Summaries retain all three seed effects, means, ranges, and descriptive 95% Student-t intervals with two degrees of freedom. Dates, assets, sampled paths, and replay do not increase the independent seed count. No p-values, discovery declarations, or aggregate loss ranking across tasks are reported.

## Primary and ablation comparisons

Each cell below is **mean full-minus-named-comparator / mean full-minus-ablation**. Negative favors the full model for that comparison. Values are generated from [the complete comparison CSV](matrix_paired_summary.csv), rounded to four decimals; [individual paired seeds](matrix_paired_seeds.csv) retain precision. Units follow the preceding table and differ across rows.

| Model | Null | Linear signal | Regime shift | Heavy tails |
|---|---:|---:|---:|---:|
| M01 | +0.0017 / -0.0004 | +0.0041 / -0.0007 | -0.1249 / +0.0018 | -0.0097 / -0.0011 |
| M02 | -0.0450 / -0.0333 | -0.0300 / -0.0483 | +0.0567 / +0.0517 | +0.0050 / +0.0233 |
| M03 | -0.0011 / +0.0110 | +0.1198 / +0.3065 | -0.5289 / -0.5074 | +0.0285 / +0.0484 |
| M04 | +0.1114 / +0.0832 | +0.0400 / +0.0214 | -0.0744 / -0.1693 | +0.1480 / +0.1543 |
| M05 | -0.0053 / -0.0053 | +0.0868 / -0.0044 | -1.4230 / -0.0502 | +0.0300 / -0.0031 |
| M06 | +0.2623 / -0.1635 | +0.0819 / -0.0211 | -0.0972 / +0.0767 | -0.0807 / -0.0032 |
| M07 | +0.3324 / +0.6575 | -0.3280 / +0.4103 | +1.0601 / -0.1450 | +0.0116 / +0.9452 |
| M08 | -0.1248 / +0.0133 | +0.1157 / +0.0946 | -2.3220 / -0.4337 | -0.3832 / +0.0031 |
| M09 | -0.0163 / -0.0056 | +0.0738 / -0.0108 | -1.0964 / +0.1542 | +0.0786 / -0.0027 |
| M10 | -0.0031 / +0.0259 | -0.0174 / -0.1165 | -0.0940 / +0.6013 | -0.0516 / +0.0432 |
| M11 | 0 / -0.0139 | 0 / -0.0313 | 0 / -0.0287 | 0 / -0.0034 |
| M12 | +0.0523 / -0.0070 | +0.3094 / -0.0186 | +0.0081 / +0.0846 | +0.0882 / +0.0223 |

## Secondary controls change the interpretation

The first analysis omitted simpler controls already retained in raw artifacts. A reporting amendment added them without changing or refitting models, seeds, data, hyperparameters, endpoints, or ablations; the original analysis remains archived. The corrected audit verified 156 seed/control rows and 52 model/world/control summaries. Full results, including unfavorable comparisons, are in [secondary seed values](secondary_baseline_seeds.csv) and [secondary summaries](secondary_baseline_summary.csv).

The selected regime-shift comparisons below show why beating the named comparator is insufficient. Every listed full model loses to its stored secondary control on all three seeds, including M03's expanding historical mean.

| Model | Stored control | Full mean | Control mean | Full minus control | Full better seeds |
|---|---|---:|---:|---:|---:|
| M01 | Empirical training quantiles | 0.7104 | 0.6446 | +0.0658 | 0/3 |
| M03 | Expanding historical mean | 5.7503 | 5.5979 | +0.1524 | 0/3 |
| M05 | Training historical mean | 5.2780 | 4.9568 | +0.3212 | 0/3 |
| M08 | Training historical mean | 6.2004 | 5.5890 | +0.6113 | 0/3 |
| M09 | Training historical mean | 5.6045 | 4.9568 | +0.6477 | 0/3 |
| M10 | Unconditional training bootstrap | 9.4087 | 8.9834 | +0.4253 | 0/3 |

M01's nominal 90% interval coverage under this shift is only 55%, 35%, and 60%; improved pinball relative to Ridge therefore cannot be described as calibrated uncertainty. M07 also loses to training-global covariance on all three null and heavy-tail seeds. M11's equal-weight control is audited by independently reducing its saved path-loss vector to CVaR, with that narrower scope explicit; the secondary audit does not independently reconstruct that control ledger.

## Component observations

M03's historical-OLS loading target beats both its PCA comparator and legacy target on all three shift seeds, although the simpler expanding mean remains better. M02 residualization, M06 predictive loss, and M09 reweighting each improve on their ablation on all three linear-signal seeds. M05, M08, and M09 also beat historical mean on all three linear-signal seeds. These observations support identifying mechanisms for further falsification, not general superiority over Ridge.

M04 memory worsens heavy-tail loss on all three seeds. M07 conditioning loses to its no-context ablation on all nine null, linear-signal, and heavy-tail cells. M08 edges worsen null and linear-signal errors on all three seeds but improve on no-edges under the compound shift; historical mean still wins there. M09 group reweighting, M10 conditioning, and M12 memory each worsen their shift result on all three seeds.

M11's full endpoint equals cash-only in every world/seed cell. M11 and M12 report zero first risky allocation and unpaid CVaR approximately 0.03 throughout. Retaining cash under the artificial obligation schedule explains the contrast with forced investment; these results do not demonstrate useful learned allocation or global optimization. M12 is a composed stack, not a jointly trained architecture.

## Point-in-time forecasting and accounting demonstration

The separate [development demonstration](../../src/wharton_lab/research/development_demo.py) uses seed `2026101908`, three anonymous assets, and 160 artificial daily sessions. Decisions occur at 08:55, execution at 09:00, return events at 16:00, and publication at 16:05 UTC. Features retain five prior-session vintages. Labels are next-session open-to-close simple returns; overnight changes and cash interest are zero by construction.

Four chronological outer folds use minimum training 64, test size/step 20, purge one, and no lockbox. Nested chronological Ridge selection considers only alphas 0.1, 1, and 10 with train-only scaling and label-maturity checks. The 80 scored sessions yield 960 forecasts and 480 ledger rows. Each forecast arm allocates 0.4 to each of its top two assets, including negative forecasts, with deterministic ties and 0.2 cash. Equal weight and cash-only controls are separate.

Commission and slippage total 0.001 times gross traded notional, charged once. Actual drifted positions carry forward. Initial wealth is one artificial unit; endpoints mark holdings without terminal liquidation. The zero forecast's portfolio is the fixed tie allocation, not cash.

| Forecast / policy | Forecast MSE | Net return | Maximum drawdown | Total cost | Gross turnover |
|---|---:|---:|---:|---:|---:|
| Ridge | 8.9442e-05 | -7.94% | 10.52% | 0.02416 | 25.028 |
| Zero forecast | 8.6461e-05 | -4.36% | 6.11% | 0.00112 | 1.124 |
| Historical mean | 8.7602e-05 | -2.95% | 7.42% | 0.00187 | 1.892 |
| Trailing-five momentum | 1.0796e-04 | -4.79% | 8.09% | 0.02055 | 21.009 |
| Equal weight | — | -3.08% | 5.38% | 0.00116 | 1.164 |
| Cash only | — | 0.00% | 0.00% | 0.00000 | 0.000 |

[Forecast errors](demo_forecast_metrics.csv) and [portfolio metrics](demo_portfolio_metrics.csv) show zero forecasting has lower MSE than Ridge. Ridge also trades substantially more and loses more wealth than the passive controls. Costs are cumulative payments in initial-wealth units; turnover sums each period's traded notional divided by pre-trade wealth. Cost totals alone are not a counterfactual attribution of the return difference.

An independent reconstruction of post-cost wealth, positions, trading, returns, and drawdown agrees with the CSV to at most approximately 8.66e-15; the maximum accounting identity residual is approximately 4.44e-16. Demonstration v2's deterministic artifacts and journals replay identically. These checks establish accounting consistency on this fixture, not investment efficacy.

## Limits and one successor falsifier

Three seeds and twenty scored origins provide limited evidence about robustness. Five-session covariance/path labels overlap, and M11 has only four nonoverlapping held-out paths per cell; its empirical CVaR95 is their maximum. M07's five-observation covariance target for five assets is intrinsically noisy. Positive definiteness is enforced by parameterization and is not evidence of forecast skill.

The regime shift simultaneously changes autoregression from 0.5 to −0.3 and innovation scale from 0.008 to 0.022. Comparator failures can therefore reflect wrong-sign extrapolation, volatility sensitivity, or insufficient shrinkage. Some ablations change effective capacity or regularization; M05 also changes expert composition/count. Its single training batch does not establish expert lifecycle benefits. M07/M08 supervised objectives do not establish JEPA pretraining benefits. Synthetic results do not establish market provenance, trading suitability, novelty, or independent reproduction.

**Proposed, not executed:** isolate M08's claimed graph information. Freeze network size, initialization, training budget, and a fresh seed list. Compare the learned training-correlation graph with uniform off-diagonal mixing and no edges, retaining historical mean. Separate autoregressive-sign and volatility changes in a predeclared 2×2 design. The mechanism contrast is learned-graph loss minus uniform-mixing loss; forecasting usefulness also requires improvement on historical mean. Equality with uniform mixing would undermine a graph-specific explanation. Failure against historical mean would reject forecasting value even if the graph beats no-edges. Register the budget and analysis before generating new outcomes.

## Claim-to-artifact map

Linked CSVs are public curated exports. Entries below refer to the private evidence archive under `research/development/integrity_20261019/`, not public download URLs.

| Claim | Public source or table | Private archive entry |
|---|---|---|
| Boundary defects and repaired contracts | [Split](../../tests/unit/test_split_boundaries.py), [policy](../../tests/unit/test_mpc_policy_contract.py), [import](../../tests/unit/test_optional_model_imports.py) tests | `before_fix/`, `patched_integrity/` |
| Bounded numerical checks | [Gradient tests](../../tests/unit/test_numerical_gradients.py) | `model_tensor_regressions/`, `finite_difference_gradients/` |
| All 288 primary cells re-scored | [Runner](../../scripts/run_portfolio_research.py), [verifier](../../scripts/verify_portfolio_artifacts.py), [paired seeds](matrix_paired_seeds.csv) | `matrix_<world>/artifacts/`, `verify_matrix_<world>/verification.json` |
| Exact 24-cell replay | [Analysis script](../../scripts/analyze_integrity_development.py) | `matrix_replay/artifacts/`, `analysis_v2/analysis.json` |
| Complete stored secondary controls | [Secondary summaries](secondary_baseline_summary.csv) | `reporting_amendment.json`, `analysis_v2/secondary_baseline_analysis.json` |
| Demonstration accounting and provenance | [Demo tests](../../tests/unit/test_development_demo.py), [PIT tests](../../tests/unit/test_point_in_time.py), [portfolio CSV](demo_portfolio_metrics.csv) | `demo_v2_primary/artifacts/`, `demo_v2_replay/artifacts/` |

The separate verifier's largest score discrepancy is 1.78e-15. Its M11 primary re-score reuses the shared ledger; demonstration accounting uses an independent reconstruction. Source, data, configuration, prediction, failure, environment, and timing receipts are archived. Recorded matrix source fingerprint: `8369f937581a33d013e34815bdb663f0669ccac0e1268a926e641d3ec7318811`.

## Final delivery identity and checkpoint scope

Tested source commit: `5081e37dadfc9cbb0ae5e06c46385ba89c05e50c`. See [DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json) for the pinned notebook and private evidence archive identity. The original multi-seed runner preserves a representative checkpoint for each model/variant at the first seed of each invocation; every cell retains raw predictions, targets and dataset/configuration identity, but not every fitted seed state was checkpointed. The supplied Kaggle driver invokes each seed separately. The first source-change manifest is a snapshot at this source commit; later delivery documentation does not change the frozen model source.
