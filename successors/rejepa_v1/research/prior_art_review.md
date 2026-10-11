# Checkpoint 1 — idea review and prior-art boundary

Status: **IDEA READY as a falsifiable research proposal; novelty unestablished.**

## Claim that may be tested

When the training history is deliberately reduced, a causal reservoir-JEPA with
both (1) learned stochastic latent rollouts and (2) outcome-matured,
Fourier-keyed episodic retrieval will improve date-level Student-t return NLL by
more than 0.01 nat per observation relative to the identical backbone with both
mechanisms disabled. The comparison uses identical observations, dates, seeds,
tuning opportunities and bounded compute. Calibration must not materially
degrade.

This is an empirical hypothesis, not a claim that simulation creates missing
facts. The rollouts express model uncertainty conditional on observed history.

## Nearest prior art

| Component | Closest established idea | Boundary for RE-JEPA |
|---|---|---|
| Latent prediction | I-JEPA predicts target representations from context representations and uses an EMA target encoder. | RE-JEPA applies a future-window latent target to causal financial sequences; this transfer is not itself novel. |
| Time-series JEPA | TS-JEPA adapts joint-embedding prediction to time series and evaluates forecasting; a 2026 cross-backbone study reports sharply architecture-dependent gains and degradations. | The claimed value must come from the coupled worlds/engram mechanisms under low-data controls, not from the word JEPA. |
| Imagined futures | DreamerV3 learns a world model and improves behavior using imagined futures. | RE-JEPA samples latent forecast paths but does not call them calibrated economic scenarios until coverage tests pass. |
| Reservoir dynamics | Echo-state/reservoir computing uses a fixed recurrent dynamical system with a trained readout. | RE-JEPA uses a contractive fixed reservoir as one context feature generator. |
| Frequency features | FEDformer uses Fourier-domain structure for time-series forecasting. | RE-JEPA uses causal trailing Fourier-band energy as a retrieval key component, not a claim of inventing frequency-domain forecasting. |
| Episodic retrieval | Neural Episodic Control uses a differentiable lookup over stored experience. | RE-JEPA makes records immutable and unreadable until their outcomes were available, then learns only the current query/fusion path. |
| Retrieval-augmented forecasting | RAF and RAFT retrieve similar historical segments and their continuations to augment time-series forecasts. | Retrieval itself is established prior art. RE-JEPA may only test whether outcome-maturity gating and utility-ranked retention add value beyond comparable causal retrieval controls. |
| Streaming residual memory | FreshCast continuously refreshes a non-parametric memory and calibrates its residual correction from validation error alignment. | Freshness and residual alignment are mandatory contemporary controls. RE-JEPA cannot claim either online memory refresh or residual-aware weighting as novel. |
| Multiple predictive models | Deep ensembles use independently trained predictors for uncertainty and robustness. | RE-JEPA's worlds share a transition model; they are Monte Carlo paths, not independent ensemble members. A true ensemble remains a robustness baseline. |
| Decision-aware prediction | Decision-focused learning tunes predictors for downstream decision quality. | Phase 1 keeps forecast losses and allocation separate to avoid hiding forecast failure behind an optimizer. Decision-focused training is optional future work, not a current contribution. |
| Downside optimization | CVaR provides a coherent tail-risk objective and a tractable optimization formulation. | The protection head supplies downside estimates; portfolio claims still require realized CVaR/coverage evaluation and transaction-cost stress. |

## Primary sources

1. Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture,” 2023, <https://arxiv.org/abs/2301.08243>.
2. Ennadir, Golkar and Sarra, “Joint Embeddings Go Temporal,” 2025,
   <https://arxiv.org/abs/2509.25449>.
3. Feng et al., “Does Joint-Embedding Predictive Architecture Pretraining Help
   Time Series Forecasting?”, 2026, <https://arxiv.org/abs/2609.31680>.
4. Hafner et al., “Mastering Diverse Domains through World Models,” 2023,
   <https://arxiv.org/abs/2301.04104>.
5. Jaeger and Haas, “Harnessing Nonlinearity: Predicting Chaotic Systems and
   Saving Energy in Wireless Communication,” *Science* 2004,
   <https://doi.org/10.1126/science.1091277>.
6. Zhou et al., “FEDformer,” 2022,
   <https://arxiv.org/abs/2201.12740>.
7. Pritzel et al., “Neural Episodic Control,” 2017,
   <https://arxiv.org/abs/1703.01988>.
8. Lakshminarayanan, Pritzel and Blundell, “Simple and Scalable Predictive
   Uncertainty Estimation using Deep Ensembles,” 2017,
   <https://arxiv.org/abs/1612.01474>.
9. Amos and Kolter, “OptNet,” 2017,
   <https://arxiv.org/abs/1703.00443>.
10. Rockafellar and Uryasev, “Optimization of Conditional Value-at-Risk,” 2000,
    <https://doi.org/10.21314/JOR.2000.038>.
11. Zeng et al., “Are Transformers Effective for Time Series Forecasting?”,
    2022, <https://arxiv.org/abs/2205.13504>.
12. Nie et al., “A Time Series is Worth 64 Words: Long-term Forecasting with
    Transformers,” 2022, <https://arxiv.org/abs/2211.14730>.
13. Tire et al., “Retrieval Augmented Time Series Forecasting,” 2024,
    <https://arxiv.org/abs/2411.08249>.
14. Han et al., “Retrieval Augmented Time Series Forecasting,” ICML 2025,
    <https://arxiv.org/abs/2505.04163>.
15. He et al., “Retrieval Is Not Enough: Refreshing Memory for Frozen
    Time-Series Forecasters,” 2026, <https://arxiv.org/abs/2610.07834>.

Sources were checked on 2026-10-11. The FreshCast comparison materially narrows
the novelty boundary: retrieval, streaming refresh, and residual-alignment
calibration are not available as novelty claims. The candidate contribution is
therefore only the coupled, observability-safe test of matured utility retention,
latent-world conditioning, and JEPA under the stated financial chronology. This
review is not a patent search or an independent novelty determination.

## Fatal-flaw audit

| Potential fatal flaw | Why it could kill the idea | Required test or control |
|---|---|---|
| Synthetic worlds add no information | Repeated noisy transforms may only add variance or capacity. | Full model versus world-off, identical data and seeds; calibrated coverage and world diversity; parameter/compute disclosure. |
| Existing retrieval methods explain any memory gain | Similar-history continuation or a fresh residual correction may match the engram mechanism. | Add RAFT-style causal retrieval and FreshCast-style streaming residual calibration as development baselines before any freeze. |
| Engrams leak future outcomes | A record created or scored too early makes all results invalid. | Enforce `outcome_available_at <= as_of`; immutable detached records; adversarial timestamp tests and row-level audit. |
| Low-data framing is post hoc | Selecting a favorable scarcity regime would invalidate the central claim. | Freeze 10%, 25%, 50% chronological training fractions before protected access; primary conclusion aggregates the declared family. |
| Advantage is only extra parameters/compute | A larger candidate is not evidence for either mechanism. | Report trainable parameters, wall time and peak RSS; use an enlarged backbone control within ±5% parameters and a compute-capped sensitivity. |
| Gate chooses after seeing outcomes | Test-time interpretation selection would be direct overfitting. | Gate sees context only; checkpoint and seed policy frozen; state hash must be identical before and after inference. |
| Market panel is survivorship contaminated | Current constituents, adjusted-close backfills or missing delistings create false predictability. | Point-in-time universe, availability timestamps, corporate actions, delistings and FX lineage are admission gates. |
| Shared representation harms capital protection | Return gradients can erase downside signal. | Separate-head versus joint-head control; gradient-conflict diagnostics; downside Brier and expected-shortfall calibration. |
| Forecast gain has no client value | Small NLL gains may not improve the liability path after costs. | Treat client metrics as secondary; require cost stress and Laura-goal simulation without mixing WInS P&L into long-term assets. |

## Disconfirming results that end or narrow the project

- The paired 95% interval for control NLL minus full-model NLL does not clear
  0.01 nat/observation: central hypothesis unsupported.
- Its upper endpoint is at or below zero: negative result.
- Any material calibration degradation: forecasting claim rejected even if NLL
  improves.
- World-off or engram-off matches the full model within the practical margin:
  corresponding mechanism claim removed.
- Integrity, seed-completeness, provenance or compute-matching failure:
  experiment invalid, never relabeled positive.
- Only portfolio results improve while forecast endpoints fail: at most an
  optimizer finding, not evidence for RE-JEPA representation learning.

## Feasibility conclusion

The architecture is implementable on a small CPU/GPU budget and the hypothesis
is falsifiable. The largest risk is not coding; it is obtaining a legally usable,
point-in-time international total-return panel plus the team-only WInS universe.
That external dependency blocks empirical readiness but does not block a clean
Phase 1 implementation.
