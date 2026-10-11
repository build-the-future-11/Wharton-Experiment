# Mathematical specification

The complete prospective specification is maintained in
`paper/prospective_specification.md`. The executable mapping is:

| Mathematical object | Implementation |
|---|---|
| Contractive state `h_t` | `StableReservoir` |
| Past-only spectral state `s_t` | `CausalFourierBank`, `spectral_diagnostics` |
| Context/EMA target `z_t, z*_t` | `REJEPA.context_encoder`, `target_encoder` |
| Horizon condition | `REJEPA.horizon_embedding` |
| Counterfactual worlds | `LatentWorldModel`, `CounterfactualLatentSimulator` |
| Interpretation mixture | `LatentWorldModel.interpreters`, context-only gate |
| Mature associative memory | `SpectralEngramMemory` |
| Temporal relations | `EngramKnowledgeGraph` |
| Return distribution | Student-t location and positive scale |
| Protection outputs | loss probability, downside magnitude, volatility, downside quantile, drawdown exposure |
| Client decision | `optimize_scenario_cvar` under explicit constraints |

Latent worlds are conditional samples, not independent history. Memory writes
are discrete, immutable and outside backpropagation; gradients pass through the
current query and fusion only. The downside quantile is a declared conservative
location-scale transform and must be calibrated before use.
