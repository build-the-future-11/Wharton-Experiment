# FREM S10 local design review

15 October 2026. Review of the written design by the same assistant that authored it. This is not independent verification or a market test.

## Resolved during specification review

1. Continuous NLL is undefined for a finite atomic forecast at a generic continuous outcome. The design explicitly separates the new ES hypothesis from the unchanged original 0.01-nat criterion; no score conversion is allowed.
2. A mixture must be optimized using cross-expert score terms, not weighted standalone losses. The specification defines A/B coefficients and a constrained convex solve.
3. Local concentration is not effective independent sample size. n_eff is labeled a weight-concentration statistic; statistical resampling uses date blocks.
4. A 32nd leave-one-out neighbor is undefined with only 32 records. The local fallback now explicitly requires at least 33 record contexts.
5. Liquidation assumptions can charge imaginary fees repeatedly. The objective reserves a hypothetical common exit cost; the actual ledger charges only actual trades and the final exit.
6. Robustness on historical atoms does not cover unobserved crashes. The contamination set and finite-support limitation are explicit.
7. A holdings-aware pipeline has many deterministic updates, not two model fits. The budget counts issue/update origins, solves and ledgers, with measured profiling required.
8. Two chronological folds can share historical information through causal adaptation. They are not labeled independent histories; bootstrap interpretation is bounded.
9. Policy-only ablations do not change forecasts. Their zero ES differences remain explicit with p=1 in the complete secondary family.
10. Novelty overlaps existing stacking and prescriptive optimization. S10 has a narrow empirical reason to exist; algorithm novelty remains unestablished.

## Mathematical checks at design level

- Probability mixtures preserve nonnegativity and normalization.
- Finite scenarios yield finite moments and PSD covariance; they do not establish true-market moments.
- With c<1 and sum(w)<=.8, the fee-equation residual is strictly increasing: its increment is at least (1-c sum(w)) times the q increment. Endpoint signs bracket a unique feasible root.
- The energy-score quadratic is convex on the simplex tangent space; positive L2 penalty makes the mixture optimum unique within the declared finite-dimensional problem.
- The contamination robust objective equals the minimum over the stated mixture ambiguity set because its free component puts mass on an atom of minimum utility.
- Finite-grid enumeration is globally optimal only within that grid, not among all continuous portfolio weights.

These are written derivations using established mathematics. They are not executable numerical verification. The implementation plan requires independent reference checks, tolerance tests and adversarial examples.

## Remaining objections

The most serious risks are absent conditional signal; sparse/dependent history; finite return support; a stale burn-in scale; overconservative utility; idealized fills and dividend availability; unproven compute feasibility; and no admitted unused confirmation set. The strongest null is that global or equal mixing with a simple portfolio control is sufficient.

The design is coherent enough to review and implement only after its stated idea/data/precision gates are satisfied. Neither the number of documents, proposed controls, nor the mathematical identities makes it a perfect or proven financial model.
