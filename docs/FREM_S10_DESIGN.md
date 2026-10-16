# FREM S10 model design

15 October 2026. Design revision 1. **Design complete enough for technical review; no implementation, market run, scientific freeze, or performance claim.**

S10 will test whether delayed, context-dependent distribution mixing improves a horizon-matched joint forecast, and whether a holdings-aware decision rule turns that forecast into useful net portfolio outcomes. The proposed system uses a single distribution of price changes and cash dividends for prediction and decisions. It treats cash and keeping existing holdings as explicit alternatives.

A perfect market model cannot be guaranteed: a positive risky position can lose to cash, while cash can lose to a rising benchmark. The attainable design objective is a complete, falsifiable model whose accounting, chronology and numerical properties can be verified and whose performance can fail an honest test.

The recommended architecture is:

    dated observations and availability
        -> matured four-interval price and dividend scenarios
        -> global, recent and contextual empirical experts
        -> delayed mixture-level scoring and shrinkage
        -> joint predictive scenario distribution
        -> holdings-aware utility with explicit costs and optional robustness
        -> trade or retain holdings
        -> immutable forecast, maturity, decision and accounting records

## Design decisions

| Decision | Reason | Required evidence |
|---|---|---|
| Forecast the actual four-interval holding horizon | S8 extended weekly moments into longer decisions | Paired direct/repeated-horizon ablation |
| Keep price gross returns and dividends separate | They have different fee and cash treatment | Exact accounting and actual source-event reconciliation |
| Use finite synchronized scenarios | Avoid undefined exponential moments and retain joint dependence | Proper-score, calibration and support diagnostics |
| Start with three transparent distribution experts | Memory cannot create signal absent from the expert set | Every expert, equal mixing and global stacking retained |
| Fit mixture quality, rather than reward experts separately | Complementary distributions can be useful despite weak standalone scores | Same-support global and contextual mixture controls |
| Shrink contextual weights toward global weights | Available conditional evidence is small and dependent | No-shrink and context-off comparisons |
| Retain holdings between decisions | Mandatory liquidation can create unnecessary turnover | Matched forced-liquidation ablation |
| Compare nominal and conservative decisions | Conservatism may prevent mistakes or reproduce S7's cash trap | Same forecasts, limits and costs; publish both results |
| Keep neural and spectral components outside revision 1 | Their useful increment has not been established | A later separately specified component experiment |

This is a deliberately bounded distribution-and-decision model. It is not an implementation of the original Fourier/JEPA mechanism. The FREM-S10 name preserves project lineage; it does not establish that the original hypothesis succeeded.

## Full design package

- [Model and mathematical specification](../research/frem_s10_20261015/MODEL_SPECIFICATION.md): data, equations, parameters, training, delayed updates, decisions and failure behavior.
- [Experiment protocol](../research/frem_s10_20261015/EXPERIMENT_PROTOCOL.md): 176 core ledgers, 16 fixed ablation ledgers, comparisons, uncertainty, budgets and stopping rules.
- [Failure and repair map](../research/frem_s10_20261015/FAILURE_REPAIR_MAP.md): predecessor evidence, proposed cause, severity, repair and discriminating evidence.
- [Prior art](../research/frem_s10_20261015/PRIOR_ART.md): established methods, closest overlap, contradictory considerations and novelty boundary.
- [Implementation and review plan](../research/frem_s10_20261015/IMPLEMENTATION_PLAN.md): modules, invariants, acceptance checks, release and unresolved gates.
- [Machine-readable design](../research/frem_s10_20261015/design.json): exact proposed configuration and explicit non-executable status.

## What counts as success

There are separate conclusions, with no substitution between them:

1. **Engineering correctness:** causality, numerical identities, checkpointing, controls and accounting pass.
2. **S10 distribution skill:** contextual mixing clears its new, explicitly separate proper-score comparison against both global stacking and equal mixing.
3. **Economic success:** the primary net-wealth comparison clears the retained 6% uplift objective against both equal-asset rebalancing and buy-and-hold, in both folds, and beats cash. Risk, exposure and uncertainty must also be reported.
4. **Original memory claim:** the existing 0.01-nat density criterion remains unmet. A finite scenario model has no continuous density NLL; S10 cannot satisfy that criterion by relabeling energy-score gains.
5. **Confirmation and practical use:** require admitted untouched evidence, independent reproduction and verified execution/mandate inputs.

## Readiness and sequence

The design selects an empirical successor question, specifies its mechanism, and gives it a reason to exist without claiming algorithmic novelty. The broader novel-method gate remains unresolved. The existing pre-2017 ETF histories are proposed for retrospective development only; S10-specific admission and precision checks remain pending.

Next: adversarial technical review of this design; data metadata/admission and precision feasibility; implementation of the exact reviewed specification; constructed end-to-end verification; a source/data/environment freeze; then one admitted, bounded development campaign. Confirmation needs its own feasible protocol and untouched data.

S9 is preserved, unconsumed and unrun. S10 does not assume S9 works, consume its freeze or reopen S6-S8. The canonical 2025+ protected boundary stays closed. The present request maps S10; it does not launch the proposed experiment.
