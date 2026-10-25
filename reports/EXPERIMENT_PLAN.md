# Frozen runnable matrix and broader requirements

The machine registry is `kaggle/experiment_manifest.json`; every row belongs to
existing S14 entrypoints. No new scientific trial was introduced in this pass.

| Stage | Cells / data | Seeds / budget | Entrypoint and status |
|---|---|---|---|
| Local tests | 109 tests, generated fixtures | 108 pass, 1 CUDA-only skip; 17.29 s observed | pytest; executed |
| Generated target qualification | 19 arms; generated engineering data | seed 11; two epochs; 1,800 s inner ceiling | frem_s14 qualify; CUDA pending |
| Historical matrix | 19 arms × 3 folds × 3 seeds = 171; seven exact ETFs | 11/29/47; 10 epochs, patience 3; 4,000 s/fold | frem_s14.suite; pending |
| Independent verification | All complete families; no seed exclusions | 20-date blocks, 2,000 bootstrap replicates; full-family Holm | verify-suite; pending |
| Confirmation | Protected 2025 onward | No authorized S14 confirmatory entrypoint | NOT IMPLEMENTED / boundary closed |

Arms: full, no_memory, no_jepa, no_neurogenesis, no_reservoir, no_spectral,
uniform_router, no_risk, zero, ridge, xgboost, hybrid_xgboost,
fixed_full_capacity, no_ngram, gru, no_spde, no_multihorizon, no_graph, no_mask.
The full model combines the existing reservoir, episodic graph, conditional
Gaussian field, masked JEPA, bounded growth and three objective routers.

Data: SPY/IWM/DIA/TLT/IEF/GLD/SLV; start 2007; prefixes end exclusively before
2011/2015/2018. Nine causal features, 20-observation compounded targets,
60/20/20 chronological splits, 20-observation embargo and maturity purge.
Training-only preprocessing; no source-calendar intersection or imputation.
Seeds measure optimization variation, not independent financial histories.
The proper-score practical margin is 0.01 nats. Proxy portfolios use 1x/2x/4x
cost sensitivity but do not establish tradable profitability.

The notebook ceiling remains 13,500 seconds (3 h 45 min), inclusive of tests and
qualification. Three 4,000-second ceilings cannot all be treated as guaranteed
allocations inside this cap. The historical-suite subprocess ceiling is 11,200
seconds; remaining notebook time can reduce it. No measured current-target ETA,
quota availability or success guarantee exists. Earlier Version 8's budget
exhaustion is retained. No additional paid credits or 8–24-hour budget is assumed.

Mapping to requested E00–E16: E00 correctness; E01/E02 historical predecessor
receipts and corrected-original preservation; E03 zero/ridge; E04 XGBoost/GRU;
E05 full; E06 mechanism-off; E08 fixed-full-capacity (capacity control, not exact
trainable-parameter matching); E09 listed ablations; E14 retained runtime/RSS
and hardware metadata; E15 repeated development seeds. E07 strict matched
compute, E10 broad optimizer sensitivity, E11 harder datasets, E12 scaling,
E13 broad noise/OOD and E16 protected confirmation remain incomplete. Cost
sensitivity alone does not satisfy broad robustness. No SOTA claim or claim that
all requested experiment classes are implemented.

The existing final master checklist specifies future 24-cell S15 development,
210-cell core confirmation and 162-cell ablations. They remain proposed gated
successor work, not launched experiments or time estimates.
