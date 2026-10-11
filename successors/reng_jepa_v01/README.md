# RENG-JEPA v0.1 executable research package

This directory is an additive successor lineage implementing the 11 October 2026
pre-results manuscript, **RENG-JEPA: Evidence-Anchored Latent Simulation with
Spectral Engram Memory for Liability-Aware Global Investing**. It does not modify
or reinterpret the repository-root FREM study or `successors/rejepa_v1`.

Scientific status: **NOT EVALUATED**.

Implemented contracts:

- point-in-time panel validation, fold-local universe selection, robust 18-channel
  features, execution-aligned opening returns, and purged annual folds;
- 32-dimensional GRU/EMA JEPA encoders, three causal spectral bands, three fixed
  contractive reservoirs, signed graph context, and horizon embeddings;
- horizon-matured immutable engrams with differentiable spectral retrieval;
- three stochastic latent interpreters, market-level mixture gates, low-rank
  correlated Gaussian return laws, analytic marginal quantiles, and a separate
  growth head;
- Stage A and Stage B objectives, EMA updates, clipping, deterministic schedules,
  calibration search, scenario sampling, and frozen inference;
- date-level normalized quantile score, fold-preserving stationary bootstrap,
  Holm correction, calibration/mechanism diagnostics, and decision rules;
- scenario-CVaR allocation, exact post-fee wealth accounting, explicit
  infeasibility, international-return conversion, and separate long-horizon
  liability simulations;
- immutable experiment/failure receipts and a bounded engineering dataset matrix.

The included engineering datasets are deterministic synthetic fixtures named
`calm_trend`, `mean_reversion`, `correlation_break`, and `missing_actions`. They
test integration and failure behavior only. They are not market evidence and are
never counted as independent observations for a scientific claim.

Run the package tests from the repository root:

```bash
PYTHONPATH=successors/reng_jepa_v01/src .venv/bin/python -m pytest -q successors/reng_jepa_v01/tests
```

Run the bounded multi-dataset compatibility matrix:

```bash
PYTHONPATH=successors/reng_jepa_v01/src .venv/bin/python -m reng engineering-matrix \
  --output successors/reng_jepa_v01/runs/engineering_compatibility
```

Current retained verification: 43 tests passed with zero warnings. The latest
source-bound matrix is
`runs/engineering_compatibility_20261011_v4/SUMMARY.json`; all four manifests
set `dataset_synthetic=true` and `scientific_claims_allowed=false`. The random,
untrained model triggered the declared representation-collapse diagnostic on
all four fixtures, which is retained as a negative engineering diagnostic.

No command acquires market data, opens the protected period, places an order,
or creates a competition submission. Scientific execution requires an admitted
dataset manifest, frozen dates, verified competition rules when relevant, and a
separate resource admission.

