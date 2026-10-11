# RE-JEPA v0.3: Reservoir-Engram Joint-Embedding Predictive Architecture

RE-JEPA is a clean successor research lineage for the 2026-2027 Wharton Global
High School Investment Competition. It does not overwrite or reinterpret the
earlier Wharton lab or FREM evidence.

Scientific status: **NOT EVALUATED**.

The complete local research system has these layers:

1. A fixed, contractive reservoir turns a causal market history into a fading
   nonlinear state.
2. A causal Fourier bank supplies multi-band spectral descriptors.
3. A delayed-credit engram store retrieves only episodes whose outcomes were
   already available at the decision time.
4. A JEPA predictor learns future representations, while a stochastic latent
   world model creates several possible future latent trajectories.
5. Separate growth and capital-protection experts produce expected-return and
   downside signals for a liability-aware international allocation layer.
6. A point-in-time data validator writes hash-bound deterministic snapshots.
7. A funding engine allocates dynamically among FUND, GUARD and GROW sleeves.
8. Purged walk-forward evaluation, simple baselines, immutable experiment
   manifests and complete per-cell receipts enforce fair tests.
9. An exact empirical scenario-CVaR linear program applies cash, security,
   sector, country, currency,
   liquidity, turnover, loss-proxy and stale-data constraints.
10. The CLI runs validation, deterministic synthetic-data generation,
    baseline/model/simulator training, end-to-end engineering smoke execution,
    memory analysis, walk-forward planning, portfolio optimization and
    evidence-bound reporting.

The latent-world interpreter is learned during development and frozen before
protected evaluation. It cannot choose an interpretation after inspecting test
returns. The engram store is also causal: gradients can train how the current
query uses already-matured memories, but cannot propagate through unknown future
outcomes or rewrite historical records.

## Start here

- `RESEARCH_STATE.json`: canonical state for this successor.
- `research/protocol.json`: falsifiable scientific protocol.
- `research/checkpoint_ledger.json`: pipeline gate status and blockers.
- `research/implementation_ledger.json`: phase-by-phase engineering coverage.
- `research_state.yaml`: human-readable canonical state requested by the v0.2 directive.
- `research/prior_art_review.md`: primary-source prior art, fatal flaws, and falsifiers.
- `paper/prospective_specification.md`: equations, algorithms, complexity, and safety contract.
- `research/methodology.md`: splits, controls, endpoints, statistics, and freeze prerequisites.
- `docs/ARCHITECTURE.md`: exact mechanism and novelty boundary.
- `docs/CASE_CONTRACT.md`: Laura Gao cash flows and competition scope.
- `docs/DATA_CONTRACT.md`: international point-in-time data requirements.
- `src/rejepa/`: executable core implementation.
- `tests/`: mathematical, causal, serialization, and mandate checks.

## Current pipeline verdict

- Checkpoint 0 Initialize: pass.
- Checkpoint 1 Idea Ready: pass as a testable proposal; novelty is not established.
- Checkpoint 2 Model Specification Ready: pass as a prospective, no-results paper.
- Checkpoint 3 Methodology Ready: specification complete, formal gate blocked by the absent admitted dataset, actual split boundary, and team-only WInS rules.
- Checkpoint 4 Model Implemented: engineering complete, but the formal checkpoint cannot pass its upstream dependency until Checkpoint 3 is frozen.

The implemented system includes deterministic training, validation-only early stopping,
gradient clipping, EMA targets, executable mechanism controls, frozen inference,
immutable memory serialization, atomic hash-bound checkpoints, and fail-closed
numerical checks. Phase 2 experiment manifests and receipts are implemented,
but no scientific market experiment or protected evaluation has been run.

## CLI

```bash
PYTHONPATH=successors/rejepa_v1/src .venv/bin/python -m rejepa --help
```

Commands: `data-validate`, `baseline-train`, `jepa-train`, `simulator-train`,
`memory-experiment`, `full-train`, `walk-forward`, `portfolio-optimize`, and
`report-generate`, plus `synthetic-generate` and `engineering-smoke`. Synthetic
inputs are permitted only for labeled engineering compatibility checks.

## Bounded compatibility check

From the repository root, using the existing verified Python environment:

```bash
PYTHONPATH=successors/rejepa_v1/src .venv/bin/python -m pytest -q successors/rejepa_v1/tests
```

Generate and execute the bounded local fixture without accessing market or
protected data:

```bash
PYTHONPATH=successors/rejepa_v1/src .venv/bin/python -m rejepa synthetic-generate successors/rejepa_v1/data/engineering/v0_3
PYTHONPATH=successors/rejepa_v1/src .venv/bin/python -m rejepa engineering-smoke successors/rejepa_v1/data/engineering/v0_3/training.npz successors/rejepa_v1/runs/engineering_v0_3
```

The verified v0.3 suite contains 67 tests. These commands are
engineering evidence only. No market-data experiment, portfolio recommendation,
WInS trade, or competition submission is produced.

