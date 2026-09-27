# Research Audit — Wharton-Experiments

**Audited:** 2026-09-27. **Ground truth:** code, `protocol/EXPERIMENT_MANIFEST.csv`, receipts, and the re-run artifacts listed below. Prior summaries were not trusted.

## Protocol inventory

| Item | Location | State |
|---|---|---|
| Seeds / horizons / costs | `protocol/DECISIONS.md` D-010..D-041 | FROZEN; unchanged |
| Profiles | `src/wharton_lab/orchestration/profiles.py` (authoritative); `configs/lean_matrix.yaml` (mirror, corrected) | unchanged |
| Lockbox freeze | `protocol/LOCKBOX_FREEZE.yaml` | FROZEN, opened once; not reopened |
| Legacy splits | `orchestration/datasets.py::_train_test` | unchanged; degenerate (D-052) |
| Legacy features | `orchestration/datasets.py::load_tabular` | unchanged; leaky (D-050, D-051) |
| Repaired path | `data/causal_tracks.py`, `backtesting/walk_forward.py`, `evaluation/repaired.py` | NEW (D-056) |
| ETF data | `data/raw/etf/…npy`, sha256 `6c234590…`; `data/manifests/DATA_MANIFEST.yaml` | not in git; columns AGG, EFA, GLD, SPY, VNQ (D-053) |
| Metrics | `evaluation/metrics.py`, `orchestration/executor.py` | MSE on 1-step label; "h20" nominal |

## Claims

| Claim | Evidence Exists? | Artifact | Reproducible? | Status |
|---|---|---|---|---|
| Twelve models implemented and unit-tested | Yes | `src/wharton_lab/models/`, 50 tests pass | Yes | VERIFIED |
| Lean matrix ran 1181/1181 cells | Yes | manifest + receipts | Yes (with the exact ETF cache) | VERIFIED (execution only) |
| Matrix provides fold/seed replication | No — 1 distinct ETF value, ≤ 5 synthetic, per model per 20 cells | manifest analysis; `tests/causal/test_dataset_leakage.py` | Yes | FAILED (D-052) |
| Selection of B_RIDGE reflects forecasting skill | No | pooled MSE dominated by leaky synthetic track | Yes | UNSUPPORTED |
| Lockbox H1 synthetic: ridge beats persistence | Yes, but leaky | `runs/lockbox/LOCKBOX_RESULTS.json` | Yes | FAILED — leakage (D-050) |
| Lockbox H1 ETF: ridge beats persistence | Yes | same | Yes | UNSUPPORTED (negative) |
| Lockbox H2: EWMA Frobenius error finite | Yes | same | Yes | VERIFIED (trivial) |
| Lockbox integrity (code frozen at fingerprint) | Partial | `LOCKBOX_FREEZE.yaml` | No commit matches | PARTIALLY VERIFIED (D-058) |
| Multiplicity-corrected legacy claims | Yes | `reports/evidence_bank/MULTIPLICITY_*` | Yes | 0/10 testable — no claim possible |
| Factorial: MPC > myopic; cost sensitivity | Yes, but invalid design | `runs/factorial/` | Yes | FAILED (D-054) |
| Repaired: ridge vs historical mean (primary) | Yes | `runs/repaired/REPAIRED_RESULTS.json` | Yes (`scripts/run_repaired_eval.py`) | VERIFIED NULL — no skill (exploratory) |
| Repaired: ridge beats 1-step persistence | Yes | same | Yes | VERIFIED but uninformative (weak benchmark) |
| Legacy synthetic features explain the lockbox win | Yes | leak-demo arm, OOS R² ≈ 99.5% | Yes | VERIFIED |
| M01–M12 vs baselines on valid data | No | — | — | NOT RUN |
| Distributional (M01 pinball) and covariance (M07) quality | Legacy only | manifest | — | UNSUPPORTED (leaky/degenerate data) |
| Client IPS / trading notes | No | drafts only | — | BLOCKED — WGY PDFs, WInS executions |

## Result generations (never mix without labels)

1. **historical / superseded** — 1916 orphan receipts under `runs/` (`reports/generated/HISTORICAL_ORPHANS.json`).
2. **legacy lean matrix — leakage-contaminated** — 1181 manifest cells.
3. **preregistered lockbox — single open** — synthetic arm invalid, ETF arm negative.
4. **invalid** — factorial.
5. **repaired pipeline — exploratory, post-lockbox** — `runs/repaired/`.

## What would make a confirmatory claim possible

A new preregistration (amendment) that fixes the causal loader, disjoint folds, target asset (by ticker, not column index), primary metric (OOS R² vs historical mean), and a **fresh** holdout never touched by any run: e.g. ETF data after the cache end date, collected prospectively. Not executed here, because it requires your authorization and new data.
