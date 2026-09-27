# Master Execution Queue — Wharton-Experiments (2026-09-27)

## P0 — Critical

| # | Task | Status |
|---|---|---|
| 1 | Synthetic Track C contemporaneous-latent leak (lockbox H1 synthetic invalid) | DONE — documented D-050, regression test, claims withdrawn |
| 2 | ETF expanding-std includes label; "h20" is really 1-step | DONE — documented D-051, structural causality test |
| 3 | Degenerate folds + identical ETF seeds (pseudo-replication) | DONE — documented D-052, test pins legacy behaviour |
| 4 | ETF target mislabelled (AGG, not SPY); synthetic fallback cached as "real" | DONE — loader fixed, `DATA_MANIFEST.yaml` (D-053) |
| 5 | Scoreboards/evidence table mixing 1916 orphan receipts | DONE — manifest-only builder, `HISTORICAL_ORPHANS.json` (D-055) |
| 6 | Source packages untracked (fresh clone incomplete); `.pyc` tracked | DONE — committed / untracked |
| 7 | Report, brief, and IPS draft asserting withdrawn claims | DONE — rewritten; PDFs regenerated from `scripts/build_pdfs.py` |
| 8 | Factorial look-ahead + unused forecasts | DONE — labelled INVALID (D-054); not re-run on lockbox seed |
| 9 | Secrets / unsafe execution scan | DONE — none found (local `pickle` checkpoints only) |

## P1 — Must finish

| # | Task | Status |
|---|---|---|
| 10 | Multiplicity-corrected legacy claims | DONE — 0/10 testable (D-057) |
| 11 | Repaired causal evaluation of the selected baseline | DONE — exploratory null (D-056) |
| 12 | Figure from real data + source CSV | DONE — `reports/figures/repaired_oos_r2.*` |
| 13 | Reconcile stale docs (`lean_matrix.yaml`, repro audit, gaps, README, checklist) | DONE |
| 14 | Client IPS FINAL | BLOCKED — `2026_WGY_*.pdf`, roster/wealth/obligations |
| 15 | Trading Notes FINAL | BLOCKED — 3 WInS executions |
| 16 | Submit / trade | BLOCKED — explicit authorization |

## P2 — High value (not done)

| # | Task | Status | Exact next step |
|---|---|---|---|
| 17 | M01–M12 on repaired pipeline (exploratory) | NOT RUN | Add a causal-panel adapter to `orchestration/executor.run_model` and call it from `evaluation/repaired.py`. CPU is fine; the neural models take minutes each |
| 18 | Repaired factorial | NOT RUN | Pass forecasts to the controllers, centre scenarios on information < t, use non-lockbox seeds and ≥ 200 paths, and report CIs |
| 19 | Version ETF cache or a prospective holdout | DECISION NEEDED | Either `git add -f` the 67 KB cache (Yahoo terms are your call) or preregister a fresh post-cache holdout |
| 20 | Confirmatory amendment | NOT RUN | New preregistration: causal loader, target by ticker, OOS R² vs mean, fresh holdout |
| 21 | Lint / type check | NOT RUN | No ruff/mypy configured; add them to the `dev` extra if wanted |
| 22 | Placeholder git author email | OPEN | `git config user.email <you>` (not changed automatically) |

## P3 — Optional
- Remove or archive the 1916 orphan receipts under `runs/` (kept for provenance; excluded from all tables).
- Pin exact dependency versions (lockfile).
