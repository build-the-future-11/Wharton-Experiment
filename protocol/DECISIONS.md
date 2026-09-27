# Protocol decisions

Labels: **RECOVERED** | **NEW_DEFAULT** | **FROZEN** | **BLOCKED**

## Workspace and recovery

| ID | Label | Decision |
|----|-------|----------|
| D-001 | RECOVERED | Workspace was empty; greenfield implementation under `src/wharton_lab/` on 2026-09-26. |
| D-002 | FROZEN | Do not rewrite model implementations M01–M12 without a new protocol amendment. |

## Seeds and profiles

| ID | Label | Decision |
|----|-------|----------|
| D-010 | FROZEN | Smoke seeds: `[11]`. |
| D-011 | FROZEN | Pilot seeds: `[11, 23, 47]`. |
| D-012 | FROZEN | Full seeds: `[11, 23, 47, 89, 131]`. |

## Horizons and windows

| ID | Label | Decision |
|----|-------|----------|
| D-020 | FROZEN | Forecast horizons 1 / 5 / 20; **primary horizon = 20**. |
| D-021 | FROZEN | Covariance windows 20 / 60. |
| D-022 | FROZEN | Sequence context length **128** (profiles carry `context_len`; smoke uses truncated steps). |

## Client and data

| ID | Label | Decision |
|----|-------|----------|
| D-030 | BLOCKED | Client PDFs `2026_WGY_*` **NOT FOUND** → BLOCKED for IPS / trading notes. |
| D-031 | BLOCKED | Track B equities without point-in-time fundamentals → **BLOCKED_DATA** for unbiased selection. |

## Costs and compute

| ID | Label | Decision |
|----|-------|----------|
| D-040 | FROZEN | Transaction cost research scenarios: 0 / 5 / 10 / 25 / 50 bps; **base = 10 bps**. |
| D-041 | FROZEN | Device: **CPU default**; **MPS optional** when torch available. |

## Lineage

| Component | Label | Notes |
|-----------|-------|-------|
| M04 APEN | RECOVERED | Synthica financial APEN; mechanism reimplemented. |
| M05 Q-APEN pending queue | NEW_DEFAULT | Extension beyond recovered World-Series causal reference. |
| M06 FI-JEPA | RECOVERED | GitHub-Every-Repo lineage; lab torch reimplementation. |
| M07 Eigen-JEPA | RECOVERED | GitHub-Every-Repo lineage. |
| M08 Graph JEPA | NEW_DEFAULT | No upstream; graph JEPA NEW_DEFAULT in this repo. |
| M12 LAF-GMJEPA | NEW_DEFAULT | Composed stack; ablation surface NEW_DEFAULT. |

## 2026-09-26 FAST_MATRIX
- NEW_DEFAULT: lean folds/steps (pilot 2 folds/96 steps; overnight 3/128; full 4/160).
- NEW_DEFAULT: M01 regime-as-feature; reduced estimators/epochs; ETF memo; manifest flush every 25.
- Prior 3071-cell receipts deleted as redundant vs new lean matrix (~1181 cells). **Correction (D-055):** 1916 of those receipts were not deleted and remain under `runs/`.

## 2026-09-27 Integrity audit amendments

No frozen seed, split, threshold, or lockbox was modified or re-run. Legacy loaders in `orchestration/datasets.py` are left byte-identical so historical receipts stay reproducible; repairs live in new modules.

| ID | Label | Decision |
|----|-------|----------|
| D-050 | FINDING | Legacy Track C (`synthetic_track_c`) features are `[latent_t, latent_{t-1}]` while the label is `β·latent_t + noise` — a contemporaneous leak. All synthetic MSE results (matrix, validation selection, lockbox H1 synthetic, factorial) are **leakage-contaminated**. Test: `tests/causal/test_dataset_leakage.py`. |
| D-051 | FINDING | Legacy Track A (`etf_track_a`) expanding-std feature at row t includes the label r_t. Minor leak (magnitude only). All tabular "h20" cells use a 1-step label; `horizon` is not passed to tabular baselines. |
| D-052 | FINDING | Legacy `_train_test` clamps every fold to the same split at lean sizes (n=160 → split=112 for folds 0–3). ETF seeds only affect the synthetic fallback, so all ETF seeds read the same panel. Full profile: every model has 1 distinct ETF value and ≤5 distinct synthetic values across 20 cells. "Validation folds 0–2" are the same split as full fold 3. |
| D-053 | FINDING | ETF cache columns are yfinance-alphabetical (AGG, EFA, GLD, SPY, VNQ), not the requested order; the legacy target (asset 0) is **AGG**, not SPY. Recorded in `data/manifests/DATA_MANIFEST.yaml` (INFERRED from vol/correlation signature). Loader fixed for future downloads. |
| D-054 | FINDING | Factorial (`orchestration/factorial.py`): forecasts never reach the controllers; MPC scenarios at step t are centred on the return earned at t (look-ahead); run on the spent lockbox seed. Status INVALID AS EVIDENCE; not re-run. |
| D-055 | FINDING | 1916 orphan receipts (pre-lean generation, incl. `dbg_*`) remained under `runs/` and were being aggregated into scoreboards and `EVIDENCE_TABLE.csv`. Report builder now reads manifest rows only; orphans listed in `reports/generated/HISTORICAL_ORPHANS.json`. |
| D-056 | NEW_DEFAULT | Repaired exploratory pipeline: `data/causal_tracks.py` (features from rows < t), `backtesting/walk_forward.py` (8 × 100 disjoint expanding-window test folds), `evaluation/repaired.py`. Synthetic seeds = frozen full seeds only (never 997); ETF test folds overlapping lockbox rows [800, 960) dropped. Primary contrast pre-declared: standardised ridge vs historical mean. **Exploratory, post-lockbox, not confirmatory.** |
| D-057 | NEW_DEFAULT | Multiplicity over the legacy matrix collapses exact-duplicate cells before testing; comparisons with < 6 unique pairs are untestable (exact two-sided Wilcoxon floor). |
| D-058 | FINDING | Lockbox freeze `code_tree_fingerprint` matches no commit (none existed at freeze). Lockbox-relevant files (`lockbox.py`, `datasets.py`) predate the freeze by mtime. Freeze integrity: PARTIALLY VERIFIED. |
