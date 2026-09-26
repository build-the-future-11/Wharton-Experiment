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
- Prior 3071-cell receipts deleted as redundant vs new lean matrix (~1181 cells).
