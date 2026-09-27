# Workspace Audit

**Audited:** 2026-09-26  
**Path:** `/Volumes/PRO-BLADE/Wharton-Experiments`

## Recovery findings

| Item | Finding | Label |
|---|---|---|
| Directory at open | Empty (created 2026-09-26 15:02) | RECOVERED |
| Git | `git init` only; **no commits** | NEW_DEFAULT |
| AGENTS.md | None in workspace | — |
| Twelve-model decision record | Not present locally; identities taken from execution brief | FROZEN (identities) |
| Client PDFs `2026_WGY_*` | Not found on PRO-BLADE / Downloads / Documents | BLOCKED |
| Hardware | Apple M4, 16 GB, MPS available | RECOVERED |
| Disk free | ~609 GB on PRO-BLADE | RECOVERED |

## Lineage (authorized siblings; not vendored)

| Mechanism | Source path | Status |
|---|---|---|
| APEN memory | `GitHub-Every-Repo/THE-BU1LD-APEN-Synthica-*` | RECOVERED mechanism → reimplemented + pending queue NEW_DEFAULT |
| Q-APEN | `World-Series/world-series/src/world_series/qapen/` | RECOVERED |
| FI-JEPA | `GitHub-Every-Repo/FI-JEPA` | RECOVERED concepts |
| Eigen-JEPA | `GitHub-Every-Repo/Eigen-JEPA` | RECOVERED concepts |
| Graph JEPA | No upstream implementation | NEW_DEFAULT |

Unavailable upstream binaries are **evidence gaps**, not permission to claim identity with a generic net.

## Current delivery state (2026-09-27)

- Package `src/wharton_lab/` with M01–M12, baselines, splits, orchestration, reports, audit, and the repaired causal path.
- Manifest: smoke 34, pilot 187, overnight 360, full 600 — all COMPLETED (legacy, leakage-contaminated).
- Git: 4+ commits on `master`; no remote configured.
- Internal disk (`/`) was full (~116 MiB free) during the 2026-09-27 pass; temp files were redirected to PRO-BLADE.
- Client competition package blocked on missing sources.
