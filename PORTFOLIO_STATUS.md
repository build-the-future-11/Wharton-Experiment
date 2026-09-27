# Portfolio Status (2026-09-27)

Scope: this workspace (`/Volumes/PRO-BLADE/Wharton-Experiments`) only. About 40 sibling directories on `/Volumes/PRO-BLADE` (e.g. `IRIS-Project`, `World-Series`, `Kyrlov-JEPA`, `OLYMPUS`) are outside this workspace and were **not audited**.

| Project | Path | Type | Current State | Critical Blocker | Evidence | Next Action | Priority |
|---|---|---|---|---|---|---|---|
| Wharton-Experiments (wharton-lab) | `/Volumes/PRO-BLADE/Wharton-Experiments` | research / finance ML + competition decision support | experimental → negative-result package; `master`, no remote; 50/50 tests; fresh clone passes | Client WGY PDFs, WInS executions, and submit authority missing; legacy evidence leakage-contaminated (documented) | `RESEARCH_AUDIT.md`, `protocol/DECISIONS.md` D-050..D-058, `runs/repaired/` | Supply client sources, then finalize the IPS on the strategic-allocation + EWMA stack; optional repaired M01–M12 run | P0 (client-blocked) |

Environment note: the internal disk was full (≈ 116 MiB free) during this pass. Only PRO-BLADE had space.
