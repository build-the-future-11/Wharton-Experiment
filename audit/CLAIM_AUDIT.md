# Claim Audit (revised 2026-09-27 — see `RESEARCH_AUDIT.md` for the full table)

| claim_id | statement | status | evidence |
|---|---|---|---|
| C001 | Twelve models implemented | VERIFIED (code exists, unit tests pass) | `src/wharton_lab/models/`, `tests/unit/` |
| C002 | Lean matrix 1181/1181 complete | VERIFIED as execution; **not** as evidence (leakage, degenerate replication) | manifest; D-050..D-052 |
| C003 | Lockbox frozen then opened once | PARTIALLY VERIFIED (freeze fingerprint matches no commit) | `protocol/LOCKBOX_FREEZE.yaml`; D-058 |
| C004 | Selected stack = B_RIDGE + EWMA | VERIFIED as a recorded decision; UNSUPPORTED as skill | freeze; D-050, D-052 |
| C005 | Lockbox ridge beats persistence (synthetic) | **FAILED — leakage** (was SUPPORTED) | `runs/lockbox/`; D-050 |
| C006 | Lockbox ridge beats persistence (ETF) | UNSUPPORTED (negative result) | ridge 2.14e-5 > persistence 2.12e-5 |
| C007 | Client IPS ready | BLOCKED — WGY PDFs / client facts | — |
| C008 | WInS trading notes FINAL | BLOCKED — no executions | — |
| C009 | Investment outperformance | UNSUPPORTED (not claimed) | — |
| C010 | Funding shortfall guarantees | UNSUPPORTED (research proxies; shortfall never binds) | M11 cells all 0.0 |
| C011 | MPC > myopic on terminal wealth after costs | **FAILED — look-ahead, forecast unused** | `runs/factorial/`; D-054 |
| C012 | Ridge has skill over the historical mean (repaired, exploratory) | UNSUPPORTED — 0/24 ridge-vs-mean/zero contrasts significant | `runs/repaired/`; D-056 |
| C013 | Complex models (M01–M12) do not beat ridge | NOT RUN on valid data (legacy comparison leakage-contaminated) | — |
