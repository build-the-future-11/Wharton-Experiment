# Final Research Report (internal)

**Workspace:** `/Volumes/PRO-BLADE/Wharton-Experiments`  
**Date:** 2026-09-26  
**Git:** no commits yet  
**Status:** Lean FAST_MATRIX complete (1181/1181). **Lockbox opened once.** Client IPS still BLOCKED.

## Distinctions
Implemented ≠ validated ≠ investable. Lockbox ≠ client funding guarantee. Proposed trade ≠ WInS execution.

## Matrix (lean)
| Profile | COMPLETED |
|---|---:|
| smoke | 34 |
| pilot | 187 |
| overnight | 360 |
| full | 600 |

## Validation selection (folds 0–2 only)
Rule: min MSE among point-forecast models; **prefer B_RIDGE if within 1% of best** (simplicity).

| Model | Val MSE (folds 0–2) |
|---|---:|
| M04 | 5.356e-05 |
| B_RIDGE | 5.365e-05 |
| M05 / M09 / M03 | worse |

**Selected stack:** `B_RIDGE` + `B_EWMA_COV` (+ optional `M01` for pinball only). Controller not selected (no client liabilities).

Freeze: `protocol/LOCKBOX_FREEZE.yaml`.

## Lockbox (single open)
Virgin synthetic seed=997 + ETF offset=800.

| Dataset | B_RIDGE MSE | Persistence MSE | Beats persistence? |
|---|---:|---:|---|
| synthetic | 9.89e-05 | 0.0281 | **Yes** |
| ETF | 2.14e-05 | 2.12e-05 | **No** (slightly worse) |

- H1: **SUPPORTED** on synthetic; **UNSUPPORTED** on ETF lockbox.
- H2 (EWMA finite): **SUPPORTED**.

Evidence: `runs/lockbox/LOCKBOX_RESULTS.json`, `reports/evidence_bank/`.

## Negative / inconclusive
- Complex models do not beat ridge on MSE in validation.
- ETF lockbox: ridge ≈ persistence (no meaningful edge).
- M11/M12 funding = research proxies only.

## Blocked
WGY PDFs, WInS executions, client wealth/obligations, Track B PIT.
