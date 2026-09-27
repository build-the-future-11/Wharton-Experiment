# Next action

## Shipped (2026-09-27 integrity pass)
- Found and documented leakage in both legacy data tracks, degenerate folds/seeds, a mislabelled ETF target (AGG, not SPY), and an invalid factorial (D-050..D-058)
- Lockbox synthetic "win" withdrawn; ETF lockbox negative; lockbox **not** reopened
- Repaired causal evaluation: no forecast beats the historical mean (exploratory)
- Decision brief and BLOCKED IPS draft rewritten: strategic allocation + EWMA risk + cash reserve; no forecasting edge claimed

## Only you can unblock
1. Drop the `2026_WGY_*.pdf` files into the workspace
2. Export 3 WInS executions → `data/wins/executions/`
3. Explicitly authorize any submit/trade
4. Decide whether to version the ETF cache in git (data-licence call) — see `MASTER_EXECUTION_QUEUE.md`
5. Free space on the internal disk (≈ 0.1–0.4 GB free during this pass)

Until then: do not treat drafts as FINAL.
