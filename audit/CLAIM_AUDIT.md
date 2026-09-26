# Claim Audit (post-lockbox)

| claim_id | statement | status | evidence |
|---|---|---|---|
| C001 | Twelve models implemented | SUPPORTED | `src/wharton_lab/models/` |
| C002 | Lean matrix 1181/1181 complete | SUPPORTED | manifest |
| C003 | Lockbox frozen then opened once | SUPPORTED | `protocol/LOCKBOX_FREEZE.yaml`, `runs/lockbox/` |
| C004 | Selected stack = B_RIDGE + EWMA | SUPPORTED | freeze selected_stack; val folds 0–2 |
| C005 | Lockbox ridge beats persistence (synthetic) | SUPPORTED | LOCKBOX_RESULTS.json |
| C006 | Lockbox ridge beats persistence (ETF) | UNSUPPORTED | ridge MSE 2.14e-5 > persistence 2.12e-5 |
| C007 | Client IPS ready | BLOCKED | missing WGY PDFs / facts |
| C008 | WInS trading notes FINAL | BLOCKED | no executions |
| C009 | Investment outperformance | UNSUPPORTED | not claimed |
| C010 | Funding shortfall guarantees | UNSUPPORTED | research proxies only |
