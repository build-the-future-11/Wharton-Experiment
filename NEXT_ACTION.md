# Next action

## Done
- Lean matrix complete; lockbox opened once (`protocol/LOCKBOX_FREEZE.yaml`, `runs/lockbox/`).
- Selected: **B_RIDGE + B_EWMA_COV**. ETF lockbox: ridge does **not** beat persistence.

## Do not
- Re-run lockbox without an amendment log.
- Invent client facts / fake WInS trades.

## Needs from you
1. WGY PDFs + client constraints
2. WInS blotter (3 executions)
3. Optional: authorize git commit

```bash
bash scripts/audit_completion.sh
cat reports/evidence_bank/LOCKBOX_SUMMARY.md
```
