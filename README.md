# Wharton-Experiments

Research lab for twelve forecasting/decision models + Wharton competition decision support.

## Quick start
```bash
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/run_pilot.sh
bash scripts/run_overnight.sh --hours 8 --resume
bash scripts/run_full.sh --resume
bash scripts/run_lockbox.sh
bash scripts/build_reports.sh
bash scripts/audit_completion.sh
```

## Status (2026-09-26)
- Lean matrix **1181/1181** COMPLETED
- Lockbox opened once → select **B_RIDGE + B_EWMA_COV**
- ETF lockbox: ridge does **not** beat persistence
- Client WGY PDFs / WInS executions: **BLOCKED**

See `NEXT_ACTION.md`, `reports/FINAL_RESEARCH_REPORT.md`, `protocol/LOCKBOX_FREEZE.yaml`.
