# Wharton-Experiments

Research lab for twelve forecasting/decision models + Wharton competition decision support.

## Quick start
```bash
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/run_pilot.sh
bash scripts/run_overnight.sh --hours 8 --resume
bash scripts/run_full.sh --resume
bash scripts/run_lockbox.sh          # refuses: lockbox already opened
bash scripts/build_reports.sh
bash scripts/audit_completion.sh
```

Repaired (causal) exploratory evaluation:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_repaired_eval.py
PYTHONPATH=src .venv/bin/python scripts/run_multiplicity.py
.venv/bin/python scripts/figures/repaired_oos_r2.py
.venv/bin/python scripts/build_evidence_bank.py
.venv/bin/python scripts/build_pdfs.py
```

## Status (2026-09-27)
- Legacy lean matrix 1181/1181 ran, but it is **leakage-contaminated** and its folds/seeds are not independent (`protocol/DECISIONS.md` D-050..D-055).
- Lockbox opened once. Synthetic "ridge beats persistence" was a **leakage artefact**; ETF result negative.
- Repaired causal pipeline: **no forecast beats the historical mean** (exploratory, post-lockbox).
- Client WGY PDFs / WInS executions: **BLOCKED**.

See `RESEARCH_AUDIT.md`, `MASTER_EXECUTION_QUEUE.md`, `reports/FINAL_RESEARCH_REPORT.md`, `NEXT_ACTION.md`.
