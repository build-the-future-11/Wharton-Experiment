# Reproduce

```bash
cd /Volumes/PRO-BLADE/Wharton-Experiments
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/run_pilot.sh --resume
bash scripts/build_reports.sh
bash scripts/audit_completion.sh
```

Overnight / full (resumable):

```bash
bash scripts/run_overnight.sh --hours 8 --resume
bash scripts/run_full.sh --resume
bash scripts/build_reports.sh
```

Key artifacts:

- `protocol/EXPERIMENT_MANIFEST.csv`
- `runs/{smoke,pilot,overnight,full}/<run_id>/receipt.json`
- `reports/generated/{SUMMARY.md,scoreboard.json,index.json}`
- `reports/FINAL_RESEARCH_REPORT.md`
- `EXECUTION_STATE.json` / `NEXT_ACTION.md`
