# Reproduce

Requires the exact ETF cache `data/raw/etf/SPY_AGG_GLD_VNQ_EFA_returns.npy` (sha256 in `data/manifests/DATA_MANIFEST.yaml`). Without it, ETF numbers will differ.

```bash
cd /Volumes/PRO-BLADE/Wharton-Experiments
bash scripts/bootstrap.sh
.venv/bin/python -m pytest -q
bash scripts/run_smoke.sh
bash scripts/run_pilot.sh --resume
bash scripts/build_reports.sh
bash scripts/audit_completion.sh
```

Legacy overnight / full (resumable; leakage-contaminated generation):

```bash
bash scripts/run_overnight.sh --hours 8 --resume
bash scripts/run_full.sh --resume
bash scripts/build_reports.sh
```

Repaired exploratory evidence, figures, PDFs:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_repaired_eval.py
PYTHONPATH=src .venv/bin/python scripts/run_multiplicity.py
.venv/bin/python scripts/figures/repaired_oos_r2.py
.venv/bin/python scripts/build_evidence_bank.py
.venv/bin/python scripts/build_pdfs.py
```

Key artifacts:

- `protocol/EXPERIMENT_MANIFEST.csv`, `protocol/DECISIONS.md`
- `runs/{smoke,pilot,overnight,full}/<run_id>/receipt.json` (manifest rows only; orphans listed in `reports/generated/HISTORICAL_ORPHANS.json`)
- `runs/lockbox/`, `runs/repaired/`
- `reports/evidence_bank/`, `reports/figures/`, `reports/FINAL_RESEARCH_REPORT.{md,pdf}`
- `RESEARCH_AUDIT.md`, `EXECUTION_STATE.json`, `NEXT_ACTION.md`
