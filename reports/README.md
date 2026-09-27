# Reports

- `generated/` — scoreboards from manifest rows only (`bash scripts/build_reports.sh`); legacy, leakage-contaminated generation.
- `evidence_bank/` — lockbox, multiplicity, repaired-eval summaries; `EVIDENCE_TABLE.csv` and `ARTIFACT_INDEX.json` via `scripts/build_evidence_bank.py`.
- `figures/` — `scripts/figures/*.py` (source CSV alongside each figure).
- `FINAL_RESEARCH_REPORT.{md,pdf}`, `competition_drafts/IPS_DRAFT_BLOCKED.pdf` — PDFs via `scripts/build_pdfs.py`.
