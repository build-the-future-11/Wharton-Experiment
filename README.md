# Wharton-Experiments

Investment research and preparation for the Wharton Global High School Investment Competition.

## Research source and reproduction

For the repaired twelve-model synthetic study, start with [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md). It identifies the pinned repaired source and the separate September 30 paper archive, with their different reproduction commands and data requirements.

The September 27 preparation material below is retained as a historical record. It does not describe the later research release or establish competition registration.
## Historical preparation state - 2026-09-27

Local preparation tools and documents are ready for review. **Competition submission remains BLOCKED** on the complete client case, trading/eligibility sources, real WInS records, student review and authority. IPS and Trading Notes instruction PDFs were recovered from Downloads and verified; see `protocol/VERIFIED_DELIVERABLE_REQUIREMENTS.md`.

The earlier synthetic forecasting win was withdrawn for leakage. Repaired tested baselines do not show a statistically supported edge over the historical mean. The frozen M11 controller has an additional accounting defect. No client suitability, investable alpha or funding guarantee is claimed.

## Open the work

- `reports/competition_package/DECISION_ROOM.html` - offline interactive sensitivity viewer, all inputs labelled assumptions.
- `output/pdf/WHARTON_EXECUTIVE_BRIEF.pdf` - one-page brief.
- `output/pdf/WHARTON_PREPARATION_PACK.pdf` - written responses, pitch content and scripts, financial sensitivity, 60 judge questions and evidence appendix.
- `output/pdf/WHARTON_IPS_DRAFT_BLOCKED.pdf` - three-page working IPS, verified font/length structure; case and roster still missing.
- `ASTRA_FINAL_REPORT.md`, `STATUS.md`, `CLAIMS.md`, `LIMITATIONS.md` - exact scope, evidence and blockers.

## Reproduce the historical preparation package

```bash
PYTHONPATH=src .venv/bin/python scripts/run_competition_stress.py
.venv/bin/python scripts/build_competition_artifacts.py
PYTHONPATH=src .venv/bin/python scripts/verify_astra.py
.venv/bin/python scripts/verify_competition_package.py
.venv/bin/python -m pytest -q
```

See `REPRODUCE.md` for dependencies, cache placement and browser/PDF checks. The stress calculator needs no market download. The historical ETF replay needs the exact ignored cache. No lockbox run is part of this workflow.

## Historical research

`RESEARCH_AUDIT.md`, `reports/FINAL_RESEARCH_REPORT.*`, and `runs/` preserve the earlier negative-result investigation. The 1181 legacy matrix cells and factorial are invalid as forecasting/decision evidence. The original lockbox remains spent. Old competition PDFs and the legacy IPS builder are superseded; only the current package should be used for preparation.
# Wharton-Experiment
