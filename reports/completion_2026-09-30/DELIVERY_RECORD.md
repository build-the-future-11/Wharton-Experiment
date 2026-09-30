# Completed twelve-model research delivery — 2026-09-30

## Scope and storage

The M01–M12 synthetic research study, principal manuscript, twelve companion model reports, source package and reproducibility files were completed and delivered as downloadable conversation artifacts. This record is not a claim that the entire generated package has been synchronized into this branch. The full research ZIP contains the complete new source, raw canonical results, verification certificates and a source patch that was checked and applied successfully against the recovered c72c462 source; all 86 patched files matched the completed workspace.

Recovered scientific source commit: `c72c46232719b5fd144fa992f5f605ea00e8923f`.

Scientific source fingerprint (122 files):
`39292d20615c26dc7e310b6543ca36651063ded1a6369f9255e82897f70ad045`.

## Verified execution

- Primary experiment: 480/480 cells completed, zero failures. Twelve models, two variants, four synthetic settings and five seeds; twenty distinct panels.
- Tests: 99 passed.
- Independent raw-score recomputation: all 480 cells; maximum discrepancy 1.7763568394002505e-15.
- Fresh-process replay: all 480 cells bitwise identical across 5,000 saved numerical arrays. Replays are not additional independent statistical samples.
- Representative checkpoint reloads: all 24 passed.
- Full master workflow: exit code zero, including experiments, tests, replay, figures and all thirteen PDFs.
- Deliverables: one 12-page principal manuscript; twelve two-page model reports; fourteen evidence figures; generated tables and exact configurations.
- Final PDFs: 36 pages visually inspected; no unresolved references or TeX layout warnings.
- Clean manuscript ZIP: extracted into an empty directory and compiled; every page rendered identically to the reviewed manuscript.
- Full research ZIP: CRC check passed; all 4,107 manifested files verified after a fresh extraction.

## Scientific findings and limits

M04's memory correction increases mean error relative to its own backbone in all four settings. M10's affine Gaussian implementation has context-independent covariance and loses in mean energy score to its conditional residual-bootstrap comparator in all four settings. M11 selects cash in all twenty full-model fits and matches cash-only unpaid obligations of 0.03, rather than demonstrating investment skill. Several ridge-relative advantages disappear against a historical mean. No positive discovery survives the reported 96-comparison Holm-adjusted sign-flip sensitivity analysis.

These are explicit synthetic results, not market alpha, client suitability, or twelve independently novel methods. The original lockbox was not reopened. The exact ignored historical ETF cache is absent from the recovered tracked source, so the historical market replay was not regenerated. No trade, competition submission, arXiv submission or acceptance occurred. Installation in a new environment was not tested; the recorded installed dependency versions ran successfully. The master log retains non-fatal environment startup warnings from an unrelated spreadsheet runtime; all research gates passed.

## Delivery integrity

`WHARTON_RESEARCH_PACKAGE.zip`
- Bytes: 51,950,163
- SHA256: `2706463f096ca81866e6e6beca55cdcc37c437d339476b4d1879db1f7bbf0ebc`

`WHARTON_ARXIV_SOURCE.zip`
- SHA256: `676f5410b3f4b5f736371b0dd2fc0301734bb2c2b1835919d8944bda0a327009`

The other delivered artifacts are `WHARTON_TWELVE_MODEL_RESEARCH.pdf`, `WHARTON_MODEL_REPORTS.zip`, `RESEARCH_VERIFICATION.md`, `REPRODUCE_RESEARCH.md`, `COMPLETION_STATE.json`, `WHARTON_DOWNLOAD_VERIFICATION.json`, and `WHARTON_SHA256SUMS.txt`.

After extracting the full package and entering its root:

```bash
python scripts/verify_research_package.py .
bash scripts/reproduce_completed_research.sh runs/reproduction
```

The full package preserves original historical evidence, plus canonical primary arrays and checkpoints. It omits Git/interpreter caches, standalone fonts, redundant replay/master raw copies and intermediate page renders; the corresponding verification certificates and logs remain included. Raw primary results and unfavorable cells are not omitted.
