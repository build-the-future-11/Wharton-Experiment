# Reproducibility Audit (revised 2026-09-27)

## Environment

- Host: Apple M4, 16 GB RAM, MPS available (orchestration defaults to CPU for determinism)
- Python 3.14.7 in project `.venv` (`scripts/bootstrap.sh`); numpy/scipy versions recorded per run in `runs/repaired/REPAIRED_RESULTS.json`
- matplotlib (figures) and reportlab (PDFs) are required for `scripts/figures/` and `scripts/build_pdfs.py`

## Determinism controls

- Seeds: smoke `[11]`; pilot `[11,23,47]`; overnight `[11,23,47,89]`; full `[11,23,47,89,131]` (`orchestration/profiles.py`)
- Manifest rows carry `run_id`, `result_path`; resume skips `COMPLETED` only
- Repaired runs record the git HEAD, the environment, and the ETF sha256

## Verified on 2026-09-27

- Full test suite: 50 passed (41 legacy + 9 new leakage/evaluation tests)
- A fresh `git clone` of HEAD passes the suite (source packages that had been untracked are now committed)
- `scripts/run_repaired_eval.py` and `scripts/run_multiplicity.py` are deterministic given the ETF cache

## Gaps

- **ETF cache is not in git** (`data/raw/` ignored). A fresh clone re-downloads with a later end date, which changes every ETF number. Exact reproduction needs the file with sha256 `6c234590…` (`data/manifests/DATA_MANIFEST.yaml`).
- Lockbox freeze fingerprint matches no commit (D-058).
- Legacy "folds" and ETF "seeds" are not independent replications (D-052).
- The git author email is a placeholder (`youremail@example.com`).
- Upstream lineage trees are not vendored; mechanism reimplementation is disclosed in `protocol/DECISIONS.md`.
