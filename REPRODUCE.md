# Reproduce the current Wharton package

Run from the repository root with Python 3.10+ and the project's installed environment. The tested environment is recorded in requirements-verified.txt (an environment snapshot, not a portable hashed lockfile). Existing bootstrap is scripts/bootstrap.sh. Core package requirements are in pyproject.toml; PDF checks need reportlab and pypdf, rendering needs Poppler, and optional browser QA needs Playwright and Chrome.

```bash
mkdir -p .local_tmp/astra .local_tmp/cache .mplconfig
export TMPDIR="$PWD/.local_tmp/astra"
export MPLCONFIGDIR="$PWD/.mplconfig"
export XDG_CACHE_HOME="$PWD/.local_tmp/cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=src
.venv/bin/python -m pytest -q --basetemp="$PWD/.local_tmp/astra/pytest"
.venv/bin/python scripts/run_competition_stress.py
.venv/bin/python scripts/build_competition_artifacts.py
.venv/bin/python scripts/verify_astra.py
.venv/bin/python scripts/verify_competition_package.py
bash scripts/audit_completion.sh
```

The stress matrix and viewer need no network or market dataset. The artifact builder uses genuine Times New Roman from the installed macOS supplemental fonts; change only the font location for another system and preserve the required font.

The replay checker needs the exact ETF cache at data/raw/etf/SPY_AGG_GLD_VNQ_EFA_returns.npy, SHA-256 in data/manifests/DATA_MANIFEST.yaml. It reproduces into .local_tmp and does not overwrite the old result. The package checker requires the two local instruction PDFs listed in protocol/RECOVERED_DOCUMENTS.json. Those files are excluded from git; restore the user's source copies with the recorded hashes. A clone without those sources must report missing-source failure.

Browser validation, when Playwright is available to Node:

```bash
node scripts/verify_competition_viewer.cjs
```

On this host, NODE_PATH was set to the bundled runtime's node/node_modules directory. Chrome launches with a new temporary profile, not the user's browser session. The sandboxed first attempt failed; an approved isolated run checked all 75 selections and mobile layout. Browser receipt: audit/astra_2026-09-27/BROWSER_QA.json.

Render the PDFs with Poppler using a writable fontconfig cache if needed. The retained artifact QA receipt has output hashes, page counts, word counts and source hashes. Financial JSON/CSV and Markdown are deterministic for the recorded environment; PDF timestamps may differ on rebuild.

Legacy smoke/pilot/full scripts remain for historical work and use contaminated generation paths. They are not the current evidence workflow. Do not run the spent lockbox again. No remote submission or trading command is included.
