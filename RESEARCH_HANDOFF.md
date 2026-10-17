# Wharton research source and reproduction handoff

Source check: 6 October 2026. This document identifies existing research artifacts and commands; it is not a new experiment or a competition submission.

## Use the source generation named by the research release

The repaired scientific source is commit [`c72c46232719b5fd144fa992f5f605ea00e8923f`](https://github.com/build-the-future-11/Wharton-Experiment/tree/c72c46232719b5fd144fa992f5f605ea00e8923f). It contains the twelve-model repairs, `protocol/PORTFOLIO_RESEARCH_V1.md`, and `scripts/run_portfolio_research.py`.

At this source check, the default `main` branch still pointed to upstream commit `8e3d790dd513a93b7d80d5726a8b2708e8615b71`. A checkout of that older generation does not provide the repaired research runner. The dated September 27 preparation workflow in `README.md` and `REPRODUCE.md` remains a historical record.

The September 30 paper release adds completion scripts, tests, manuscripts, figures, and evidence to the repaired source. Those additional files are distributed in `WHARTON_RESEARCH_PACKAGE.zip`; the repaired Git commit alone is not the complete paper release.

## Where the recorded commands belong

| Purpose | Required source | Recorded entry point |
| --- | --- | --- |
| Inspect the repaired scientific protocol and implementations | Pinned repaired commit above | `protocol/PORTFOLIO_RESEARCH_V1.md` and `src/wharton_lab/` |
| Execute the existing exploratory synthetic protocol | Pinned repaired commit, scientific dependencies, a new output directory | `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_portfolio_research.py --output runs/reproduction_new` |
| Verify the full paper archive's included files | Extracted `WHARTON_RESEARCH_PACKAGE.zip` | `python scripts/verify_research_package.py .` |
| Reproduce the paper's numerical study, replay, figures and PDFs | Full paper archive, its scientific environment and TeX dependencies | `bash scripts/reproduce_completed_research.sh runs/reproduction_new` |

The archive-only commands are documented by the release's `REPRODUCE_RESEARCH.md` and `RESEARCH_VERIFICATION.md`. They are not present in the pinned Git tree. Do not infer their availability from a repository checkout.

The numerical entry points execute experiments. They are recorded here for a separately authorized reproduction, not as metadata checks. Existing nonempty evidence directories must be preserved. The runner requires `--output` and refuses to overwrite a nonempty directory.

## Archive identity and environment

The release's `WHARTON_SHA256SUMS.txt` records this SHA-256 for `WHARTON_RESEARCH_PACKAGE.zip`:

```text
2706463f096ca81866e6e6beca55cdcc37c437d339476b4d1879db1f7bbf0ebc
```

Compare the downloaded archive with that checksum before extraction, then use its manifest verifier. Archive bytes were unavailable during this documentation check, so their integrity and the archived workflow were not re-certified here.

The paper's reproduction document records CPython 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, pandas 2.2.3, PyTorch 2.10.0+cpu and pytest 9.0.2, with numerical thread counts set to one. It uses `requirements-research-completion.txt` inside the archive. Its recorded environment is distinct from this repository's historical `requirements-verified.txt` snapshot and the minimum dependency ranges in `pyproject.toml`. The pinned repository declares PyTorch as an optional dependency, although its neural-model research needs it. No fresh dependency installation was tested by this handoff.

## Data and competition boundaries

The repaired twelve-model protocol generates synthetic panels. The documented study has 480 primary cells and a separate 480-cell replay; the replay does not double the number of independent observations. Its five simulation seeds per setting are the replication units. Synthetic performance does not establish market alpha, client suitability, or competition readiness.

The older ETF replay needs the exact ignored cache at `data/raw/etf/SPY_AGG_GLD_VNQ_EFA_returns.npy`, bound by `data/manifests/DATA_MANIFEST.yaml`. A replacement market download would be a different input. Preserve the original spent lockbox and withdrawn legacy results.

The historical preparation checker also needs the two source PDFs listed in `protocol/RECOVERED_DOCUMENTS.json`. That manifest describes recovery from the user's source copies and explicitly does not authenticate them through SurveyMonkey Apply. The complete current client case, approved universe, point-in-time data and genuine WInS activity remain separate evidence requirements.

Research artifacts and candidate lists do not establish a registered team or an eligible advisor. Use the current [registration rules](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/register-now/) and [team rules](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/) together with actual team/account receipts. This handoff does not select team members, place trades, authorize a new backtest, or submit work.

## Scope of this correction

The correction adds an entry point and labels older preparation instructions by their date. It changes no scientific implementation, protocol, dataset, result, checkpoint or historical evidence. Git source identity and the presence of the referenced repository paths were checked; no numerical or report-generation command was run.
