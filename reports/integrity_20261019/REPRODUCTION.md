# Reproduce the integrity study

## Source and evidence identities

Use the isolated `research/wharton-integrity-20261019` branch and the exact source commit pinned by `notebooks/wharton_integrity_kaggle.ipynb`. Its parent is `fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c`. The paper's model matrix is bound to the 124-file manifest in `research/development/integrity_20261019/matrix_source_freeze.json`; its aggregate SHA-256 is `8369f937581a33d013e34815bdb663f0669ccac0e1268a926e641d3ec7318811`.

The local environment is recorded in [ENVIRONMENT_OBSERVED.json](ENVIRONMENT_OBSERVED.json): Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, pandas 2.2.3 and CPU PyTorch 2.14.0+cpu. This is an observed inventory, not a portable lock or proof that the same wheel is available on another host. No global package installation was performed; a local virtual environment referred to existing cached dependencies. A different library or BLAS version may change floating-point results. Same-source/same-seed replay on a new host remains an execution replay, not a new independent simulation seed.

The separate evidence ZIP contains original synthetic arrays, datasets, logs, receipts, checkpoints, pre-amendment failures and analysis outputs. `EVIDENCE_MANIFEST.json` lists every included file's bytes and SHA-256. `PUBLIC_FILES.json` maps identical public code/contracts/tables omitted from that ZIP to their repository paths and hashes. Private competition references are separate from synthetic research in the ZIP. Do not publish those references by copying the archive into Git.

## Local checks

From a checkout with the recorded dependencies available, use one numeric thread. On Apple Silicon, also set `VECLIB_MAXIMUM_THREADS=1`; the Kaggle driver is designed for Linux `/proc` and cgroup memory admission, while the source tests and demonstration can be invoked directly on other supported Python hosts.

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONHASHSEED=0
python -B tests/test_kaggle_driver_contract.py -v
python -m pytest -q -c /dev/null tests/unit/test_split_boundaries.py tests/unit/test_mpc_policy_contract.py tests/unit/test_optional_model_imports.py tests/unit/test_point_in_time.py tests/unit/test_development_demo.py tests/unit/test_verified_accounting.py tests/unit/test_causal_portfolio.py
python -m pytest -q -c /dev/null tests/unit/test_m01.py tests/unit/test_m02.py tests/unit/test_m03.py tests/unit/test_m04.py tests/unit/test_m05.py tests/unit/test_m06.py tests/unit/test_m07.py tests/unit/test_m08.py tests/unit/test_m09.py tests/unit/test_m10.py tests/unit/test_m11.py tests/unit/test_m12.py tests/unit/test_research_regressions.py tests/unit/test_numerical_gradients.py
```

The retained current evidence comprises 58 passing initial integrity tests and 70 subtests, 44 passing model/tensor tests, three gradient tests, a nine-test journal suite (eight repeated, one new), and eight replay-driver tests. That is 114 distinct passing test methods under this counting convention, plus the recorded subtests. Read each receipt rather than summing repeated suites as independent checks. The before-fix execution intentionally reports failures; it is not a successful suite. No claim is made that every unrelated legacy test in the full repository was run.

To reconstruct the original regression witness, use a second checkout at the parent commit, copy only the three new split/policy/import test files into a temporary test directory, and set `PYTHONPATH` to that parent checkout's `src`. Do not overwrite the repaired source or rerun protected evaluation scripts. The original before-fix command, source identity, stdout and stderr are already retained in the evidence ZIP.

## Deterministic demonstration

Use a new output directory for every invocation; existing outputs are intentionally rejected. The command uses the fixed development seed 2026101908 and the source-declared alpha grid and clocks.

```bash
python scripts/run_development_demo.py --output-dir /tmp/wharton_demo_primary
python scripts/run_development_demo.py --output-dir /tmp/wharton_demo_replay
```

Compare the two `hashes.json` files and verify every file declared under `deterministic_artifacts_sha256`. The current demonstration retains 13 deterministic payload files. Variable execution clocks belong to the separate context/terminal metadata. The Kaggle driver performs this comparison automatically. A replay must not be counted as a second independent result.

The fit/selection journal is append-only. On failure, retain the terminal receipt, completed model states, attempted configuration events and partial outputs. A new run belongs in a new directory. Never silently drop a failed alpha or replace its seed.

## Frozen development matrix

The matrix contract was recorded before the matrix outcomes. For one world, the unchanged scientific runner executes 12 models, full and existing ablation variants, three seeds and 20 held-out origins:

```bash
python scripts/run_portfolio_research.py --output /tmp/wharton_matrix_null --worlds null --seeds 2026101911 2026101912 2026101913 --models M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 --n-test 20
python scripts/verify_portfolio_artifacts.py /tmp/wharton_matrix_null --output /tmp/wharton_matrix_null_verification.json
```

Repeat only for the other declared worlds `linear_signal`, `regime_shift` and `heavy_tails`, each in a new directory, to replay the original 288 cells. This reuses the original development seeds and adds no independent evidence. Preserve every cell terminal record, saved prediction, target, training dataset and checksum. The original preflight used only null/seed 2026101910; the original matrix replay used only null/seed 2026101911 (24 cells). The notebook can replay all 288 if its 900-second budget permits; that future replay is not claimed as already executed.

The verifier loads `.npz` with pickle disabled, reconstructs task losses from predictions and truth, checks dataset/target alignment and horizon maturity, and replays M11's declared fixture through the existing ledger. It does not infer market validity from a passing checksum. Saved `.pkl` model checkpoints are preserved as bytes from this trusted source; neither the analysis nor archive verification needs to unpickle them. Never unpickle artifacts from an untrusted source.

## Regenerate tables and figures from the retained study

Extract the evidence ZIP into a separate directory. Restore the five small public contract/source-freeze files into its `research/development/integrity_20261019/` directory according to `PUBLIC_FILES.json` if they were omitted as public copies. The analysis input root is the session directory, not the repository root. Run the analysis into a new output directory:

```bash
MPLCONFIGDIR=/tmp/wharton_mpl python scripts/analyze_integrity_development.py --session-dir /tmp/wharton_evidence/research/development/integrity_20261019 --output /tmp/wharton_analysis_rebuilt
```

This validates the original demo hash manifests, recomputes forecasts and the carried-holdings ledger, verifies matrix cell coverage and saved loss reductions, compares the declared 24-cell replay byte-for-byte at the array level, and rebuilds primary/ablation and secondary-control tables and plots. Its main output is `analysis.json`; all failures remain in `failures.json`. Timing metadata and the expanded execution-history list can differ from the first analysis; compare the numerical tables and the declared data/source hashes for scientific equivalence.

The first `analysis/` output is preserved unchanged. `reporting_amendment.json` explains the post-outcome expansion to `analysis_v2/`: all already saved simpler controls are now reported, with no new fit, seed, tuning or changed primary comparator. There are 156 seed/control rows and 52 model/world/control summaries. M11's secondary equal-weight control has narrower verification: the saved path-loss vector is independently reduced to CVaR; that control ledger is not independently reconstructed by this analysis.

## Kaggle launch and retrieval

Open the supplied notebook in a Kaggle account you control, select **CPU / no accelerator**, and enable internet only if you need to fetch the pinned public source. The first code cell performs a sparse checkout of the exact commit and verifies `git rev-parse HEAD`. It fetches only source/configuration/test paths rather than historical repository archives. There are no embedded tokens or account credentials.

The second execution step runs `scripts/run_integrity_kaggle.py` with the exact source SHA, a fresh `/kaggle/working/` output directory and `--matrix`. The driver checks source cleanliness, installed dependencies, memory headroom, one numeric thread, per-command timeout and total wall-time budget. It stops after an error, timeout or admission denial and reports every planned matrix cell as completed, failed, nonterminal or unstarted. A `.partial` archive is explicitly unfinished; the normal ZIP name is published only after checksum creation.

Download the output `replay_bundle.zip` and `replay_bundle.sha256`, and preserve the actual Kaggle notebook/version/run URL with them. Inspect `execution_summary.json`, command receipts and `matrix_receipts.json` before interpreting any result. Missing dependencies or resources are a recorded failure, not permission to change the model or silently reduce the design. This session did not authenticate a Kaggle notebook or execute a remote job; the helper tests do not establish remote availability.

## Decision rule and scope

All results remain development evidence. The independent unit is the simulation seed within a world; dates, assets, paths and replays are not extra samples. Keep task-specific units separate. Three seeds and descriptive t intervals with two degrees of freedom do not establish discovery. Preserve the original spent lockbox and invalid legacy results. No successor experiment, protected evaluation, market allocation, trade or submission is part of this reproduction procedure.

## Final delivery identity and checkpoint scope

Tested source commit: `5081e37dadfc9cbb0ae5e06c46385ba89c05e50c`. See [DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json) for the pinned notebook and private evidence archive identity. The original multi-seed runner preserves a representative checkpoint for each model/variant at the first seed of each invocation; every cell retains raw predictions, targets and dataset/configuration identity, but not every fitted seed state was checkpointed. The supplied Kaggle driver invokes each seed separately. The first source-change manifest is a snapshot at this source commit; later delivery documentation does not change the frozen model source.
