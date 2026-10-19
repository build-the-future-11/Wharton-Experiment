# FREM-S12 P0v4 engineering verification

Date: 18 October 2026

## Candidate

The implemented package is `successors/frem_s12`. Its operative source identity
`019357f8bb87118ee31f29eb361dc5284f10186e7eeb795f331f8bd3f6c6a3cc` is
recorded in `IMPLEMENTATION_RECEIPT_P0V4_READY.json`. All P0v2/P0v3 receipts and
generated artifacts remain preserved. P0v4 was frozen prospectively before S12
market archive acquisition or numeric outcome access and changes only the exact
runtime/input-mount contract, not the question, model, cells, controls,
thresholds, inference or budgets.

## Passing checks

- 43 generated/unit/reference/integration tests under Python 3.12 / NumPy 2.5.4 with warnings promoted to
  errors.
- Ruff lint and format checks across source, tests and the Kaggle notebook.
- Strict mypy across 20 package modules.
- Source distribution and wheel build from the package metadata.
- Clean wheel installs under Python 3.12 / NumPy 2.5.4 and Python 3.13 / NumPy
  2.1.3. Their installed-wheel generated arrays, summary and manifest are
  byte-identical.
- Wheel inventory verification that the preserved P0v2/P0v3 files and operative
  P0v4 amendment are embedded.
- Valid Kaggle notebook JSON.
- Exact wheel and notebook SHA-256 gates plus recursive unique Kaggle input discovery.

Tests cover finite-law score identities and gradient finite differences,
variogram behavior, parser rejection, calendar continuity, causal reservoir
prefix invariance, router caps/burn-in, complete router and MLP optimizer-state
round trips, fixed/search registries, portfolio constraints, exact-origin
checkpoint resume, corruption/no-overwrite behavior, Fold-A-only search and
complete-search selection.
The one-use consumption tests also prove duplicate refusal, exact-resume reuse
and changed-gate rejection.
P0v3 additions prove reconstruction of every fixed and search score from the
retained outcomes/finite laws, separate target/holding paths, raw-mean MLP
residual updates, seed-first hierarchical aggregation and rejection of local
qualification receipts at the scientific execution gate.
The final statistical reference also proves that fold identities and circular
month blocks are shared across families, preserving contemporaneous covariance.

## Generated smoke

The P0v2 smokes remain preserved. A new P0v3 local generated smoke retained all
nine arms and their complete finite laws for 24 Fold-A test origins and
completed with a hash-bound manifest. The candidate did not beat
S11, VAR or the recent empirical law on that generated panel. This is preserved
as a negative/mixed engineering observation. Generated data neither consumes
the market protocol nor supports efficacy.

## Kaggle capability evidence

Private saved version `359681231` completed an engineering-only 24.2-second
probe. It observed Python 3.13.15, NumPy 2.1.3, Linux x86_64, four CPUs, a
read-only `/kaggle/input`, blocked direct TCP egress, and the dataset wheel under
`datasets/ryangomez010/frem-s12-input/`. The probe explicitly recorded
`scientific_admitted=false` and executed no model or market code. See
`KAGGLE_CAPABILITY_359681231.json`.

The local macOS arm64 NumPy 2.1.3 build emitted floating-point status warnings
on finite matrix products, so its warning-strict run retained 36 passes and
seven warning failures. This was not hidden or waived. The installed-wheel
generated outputs were nevertheless byte-identical across 3.12 and 3.13. The
Kaggle Linux qualification remains the authoritative discriminator.

Final wheel SHA-256:
`d4d65ad0913393b69b70f6a9f3b12f1960ab5905529695e22757e1a2024cee4e`.
Final notebook SHA-256:
`e5f5a6daa767f963a1eb49025093bf4ed5caa5270e79da827b847317082e0b13`.
The durable upload artifact was built with `SOURCE_DATE_EPOCH=1792321200`;
earlier byte-different timestamped builds are retained as non-candidates.

## Boundary

The v3 rejected qualification and v4 capability probe ran privately; no v4
scientific qualification has completed. No official market archive was downloaded or parsed. No
search/fixed market cell, model fit, scientific ledger, protected result,
profitability evidence, confirmation or independent reproduction exists.
Local and clean-wheel environment receipts carry `scientific_admitted=false`
and cannot qualify Kaggle. Scientific execution accepts only an exact-source
`kaggle_scientific` receipt produced after the target-host probe and conservative
campaign projection. The final probe measures trial 23 separately on Fold-A
search and latest Fold-H fixed execution, then projects 144 and 120 cells by
their respective costs. GPU support is not admitted; a scientific campaign may
remain CPU-only unless actual Kaggle equivalence and at least 1.5x end-to-end
speedup are demonstrated. The exact v4 upload was attempted, but Chrome exited
repeatedly at the private dataset file chooser. No new dataset version or v4
notebook run is claimed until the authenticated browser session stays open and
the resulting hashes are verified.
## P0v5 delivery-hardening verification

P0v5 was implemented prospectively before market access. It changes only the
Kaggle delivery and resource-admission contract. The model, registries, controls,
thresholds, inference and budgets remain unchanged.

- Python 3.12.14 / NumPy 2.5.4: 48 tests passed with warnings treated as errors.
- Ruff lint and format: passed.
- Strict mypy over 20 package source files: passed.
- `uv lock --check --offline`: passed after the local version-only lock update.
- Notebook JSON parsing and compilation of every code cell: passed.
- The actual notebook cells completed an end-to-end generated smoke under
  Python 3.13 / NumPy 2.1.3 using the exact wheel, isolated target, input
  inventory and final run manifest. The retained run-manifest SHA-256 is
  `3051dea1199a6dffe24971e598d985c11a4c96e5147398013dd79dd030f286cc`.
- Reproducible v0.1.2 wheel: two `SOURCE_DATE_EPOCH=1792321200`
  builds both produced
  `dac39add6e47f0bb5c6c47ebdd246c161a44d7b25c4ee2f43b52c3e6a1e50a06`.
- The sdist hashes differed across those builds; the retained sdist is not an
  upload candidate and this negative packaging observation is preserved.
- Clean-wheel generated evidence under Python 3.12 / NumPy 2.5.4 and Python
  3.13 / NumPy 2.1.3 was byte-identical: arrays `2f8b70ec...`, summary
  `631641c8...`, manifest `f8a00153...`.
- The known local macOS arm64 Python 3.13 / NumPy 2.1.3 floating-point status
  warnings remain visible. Outputs were finite and identical, but only the
  Kaggle Linux qualification may admit science.

The hardened notebook uses a fresh isolated install target, offline/no-deps
installation, a canonical input-inventory digest, fixed single-thread reference
settings, fresh output roots and a final no-overwrite hash/size run manifest.
Qualification repeats the largest declared search and latest fixed probe three
times and uses the maximum duration; artifact byte counts must agree across
repeats. Kaggle execution remains pending.

## P0v6 evidence-integrity verification

P0v6 was frozen prospectively before official archive access or any scientific
cell. It changes no model, cell, control, threshold, inference or budget.

- 53 tests passed with Python warnings treated as errors under Python 3.12 /
  NumPy 2.5.4, including semantic tamper cases for search summaries, fixed
  scores and downloaded manifests.
- Ruff lint/format over source/tests and strict mypy over 21 source modules passed.
- `uv lock --check` resolved all 15 locked packages against current metadata.
- The notebook parses and every code cell compiles. Target subprocesses set
  `PYTHONWARNINGS=error`; output symlinks and special files become retained
  postcondition failures.
- Search selection reloads retained atoms, probabilities, outcomes and scores;
  fixed analysis reconstructs energy, variogram, post-return holdings and
  cumulative wealth at the frozen 10 bps cost before accepting summaries.
- `verify-run-bundle` enforces a complete regular-file inventory, SHA-256 and
  byte sizes, candidate identity, input inventory, single-thread environment,
  command receipt and qualification disposition. It independently accepted the
  retained P0v5 smoke bundle with manifest SHA-256
  `3051dea1199a6dffe24971e598d985c11a4c96e5147398013dd79dd030f286cc`.
- Two final fixed-epoch v0.1.3 wheel builds were byte-identical at SHA-256
  `e86605be211ca5ba922cf6742c0af4c88d575915b149cb36da4d20254192b24d`.
  The wheel is 50,472 bytes. The exact notebook SHA-256 is
  `12d769b45107796b43f1d8237a45b940d42e9d8fad7c501c6997eb849298bced`.
- A clean installation of the final wheel completed the generated smoke with
  warnings as errors. Raw arrays retained the prior SHA-256
  `2f8b70ec02f9f3090ce7e73b73f472ec894ba03866f89032416de632db9aa92f`.

Preserved failures include the network-isolated build attempt, a no-isolation
build attempted from an environment without setuptools, a concurrent test run
whose temporary artifacts became source-stale while source edits continued,
and a reproducible wheel pair superseded after final README synchronization.
None produced or altered scientific evidence. Kaggle P0v6 qualification and
all market execution remain unrun; scientific status is `NOT_EVALUATED`.

## P0v7 end-to-end evidence-chain verification

P0v7 was frozen before official archive access or scientific consumption. It
does not change the model, cells, controls, thresholds, inference or budgets.
It adds raw validation-law retention and reconstruction, exact frozen-calendar
checks, selected-control reconstruction, exact complete-campaign inventories,
acquisition/data/environment/source/selection/consumption chaining, cross-seed
outcome identity, bounded array intake, and semantic reconstruction of isolated
installation and qualification projections.

Verification completed:

- 60 tests passed with warnings as errors under Python 3.12 / NumPy 2.5.4.
- Ruff lint/format and strict mypy over 21 source modules passed.
- Offline lock validation resolved 15 packages.
- Notebook JSON and every code cell parse.
- Two fixed-epoch v0.1.4 wheel builds were byte-identical at SHA-256
  `e9596b134c8bbffebb5688a2627775b449ad87c805897af46503eb7dcda4cd72`.
- A clean-wheel warning-strict generated smoke independently reconstructed;
  arrays SHA-256 is
  `b04c37b7fdbd14cc4821a69632802fc6bbb0bd1503fc16fbb1fc08ef0acf53b8`.
- The stronger verifier still accepts the retained P0v5 generated bundle as
  integrity-valid and correctly reports `eligible_for_next_stage: false`.

Preserved failures: two no-isolation builds lacked setuptools; two isolated
offline builds lacked cached pinned build requirements; and the first
clean-wheel command used a Python interpreter without NumPy and failed before
output creation. None consumed data or scientific cells. Kaggle P0v7
qualification remains unrun, and cross-saved-version attempt reconciliation is
still required before scientific shards.

