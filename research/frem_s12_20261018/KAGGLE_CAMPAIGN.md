# FREM-S12 Kaggle heavy-compute campaign draft

Status: P0v3 private qualification was rejected before package import; the
subsequent engineering-only capability saved version `359681231` passed and
observed Kaggle Python 3.13.15 / NumPy 2.1.3 plus the recursive read-only input
layout. P0v7 now prospectively supersedes the evidence-chain and qualification
contract and is locally verified, but its private wheel/notebook upload is not complete because Chrome
local-file access is disabled. No
scientific consumption exists. See `KAGGLE_QUALIFICATION_359671420.json`,
`KAGGLE_CAPABILITY_359681231.json`, and `IMPLEMENTATION_RECEIPT_P0V6_READY.json`.

## Platform facts and evidence boundary

The official Kaggle notebook documentation observed on 18 October 2026 states
that CPU/GPU notebook sessions have a 12-hour execution limit, `/kaggle/working`
has 20 GiB of autosaved space, CPU sessions have 30 GiB RAM, and documented GPU
choices include a P100 or two T4s with 29 GiB main RAM. GPU quota is weekly and
may float with demand. These values are planning ceilings, not account-specific
capacity receipts:

- https://www.kaggle.com/docs/notebooks
- https://www.kaggle.com/docs/efficient-gpu-usage

Before each launch, save a fresh observed receipt for accelerator type, Python,
installed package hashes, CPU/RAM/GPU/disk, remaining quota and session deadline.
The smaller observed allowance is always binding.

S10's strict production contract failed on Kaggle because required namespaces
could not be created, bubblewrap was absent, and parent cgroup constraints were
hidden. S12 must not disable those S10 checks or claim equivalence. Instead,
S12 defines a separate Kaggle-native threat model suitable only for its public
research data: private notebook; no secrets; source and inputs attached
read-only; internet disabled for scientific shards; exact byte inventories;
bounded `/kaggle/working`; process, network and environment observations; and
downloaded output verification. Failure to observe this contract blocks S12.

## Notebook separation

1. **S12-Q qualification:** metadata, environment, device, resource, quota,
   network-setting and persistence probe. No market model fit.
2. **S12-E engineering:** generated data, CPU/GPU equivalence, interruption,
   resume, corruption, budget and artifact tests. No market data.
3. **S12-D search shards:** nested-development trials only, with trial IDs and
   an immutable search manifest.
4. **S12-M fixed matrix shards:** one frozen candidate plus mandatory controls,
   seeds, folds, ablations and robustness cells. No search decisions.
5. **S12-A audit:** rebuild metrics/tables from downloaded immutable raw
   outputs without training or accessing a new outcome.
6. **S12-C confirmation:** nonexistent and unauthorized until a separate
   post-development freeze.

Engineering, search, fixed matrix, audit and confirmation must never share a
consumption namespace.

S12-Q must invoke `s12-research qualify-kaggle`. The local `qualify` command is
engineering-only and its receipt is structurally rejected by search/fixed
execution. The Kaggle command measures the largest declared configuration on a
complete Fold-A search cell and separately on latest Fold-H fixed execution,
then projects all 144 search and 120 fixed cells. It admits only when CPU-hour,
shard-hour, shard-output and live-free-space limits all pass.
The notebook raises after saving the receipt when `scientific_admitted` is not
true. This code receipt does not replace the account quota/session observation
or downloaded-output hash match required below.

## Stage gates

### K0 — local generated verification

- Exact source archive and lockfile build.
- Unit, mathematical, leakage, chronology, receipt and persistence tests.
- Generated end-to-end interruption/resume and raw-output reconstruction.
- CPU reference run with strict finite checks.

### K1 — Kaggle capability probe

- Fresh private notebook, accelerator None.
- Capture platform/environment/quota without importing model dependencies.
- Verify read-only input mounts, bounded write root, internet-disabled setting,
  clock/deadline availability, checkpoint flush and output retrieval.
- Save-and-run output must be downloaded and hash-matched locally.

Saved version `359681231` satisfies the runtime/input/egress observation portion
of K1 and remains explicitly engineering-only. The displayed receipt was
transcribed locally; a direct output download was attempted but did not complete.
Therefore K1 is not promoted to a full downloaded-and-hash-matched pass, and it
does not admit science. At the P0v5 stage, target qualification still required
the exact wheel SHA-256
`dac39add6e47f0bb5c6c47ebdd246c161a44d7b25c4ee2f43b52c3e6a1e50a06`;
that unrun candidate is now preserved and superseded by the operative P0v7
contract below.

### K2 — numerical equivalence and throughput

- Generated representative workload on CPU and one available GPU option.
- Compare forecasts, scores, gradients and checkpoint resume within frozen
  tolerances.
- GPU is admitted only if numerical equivalence passes and measured end-to-end
  throughput is at least 1.5x CPU after data/loading overhead.
- Choose one accelerator class for each compute-matched comparison; record any
  unavailable hardware rather than substituting silently.

### K3 — costed dry run

- One dataset-family-shaped generated shard with every model family.
- Measure cell runtime, peak RAM/VRAM, output growth and recovery time.
- Project the complete campaign. Proceed only if the worst credible projection
  fits the live quota, 60 GPU-hour/120 CPU-hour development ceiling, two-week
  calendar ceiling and 2,000 cell-equivalent ceiling.

### K4 — nested development search

- Maximum 24 trials and seeds 11/29.
- Fold A validation (1998-1999) only, across all three families: exactly 144
  cells. Later-fold validation is excluded from selection so it cannot leak
  into earlier fixed-matrix tests.
- Search controller sees only admitted development metrics.
- Failed trials remain in the denominator and artifact ledger.
- Select exactly one candidate; publish selection receipt before the fixed
  matrix begins.

### K5 — fixed heavy matrix

- Run the selected candidate, S11 stacker, strong non-FREM controls, mechanism
  off arm, frozen ablations and authorized robustness/scaling cells.
- Final seeds: 11, 29, 47, 71 and 97.
- Each shard is at most 8.5 active hours and 15 GiB retained output, leaving
  platform headroom.
- Each `(family, fold, seed, arm, condition)` cell has exactly one owner shard.
- Do not interpret partial headline metrics while the matrix is incomplete.

### K6 — audit and disposition

- Download every saved-version output bundle.
- Match producer and local SHA-256 inventories.
- Reconstruct all scores, intervals, multiplicity corrections, tables and
  figures from raw arrays.
- Reconcile planned versus completed/failed/invalid cells.
- Apply the frozen decision rule once. Preserve negative and mixed outcomes.

### K7 — confirmation decision

- If development is negative or inconclusive, close S12.
- If invalid, preserve the attempt and prospectively version a corrected freeze.
- If development supports the hypothesis, prepare a new confirmation protocol,
  contamination audit and budget. Do not access confirmation outcomes in S12-D.

## Shard and checkpoint contract

Every shard manifest binds:

- protocol/source/environment/data/split identities;
- exact cell registry and deterministic seed;
- accelerator class and numerical mode;
- wall-clock and active-compute ceiling;
- input and expected-output hashes;
- checkpoint frequency and resume compatibility;
- consumption path and attempt number;
- parent campaign identity and predecessor shard hashes.

Checkpoints publish model, optimizer, RNG, data-loader position, matured-feedback
state, cell status and cumulative charged time atomically. Resume is allowed
only for an incomplete shard with exact identities. A completed shard is never
retrained merely because its result is unfavorable. Hard termination without a
final receipt leaves the shard `UNKNOWN_CONSUMPTION` pending forensic review.

## Heavy budget

Development ceiling, not a target:

- 60 GPU-hours total;
- 120 CPU-hours total;
- 2,000 model cell-equivalents;
- 24 search trials;
- 8.5 active hours per scientific shard;
- 15 GiB retained output per shard;
- two calendar weeks from first scientific consumption.

No paid cloud spend, additional accounts, parallel quota circumvention or
confirmation compute is authorized. Live account quota or measured projections
may reduce this plan. The existence of all presets does not require every
preset to run; stage gates decide which declared work becomes admissible.

## Required closing receipts

- notebook URL and saved version ID;
- session start/stop and accelerator observations;
- quota before/after when visible;
- source, input, environment, manifest and output hashes;
- planned/completed/failed/invalid cell counts;
- active compute, wall time, peak RAM/VRAM and disk;
- interruption/resume history;
- producer archive hash and independent local readback hash;
- scientific disposition with zero silent omissions.

## P0v5 delivery contract

The upload candidate is exclusively
`frem_s12-0.1.2-py3-none-any.whl` with SHA-256
`dac39add6e47f0bb5c6c47ebdd246c161a44d7b25c4ee2f43b52c3e6a1e50a06`.
The qualification notebook SHA-256 is
`605bdd9d99d7e421ca088c4161ea18589a99ab0e2a53177f609363da4ac2c5ce`.
Do not upload the sdist or any earlier v0.1.0/v0.1.1 wheel as the P0v5 package.

The notebook installs the exact wheel into a fresh isolated target with no
index, dependency resolution or bytecode compilation. It rejects input
symlinks, records the complete regular-file inventory and digest, fixes the
reference numerical thread count to one, preserves installation verification,
refuses an existing output root and writes a final no-overwrite manifest of
every retained output hash and byte count.

Qualification uses three identical repetitions of the largest declared search
configuration and latest fixed fold. Projections use the maximum observed
duration, and artifact byte counts must be stable. A capability run, local
receipt, successful upload or running notebook is not qualification; only the
downloaded, hash-verified P0v5 environment receipt can admit science.

## P0v6 evidence-intake contract

The only P0v6 upload candidate is
`frem_s12-0.1.3-py3-none-any.whl` with SHA-256
`e86605be211ca5ba922cf6742c0af4c88d575915b149cb36da4d20254192b24d`.
The notebook SHA-256 is
`12d769b45107796b43f1d8237a45b940d42e9d8fad7c501c6997eb849298bced`.
Do not upload the preserved pre-README-sync v0.1.3 wheel or any v0.1.0-v0.1.2
candidate.

P0v6 target commands treat Python warnings as errors. Before the final manifest,
the notebook detects output symlinks and special files and marks the run failed
closed. After downloading a saved-version output, run
`s12-research verify-run-bundle` with both expected candidate identifiers. It
rejects missing, extra, changed, symlinked or special files; checks byte counts
and SHA-256 values; reconstructs the input-inventory digest; and cross-checks
the preflight, command and final manifest. Search selection and fixed analysis
then independently reconstruct their raw scores and policy paths before any
headline statistic is accepted. A passing P0v5 receipt cannot admit P0v6.

## P0v7 end-to-end evidence-chain contract

The only P0v7 upload candidate is
`frem_s12-0.1.4-py3-none-any.whl` with SHA-256
`e9596b134c8bbffebb5688a2627775b449ad87c805897af46503eb7dcda4cd72`.
The notebook SHA-256 is
`6db7e62e45fa90b29fe9bafda454239f411cc3227b50aa668fa9b0873916afb3`.
Do not upload P0v6 or either retained failed/superseded P0v7 build attempt.

P0v7 fixed artifacts retain validation outcomes and laws in addition to test
evidence. Consumers reconstruct validation scores, the selected non-FREM
control, test scores, holdings and wealth, and require the exact frozen months.
Search and fixed campaign roots are exact inventories: foreign files or attempt
directories are blockers, not ignored material. Every scientific cell requires
the original acquisition receipt as well as data/environment receipts; fixed
cells also require the complete v2 selection receipt binding all 144 search
manifests.

Downloaded qualification verification must include the exact wheel, version
and `--expected-mode qualify-kaggle`. Bundle integrity alone is insufficient:
`eligible_for_next_stage` must be true after reconstruction of installation
identity, runtime allowlist, three timing maxima, artifact-size stability and
all resource projections. Save verifier output outside the immutable bundle.

Kaggle saved versions do not share an atomic writable ledger. Therefore K4 is
blocked until the complete saved-version inventory is reconciled into a
fork-free cumulative attempt ledger. Per-run `consumption/` claims remain
necessary but are not, by themselves, global one-use proof.

