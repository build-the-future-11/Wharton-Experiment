# FREM-S14 0.2.4 exact execution handoff

Status on 25 October 2026: **BLOCKED for the complete requested intensive system**.
The exact frozen S14 notebook and sources are packaged. Local source tests passed
108 with one CUDA-only skip. A separate CPU end-to-end qualification attempt
failed resource admission before imports. CUDA qualification, sufficient account
quota, full suite execution and whole-suite cross-session resume are unverified
or unavailable. These limits prevent claiming fully validated manual readiness.

## Current saved source

Account access is restored. The exact candidate is privately saved as
[Version 9 / 362065796](https://www.kaggle.com/code/ryangomez010/frem-s14-gpu-benchmark-20261019?scriptVersionId=362065796).
Downloaded saved cells and embedded archive match the trusted release. This was
Quick Save only; blank execution counts and zero outputs confirm no model run.
GPU quota displayed **00:44 / 30 hrs** at 13:53 IST on 25 October. Restore enough
quota for the 13,500-second campaign before Save & Run All. Do not re-import or
create another source-only version unnecessarily.

## Exact next launch, once sufficient target quota is available

1. Sign into the existing Kaggle owner `ryangomez010`. Open
   https://www.kaggle.com/code/ryangomez010/frem-s14-gpu-benchmark-20261019
   and inspect saved versions. Continue any already running exact 0.2.4 run;
   never create a duplicate or rerun Version 8.
2. Use the verified saved Version 9. Only if recreating it elsewhere, import **FREM_S14_GPU_0_2_4.ipynb** from this directory. It embeds the exact
   source archive, wheel, tests and configuration. No GitHub credentials, separate
   private source dataset, internet install or model-code edits are needed.
3. Attach **borismarjanovic/price-volume-data-for-all-us-stocks-etfs** (Huge Stock
   Market Dataset). Intake recursively resolves the seven exact `Data/ETFs`
   members and checks their hashes in `input_manifest.json`. Never substitute
   another asset/version or fill a missing date to make the run pass.
4. Keep visibility **Private**, Internet **off**, accelerator **GPU T4 x2**.
   Require Python 3.11–3.13 and the installed versions listed in
   `requirements-kaggle.txt`. Confirm sufficient remaining GPU quota for the
   13,500-second (3 h 45 min wall-clock) ceiling. Quota accounting is provider
   controlled; two available GPUs do not imply two-device model execution.
5. For the existing verified Version 9, proceed directly after confirming settings; repeat source readback only if source changes. Keep the configuration cell at `MODE = "historical_suite"`, `DEVICE = "cuda"`.
   Save source without execution, download the saved notebook, retain its actual
   version ID and run the repository readback command below. Verify all cells,
   embedded release, owner, privacy, attached input and settings before launch.
6. Save Version with **Save & Run All**. Record the actual new version URL and
   terminal status. Submitted or queued is not running. The fixed chain is
   admission → tests → 19-arm generated qualification → independent qualification
   verifier → strict intake → 171 historical cells → full-suite verifier.
7. Save/download **all output**, including failed stages. The notebook prints the
   exact `/kaggle/working/frem-s14-<UTC>` run directory. Download Kaggle's complete
   output archive where available. Do not copy only selected metrics.

From the repository root, compare downloaded saved source:

```sh
rtk proxy .venv/bin/python scripts/verify_s14_saved_notebook.py \
  --saved-notebook /absolute/path/to/downloaded.ipynb \
  --output /absolute/path/to/new-saved-source-receipt.json
```

## One archive to return

If Kaggle supplies separate downloaded files, preserve their one run-directory
layout and use the standalone `pack_results.py` shipped here:

```sh
python3 pack_results.py --run /absolute/path/to/frem-s14-run \
  --output /absolute/path/to/frem_s14_results.zip
```

Paths above refer to the actual downloaded run and a new destination you choose.
Return `frem_s14_results.zip` and its `.receipt.json` sidecar. Packing is streaming,
refuses overwrite and symlinks, compares every archived byte, and detects input
changes. A transport pass does not validate scientific results. Keep all
checkpoints, raw laws, score vectors, preprocessing/split records, manifests,
logs, configuration, input hashes, environment and failure records. No historical
result or numerical metric is shipped in this launch package.

## Mandatory independent result check

Use the trusted source archive shipped here, not a digest supplied by a remote
run. From the repository root:

```sh
rtk proxy env PYTHONPATH=successors/frem_s14/src .venv/bin/python -m frem_s14.archive \
  --archive /absolute/path/to/frem_s14_results.zip \
  --release kaggle/frem_s14_kaggle_bundle-0.2.4.zip \
  --expected-release-sha256 ddc04e93b37a086684624a2ad8d03de9269df1fa7f23ed14b9fcc8b2943f7c21 \
  --output /absolute/path/to/new-integrity-receipt.json
```

Only after integrity passes, inspect the archive paths and extract normally to a
fresh directory. On a resource-admitted numerical host, run:

```sh
rtk proxy env PYTHONPATH=successors/frem_s14/src .venv/bin/python -m frem_s14 verify-suite \
  --bundle /absolute/path/to/frem-s14-run/historical_suite
```

Both checks must pass before interpreting comparisons. The latter reconstructs
all 171 cells, raw law/ledger arithmetic, splits and multiplicity correction.
If either fails, retain the failed evidence and record its exact defect. Do not
weaken tolerances or discard seeds. Current manuscript results remain pending.

## Recovery and limits

- Missing attachment or member/hash/calendar mismatch: stop and retain `intake`
  failure. Attach the declared unchanged source; an incompatible source requires
  a prospective versioned protocol, not a relaxed check.
- CPU-only session: select the required GPU before running. No CPU substitution
  for this target campaign.
- Incompatible Python/CUDA/dependencies: select a compatible existing Kaggle
  environment; retain its exact version. Do not enable network or upgrade inside
  the frozen run. Target tests must pass on the actual selected environment.
- OOM, nonfinite values, failed tests or verifier: retain logs/checkpoints. No
  batch-size, precision, architecture or tolerance edits in this freeze.
- Browser disconnect: inspect the saved server run before doing anything; a
  running saved version continues remotely. A disconnected interactive session
  is not a saved completion receipt.
- Timeout/exhausted quota: retain the failed/partial run. **The frozen notebook
  does not implement whole-suite cross-session resume.** Its per-cell training
  resume API does not prove campaign-level recovery. Do not rerun completed cells
  or claim an unchanged cross-session restart is available. A separately
  versioned successor needs an audited ledger, charged budgets and new-process
  equivalence before that capability can be advertised.
- Oversized output: save a durable private output version; download all shards
  and reconstruct the complete tree before packing. Do not omit losing cells or
  checkpoints merely to reduce the transfer. Confirm local free space first.
- Confirmation/OOD/full-world equities: outside this frozen historical-proxy
  contract. The 2025+ boundary remains closed; no confirmatory mode is offered.

The intended duration is bounded by protocol, not a promise of a complete run.
The predecessor exhausted its smaller fold budget. A launch is not evidence of
positive results, profitability, novelty or publication readiness.

## Portable static validation

Run `python3 validate_kaggle_package.py --directory . --output new-package-check.json` from the extracted handoff directory. It checks pinned notebook/archive/wheel identities, all payload members, private/GPU/offline metadata and the dataset member allowlist. Its pass explicitly does not certify runtime or manual-launch readiness.
