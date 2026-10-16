# Reproduce the Wharton/S10 campaign evidence

**PARTIAL CAMPAIGN — one CPU metadata/persistence probe; numerical and scientific work blocked.** Read REPORT.md and RUN_LEDGER.json first.

## Delivered and observed

- FREM_S10_TONIGHT_CPU.ipynb is the self-contained 12-section notebook. Source and helper are embedded and hash-checked; no Mac runtime path is needed.
- PASTE_CPU_PROBE_BOOTSTRAP.py reconstructs that byte-identical notebook and runs its exact eight code cells in probe-only mode. It was used through Kaggle's code editor because file import permissions were unavailable.
- Private version 1 is saved at [Kaggle](https://www.kaggle.com/code/ryangomez010/wharton-s10-cpu-readiness-2026-10-16/output?scriptVersionId=358930173). It contains the actual probe and source, not model results.
- retrieved/version1/kaggle-results.zip is the actual downloaded 2.45MB archive. RETRIEVAL_VERIFICATION.json records CRC checks and three hashes matched against producer observations. retrieval-receipt.json records the verified external attestation.
- The observed target Python3.13.15/NumPy2.1.3 fails the sealed Python3.12/NumPy2.5.4 contract. The successful retrieval receipt cannot override that failure.
- Stop Session was clicked and the UI displayed Off. Final account inactivity/quota confirmation was blocked by the local disk-space failure. Check SESSION_CLOSEOUT.json and confirm manually before any new session.

## Exact local checks

From the repository root, the bounded stdlib test commands are:

    rtk proxy python3 -B -W error::ResourceWarning -m unittest discover -s research/frem_s10_tonight_20261016 -p test_tonight_runner.py -v
    rtk proxy python3 -B -m unittest discover -s research/frem_s10_tonight_20261016/budget -p test_account_budget.py -v

Nineteen wrapper tests and sixteen budget tests passed in their retained receipts. These do not run the numerical model or certify Linux production. The exact notebook/runner hashes are recorded in NOTEBOOK_IDENTITY.json; the separate-agent same-session review is REVIEW.md, with a separate paste-transport review.

The actual external retrieval attestation was produced using:

    rtk proxy python3 -B research/frem_s10_tonight_20261016/tonight_runner.py attest --downloaded-probe research/frem_s10_tonight_20261016/retrieved/version1/session/persistence-probe.json --expected-probe-sha256 18229e7b5e29fa9416233d6d383d5aa400555eae20d97681f2a2d9fff361c67a --kernel-version ryangomez010/wharton-s10-cpu-readiness-2026-10-16/1 --output research/frem_s10_tonight_20261016/retrieved/version1/retrieval-receipt.json --ack-retrieved

The output already exists; preserve it. A new independent retrieval needs a new output path and a truthful observation of the private version. The helper has no provider signature and cannot authenticate an invented download claim.

## Future launch and restore

Read NOTEBOOK_USAGE.md for the full exact flow. Keep the notebook Private, accelerator None, Internet Off and MODE probe. Do not run the GPU path: none exists. A compatible target must be established before smoke; this package performs no blind installation or upgrade.

Use one interactive CPU session. Quick Save defaults to Never save output: explicitly choose Save output for this version, then download the exact private version and hash-check it. Do not use Save & Run All simply to preserve completed work, because it starts a new execution.

The known state ZIP is retrieved/version1/exports/state-5c6c61614e45491fbe7726bf2015e12b.zip with SHA256 8f87cbf3a5e081732ae3a16588d07a4714296ad632021bb3376726ceaa1b0733. It contains probe metadata only. For a genuine later checkpoint, attach the retrieved ZIP privately, set RESTORE_ARCHIVE and its independently recorded hash, and use a fresh session directory. Source/runtime/config/data identities and consumed time must match. Do not merge outputs, delete failed receipts or reset budgets. Cross-machine numerical resume has not been demonstrated.

The absolute deadline is **16 October 2026, 22:00 IST / 16:30 UTC**. Later execution requires a new recorded request and candidate. The prepared smoke sequence does not launch the old scientific production plan. All scientific cells remain incomplete and SCIENTIFIC_MANIFEST_UNFROZEN.json is intentionally not a scientific freeze.

## Package map and claims

manifests/ contains source, actual environment, proposed generated data and the unfrozen scientific question. RUN_LEDGER.json distinguishes the successful metadata/persistence cell from unexecuted model/scientific cells. budget/ contains a disabled account-level ledger and a tested accounting appendix; no provider-wide lock or final postrun quota observation is claimed.

observations/ preserves the initial UI failure, recovery, upload refusal, exact editor readback, actual cloud console/environment, failed local host gate and incomplete shutdown closeout. analysis/ contains evidence-derived status/claim tables; no numerical results or fabricated plots. MANUSCRIPT_TEXT.md supplies appropriately limited methods/results wording.

RECEIPT.json, CLOSURE_RECEIPT.json, CHECKSUMS.json and evidence.zip identify the final local delivery. ARCHIVE_RECEIPT.json is kept beside the ZIP to record its final hash and readback; it is not inside its own archive. Prior negative studies and the scientific source were not changed. No commit, push, public release, dataset publication, paper submission or automation was performed.
