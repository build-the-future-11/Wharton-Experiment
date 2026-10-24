# Final program execution: preflight and evidence-tooling repairs

24 October 2026. Scientific status: **NOT_EVALUATED**. S14 0.2.4 remains at **0/19 qualification cells and 0/171 historical cells**. S15 has not started.

The requested program has started. Its declared dependency chain stops at the specific browser-upload confirmation. The private Kaggle import dialog is ready; no file has been selected, uploaded, or executed.

## Completed

- Captured branch, HEAD, dirty-state digest, host, operator, purpose and pre-edit state snapshots in `SESSION_START.json` and `before/`.
- Reverified the unchanged source identity, wheel, archive and notebook against the immutable 0.2.4 release. `RELEASE_VERIFICATION.json` also checks 15 wheel modules, 17 payload hashes, version agreement and the embedded archive.
- Reverified existing owner `ryangomez010`, private notebook, Versions 1–8, draft session off, T4 x2, Internet off, no persistence, pinned 2026-10-03 environment and attached Huge Stock Market Dataset. See `KAGGLE_PREUPLOAD_OBSERVATION.json`.
- Enumerated the frozen campaign into `EXECUTION_REGISTRY.json`: 19 arms × seed 11 for generated qualification, then 19 arms × 3 seeds × 3 folds = 171 historical cells. All are `NOT_STARTED`.
- Corrected the master checklist's qualification-seed transcription error. The frozen package was not edited. Corrected stale top-level launch pointers to 0.2.4; historical 0.1.9 and superseded-release objects remain preserved.
- Reproduced two inventory-tool failures before repair: existing-receipt overwrite and overwrite through an output symlink. The retained failure log is `inventory_regression_before.log`.
- Repaired the external inventory tool with exclusive writes, symlink/path checks, regular-file checks, streaming SHA-256, and baseline comparison. It rejects missing evidence, duplicate baseline paths, wrong counts, changed sizes and changed hashes. New untracked paths are listed separately instead of silently covered by an old baseline.
- Added `scripts/verify_s14_saved_notebook.py` to compare downloaded notebook cells and the embedded archive with the trusted release without executing notebook code. It accepts only equivalent string/line-array serialization; changed source, cell order, types, missing/extra cells or altered archives are rejected. It does not certify Kaggle ownership or runtime.
- Passed **20 dependency-free tooling tests**, warning-strict, plus changed-file Ruff lint and formatting. See `tooling_tests_final.log`. These are engineering results, not target CUDA qualification.
- Recorded the correction through `frem record-decision`; see `DECISION_RECEIPT.json` and the appended research journal event.

## Preservation evidence

`EVIDENCE_BASELINE_HASHES.json` covers all **4,902** retained untracked evidence files. Original inventory paths and sizes match. Because the earlier inventory had no content hashes, this establishes current byte identities without claiming a retrospective hash comparison.

`EVIDENCE_PRESERVATION_FINAL.json` compares every file with the newly captured hashes. No retained experiment output is edited, deleted or bulk-staged. No S14 source, notebook, release archive, wheel, configuration or frozen tolerance is changed.

## Current gates

Local numerical admission failed at **115.0, 162.65625 and 228.015625 MiB** against the **512 MiB** minimum; pageouts were stable at 434384. See `LOCAL_ADMISSION.json`. No NumPy/Torch/SciPy/XGBoost work was started.

The computer-use tool requires permission for the specific browser file upload; the pending question names the exact 0.2.4 notebook and embedded archive, private destination, and bounded frozen campaign. Broad checklist execution does not satisfy that upload rule. No new API token or credential was created.

S15 implementation, its 24-cell development screen, 210-cell confirmation, 162-cell ablations, later benchmarks, point-in-time data admission, protected freeze, independent reproduction and final results paper remain downstream. This session does not mark any of them complete or claim every repository flaw is closed.

## Next execution

1. After the pending confirmation, upload only `successors/frem_s14/kaggle/FREM_S14_GPU_0_2_4.ipynb` to the existing private notebook. The release archive is embedded in that file.
2. Save source without launching; download it, retain the actual Kaggle version identity and settings, and run the new source verifier with a fresh receipt path. `READBACK_TOOL_LOCAL_SELF_CHECK.json` is only a local tool check, not remote readback evidence.
3. Run the unchanged notebook under the frozen T4/Internet-off contract. Its tests and qualification must pass before intake and the 171-cell suite. Enforce the 13,500-second total ceiling and retain failures.
4. Retrieve and independently verify all output before classifying S14. Start downstream S15 work only under the checklist's prerequisite gates.

The exact upload and saved-source comparison are the next actions; additional planning is not a substitute for them.
