# Kaggle full S10 test suite — 16 October 2026

Scope: every test collected from `successors/frem_s10/tests` using its strict pytest configuration, warnings as errors and no selection filters; every `tools/kaggle/test_*.py` unittest. Root predecessor/legacy and other successor packages are separate suites, not included in this S10 result. Tests use generated fixtures; no market-data input was uploaded.

Private source dataset: https://www.kaggle.com/datasets/ryangomez010/wharton-s10-full-suite-source-2026-10-16

Notebook: https://www.kaggle.com/code/ryangomez010/wharton-s10-full-test-suite-2026-10-16

## Attempts

1. Source preparation stopped before numerical execution because Kaggle recursively expanded the nested original source ZIP. `attempt1-upload-error.txt` retains the observation. The exact original archive is embedded and hash-checked in subsequent drivers.
2. Full suite completed: 688 passed, 11 failed in 163.25 seconds. All 37 wrapper tests passed. Console and exact driver are retained. Eight smoke failures arose from omitted NUMEXPR/VECLIB thread controls, two isolated-import failures from not installing the package, and one causal replay regression from comparing unique forecast origins against repeated origins across overlapping generated windows.
3. Corrected candidate: all five declared thread controls set before import, editable installation of exact source without replacing NumPy, and replay origin validation uses the sorted unique origin union. Original tests/assertions are unchanged. Fresh source/runtime/output directories preserve the failing attempt. All 699 S10 cases passed in 183.95 seconds, and all 37 wrapper cases passed in 1.291 seconds. No tests/assertions were skipped or edited. The result ZIP printed SHA-256 `76a49f321c8090bbb84fda56edbe7adddcdce80054733144e1b4da61f487541d`; saved Version 1 (script ID 359065132) was retrieved separately and matched that digest. Per-node/JUnit readback confirms the identical 699 node IDs, zero skips/failures and exact corrected source inventory. Both failed and passing archives are retained. The CPU session is stopped.

## Evidence boundary

Separate generated-test namespace contract; three stable observations, 1 GiB per-process address ceiling, 600 CPU seconds per process, 900 seconds per command wall limit, one numerical thread and 512 MiB visible reserve. Visible namespace pressure is monitored. Hidden ancestor constraints remain UNKNOWN. This is not production host admission, independent reproduction, a completed mutation/packaging qualification, a market study or scientific confirmation. Previous sealed smoke and failed scientific histories remain intact.
