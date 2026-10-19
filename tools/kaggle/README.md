# FREM-S10 Kaggle CPU notebook

Upload `FREM_S10_KAGGLE_CPU.ipynb` to a **private** Kaggle notebook. Select
**Accelerator: None** and **Internet: On** for setup. Start in `MODE = "probe"`.
This delivery retains the original 16 October 2026, 22:00 IST deadline.

Use one copy of the notebook source per Kaggle draft. The delivered notebook is
657201 bytes; appending another embedded copy can exceed Kaggle's observed 1 MB
kernel-source limit and prevent saving versions. Preserve old attempts in their
saved versions and evidence files instead of duplicating their payload cells.

The notebook installs uv 0.12.5 in `/kaggle/working/frem_s10_runtime/tools`,
creates a managed Python 3.12 virtual environment, and installs the original
hash-pinned NumPy 2.5.4 wheel. Every model subprocess uses that environment's
Python, even when the notebook kernel is Python 3.13. No kernel downgrade or
restart is needed. Internet may be disabled after setup. Downloads must succeed;
there is no fallback to a different NumPy version. Setup is capped at 300 seconds
separately from the existing 600-second engineering wrapper allowance.

Setup logs and `READY.json` or `FAILED.json` remain in the runtime directory.
An interrupted/failed setup is never silently overwritten or retried. Inspect its
logs and retain them before preparing a separately identified attempt. A READY
runtime is checked again without reinstalling. uv's interpreter downloads are
managed by uv; only the NumPy wheels use the supplied research hash allowlist.
Actual interpreter and NumPy bytes are bound by the existing probe.

Save the probe output in a private version, download it, and use the **new**
`runner.py attest` command described in section 8. The old notebook's receipt
cannot authorize this changed runner. Then set `MODE = "smoke"` with the new
retrieval receipt to execute the existing 12-origin interruption/resume check.
The source bundle, model, seeds and thresholds are unchanged. The production
resource gate remains unchanged. Kaggle's cgroup namespace hides ancestor
constraints, so the wrapper uses the separately recorded
`s10-kaggle-namespace-smoke-v1` contract for this generated 12-origin check only.
It requires three stable samples, at least 1536 MiB visible headroom, stable swap
and pressure counters, and a finite visible memory ceiling. Each numerical worker
has a hard 1 GiB address-space ceiling, 75-second CPU ceiling and parent wall
deadline. Hidden ancestor limits remain UNKNOWN; this is not production admission.

Intermediate state exports explicitly say `resumable: false` while a wrapper
invocation is open. Only a terminal snapshot with `resumable: true` can be used
for restore; a failed command or unknown hard-kill time still blocks continuation.
Download the final receipt separately alongside the final state archive, whose
hash the receipt names. Stop the Kaggle session explicitly after saving outputs.

Build: `python tools/kaggle/build_notebook.py`.
Tests: `python -m unittest discover -s tools/kaggle -p 'test_*.py' -v`.

Actual Kaggle setup and namespace probe passed on 16 October 2026 using Python
3.12.14 and NumPy 2.5.4. See `research/frem_s10_kaggle_live_20261016/` for target
evidence. The 12-origin interruption/resume/evaluation/reuse/export sequence
completed on Kaggle; final Version 3 output hashes and inventories were verified
locally, and the CPU session was confirmed stopped. Full Linux production qualification
and scientific evaluation remain unverified.
This is a CPU NumPy model; GPU execution is unsupported.

Environment setup follows the official [uv Python documentation](https://docs.astral.sh/uv/concepts/python-versions/)
and [hash-checked installation documentation](https://docs.astral.sh/uv/pip/compile/).


## Full S10 suite on Kaggle

The full generated test notebook is `research/frem_s10_kaggle_full_tests_20261016/S10_FULL_TEST_SUITE.ipynb`; its private input dataset and immutable attempts are documented in that directory's `REPORT.md`. It runs all S10 pytest tests with strict markers and warnings as errors, followed by every Kaggle-wrapper unittest. This is separate from the original sealed smoke notebook and its consumed ledger.

Kaggle recursively extracts nested ZIP inputs. The full-suite notebook therefore embeds the exact historical archive needed by wrapper fixtures and verifies its original digest. It verifies every uploaded source/test file, applies the recorded corrected replay file, writes the corrected inventory, and installs that exact package with `--no-deps -e`. All five `THREAD_ENV` controls must be set before importing NumPy, including `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS`; `PYTHONPATH` alone does not support isolated import tests that deliberately remove it.

The retained notebook is a dated execution receipt. A later run requires a fresh attempt directory and reviewed setup deadline; do not overwrite these outputs or silently change the old smoke budget. Passing S10 tests does not claim the separate root/legacy/successor suites passed or establish production/scientific qualification.
