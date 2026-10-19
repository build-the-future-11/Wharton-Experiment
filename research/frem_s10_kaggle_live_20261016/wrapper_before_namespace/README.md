# FREM-S10 Kaggle CPU notebook

Upload `FREM_S10_KAGGLE_CPU.ipynb` to a **private** Kaggle notebook. Select
**Accelerator: None** and **Internet: On** for setup. Start in `MODE = "probe"`.
This delivery retains the original 16 October 2026, 22:00 IST deadline.

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
The source bundle, model, seeds, thresholds and resource gate are unchanged.

Intermediate state exports explicitly say `resumable: false` while a wrapper
invocation is open. Only a terminal snapshot with `resumable: true` can be used
for restore; a failed command or unknown hard-kill time still blocks continuation.
Download the final receipt separately alongside the final state archive, whose
hash the receipt names. Stop the Kaggle session explicitly after saving outputs.

Build: `python tools/kaggle/build_notebook.py`.
Tests: `python -m unittest discover -s tools/kaggle -p 'test_*.py' -v`.

The new notebook is locally contract-tested, not yet executed on Kaggle. Full
Linux production qualification and scientific evaluation remain unverified.
This is a CPU NumPy model; GPU execution is unsupported.

Environment setup follows the official [uv Python documentation](https://docs.astral.sh/uv/concepts/python-versions/)
and [hash-checked installation documentation](https://docs.astral.sh/uv/pip/compile/).
