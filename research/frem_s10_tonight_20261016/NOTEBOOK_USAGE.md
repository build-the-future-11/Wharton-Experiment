# Wharton/S10 tonight notebook

Import `FREM_S10_TONIGHT_CPU.ipynb` into the private Wharton notebook. Select **accelerator None**, Internet Off, and keep the default `MODE = "probe"`. Source and helper bytes are embedded and hash-checked; no laptop path or cloud credentials are required. No dependency installation or upgrade occurs. All twelve sections must run in their written order.

The first pass performs stdlib metadata/persistence work. Python 3.12 and NumPy 2.5.4 are required for the maintained numerical path. Incompatible Python does not import the package; incompatible NumPy does not import numerical libraries. On Python 3.12, the probe also attempts the maintained `check-host` command and retains pass/refusal. Otherwise it records `NOT_RUN_PYTHON_INCOMPATIBLE`. A blocked environment can produce a persistence receipt; that receipt does not override the compatibility gate.

Run the probe once in a single interactive CPU session. Record the displayed `probe_sha256`. In **Quick Save → Advanced Settings**, choose **Save output for this version**; the default Never save output does not preserve this evidence. Confirm the saved version, then download its `frem_s10_tonight_20261016/session/persistence-probe.json` using that private version's Output view. Do not use Save & Run All merely to save already generated output: it launches a fresh execution.

On the local workstation, after actual retrieval, produce the explicit retrieval attestation:

```sh
rtk proxy python3 -B research/frem_s10_tonight_20261016/tonight_runner.py attest \
  --downloaded-probe /ABSOLUTE/PATH/TO/DOWNLOADED/persistence-probe.json \
  --expected-probe-sha256 PRODUCER_DISPLAYED_SHA256 \
  --kernel-version ryangomez010/wharton-s10-cpu-readiness-2026-10-16/VERSION \
  --output /ABSOLUTE/NEW/PATH/retrieval-receipt.json \
  --ack-retrieved
```

This compares downloaded bytes to the separately observed producer digest and checks source/runner/deadline identity. The operator explicitly attests the private notebook version; the helper has no platform signature and cannot independently prove where a file was downloaded. Never create that attestation inside the producer notebook.

Only after that receipt exists, paste its exact JSON into `RETRIEVAL_RECEIPT`, select `MODE = "smoke"`, and run the notebook in order. Fresh Linux admission must pass before each numerical subprocess. The maintained sequence is: stop after exactly three committed origins (exit 130), resume to twelve, reconstruct saved scores, validate completed reuse, and export. Model settings remain seed 11, primary variant, float64, one thread and 60 cumulative model seconds. Output semantics and counts are checked as well as return codes. No market/development/confirmatory campaign is available through this notebook.

The wrapper has independent 600-second command-reservation and total-wall ceilings, including conservative setup/cleanup/export charges. Each repeated bootstrap is charged again. Absolute cutoff is **2026-10-16 16:30 UTC / 22:00 IST**; source setup, commands and final export have watchdogs. These bounds do not prove a Kaggle platform session has stopped or establish quota accounting.

Live stdout is streamed and retained in command logs. The `ENGINEERING_RECEIPT.json` reports model stages with final export pending. Only `FINALIZATION_RECEIPT.json`, created after a successful full-state export, reports generated CPU engineering completion. It explicitly leaves platform-session termination unobserved.

For recovery, retrieve a `state-*.zip` and its JSON digest, attach the ZIP as a private input, then set `RESTORE_ARCHIVE` and `RESTORE_ARCHIVE_SHA256` in a fresh session. Exact source/runtime/config/data bindings are rechecked. Restore never merges existing directories or resets consumed time. Intermediate snapshots from an invocation still in progress retain its unfinished start receipt and deliberately refuse automatic continuation. Gracefully finalized exports contain the end receipt. Hard kills, unknown duration, prior failed commands and changed Python/NumPy/OS identity require review. Completed campaigns are retained rather than automatically rerun.

Retrieve the final ZIP, its digest, finalization receipt, command logs and smoke export. Then use Kaggle's explicit **Stop Session/Cancel** for the notebook's interactive and batch sessions and inspect Active Events. Python exit, closing the tab, saving a version or writing a receipt is not confirmation of platform termination. No GPU runtime, Linux production qualification or scientific success is claimed by this handoff.
