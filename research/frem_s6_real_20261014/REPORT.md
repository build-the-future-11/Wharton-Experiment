# S6 real-price experiment readiness — 14 October 2026

Real market data acquired and reconciled; protocol frozen; **zero market fits**.

The fixed SPY/TLT/GLD benchmark contains 422 training dates (2005–2014), 84 calibration dates (2015–2016), and 84 outer development dates (2017–2018), with three assets per date: 1,770 rows. Each observation predicts an unadjusted Monday-open to next-Monday-open price return from eight earlier daily close returns. Training JEPA targets are eight later daily return observations, purged before the fit cutoff. No missing price is filled. The source request excludes 2019 onward. Current quote metadata returned by the provider is discarded without using it.

This is explicitly retrospective development: current-vintage quotes, three surviving ETFs, assumed conservative availability times, no historical point-in-time certification, and no approved WInS eligibility claim. Dividends are excluded. Holiday endpoint exclusions leave gaps: economic outputs would describe included episodes, not a continuous total-return portfolio or executable fills. No Laura funding confidence or investment recommendation follows from this dataset.

The fixed protocol is `freeze2/protocol.json`, with 24 cells, seeds 11/29/47, full/half/zero contexts, widths 8/16, JEPA and ripple ablations, and the real XGBoost expert enabled. The primary family is full-context width 8 against equal routing, measured by date-level mixture NLL. Every control, both prescribed date-block lengths, and all costs must be reported. The precision analysis explicitly does not claim power for an unknown NLL effect. The 84 evaluation dates provide limited regime coverage. Prior other-asset training exposure prevents treating this history as independent confirmation.

`freeze1` is retained: it captured the same scientific matrix before the portable runner existed. `freeze2` adds the final executable source identities; no model was fitted between them. Old S6 and predecessor source files were not edited.

Verification completed:

- The S6 structural validator and exact configuration/fold validator passed.
- Every label and feature was independently reconciled against retained real quotes (`data_reconciliation.json`).
- Six new engineering tests passed, including resource rejection and cgroup headroom. Simulated OS counters in those tests are not market data or performance results.
- Linux resource handling is unit-tested, not yet exercised on a real Linux host. The full S6 numerical suite will run on the admitted execution host before fitting.

Compute attempts:

- Configured SSH runner at its saved address timed out after network escalation.
- Google Cloud instance inspection failed because credentials returned `invalid_grant`.
- Hugging Face identity worked, but the Jobs backend returned UNAVAILABLE. No job was created or charged.
- Local portable launch wrote `outputs/frem_s6_real_20261014_attempt1/admission.json`: 96.06, 146.50, 160.66 MiB available, below 512 MiB. Zero model fits started.

The next required user action is `gcloud auth login` on this Mac. Once authentication works, inspect the existing instance and its current address, available memory, cgroup limits and Python environment before transfer. Do not start a new billable instance or change machine type without a concrete cost/authorization decision. The provided package is for private transfer to the user's compute, not public redistribution.

On an admitted host with Python 3.11 or later, install the pinned requirements in an isolated environment, preserve the package directory structure, and run from its root:

```sh
python -m successors.frem_s6_real.run \
  --root data/s6_real_etf_20261014 \
  --frozen research/frem_s6_real_20261014/freeze2 \
  --output outputs/s6_real_remote_attempt1
```

Use a new output directory for every attempted execution. The runner checks source/data identities and resource admission twice, caps a worker at 768 MiB and two hours, records all 24 fits and checkpoints, and preserves failures. Copy the complete output directory back before deleting any remote workspace. Package installation and the remote experiment remain unexecuted.

After results exist, retain any negative result and diagnose mechanisms. A change justified by those results requires a new recorded hypothesis, source version and freeze; do not rerun the same outer dates to manufacture validation. The newly unacquired 2019 period is a possible later development validation, not a presently certified untouched global holdout. Protected 2025+ observations remain closed.
