# S14 0.1.2 hardening audit

The reservoir, engram, masked multi-offset EMA JEPA, finite 2D stochastic field,
bounded neurogenesis and multiobjective routing remain. All prior evidence is retained.
The scientific family remains 19 arms x 3 seeds x 3 folds = 171 cells, with identical
configuration bytes, budgets, blocks, hypotheses and thresholds. No market cell started.

| Finding | Consequence | Repair | Acceptance evidence |
|---|---|---|---|
| Nested manifests excluded by basename | A changed child inventory could escape the parent hash | Exclude only the root inventory; hash child inventories | Prior defect reproduced; new tamper test passes |
| Special files ignored | A bundle could appear complete despite unsupported entries | Reject symlinks and special files, including invalid roots | Local hostile-file tests pass |
| Delayed decision after target observation | Already realized returns could enter the forecast target | Strict decision-before-first-target contract | Local clock test passes; integrated Samples regression awaits GPU |
| Configurable protected cutoff after 2025 | Protected observations could be admitted accidentally | Fixed maximum cutoff in config and adapters | Prior defect reproduced; rejection tests pass |
| Extra/missing CSV fields and unchecked OpenInt | Malformed input could be accepted silently | Exact row width and finite nonnegative integer OpenInt | Prior extra-field defect reproduced; all hostile input tests pass |
| Derived feature overflow | Valid finite inputs could produce nonfinite features | Check every derived feature before publication | Implemented; normal end-to-end generated intake passes |
| Floating-point timestamp conversion | Microsecond identity could be lost | Exact integer timedelta conversion | Prior defect reproduced at year 2500; integer tests pass |
| JSON exponent overflow | `1e999` could enter metadata as infinity | Strict finite parser for exponents and named constants | Prior defect reproduced; parser tests pass |
| Admission validates only endpoints / tolerates NaN | Invalid middle samples could be accepted | Check all samples and all pageout counts | Prior NaN defect reproduced; bad-middle tests pass |
| Resume trusts unverified/nonfinite time | Cumulative budget could be bypassed | Verify prior manifest and finite time below unchanged cap | Tampering, exhausted and nonfinite cases rejected locally |
| Half-precision graph logits | A populated AMP memory bank could overflow its mask sentinel | Disable autocast locally and use float32 retrieval | Regression written; numerical execution pending |
| Ineligible extreme memory values | Invalid neighbors could poison moment arithmetic | Mask values before moments; zero absent-memory uncertainty | Numerical regression written; pending |
| Second-moment subtraction | Small positive variance lost at large locations | Centered within/between mixture variance | Neural/hybrid regressions written; pending |
| Unnecessary Python products and metric transfers | Avoidable preprocessing and synchronization overhead | Read-only strided-window products and optional metric collection | Equivalence/gradient regressions written; no speed claim |
| Weak completion/cell accounting | Missing or reordered members could look complete | Explicit COMPLETE/FAILED/NOT_STARTED inventory and exact completion contract | 19/57/171 cardinality and hostile-completion tests pass |
| Incomplete suite readback | Cross-fold family or control settings could drift | Balanced calendars, control contracts, hardware binding and complete Holm reconstruction | Structural rejection tests written; full numerical readback pending |
| Manual downloaded-output inspection | Source/payload tampering could be missed | Streaming ZIP readback anchored to trusted release, installed source and nested manifests | 11 archive tests and real-wheel generated contract fixture pass |

## Evidence

- 48 unique dependency-free tests pass from source and a fresh installed wheel, with warnings as errors. These are 48 unique cases, not 96.
- Six concrete defects were executed against the preserved 0.1.1 source and all six repaired cases pass against 0.1.2; see `regression_reproduction.json`.
- Ruff, format including the notebook, AST/compile, wheel RECORD and 15-module source equality pass. Two builds produce identical wheel, ZIP and notebook bytes.
- The notebook embeds the exact release ZIP. The generated archive fixture tests packaging only and explicitly carries a failed-test marker; it is not Kaggle or numerical execution.
- Static inventory declares 84 tests: 48 dependency-free plus 36 numerical. Actual target collection/pass/fail/skip counts must be retained when executed.
- Latest numerical admission failed at 293.08/417.56/220.45 MiB available with pageouts 441851/441851/442624. The 512 MiB/stable-pageout guard is unchanged.
- Kaggle's editor still reports unavailable reCAPTCHA after one supported reload. `kaggle_blocked.txt` and `kaggle_blocked.png` retain the observation.
- GPU version 359930995 remains a preserved 41-pass/1-fail engineering result for 0.1.0. It does not qualify 0.1.2. The 0.1.1 candidate is fully retained in `before/`.

## Remaining execution

Upload the exact current notebook and verify saved code identity. Execute all target tests,
then generated 19-arm CUDA qualification, then strict intake and all 171 declared historical
proxy cells. Download and independently verify the full archive and suite, including every
failure. The active 30-minute follow-up requires candidate readiness and exact source/release
identity before upload. Never duplicate a running/completed candidate or retry a valid market
result to seek improvement. Broader whole-market, macro, news and Jane Street studies still
need their own data contracts and prospective scope; they are not represented as completed.

Canonical receipt: `RECEIPT.json`. Scientific status remains `NOT_EVALUATED`.
