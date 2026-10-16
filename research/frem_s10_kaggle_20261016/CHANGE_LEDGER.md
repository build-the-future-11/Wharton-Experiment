# Change and evidence ledger

This ledger describes the 16 October local engineering revision. Status **BLOCKED** for a scientific Kaggle campaign; scientific status **NOT_EVALUATED**. Exact changed-file hashes are in `CANDIDATE_IDENTITY.json` (22 package files, including 21 Python files); canonical documentation/state and two generated configs are separate.

| Flaw / cause | Severity | Implemented repair | Evidence / limit |
|---|---|---|---|
| No portable bounded real-kernel execution interface; production commands require exact scientific admission | High for debugging and portability | Extended existing CLI with preflight, finite smoke plan, run/resume/evaluate/export; strict CPU/float64 schemas | Actual 11-stage CLI sequence passed; larger generated and market execution not run |
| Interruption could strand committed causal state or reset elapsed budget | High | Safe JSON full-state checkpoints, contiguous commits, same source/data/runtime binding, persistent session charges, exclusive lock | Exact uninterrupted/resumed state and metric equivalence; missing/corrupt/stale evidence, abandoned sessions and time regression rejected |
| Ctrl-C between checkpoint and metrics publication left unrecoverable ordinary interruption | High | Defer SIGINT across both durable writes and restore signal mask afterward | Actual SIGINT injected after checkpoint publication; frozen partial evaluation and resume pass |
| Complete origin count could mask unknown or over-budget session work | High | Evaluation/export enforce known session endings and cumulative budget | Negative regression cases reject abandoned/over-budget artifacts |
| Version-only NumPy binding missed replaced native backend files | High for replay identity | Streamed bounded actual package/native/bundled-library hashes, inventory equality, interpreter/ABI/build identity | Same-version binary changes, omitted/extra files, symlinks, shadow import and size excess rejected; external system libraries remain OS/build observations |
| No durable live progress from archived production origins | Medium | Actual worker archive sink and generated loop emit hash-linked JSONL plus flushed console, timing/RSS and real objectives/scores | Production sink integration and failure tests passed; no fabricated dashboard or external tracker |
| Linux memory semantics and container limits differed from macOS | High for production portability | Three samples with MemAvailable/pswpout and visible cgroup-v2 ancestor hard-limit headroom, fail closed on missing/unstable constraints | 29 Linux fixture cases in 55-test portability batch; actual Linux sampling not run |
| macOS-only verification isolation and wheel choice | High for platform portability | Mandatory bubblewrap namespace/mount probe, read-only evidence masks, exact offline Linux wheel selection and writable source staging | Simulated command/probe contracts and existing Darwin inventory tests passed; no actual Linux sandbox/build claim |
| Generic GPU workflow could imply unsupported CUDA or silent fallback | Medium | Explicit unsupported-device refusal and CPU metadata preflight; preserve locked NumPy and existing environments | Actual CUDA preflight exit 3; GPU checks absent, no dependency upgrade |
| Repeated generated data per variant or hidden tuning would waste compute/change the study | Medium | One private seeded generation per session, shared immutable inputs, fixed burn-in, one thread; original scientific model/thresholds unchanged | Current-target perturbation changes reporting score but leaves issuance state/probabilities identical; no measured speedup claim |

## Final executed checks

| Receipt | Actual outcome | Scope |
|---|---|---|
| `final_tests_attempt1.json` / `.log` | **108 passed**, 22.95 s pytest; 23.54 s command; 56.08 MiB child peak | Seven selected test modules: generated kernel/resume, runtime identity, preflight, logging, wheel selection, isolation and production verification contracts |
| `portable_tests_attempt1.json` / `.log` | **55 passed**, 0.56 s pytest; 39.45 MiB child peak | Linux host/isolation/wheel fixtures and preserved Darwin command inventory; overlaps the final batch |
| `final_static_attempt3.json` | Ruff format/lint and strict mypy passed, **21 changed Python files** | Static quality only; does not establish runtime or science |
| `CLI_RECEIPT_attempt1.json` | All **11 expected exits** matched; 43.92 MiB child peak | CPU metadata, CUDA refusal, tiny/larger/192-ledger dry plans, interrupt at 4, partial evaluation, resume to 12, frozen evaluation, completed reuse, local export |
| `preservation_attempt1.json` | **122** saved original file hashes verified; scientific/environment contracts and old production plan unchanged | Specified snapshot preservation, not a new recursive verification of every historical result |
| `host_admission.json` | **BLOCKED**, all three memory observations below 512 MiB | Actual macOS host observations; no full or scientific worker launched |

The exported tiny exercise has 48 retained artifact files plus its evaluation and hash inventory. Its energy score is a generated diagnostic, not a market result. Test counts overlap; do not add them into a whole-suite count. Earlier 87-test integration and separate identity/resource/isolation receipts remain evidence for their intermediate scopes, not a current full-suite pass.

## Retained failures and unexecuted work

Original `smoke_static_attempt1`, `smoke_tests_attempt1`, Linux-resource attempt1, isolation attempt1 and `final_static_attempt2` retain their actual failures. Repairs cover the argparse generic annotation, strict typing/import usage, platform branching, lint and formatting; final bounded checks passed without deleting tests or weakening required checks. `AUXILIARY_FAILURES.json` records one corrected report-log parsing mistake. All original run/result evidence and prior user changes remain.

No current complete suite, clean wheel/source build and installation, eleven-mutant campaign, full-scale qualification, Linux execution, GPU execution, precision pilot, scientific freeze, market development or confirmation occurred. The source bundle is not a package-install receipt. No external submission, push, purchase, dependency replacement or automation was performed.

## Maintained files

- Execution: `src/frem_s10/cli.py`, `runtime/smoke.py`, `smoke_checkpoint.py`, `smoke_contract.py`, `smoke_data.py`, `smoke_export.py`.
- Observation and portability: `runtime/preflight.py`, `progress.py`, `worker.py`, `resources.py`, `isolation.py`, `verification.py`, `packaging.py`.
- Targeted regressions: `tests/test_smoke_audit.py`, `test_smoke_identity_audit.py`, `test_preflight_audit.py`, `test_progress_audit.py`, `test_linux_resources_audit.py`, `test_isolation_audit.py`, `test_portable_packaging_audit.py`, and the explicit Darwin fixture in `test_production_verification_audit.py`.
- Package README, root generated manifests, canonical state/handoff/static-verification additions, and this evidence directory. All source/test paths above are relative to `successors/frem_s10`.
