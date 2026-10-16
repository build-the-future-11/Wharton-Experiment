# FREM-S10 campaign handoff — 16 October 2026

**Readiness: BLOCKED.** The maintained S10 model uses CPU NumPy float64. Choose accelerator **None** for the first portable engineering check. CUDA/MPS are unsupported backends and requested GPU execution fails without CPU fallback. No GPU check or cloud campaign ran; this session did not authorize or use cloud GPU quota. Campaign runtime and GPU benefit remain unmeasured.

The generated runner exercises the actual forecasting kernel, saves committed causal state, supports checked recovery at completed origins, and reconstructs saved scores. Linux host-admission, isolation, and offline wheel-selection adapters have now been authored. Bounded fixture checks distinguish their contracts from observed Linux execution; the production numerical supervisor, exact Linux environment binding, integration, and actual Linux validation remain blocked. Read the final [session receipt](RECEIPT.json) for the settled source/test inventory; that receipt was pending when this handoff was finalized.

- `check_host` now has an explicit Linux contract: three samples at least one second apart, at least 512 MiB from the minimum of host `MemAvailable` and every visible cgroup-v2 ancestor's `memory.max - memory.current`, stable `pswpout`, and stable constraint identity. Missing/hidden ancestry, cgroup-v1/hybrid layouts, and ambiguous counters fail closed. This preserves Linux counter names and the existing macOS contract. [30 fixture checks passed](linux_resources_checks_attempt2.json); no live Linux admission ran. The lighter `preflight` observation remains descriptive and does not itself perform these checks.
- The Linux verification adapter requires bubblewrap user/PID/network namespaces, a read-only root, tmpfs masks over repository evidence, and explicit work/wheel exceptions. Its probe requires observed namespace and mount isolation; generic connection failure cannot pass. Missing bubblewrap or unsupported namespaces cause refusal. [21 scoped checks passed](isolation_checks_attempt2.json), including simulated Linux probes and Darwin argument preservation; no Linux bubblewrap process ran.
- Offline packaging selects the exact CPython 3.12 manylinux NumPy wheel/hash from `uv.lock` for x86_64 or aarch64, requires that wheel in `research/frem_s10_wheelhouse`, and stages build inputs inside the writable output directory. It never downloads a replacement. Selection/staging regression tests are authored; their final execution status and any later verification belong in [RECEIPT.json](RECEIPT.json). No Linux clean installation or production package execution is claimed.

## Environment and first launch

Start in the repository root. Use an existing **Python 3.12** interpreter with **NumPy 2.5.4**. [requirements-cpu.txt](requirements-cpu.txt) pins that existing NumPy version and accepted wheel hashes for a separately isolated CPU environment; it is not the complete production environment. `successors/frem_s10/uv.lock` preserves the full existing specification. No dependencies were installed or upgraded in this work. Inspect first; do not upgrade PyTorch or replace a CUDA build. S10 does not require PyTorch. Production still requires an exact supported environment and current-source verification; matching NumPy alone is insufficient. These are underlying portable commands; on the configured workstation, prefix commands with `rtk proxy` as required locally.

```sh
S10_REPO="$PWD"
S10_PYTHON="python3"
S10_RESULTS="$S10_REPO/outputs/kaggle_engineering_attempt1"
export PYTHONPATH="$S10_REPO/successors/frem_s10/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir "$S10_RESULTS"
"$S10_PYTHON" -u -B -m frem_s10 preflight --device cpu --output "$S10_RESULTS/preflight.json"
"$S10_PYTHON" -u -B -m frem_s10 smoke-plan --manifest "$S10_REPO/configs/generated_smoke.json"
"$S10_PYTHON" -u -B -m frem_s10 smoke --manifest "$S10_REPO/configs/generated_smoke.json" --directory "$S10_RESULTS/tiny" --execute
"$S10_PYTHON" -u -B -m frem_s10 smoke-evaluate --directory "$S10_RESULTS/tiny"
"$S10_PYTHON" -u -B -m frem_s10 smoke-export --directory "$S10_RESULTS/tiny" --archive "$S10_RESULTS/tiny.zip"
```

Select the inspected interpreter by changing `S10_PYTHON` before execution; `python3` must not be assumed to be Python 3.12. The repository and results paths can be changed to mounted source and writable output locations. Use a new results parent for each attempt. Run each stage only after its predecessor succeeds; preflight's `READY_FOR_CPU_SMOKE_METADATA_ONLY` is metadata compatibility, not a runtime or production pass. Source/configuration/outputs must be retained together. The export is a local archive of verified generated artifacts, not a repository archive or a published result.

The tiny manifest specifies `scope=smoke`, CPU, float64, seed 11, 12 origins, variant `primary`, and a cumulative 60-second budget. The larger manifest specifies `scope=development-engineering`, seed 11, 72 origins, all six variants (`primary`, `repeat_h1`, `no_shrink`, `window260`, `neighbors16`, `neighbors64`), and 300 seconds. Both use dataset `s10-generated-smoke-data-v1`, one thread, and `checkpoint_every_origins=1`. Unknown fields and ignored configuration overrides are rejected. No batches, epochs, optimizer, scheduler, AMP, or validation-selected best checkpoint exist in this deterministic online kernel. Data and its burn-in transform are generated once per session from a private seeded PCG64 stream, then identity-checked on recovery.

`smoke-plan --manifest configs/generated_development.json` enumerates the larger generated exercise without running it. Execution calls `check_host`; Linux is now supported through the conservative cgroup-v2 contract above, rather than rejected solely for its platform. The target host must actually pass that gate. Unsupported container ancestry or failed headroom/swap checks still block the run, and this session has no actual Linux execution evidence. `check-host --output "$S10_RESULTS/host.json"` can save the fresh observations before deciding whether to attempt the larger generated check. Do not remove the guard, lower the 512 MiB floor, or treat a historical resource waiver as current admission.

## Interruption, saved evaluation, and evidence

To check intentional interruption, substitute a new run directory such as `"$S10_RESULTS/resume-check"` and append `--stop-after 3` to the `smoke` command. A controlled interruption returns exit **130** after a committed origin. Continue explicitly with:

```sh
"$S10_PYTHON" -u -B -m frem_s10 smoke-resume --manifest "$S10_REPO/configs/generated_smoke.json" --directory "$S10_RESULTS/resume-check" --execute
"$S10_PYTHON" -u -B -m frem_s10 smoke-evaluate --directory "$S10_RESULTS/resume-check"
```

Recovery verifies source bytes, Python build/ABI/interpreter and installed NumPy artifact identities, platform, complete configuration, regenerated data, checkpoint hashes, and contiguous origin commits. Ctrl-C is deferred while the checkpoint and its durable metrics are being published together. All variants commit together. Installed NumPy runtime/bundled-library bytes and interpreter bytes are hashed; external system libraries are represented by OS/build metadata, so arbitrary cross-machine bitwise recovery is not promised. Each checkpoint contains model and pending-feedback state; there is no mutable training RNG or optimizer state to omit. Repeated sessions retain cumulative active time. Budget checks are cooperative between origins, so the configured ceiling is not a hard preemption guarantee. A hard-killed session missing its end receipt has unknown charged time and is refused pending review. Do not delete session records or reset the budget. Corrupted/stale checkpoints, concurrent writers, and existing new-run directories fail. Reusing a completed run still validates it. `smoke-evaluate` reconstructs saved scores without fitting; partial evidence returns exit 4, and export requires completion. Evaluation and export also reject abandoned sessions and exhausted cumulative budgets; a full origin count alone does not establish successful completion.

Timestamped console messages and durable per-session `metrics-*.jsonl` carry actual origin progress, scores, throughput, and checkpoint identity. Immutable status history accompanies the replaceable `status.json` convenience view. These are generated engineering metrics; no placeholder curves, profitability, or scientific support are implied.

## Finite follow-up campaign

`PRELIMINARY_CAMPAIGN.json` is planning metadata, **not** an executable production manifest. It recommends a stage-gated session envelope below six hours to respect the requested 4–6-hour planning target, with no runtime promise or statement about platform quotas. Stop on any failed admission or unmeasured cost that no longer fits the envelope. No automatic retries or open-ended sweeps are planned.

After the remaining production supervisor/environment integration and actual Linux platform checks pass, the existing production ceilings remain: precision 600 seconds, full-scale generated qualification 3,600 seconds, and development campaign 3,600 seconds. Verification separately limits tests/packaging/mutations to 600 seconds per command and other commands to 120 seconds, with bounded cleanup. These limits are not increased by the proposed envelope. Real execution additionally needs exact source/data/environment admission, current verification, precision success, qualification, and a new freeze. The fixed 192-ledger design remains inspectable with:

```sh
"$S10_PYTHON" -u -B -m frem_s10 plan --manifest "$S10_REPO/successors/frem_s10/src/frem_s10/science.json"
```

Previous market attempts cannot use `smoke-resume`. Production consumption rules remain unchanged. This source revision invalidates the old frozen production plan; preserve that plan and its refusals, then create a new exact candidate and receipts through the maintained admission/freeze/controller path. Historical S6/S7/S8 results, S9 state, and the earlier S10 generated-suite results remain historical evidence. None is current-source confirmation. The smallest useful next platform step is the tiny **CPU** check above after metadata preflight; there is currently no legitimate CUDA training/evaluation command to run.

## Verified local handoff evidence

The final local batch passed 108 selected tests; the separate 55-test portability batch overlaps it. All 21 changed Python files passed scoped Ruff/format/strict mypy. The actual 11-stage CLI check interrupted at four origins, resumed to twelve, reconstructed frozen scores, validated completed reuse, and exported 48 generated artifacts. See `RECEIPT.json`, `CLI_RECEIPT_attempt1.json`, and `CHANGE_LEDGER.md`. These results do not satisfy complete production verification or Linux/GPU validation. `source_bundle.zip` is a local source handoff only.
