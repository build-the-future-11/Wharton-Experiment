# FREM-S10 engineering and Kaggle readiness — 16 October 2026

**Status: BLOCKED. Scientific status: NOT_EVALUATED.** This session implemented and exercised local engineering improvements in the maintained S10 package. It did not run Kaggle, CUDA, a market fit, a research ledger, protected confirmation, or a new scientific freeze. The numerical model uses CPU NumPy float64. Accelerator **None** is the appropriate first platform check; there is no implemented CUDA model to validate and no measured GPU benefit.

## Project and evidence identity

The selected checkout is the executable Wharton repository, on `codex/repo-cleanup-efficiency`, based on Git `09268ecb01ac645eeee18cc9fcbc6db247b6ec89`. It already contained extensive user changes, including the untracked S10 package. Git HEAD alone does not identify this implementation. `INITIAL_IDENTITY.json` records the starting dirty state and hashes; `before/` preserves 122 original files. `CANDIDATE_IDENTITY.json` and `source_bundle.zip` identify the delivered source. The ZIP is a source handoff, not evidence of wheel build or clean installation.

The maintained question is whether delayed contextual mixture fitting with a proper score improves total-return forecasts and fee-aware decisions against the fixed reference forecasts and policies. S10 is an online finite-scenario model with constrained mixture optimization, not a neural training loop. Epochs, batch size, backward gradients, optimizer/scheduler/scaler state, AMP, accumulation, and multi-GPU controls are inapplicable. Adding them would not repair this implementation.

The existing scientific design retains six variants, 192 policy ledgers, costs of 0/10/20/40 bps, pre-2010 pilot values and pre-2017 development values. Existing SPY/TLT/GLD source admission covers retrospective current-vintage 2005–2016 observations, not point-in-time certification. Source timestamps, corporate-action decomposition, missing-row/calendar admission, overlapping maturities, cash/equal-weight references, turnover and entry/exit fees remain governed by that design. No thresholds, assets, horizons, bootstrap blocks, eligibility rules, baselines or outcomes were selected again in this session.

## A. Engineering delivered

The existing `python -m frem_s10` entrypoint now exposes metadata preflight, a strict finite generated manifest planner, real-kernel CPU smoke execution, checked origin-boundary resume, frozen-state score reconstruction, and verified local export. CUDA/MPS requests fail explicitly with no CPU fallback. Manifest fields cannot be silently ignored; zero/NaN budgets and invalid variants are rejected. New runs refuse existing directories, writers use an exclusive process lock, and commits/status histories are retained.

Generated inputs use a private seeded PCG64 stream, fixed burn-in preprocessing, and matured feedback. Data is constructed once per session and shared across variants; resumed sessions regenerate and verify it. Current targets are used only for reporting scores after issuance. No future target enters the issue request or model update. No throughput improvement or market benefit is claimed without a matched measurement.

Every complete origin saves all selected variants and pending-feedback state through safe JSON serialization. Checkpoints bind configuration, generated data, source, actual installed NumPy runtime/bundled-library bytes, interpreter bytes and Python/OS build metadata. External system libraries are represented by OS/build observations, not independently hashed; this is not a cross-machine bitwise guarantee. Ctrl-C is deferred during checkpoint/metrics publication, preventing a committed origin with a missing metrics log. Resume retains cumulative active time; a hard-killed session without a final duration is refused for budget review. Budgets are cooperative between origins, so they are not hard preemption guarantees. No selected-best checkpoint exists because this path performs no validation-based selection. Market runs remain one-use and cannot use generated-run resume.

Live logging is connected to the actual production worker archive sink and generated loop. Hash-linked local JSONL and flushed console messages record real steps, state/checkpoint identities, solver objectives or generated scores, timing, throughput and peak RSS. Optional console failure cannot lose durable logs. A disk failure remains an explicit failure; no mandatory online tracker or credentials were added.

Linux preparation adds cgroup-v2-aware three-sample memory admission, mandatory bubblewrap isolation, strict network-namespace/mount probes, and offline selection of the target NumPy wheel from the existing lock. Linux memory/counter semantics retain their own names. Unsupported mounts, hidden cgroup ancestors, absent namespaces/tools and missing wheels fail closed. These adapters have fixture tests; actual Linux admission, sandboxing and installation have not been demonstrated.

## B. CPU execution evidence

Read `RECEIPT.json` for the final command-level evidence and exact source identity. Earlier failed formatting, typing, collection and test attempts remain alongside repaired attempts. Focused counts overlap and must not be summed into a whole-suite claim. This session's tests and CLI output are generated engineering evidence only.

Final focused test/static and command outcomes are listed in `CHANGE_LEDGER.md`. The CLI exercise checks intentional interruption, resume, frozen score reconstruction, completed-cell validation and local export. Resume-versus-uninterrupted regression requires byte-identical causal state and exactly equal binary64 metrics in the bound environment.

The real host admission receipt failed: 78.734375 / 89.671875 / 120.703125 MiB free-plus-speculative, below the unchanged 512 MiB floor; Pageouts stayed 1058674. Therefore no full suite, complete package build/install, semantic mutation campaign, full-scale generated qualification or scientific worker was launched. Small bounded engineering tests are not substitutes for those gates.

## C. GPU execution evidence

**None.** No CUDA import/runtime test, GPU training, Kaggle session or cloud quota use occurred. Metadata-only inspection found no S10 PyTorch dependency or installed S10 torch package; the separate existing root environment's torch version was read without changing it. No dependencies were installed or upgraded. `preflight --device cuda` is a tested refusal, not a GPU pass.

## D. Remaining blockers and precise limits

1. The production numerical supervisor and exact environment contract still require Darwin arm64. Linux adapter code does not extend the scientific execution contract by itself. Linux receipt/catalog joins and production admission must be integrated and verified before a real campaign can use them.
2. No actual Linux bubblewrap/cgroup/offline-build probe has run. A Kaggle container may hide cgroup ancestors or disallow the required user namespaces. Those conditions must stop execution; there is no weaker fallback.
3. Current full verification, wheel/install checks and eleven semantic mutants remain unexecuted behind resource admission. The old 274-test/wheel receipt belongs to an older source candidate.
4. The changed source invalidates the earlier production plan `852e85d6e4889687d7ae555afdadb21b1a563a8392c206a17ca75ebb3a32eba2`. Preserve it and its refusals. A new candidate needs exact admission, verification, precision, qualification and freeze through the existing controller before a separately authorized development run. This session created or changed no automation.
5. Research validity still requires truthful point-in-time provenance for confirmatory claims, untouched confirmation and independent reproduction. Current-vintage retrospective sources, generated tests, and same-host reviews do not establish these.

The preliminary 5.5-hour session envelope respects the requested 4–6-hour planning preference but is neither a measured estimate nor an enforced/authorized cloud budget. Existing stage ceilings remain 600 seconds for precision, 3600 for qualification and 3600 for development; verification has separate command ceilings. No automatic sweep, retry or cloud launch was added.

## E. Historical evidence retained

S6/S7 unsupported outcomes, S8's inconclusive predictive result and failed economic gate, S9's unconsumed state, all failed/interrupted attempts, prior freezes and old S10 receipts remain historical. They were not rerun, pooled with this candidate, or promoted into current confirmation. The original score/wealth thresholds, data contract and scientific manifest remain unchanged. The canonical state and append-only decision journal record this engineering change separately.

## Launch and smallest next platform step

Use `KAGGLE_HANDOFF.md` for complete commands, `configs/generated_smoke.json` for the executable tiny manifest, and `PRELIMINARY_CAMPAIGN.json` for staged planning metadata. The local interpreter is `successors/frem_s10/.venv/bin/python`; the package entrypoint is `-u -B -m frem_s10`. Start with `preflight --device cpu`, then `smoke-plan`, then the 12-origin CPU smoke in a fresh directory. The next target-platform check is CPU-only with accelerator None after inspecting Python 3.12 and NumPy 2.5.4 metadata. CUDA is unsupported; do not spend GPU quota on this model merely to satisfy a generic GPU checklist.

## Primary sources consulted

[Gneiting and Raftery (JASA, 2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) supports proper-score methodology, not a claim that fitted S10 forecasts are calibrated or that price/dividend components are separately identified. [Yao et al., hierarchical stacking (submitted 22 January 2021; revised 20 May 2021)](https://arxiv.org/abs/2101.08954) establishes prior art for input-varying partially pooled mixture weights. S10's causal matured-score records and constrained non-Bayesian finite mixture are implementation distinctions, not proof of novelty. [Official Kaggle image sources](https://github.com/Kaggle/docker-python), [Linux cgroup-v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html), and [bubblewrap's official option documentation](https://github.com/containers/bubblewrap/blob/main/bwrap.xml) informed platform checks. No current account quota or Kaggle runtime behavior was inferred from documentation.
