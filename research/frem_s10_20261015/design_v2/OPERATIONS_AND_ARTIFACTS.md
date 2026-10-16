# S10 execution operations and artifact contract

15 October 2026. Everything in this file describes future implementation. No launcher, checkpoint, CI pass or experiment is claimed to exist. Model details come from [the normative clarifications](MODEL_CLARIFICATIONS.md) and the preserved revision-1 specification.

## 1 Supported operating boundary

Use an isolated Python 3.12 package and a frozen environment. macOS arm64 and Linux x86_64 are target validation platforms; each needs an actual passing receipt before being called supported. No CUDA, MPS, distributed execution, live orders or network acquisition is part of S10 revision 2.

The runtime performs local deterministic research over explicitly admitted inputs. It imports no root FREM/Torch package. Dependency installation, source acquisition and environment resolution happen outside a consumed scientific attempt. A report renderer is separate from the numerical worker and cannot trigger model execution.

## 2 Planned command surface

The proposed future executable is s10-research from the isolated package; none of these subcommands is implemented today. All commands accept a complete manifest path, not a collection of implicit environment defaults.

| Command | Inputs and effects | Evidence boundary |
|---|---|---|
| plan | Configuration and metadata; enumerate identities, counts, budget, proposed paths | No data outcomes, fitting or output consumption |
| admit | Explicit source manifest, approved cutoff, output path; construct and verify permitted records | Reads development values only when scope allows; not a model success |
| verify-engineering | Constructed fixtures and isolated output directory | Engineering tests only; no market-data discovery |
| prepare-precision | Source/scope/configuration and precision authorization; bounded pre-2010 work | Separate real-data preparation attempt, with consumption and complete receipt |
| freeze | Exact reviewed source/data/config/environment/test/admission/precision identities | Creates immutable run declaration; grants no protected access |
| run | Freeze and new output directory with explicit execution flag | One-use declared development campaign |
| audit | Read-only saved artifacts plus frozen source identity, separate output directory | Does not rerun model or modify old receipts |
| report | Audited artifact manifest, separate output directory | Generates tables/figures/prose from actual retained evidence |
| package | Audited source/evidence allowlist and rights manifest | Local package only; no publication or raw-data redistribution by default |

There is no resume, auto-tune, run-all-presets, fetch-on-missing-data or delete-consumption command. CLI usage errors show actionable field/path/reason without dumping raw financial observations or environment secrets.

Proposed stable exit codes: 0 complete requested operation; 2 usage/configuration rejection; 3 resource admission blocked before consumption; 4 data/source/environment integrity rejection; 5 consumed-freeze refusal; 6 started attempt failed numerically or structurally; 7 elapsed-time/RSS limit; 130 user cancellation. Exit zero describes completed processing, not scientific support. The result disposition is a separate mandatory field.

## 3 Lifecycle and irreversible boundaries

Each attempt follows:

    CREATED -> PREFLIGHT -> ADMITTED -> CONSUMED -> RUNNING
      -> COMPLETE or FAILED or INTERRUPTED or BUDGET_EXCEEDED.

PREFLIGHT may end REJECTED_UNCONSUMED. No scientific outcome-dependent computation occurs before CONSUMED. Preflight includes schema, paths, source/config/environment hashes, finite budget, source admission, operation-specific verification and host resources. Precision preparation requires its reviewed declaration and tested kernel/bootstrap/artifact/consumption/resource subset; it cannot require its own unproduced result. The full campaign additionally requires the completed passing precision receipt. These shared production guard components are implemented and verified within WP06 before real precision preparation, then extended in WP10-WP12. Plan metadata may be produced without consumption.

Write the run consumption marker with exclusive creation and fsync before launching scientific processing. It binds freeze ID, attempt ID, experiment identity, permitted operation and timestamp. Precision preparation has a distinct identity/scope and cannot consume or certify the full campaign. Never reuse a marker from another operation.

An existing or ambiguous marker is a stop condition. A failure after marker creation is not converted back to unconsumed. A process crash between consumption and first model operation still produces a spent attempt requiring review. A preflight resource rejection can be retried in a new directory; it preserves the earlier receipt.

All terminal failures are retained. A scientifically valid negative or inconclusive result is COMPLETE with a negative/inconclusive scientific disposition, not FAILED infrastructure and not automatic retry permission.

## 4 Exclusive ownership and filesystem semantics

The controller owns an explicit output root resolved before use. Refuse existing attempt directories, unsafe traversal, unexpected symlinks escaping the root, and output paths inside input or protected source trees. Canonicalize allowlisted input paths and compare source identities; no broad recursive data search during execution.

Use an exclusive run lock and consumption marker. Lock metadata contains PID, host, process start identity and attempt ID. A stale-looking lock is inspected, not automatically deleted because a PID disappeared or was recycled. Crash remnants are evidence. There is one writer for each journal/artifact.

Write a new artifact to a unique temporary file on the same filesystem, flush and fsync, validate length/hash/schema, then publish atomically to a previously nonexistent final identity. Never replace an already-published result. Internal working checkpoint pointers may advance atomically while all immutable checkpoint generations remain retained. Fsync the containing directory where supported; unsupported durability behavior must be recorded.

No pickle, arbitrary object deserialization, executable YAML tags, eval, shell expansion of manifest values or importing module paths supplied by data. JSON rejects duplicate keys, nonfinite numbers and unknown required-schema fields. Binary numeric arrays use allow_pickle=false and separate explicit dtype/shape identities. Resource sizes are checked before allocating from untrusted shape fields.

## 5 Artifact layout

Each future attempt has these logical files; exact partition counts come from the frozen manifest.

| Path | Contents and invariants |
|---|---|
| manifest.json | Experiment ID, attempt ID, operation, config/source/data/environment hashes, claim scope |
| preflight.json | Every admission check, measured resources, outcome and time |
| consumption.json | Local copy of persistent one-use identity; not its sole authoritative location |
| environment.json | Python/tool/package versions, OS/arch, BLAS/thread settings, timezone database identity |
| timeline.jsonl | Ordered lifecycle/maturity/decision/checkpoint events with prior-event hashes |
| sources.json | Raw source identities, provenance, allowed date range and data-rights statement |
| calendar.json | Exact session/anchor/index/fold identities and exclusions |
| transforms.json | Burn-in means/scales and maturity cutoffs |
| issued/ | Immutable expert probabilities, atom/state IDs, context and timestamps |
| matured/ | Once-only label linkage and A/B score coefficients |
| mixtures/ | Global/local/final weights, eligibility flags, active-face/KKT certificates |
| decisions/ | Every action, hold/cash alternatives, constraints, cost assumptions and objective certificate |
| ledgers/ | Event-level shares/cash/fees/dividends and daily actual/liquidatable NAV |
| checkpoints/ | Immutable state generations with parent generation, content identities and cursor |
| statistics/ | Shared bootstrap index arrays, paired metrics, all comparisons and disposition |
| resources.jsonl | Monotonic elapsed time, owned process-tree RSS, host memory/pageouts |
| inventory.json | Exact expected and produced 192 ledger identities; no silently skipped cells |
| errors.jsonl | Structured exceptions with stage/identity and bounded diagnostic context |
| completion.json | Terminal state, counts, scientific disposition, limitations and all final artifact hashes |

Every numeric artifact carries schema version, array dimensions, units, asset order, experiment identity and content digest. Fields that are absent, not applicable and zero are distinct. Paths alone never establish provenance. A report may link raw arrays without copying private/protected observations into terminal output.

## 6 Checkpoints and crash semantics

An immutable checkpoint binds model configuration, code/environment, source/fold transforms, scenario atom IDs, maturity queue, issued forecasts, score-record queue, mixture state, calendar cursor, RNG state/descriptors, holdings/cash, last credited dividend IDs, last trade and event-chain head. Capture a checkpoint only at a defined event boundary after all preceding outputs are durably committed.

Engineering crash tests inject failure before/after each publish/credit/maturity/consumption boundary. Replay from a valid constructed checkpoint must equal uninterrupted execution under the same environment and cumulative event ordering. Half-published checkpoints fail validation; they are not repaired by guessing missing state.

Actual market attempts remain terminal after failure or cancellation. Checkpoints enable diagnosis and same-environment engineering equivalence, not a hidden market restart. Any later recovery policy needs a separately recorded pre-outcome contract; the current CLI has no resume route.

## 7 Resource and cancellation behavior

Before importing the numerical worker, collect three free-plus-speculative memory/pageout readings one second apart. All must meet the 512 MiB floor and pageouts must not rise. Store raw counters, conversion units and timestamps. On platforms lacking an equivalent metric, record admission unavailable and stop; do not quietly substitute a more favorable measure.

A lightweight supervisor owns one numerical worker with one BLAS thread. Worker configuration must explicitly disable unplanned thread pools before scientific imports. Read-only audits/renderers run afterward under their own declared resource scope, not concurrently with market processing.

Use a monotonic clock for elapsed budget. A supervisor polls owned-process RSS and wall time at intervals no longer than 0.5 seconds; cooperative internal checks run at record/batch boundaries. Budget accounting includes all relevant worker/controller preparation, issue/update, policy and analysis work. The 256 MiB aggregate RSS ceiling is conservative when shared pages are double-counted. Instrumentation failure or unknown ownership fails closed.

At limit, stop new work, request orderly termination, allow at most two seconds for bounded failure recording, then terminate the owned worker if needed. This grace does not grant additional scientific work. OS sampling can overshoot a nominal limit between polls; report maximum observed RSS/time and overshoot rather than promise a hard kernel guarantee. Never kill unrelated processes or close user applications to make admission pass.

SIGINT/user cancellation retains the consumed marker, last complete artifacts and interrupted receipt. A supervisor crash may leave no final receipt; a read-only recovery inspector may produce an external inspection receipt in a new directory, never rewrite the missing/partial original as successful.

## 8 Logging, diagnostics and monitoring

Use structured JSON event logs with UTC time, monotonic elapsed time, severity, stage, attempt/record identity and bounded details. Numerical failures include shape, unit, min/max finite diagnostic where approved, residual and tolerance; they do not include unrestricted source rows. Never log tokens, entire environment variables, provider credentials or user-private mandate inputs.

Default console output is a concise stage/count/resource summary. Warnings must be classified: expected scientific limitation, numerical near-tolerance condition, or actual integrity failure. An integrity failure cannot be downgraded to a warning. Record fallback counts, zero-action rates, neighbor distances, mixture concentration and active constraints as diagnostics, never automatic tuning signals.

No background heartbeat, remote telemetry or external message is scheduled by this plan. Completion, negative results and failures are explicit terminal records.

## 9 Caching, storage and cleanup

Cache only deterministic derived quantities keyed by exact source/fold/transform/model/algorithm identities, array order and numerical environment where needed. Cache retrieval validates schema, size and digest before use. A cache miss recomputes within the same remaining budget; a corrupt cache fails integrity instead of silently using stale data. Changed algorithms cannot reuse the old key.

Preflight estimates disk requirements from admitted record counts and artifact shapes without inspecting evaluation outcomes. Require estimated bytes plus max(256 MiB, 20% of estimate) free on the actual target volume. Retain all scientific artifacts and failures. Delete only documented disposable temporary files belonging to the current unconsumed preparation or after their durable immutable replacement is verified. No automatic deletion of old experiments, checkpoints, consumption receipts or user caches.

## 10 Audit and release boundaries

The saved-output auditor verifies chain/order, source identities, issued-to-matured links, all 192 cells, matrix uniqueness, numerical/accounting residuals and reconstructed statistics. A second implementation should regenerate key scores/accounting without calling the production kernel. Sharing source/data is not independent research reproduction.

Release contents are allowlisted: source, exact environment/lock, documentation, schemas, tests, reproduction instructions, data acquisition/admission instructions, audit receipts, actual results where rights permit, generated figures/tables and a claim-to-evidence index. Exclude credentials, local environment folders and raw provider data unless redistribution is verified. Produce a hash manifest before packaging and inspect archive contents.

Local source delivery, an empirical negative paper, independent reproduction, public release and submission are separate states. No CI badge, package build or rendered PDF substitutes for them.
