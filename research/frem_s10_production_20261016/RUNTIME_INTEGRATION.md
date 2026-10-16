# S10 production runtime integration — 2026-10-16

This record describes local engineering work. It is not a scientific result, a passing production admission, a precision result, or full-scale qualification. No market fit, market ledger, protected evaluation, or full calendar generated qualification was executed by the runtime implementation agent.

## Fixed execution path

`runtime.pipeline` joins exact current code/configuration/data/calendar/environment identities, actual command-verification evidence, retrospective source admission, a prospective operation declaration, fresh host admission, exclusive consumption, and the fixed supervised worker. Production cannot use historical structural-only PASS fixtures as authority.

The precision worker loads its value-restricted pre-2010 source view only after consumption. Every weekly transition is archived. The saved precision audit reconstructs those transitions, pairs every retained origin and forecast ID, recomputes energy scores against admitted targets, and checks cold-start/fallback diagnostics, projected counts, and final state before publishing an operation prerequisite. A campaign independently repeats this saved-evidence check inside its supervisor before reading campaign values.

The campaign also deeply verifies the retained generated qualification inside its supervisor. It then uses six weekly model variants and the fixed192-cell matrix, preserves every transition and ledger artifact, reads the saved outputs back, and regenerates all primary/ablation statistics, paired bootstrap arrays, and the report. That full-scale execution remains subject to qualification and live resource admission; small tests do not establish that it fits the budget.

RAM remains256MiB, wall time600s for precision and3600s for campaign/qualification/saved-audit, one worker and one numerical thread, and fresh host reserve at least512MiB with unchanged pageouts. The fixed storage allowance is512MiB for precision and2GiB for the larger operations, with volume reserve checked before consumption. Workers use `-B -X pycache_prefix=<new attempt output>/empty_bytecode` so stale source bytecode cannot substitute for the hash-bound source.

Pre-consumption resource refusals remain immutable and can be retried only after reconciling the retained two-event lifecycle and confirming that the corresponding consumption marker is absent. Every consumed failure, ambiguous interruption, or post-worker finalization failure remains terminal for automatic retry. Successful receipts validate exact resource peaks and the complete ordered lifecycle, including the actual consumption hash.

## Continuation manifest

`python -m frem_s10.cli advance --manifest <absolute plan path> --execute` accepts exactly:

```text
schema: s10-continuation-plan-v1
artifact_root, source_root, consumption_root, output_root, work_root: absolute existing directories
attempt_prefix: bounded plain identifier
artifacts:
  code, configuration, environment, calendar, clock, source_admission,
  data_pilot, data_campaign, catalog: {path: relative artifact path, sha256: exact SHA256}
```

Work, outputs and consumption directories must be distinct, inside the artifact root, and outside source data. The controller locks its work directory, preserves the immutable plan, and appends a verified hash-linked stage journal. Its fixed order is engineering verification → precision pilot → generated qualification → development campaign. It creates freezes prospectively, uses the same declaration after a proven unconsumed rejection, and audits completed prerequisites before advancing. It never repeats a consumed stage or retries a failed verification command. A precision result other than PASS stops advancement. A valid negative campaign result is a terminal result, not an invitation to tune or repeat. The controller does not execute protected confirmation.

The initial verification producer independently checks its own host admission. A verification failure is automatically retryable only when the saved evidence has a failed host gate and zero executed commands. Campaign verification reuses the exact previously observed command logs under the same code/configuration/calendar/environment identities and adds the separately audited qualification; it does not invent a second test run.

## Manual operation manifests

`admit` and `freeze` use `s10-preparation-manifest-v1` with `request`, `artifact_root`, `source_root`, `consumption_root`, and `output_name`. Freeze output names may be safe relative paths below the artifact root, with existing nonsymlink parents. The freeze request is `s10-freeze-request-v1`, containing operation, scope, current UTC creation time, exact artifact registry, source-admission ref, and null or actual precision-prerequisite ref. It refuses an invented historical creation time.

`prepare-precision` and `run` require `--execute` and `s10-launch-manifest-v1`: declaration ref, artifact/source/consumption/output roots, and a new attempt ID. `qualify` requires its separate fixed qualification launch manifest. No command accepts arbitrary worker arguments or a resume switch.

Saved full-campaign regeneration is also supervised. `audit --manifest <path> --execute` accepts `s10-saved-audit-launch-v1`: `request` ref, artifact/source/consumption/output roots, and attempt ID. The request is `s10-saved-audit-request-v1` containing exact `code`, `environment`, and saved `campaign` manifest refs. This operation has its own `saved-audit-<request hash>` consumption namespace and performs zero model fits. It reconstructs saved arrays, accounting events, and statistics inside the guarded worker; it does not open market sources.

`report` is a small saved-report extraction using `s10-saved-evidence-request-v1`: existing analysis directory, its audit SHA256, and a new output directory. It verifies the audit/report reference and publishes the retained Markdown-bearing report JSON without fitting or full matrix decoding.

Both `python -m frem_s10` and `python -m frem_s10.cli` execute the CLI. An invalid explicit manifest cannot silently exit successfully.

## Evidence and limits

The controller tests inject stage producers/auditors to test order, locking, terminal states, fresh refusals, and immutable history. Those tests are not evidence that a real source was admitted or that a full numerical campaign fits its ceilings. The saved-audit tests inject dispatch or supply tiny invalid metadata; they do not execute full192 decoding.

The earlier generated precision integration test executes the real numerical kernel, weekly archive, score serialization and saved-law audit using a separately validated generated value view. Only the source-loader boundary is replaced. Its retained successful run took444.61s under host pressure (`runtime_tests_attempt5.log`); it was not repeated during the later metadata-only integration checks. That generated test is not a market precision result.

The original sandboxed supervisor test attempt retained19 failures caused by denied `/bin/ps` process-accounting access and27 passes (`runtime_existing_tests_attempt1.log`). These failures were not relabeled as successful guards. Final package-wide verification must run in the parent's authorized process-accounting context and bind the exact final candidate, catalog, environment, command logs, package build/install evidence, and result counts.

All failed typing/lint/test attempts remain in this directory. Current scoped verification commands and hashes are recorded separately once source integration stabilizes. Original negative research evidence and all previous receipts remain unchanged.
