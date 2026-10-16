# FREM S10

S10 is an isolated research implementation. It implements causal empirical forecasts, scoring, constrained mixtures, fee-aware policy selection, all six benchmarks, direct-horizon data contracts, exact event accounting, fixed-matrix replay, typed saved-output readback, paired endpoint/ablation analysis, weekly state archives, real source admission, prospective freezes and guarded one-use execution. Implementation and passing generated tests do not establish a positive market result. Consult the current canonical state and immutable run receipts for actual execution status.

The production integration audit is in `research/frem_s10_production_20261016`, with the complete predecessor repair preserved in `research/frem_s10_repair_20261015`. The original scientific design remains immutable in `research/frem_s10_20261015/design_v2`; prospective `research/frem_s10_repair_20261015/DESIGN_ADDENDUM_CL15.md` clarifies A4 ordering and `DESIGN_ADDENDUM_CL16.md` fixes minimum-variance tie units. ES claims concern the total-return distribution; score thresholds are unchanged.

## Reproduce local engineering checks

From the repository root:

```bash
rtk proxy env UV_CACHE_DIR=research/frem_s10_implementation_20261015/uv_cache uv sync --project successors/frem_s10 --locked --offline --group quality --group build
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 successors/frem_s10/.venv/bin/python -B -m pytest -c successors/frem_s10/pyproject.toml successors/frem_s10/tests -q -W error -p no:cacheprovider
rtk proxy successors/frem_s10/.venv/bin/mypy --config-file successors/frem_s10/pyproject.toml successors/frem_s10/src/frem_s10 successors/frem_s10/tests
rtk proxy successors/frem_s10/.venv/bin/ruff check --config successors/frem_s10/pyproject.toml successors/frem_s10/src/frem_s10 successors/frem_s10/tests
rtk proxy successors/frem_s10/.venv/bin/s10-research plan --manifest successors/frem_s10/src/frem_s10/science.json
rtk proxy successors/frem_s10/.venv/bin/s10-research check-host
```

Offline setup requires the pinned packages in the named cache. Process-supervisor tests require local PID/RSS/start metadata from `/bin/ps`; generated memory counters in those tests never establish live host admission. The verification producer additionally runs the fixed commands inside a macOS sandbox denying network access and reads beneath the repository's `data`, `outputs` and `research` directories, except its owned work directory and the exact pinned NumPy wheel, with an explicit boundary probe. Nested sandbox installation requires an execution context that permits `sandbox-exec`. Do not weaken that test boundary to obtain a passing receipt.

The last two commands read only design metadata or host counters. `plan` enumerates 192 proposed ledger identities; `check-host` returns exit3 when three observations fail the unchanged512MiB/non-rising-pageouts rule. Neither runs a model. The explicit `prepare-precision`, `qualify`, `run` and `advance` commands require `--execute` and exact manifests. There is no scientific resume command.

The package's numerical tests use generated examples. They do not access market data or establish independent empirical replication. The current validation platform is CPython 3.12 on macOS arm64. Other platforms and full-study time/memory remain unverified.

## Fixed production sequence

1. Audit source/code/configuration/calendar/environment identities and retained predecessor evidence.
2. Execute the complete generated engineering suite, strict static checks, exact wheel/source archive audit, offline clean installation and the eleven fixed semantic mutation checks (all ten normative faults plus the retained omitted-exit fault) on the exact candidate. Mutation copies must prove their import origin and kill the intended assertion, with the unchanged baseline passing. Bind every collected parameter case to the operation's required catalog obligations. Changed source or installed versions invalidate the receipt.
3. Freeze and consume the primary pre-2010 precision pilot once. Projected A/B counts use calendar metadata only. Recompute saved scores from archived laws and admitted pre-2010 targets.
4. Only a passing pilot can advance. Qualify the full six-variant weekly schedule and192 ledgers on a deterministic generated panel using the actual calendar. Exercise64/128/260/520-atom laws with610 actual action candidates and retain measured process resources. This is engineering evidence.
5. Bind the unchanged engineering evidence plus passing qualification to the campaign, freeze and consume the full fixed development matrix once. Audit saved pilot evidence again inside the guard before loading campaign values.
6. Retain every outcome, including a valid negative or unidentifiable result. Reconstruct statistics, tables and report data from saved artifacts. Never retune or retry a consumed study to obtain a positive result.

The conditional continuation manifest is `research/frem_s10_production_20261016/EXECUTION_PLAN.json`, created only after the candidate source inventory is fixed. The local command is:

```bash
rtk proxy successors/frem_s10/.venv/bin/python -B -m frem_s10 advance --manifest research/frem_s10_production_20261016/EXECUTION_PLAN.json --execute
```

A resource refusal does not consume a scientific attempt. Each subsequent resource-only retry gets a fresh output directory. Failed tests, changed source, a consumed failure, an interrupted or ambiguous operation, or failed precision stop advancement for review. Existing output and consumption records are never deleted. One worker/thread,256MiB sampled aggregate controller/worker RSS,600s pilot and3600s campaign ceilings remain fixed. The prospective storage reservations are512MiB for pilot and2GiB for qualification/campaign, plus the required volume reserve; those are storage allowances, not RAM allowances. Sampling can overshoot and does not constitute a hard OS memory limit.

Engineering command timeouts are enforced, with owned process-group cleanup. Their child-memory high-water marks are checked after each command; that check is an observed bound, not live memory enforcement. Numerical pilot, qualification, campaign and saved-output audit operations use the separate live supervisor. `audit --manifest <saved-audit-launch.json> --execute` regenerates saved numerical evidence under its own one-use engineering receipt, without fitting or reopening the research study.

The admitted data are retained current-vintage2005-2016 Yahoo snapshots. The pinned community-maintained XNYS calendar supplies scheduled regular-session opens/closes, including early closes. Those clocks do not prove actual fills, vendor latency or point-in-time availability. Cash dividends use the declared ex-date convention. Source integrity admission does not establish provider collection permission or redistribution rights; no new Yahoo collection is part of this pipeline.

The complete requirement/catalog/checkpoint reconciliation remains separate from a pytest count. Linux portability, independent reproduction, untouched confirmatory evidence and final paper/release remain separate obligations. The production route supports only the actually verified Python3.12/macOSarm64 environment.

A valid completed negative study is terminal for its version. The original 0.01-nat claim, 6% wealth objective and untouched-data boundary remain unchanged. Generated data, passing tests and a built wheel are engineering evidence, not positive market research.

Pure analysis and reporting import no forecast training or source acquisition. Saved replay decoding is limited to an explicit class allowlist and verifies the exact artifact inventory. The 24/25-date analysis fixtures are constructed retained records; the actual 192-ledger replay fixture uses only two blocks per fold and does not satisfy the full-study precision minimum. These distinct tests must not be described as a single completed scientific run.
