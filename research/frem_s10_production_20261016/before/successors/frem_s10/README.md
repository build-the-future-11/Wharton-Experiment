# FREM S10

S10 is an isolated **research implementation in development**, with no S10 market result. It implements causal empirical forecasts, scoring, constrained mixtures, fee-aware policy selection, all six benchmarks, direct-horizon data contracts, exact event accounting, fixed-matrix replay, typed saved-output readback, paired endpoint/ablation analysis and bounded statistics reporting. It does not yet implement the complete guarded precision/campaign/report pipeline. No profitable, novel or independently reproduced model is claimed.

The latest audit and known-flaw checklist are in `research/frem_s10_repair_20261015` at the repository root. The original scientific design remains immutable in `research/frem_s10_20261015/design_v2`; prospective `research/frem_s10_repair_20261015/DESIGN_ADDENDUM_CL15.md` clarifies A4 ordering and `DESIGN_ADDENDUM_CL16.md` fixes minimum-variance tie units. The audit narrows ES claims to the total-return distribution, leaving all score thresholds unchanged.

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

Offline setup requires the pinned packages in the named cache. Process-supervisor tests require permission to read local PID/RSS/start metadata using `/bin/ps`; generated memory counters in those tests are fixtures, never live host admission. The last two commands read only design metadata or host counters. `plan` enumerates 192 proposed ledger identities and explicitly reports zero executed ledgers. `check-host` returns exit 3 when the unchanged 512 MiB/non-rising-pageouts rule fails. Neither command runs a model. There is no market `run` or `resume` command.

The package's numerical tests use generated examples. They do not access market data or establish independent empirical replication. The current validation platform is CPython 3.12 on macOS arm64. Other platforms and full-study time/memory remain unverified.

## Remaining gates

The complete 60-requirement and ten-checkpoint status is retained in the audit directory. Generated fixtures exercise the 192-cell matrix with real policy and ledger code; separate small cases connect the real forecast kernel through policy to ledger. The whole market-length, 520-atom workload remains unprofiled. A generated pre-2010 pilot exercises the real kernel and precision calculation, with projected A/B counts derived solely from timestamps.

Required work still includes the final operation-specific admission/launcher and freeze integration, exact real source/calendar and close-clock admission, complete catalog closure, weekly certificate/state archival, complete scientific saved-output audit and release, measured full-scale feasibility, real precision preparation, then a separately frozen development campaign. The generic supervisor verifies local lifecycle/resources. A separate structural declaration validator binds exact input/configuration/code/environment/verification/precision artifacts, but does not produce their missing scientific admission or launch a market operation. There is no market CLI route before that integration is complete.

A valid completed negative study is terminal for its version. The original 0.01-nat claim, 6% wealth objective and untouched-data boundary remain unchanged. Generated data, passing tests and a built wheel are engineering evidence, not positive market research.

Pure analysis and reporting import no forecast training or source acquisition. Saved replay decoding is limited to an explicit class allowlist and verifies the exact artifact inventory. The 24/25-date analysis fixtures are constructed retained records; the actual 192-ledger replay fixture uses only two blocks per fold and does not satisfy the full-study precision minimum. These distinct tests must not be described as a single completed scientific run.
