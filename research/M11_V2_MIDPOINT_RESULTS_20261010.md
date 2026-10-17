# M11-v2 finite midpoint: closed engineering result

The explicitly imported M11-v2 ledger now retains finite high-wealth brackets
when `lo + hi` overflows during bisection. It uses the original midpoint
arithmetic whenever that addition is finite, and `lo + (hi - lo) / 2` only on
the overflowing path. The local v2 rebalance otherwise preserves the inherited
80-step fixed point, real-trade fee law, cash accounting and operation order.

The original `competition/stress.py`, legacy `mpc.py`, model/config/factory,
historical evidence and conclusions remain unchanged. The new prospective
contract is `protocol/M11_V2_MIDPOINT_AMENDMENT_20261010.md`, stacked on PR #3
commit `8dd558b6c90e96f990dbe2e63777498ad4d7f8e0`. Its exact incoming canonical
and source snapshots and closure are under
`research/development/m11_v2_midpoint_20261010/`.

## Verification and actual failures

The first bounded command **failed during collection** because the selective
snapshot lacked an eagerly imported model source. No test or optimizer ran.
The original error log, receipt and exact test bytes are preserved. All 117
Python source files were then materialized at the same exact parent identity.
The baseline was not silently rerun or assigned a numerical pass/fail count.

The repaired gate passed **61 selected cases**: 18 new and 43 inherited, with
warnings treated as errors. Three inherited optimizer tests were explicitly
excluded from this zero-optimizer allocation. New cases exercise large
cash-only balances through maximum finite binary64, purchases, sales, multiple
assets, actual notional fees, a cash-only solver objective, finite ordinary
bitwise comparisons and retained refusal of a nonfinite ledger. Independent
Fraction equations bound the high-range cash/fee errors within explicit ULP
tolerances. A direct call to the untouched frozen stress implementation still
reproduces its old high-wealth failure; it is not reinterpreted as fixed.

Before any numerical execution of these new tests, one cash-only objective
assertion was aligned with the prospectively declared ULP tolerance because
sequential binary64 liability deductions need not equal one exact rational
subtraction bit-for-bit. Its earlier bytes remain with the collection failure.
No observed financial outcome or scientific conclusion informed that correction.

The one final fixture calls real `solve_mpc_v2` with five identical generated
zero-return scenarios, two periods, one asset, an explicit full cash buffer,
cash `1e308`, fixed obligations and a declared terminal reward. A process-local
guard would fail before any optimizer invocation; its actual count is zero.
Cash and the objective agree with exact binary64-input Fraction equations to
within two ULP, and holdings, arrears, fees and turnover remain exactly zero.
All obligations are retained as paid amounts. This is a constructed accounting
sanity case, not a scenario forecast, funding claim or market result.

The raw input and ledger NPZ files are retained. The original derived CSV is
saved separately before deletion, then regenerated from the same verified NPZ
bytes. The table is byte-identical. That preservation step was added after
independent source review, before the single fixture run; the production source
and passing test cohort did not change afterward.

## Finite closure

| Command | Actual child wall seconds | Result |
|---|---:|---|
| Initial regression collection | 2.491999 | Import/materialization error; no tests |
| Selected repaired gate | 10.598174 | 61 passed; 3 optimizer cases excluded |
| Retained cash-only fixture | 8.486442 | Completed and regenerated |

The session is **closed: 3 of 3 commands, 21.576615 of 90 seconds**.
No optimizer, training, provider call, paid job, market data, historical replay,
protected outcome, trade or submission was performed. Unused time is retired.
The separate initial tool-argument size failure occurred before process
creation and is also retained. Earlier studies and closed budgets are unchanged.

The complete records include command drivers, hashes, stdout/stderr, snapshots,
archive member identities, the fixture and the final closure. Original and
regenerated table SHA-256:
`a42c308fd8f397a565a8ded1540671185df88c823008833030e9728083d2289a`.

## Reproduction and scientific limits

In a separately budgeted fresh checkout with the project's dependencies:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q -W error -p no:cacheprovider -o addopts= tests/unit/test_m11_v2_midpoint.py tests/unit/test_m11_v2_objective_numerics.py tests/unit/test_m11_v2.py tests/unit/test_competition_stress.py -k 'not test_solver_cash_feasibility_and_liability_result and not test_solver_uses_declared_scenarios_to_reduce_shortfall and not test_actual_bounded_optimizer_still_improves_constructed_funding_gap'
```

The retained fixture driver documents the exact direct `solve_mpc_v2` call,
input construction, oracle and raw-NPZ table regeneration. Historical drivers
refuse to overwrite their retained directories and receipts; a future run
requires a new session and fresh destinations. The fixture archive includes
the full executed driver and the unmodified raw arrays.

This change does not recover arbitrary overflowing trade totals, asset gains,
costs, initial wealth or objectives. Its ordinary finite midpoint path is
bitwise preserved, while high-range fallback arithmetic is qualified by the
explicit numerical tolerances. The predecessor's leakage, invalid synthetic
lockbox, negative ETF lockbox and exploratory null findings stand. A separately
frozen prospective comparison is still required; no forecasting edge, client
funding reliability, competition readiness or research completion is claimed.
