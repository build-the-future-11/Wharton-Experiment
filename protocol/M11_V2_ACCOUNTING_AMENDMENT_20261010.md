# M11 v2 accounting amendment — development only

State: SMOKE_PASSED (28 controller/accounting regression tests). This amendment
implements the accounting repair identified by D-059. The frozen M01–M12 files,
model factory, fingerprints, seeds, lockbox, historical runs and published
conclusions remain unchanged. Import `wharton_lab.models.m11.mpc_v2` explicitly.

## Model contract

Inputs are nonnegative holdings/cash, finite simple-return scenarios shaped
`[scenario,horizon,asset]`, a nonnegative liability schedule, transaction costs
below 10000 bps, and one common open-loop sequence of risky-asset weights. The
cash fraction is one minus the row sum. No future realized observation selects
a different weight sequence for an individual scenario.

Each period uses the independently tested `competition.stress.rebalance`
fixed-point accounting: post-cost wealth plus transaction cost on actual
currency trades equals pre-trade wealth. Returns change holdings, not cash.
When required, proportional liquidation covers cash needs with costs on sold
notional. Current obligations and prior arrears are paid once; unpaid amounts
remain visible as arrears, without borrowing or external capital.

The solver minimizes empirical CVaR of maximum outstanding arrears over the
horizon. The upper-tail probability mass includes a fractional boundary
observation. A separately explicit terminal-wealth reward defaults to zero.
Risky-asset caps and a minimum cash fraction are enforced. Optimizer failures
are exposed, with no fabricated allocation to asset zero.

## Falsifiers and verification

Cash-only 0.1/0.1 liabilities from wealth 1 must yield 0.9/0.8, while the legacy
0.8/0.8 result remains reproducible. Tests check exact liability accounting,
cash versus unrealized gains, holdings-dependent turnover, liquidation costs,
wealth conservation across scenarios, insolvent paths, future-shock isolation,
fractional CVaR, constraint feasibility and invalid input rejection.

Run `PYTHONPATH=src python -m pytest tests/unit/test_m11_v2.py tests/unit/test_m11.py tests/unit/test_competition_stress.py -q`.

No real-market replay, lockbox access, competition submission, model superiority,
funding reliability or client-suitability result is produced. A scientific
comparison of M11-v2 requires a separately frozen prospective experiment.

## Follow-up: objective and tail-average numerical integrity

State: DEVELOPMENT_NUMERICS_VERIFIED. This follow-up applies only to the
explicitly imported successor `mpc_v2.py` from draft PR #3, reviewed parent
`42e5aa0c7b9b2fdc5c4ab560aafefb8c30929b37`. It does not alter the legacy model
factory, frozen `mpc.py`, competition stress accounting, historical manifests,
75 retained stress cells, consumed lockbox or conclusions in D-050–D-059.

Five fictional cash-only ledgers with wealth `4e307` each are individually
finite. The previous objective nevertheless reduced their sum to infinity and
returned `success=True` with objective NaN when the optional wealth reward was
disabled (`0 * inf`), or negative infinity with reward coefficient 1. The
default objective now omits the disabled term. Enabled rewards use a
compensated mean with exact rational accumulation if the intermediate sum
overflows. Unrepresentable objectives raise an arithmetic error; a nonzero mean
or positive reward product that underflows to zero is also rejected. Independent
review supplied the product example: cash `1e-200` with reward `1e-200`.

The public empirical CVaR helper previously scaled all losses by their maximum
absolute magnitude. This erased a residual before cancellation: at alpha zero,
`[1e308, 1, -1e308]` returned 0 instead of 1/3. It now accumulates the admitted
binary64 losses and the existing binary64 fractional tail mass exactly as
rationals, then converts once. A nonzero unrepresentable CVaR fails explicitly;
genuine cancelling losses still yield zero. The alpha convention, upper-tail
mass, arrears objective, optional reward coefficient and optimization constraints
remain the same. Small last-bit differences from the former arithmetic are
possible and are confined to prospective execution.

Validation uses 20 new regressions plus the 28 existing controller/accounting
tests: **48 tests** in the combined bounded suite. The first 16 new cases gave
eight failures and eight passing controls on the parent implementation. Cases
include finite large means, disabled rewards, true objective overflow, signed
CVaR residuals, fractional mass, subnormal representability, exact cancellation,
ordinary cash/zero-wealth controls and a real three-scenario optimization that
funds a fictional 1.5 obligation from initial wealth 1 under declared +100%
scenario returns. That constructed sanity case is not a market result.

A local diagnostic of 1000 CVaR calls over 64 losses at alpha .95 took about
0.082 seconds before and 0.117 seconds after the repair; the exact accumulator
has a cost. One tiny bounded solver call took 0.056/0.039 seconds respectively.
These timings are noisy development diagnostics, not speed or scaling claims.
No performance gain is asserted. Larger scenario workloads remain unprofiled.

Run the combined suite with:

```bash
PYTHONPATH=src python -m pytest tests/unit/test_m11_v2_objective_numerics.py tests/unit/test_m11_v2.py tests/unit/test_m11.py tests/unit/test_competition_stress.py -q
```

The new bounded CI job checks out the exact PR commit and executes only these
fictional tests. It does not execute orchestration profiles, data acquisition,
historical replay, the lockbox, competition submissions or accounts. Its actual
hosted result is recorded in the PR after completion. Scientific comparison of
this opt-in version still requires a separately frozen prospective protocol.
