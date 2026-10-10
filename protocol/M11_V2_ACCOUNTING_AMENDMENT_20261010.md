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
