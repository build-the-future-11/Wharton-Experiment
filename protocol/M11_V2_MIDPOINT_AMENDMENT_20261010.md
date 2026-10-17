# M11-v2 midpoint range correction — prospective engineering contract

Recorded before changing runtime source or executing regressions on 2026-10-10.
This is a separate bounded engineering session; earlier closed verification
budgets, evidence and accounting amendments are not reopened.

## Incoming identity and preserved conclusions

Selected parent: PR #3 commit `8dd558b6c90e96f990dbe2e63777498ad4d7f8e0`,
tree `f289766b981ec4757b7ff1bc0bf7747108309fe0`.
`EXECUTION_STATE.json`, `STATUS.md`, `CLAIMS.md`, D-050–D-059, the spent
lockbox and the existing M11-v2 accounting amendment remain authoritative.
Exact incoming canonical and relevant source bytes are retained under
`research/development/m11_v2_midpoint_20261010/incoming/` with `contract.json`.

Legacy evidence remains leakage contaminated, the synthetic lockbox invalid,
the ETF lockbox negative and the repaired comparison exploratory/null against
the historical mean. No forecasting advantage, funding reliability, client
suitability, competition readiness or completed research is established.
Existing submission and external-document blockers remain. No market data,
account, trade, submission, historical replay or protected outcome is accessed.

## Exact existing law and repair boundary

The explicit opt-in `wharton_lab.models.m11.mpc_v2` uses the declared
post-cost fixed point

`V_after + c * sum(abs(w * V_after - h_before)) = V_before`,

where `c = cost_bps / 10000`, risky weights are nonnegative with their existing
cash-inclusive row-sum constraint, and holdings and cash are nonnegative.
The inherited implementation bisects `[0, V_before]` for 80 iterations.
It calculates `(lo + hi) / 2` at each iteration and once for the final target.
For finite large wealth, that intermediate addition can overflow although the
midpoint and the whole cash-only ledger are representable.

Implement the same rebalance calculation locally inside the explicit M11-v2
module, replacing only a nonfinite endpoint sum by `lo + (hi - lo) / 2`.
For finite nonnegative endpoints with `lo <= hi`, this fallback is finite and
inside the bracket. Keep `(lo + hi) / 2` bit-for-bit when its sum is finite.
Preserve the 80 iterations, bracket comparison, turnover, fee, cash equation,
validation, liquidation order, liability accounting, CVaR, objective, constraints
and opt-in selection. The frozen `competition/stress.py`, legacy `mpc.py`,
model/factory/config sources and previous results remain byte-identical.

Finite ordinary midpoint operands retain their old arithmetic exactly. The
fallback can have different last bits from a hypothetical unbounded addition;
high-range ledgers are checked against independent exact-rational cash/fee
equations with explicit binary64 tolerance. This is not a general arbitrary-
range financial solver: unrepresentable initial wealth, trade sums, costs,
gains, objectives or ledgers remain outside the numerical qualification.

Falsifiers: a representable high-wealth cash-only ledger still becomes nonfinite;
buy/sell cash, fees or wealth violate the fixed-point oracle beyond declared
rounding tolerance; ordinary results change bitwise; or frozen source bytes
change. Explicit cash-only solver calls must not enter an optimizer.

## Finite allocation and evidence

At most **3 verification commands**, each with a **30-second hard timeout**,
at most **90 seconds total child-command wall time**, at most **20 new test
cases**, and **zero optimizer calls**. Commands: new cases against unchanged
parent; new cases plus the relevant inherited ledger/objective/stress tests
(existing optimizer tests excluded); and one retained cash-only fixture after
independent source review. There is no training, search, new study or paid work.

The fixture calls the real `solve_mpc_v2` with one asset, five identical
generated zero-return scenarios, two periods, explicit full cash buffer,
finite large cash and fixed obligations. It records the full ledger, inputs,
source identities and exact-rational replay; it performs no optimization.
Keep failures, logs, actual durations and all raw fixture artifacts. A failed
command is retained and is not silently rerun. Stop at the finite allocation.

## Review and release

Source review precedes the retained fixture and publication. Publish a draft
stacked on the reviewed parent with `[skip ci]`, verify all final tree leaves,
parent identity and source/state/receipt bytes, and close the new canonical
development amendment with actual verification and unchanged scientific holds.
