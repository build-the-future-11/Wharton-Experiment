# Forecast-to-portfolio bridge: one synchronous research decision

`src/frem/decision.py` connects a saved forecast snapshot to the constrained
portfolio optimizer. It makes the return transformation and joint-distribution
assumptions explicit. It does not train, download inputs, execute trades, certify
a reserve or manufacture realized portfolio performance.

The code and tests were authored without executing scenario generation,
optimization, model inference or experiments in this delivery.

## Why this bridge is separate

The forecasting model emits cumulative **base-currency log total returns**, with
Student-t location, scale and degrees of freedom. Its scale is not its standard
deviation. A log Student-t variable has no finite positive exponential moment:
the positive-tail density decays polynomially while an exponential grows faster
than any polynomial. Consequently, `expm1(prediction)` is not an expected simple
return.

The optimizer requires expected **simple** returns, a simple-return covariance
matrix and, when used, simple-return scenarios over the same horizon and in the
same base currency. This bridge supplies a separately declared, finite scenario
model. Its expected returns belong to that scenario model.

## The implemented scenario distribution

For assets in the explicitly supplied order, let \(\mu_i,s_i,\nu_i\) be the saved
log-return location, positive Student-t scale and degrees of freedom. Let \(C\)
be a supplied Gaussian latent correlation matrix. The bridge implements

\[
Z^{(j)}\sim N(0,C),\qquad
U_i^{(j)}=\Phi(Z_i^{(j)}),\qquad
Q_i^{(j)}=t_{\nu_i}^{-1}(U_i^{(j)}),
\]

\[
L_i^{(j)}=\operatorname{clip}
  (\mu_i+s_iQ_i^{(j)},\,\ell_i,\,u_i),\qquad
R_i^{(j)}=\exp(L_i^{(j)})-1.
\]

The code uses the documented inverse CDF of
[`scipy.stats.t`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
and the CDF of
[`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html).
The empirical scenario probabilities \(p_j\) then define

\[
\widehat m=\sum_jp_jR^{(j)},\qquad
\widehat\Sigma=\sum_jp_j(R^{(j)}-\widehat m)(R^{(j)}-\widehat m)^\top.
\]

There is no Bessel correction: these are the moments of the finite probability
model supplied to the optimizer. There is no annualization and no second
application of the forecast horizon to the already cumulative parameters.

Clipping is **winsorization**: mass beyond a bound is placed at that bound. It is
not sampling from a Student-t distribution conditioned to lie within the bounds.
The bounds can materially change expected returns and tail risk. They must be
chosen prospectively and justified; the code has no economically recommended
default bounds.

`dependence="independent"` explicitly selects \(C=I\). Independence is never
assumed merely because a matrix is absent. `dependence="gaussian_copula"`
requires a dated, symmetric positive-semidefinite correlation matrix with unit
diagonal. It is a **Gaussian latent correlation**, not a simple-return
correlation and not a Student-t copula. The bridge does not estimate it from
evaluation data. Tiny accepted numerical negative eigenvalues are clipped at
zero and their count is reported.

Uniform scenario weights are the default for the declared Monte Carlo draw.
Supplying different weights requires an explicit weighting rationale. Weights
must sum to one. They are assumptions, not externally calibrated event
probabilities. The numerical normal-CDF endpoint guard and every asset's
lower/upper clipping count are recorded.

## Forecast file contract

The forecast CSV must contain these fields.

| Column | Meaning and validation |
|---|---|
| `asset` | Stable string matching `Instrument.asset_id`; exactly one row per requested asset |
| `time` | One common decision timestamp, equal to request `as_of` |
| `prediction` | Finite cumulative base-currency log-return location |
| `scale` | Positive finite Student-t scale over that same cumulative horizon |
| `student_df` | Fitted finite degrees of freedom greater than two |

`training.prediction_frame` writes `asset` alongside its model-local integer
`asset_id`. The bridge uses the stable string. It does not guess that a local
integer code refers to a particular ticker in a different model or dataset.

All instruments must be present exactly once. The bridge realigns row order to
the declared instrument order. Multiple times, model variants or seed paths in
one input are rejected. Select a prospectively declared path, or prepare an
explicitly defined ensemble forecast artifact before the bridge. It does not
silently average Student-t parameters across seeds.

If an evaluated forecast file also has `entry_time` and `exit_time`, those
timestamps must match the declared common execution interval. Its `actual` and
`label_time` columns are not used in scenario formation. The bridge works with
the unlabeled forecast output of `frem predict`.

Timestamp inputs must be timezone-qualified date-time strings or integer UTC
nanoseconds. Floating-point timestamps and date-only/naive strings are rejected.
The bridge requires exact equality between `forecast_time` and `as_of`; it does
not silently reuse an older forecast.

## Exact JSON request schema

Start with
`configs/mandates/decision_request.template.json` for forecast conversion or
`configs/mandates/allocation_request.template.json` for already supplied simple
moments. These are intentionally incomplete templates. Nulls and replacement
identifiers represent unavailable external information; they are not example
market observations or recommendations.

Both commands accept the following top-level fields. Unknown fields are rejected.

| Field | Requirement |
|---|---|
| `schema_version` | Exactly `1` |
| `mandate_path` | JSON mandate path, resolved relative to the request file |
| `instruments` or `instrument_master_path` | Supply exactly one: an inline nonempty list of instrument records, or a JSON file containing that list |
| `as_of` | Common decision timestamp |
| `nav` | Positive pre-trade NAV in the mandate base currency |
| `current_weights` | Explicit nonnegative pre-trade risky weights, in instrument order; sum at most one |
| `cash_return` | Explicit simple cash return over the common execution interval |
| `reserve_cash` | Explicit nonnegative immediate cash reserve floor in base-currency dollars |
| `return_contract` | Object defined below |
| `decision_context` | Optional `RegimeInputs` fields; quality flags must be Boolean and observed indicators need `inputs_available_at` |
| `policy` | Optional `PortfolioConfig` fields, including a nested `regime_policy`; omitted values use documented research defaults retained in the result |

The instrument master must provide the dated eligibility, market/sector
classification, liquidity, price, currency and lot fields described in
`PORTFOLIO_SPEC.md`. The single-record skeleton is
`configs/mandates/instrument_master.template.json`; a complete master file
contains a **list** of such completed records.

`current_weights` cannot be omitted. An all-cash initial portfolio is represented
by an explicit zero vector, not by null. Existing positions must not disappear
from the requested instrument/master order.

### `return_contract`

Every field in this object is required.

| Field | Meaning |
|---|---|
| `return_space` | `cumulative_base_currency_log_total_return` for `decide`; `simple` for `allocate` |
| `base_currency` | Must exactly equal the mandate's base currency |
| `asset_ids` | Ordered stable string IDs, exactly matching the supplied master and all array columns |
| `forecast_time` | Exactly equal to `as_of` |
| `inputs_available_at` | Latest availability timestamp of the forecast/risk inputs; must not exceed `as_of` |
| `horizon_bars` | Explicit positive integer; no implicit multiplication or annualization |
| `entry_time` | Explicit common future entry timestamp |
| `exit_time` | Explicit common exit timestamp, strictly after entry |
| `common_execution_horizon_attested` | Must be Boolean `true` only after the common executable interval has been verified |
| `timing_source` | Source or calendar/price alignment evidence supporting that interval |
| `input_provenance` | Forecast checkpoint, data, risk input or source identification |

The ordering constraint is `decision < entry < exit`. A declared bar count alone
does not establish a shared executable interval across markets. These fields
are explicit attestations, not independent provider verification. A synchronous
return/price adapter must justify them. The bridge cannot certify an execution
price that the supplied market data does not support.

### Additional fields for `decide`

`forecast_path` names the local forecast CSV. `scenario_config` is required and
contains:

| Field | Requirement |
|---|---|
| `n_scenarios` | Integer from 2 through 100,000; the complete scenario matrix is additionally limited to 2,000,000 entries |
| `seed` | Explicit nonnegative integer |
| `lower_log_return`, `upper_log_return` | Finite scalars or asset-order vectors, with lower strictly below upper; extreme values that would break finite arithmetic are rejected |
| `dependence` | Exactly `independent` or `gaussian_copula` |
| `assumption_rationale` | Prospective justification of the clipping and dependence choices |
| `assumptions_available_at` | Policy availability timestamp at or before the decision |
| `assumption_source` | Versioned policy/source identification |
| `correlation` | Required for Gaussian copula; absent for independent mode |
| `correlation_available_at` | Required for Gaussian copula and no later than the decision |
| `correlation_source` | Required source description for Gaussian copula |
| `probabilities` | Optional aligned probability vector summing to one |
| `weighting_rationale` | Required when probabilities are supplied |

Both the statistical choices and operational scenario budget are part of the
saved request. A large scenario count alone does not establish model calibration
or high-certainty liability funding.

### Additional fields for `allocate`

`expected_returns` and `covariance` are required fields; both represent simple
returns in the declared asset order, base currency and horizon. Optional
`scenarios` has shape `[scenarios, assets]`. `scenario_probabilities` must align
with it, and a positive CVaR penalty requires scenarios.

For an explicitly unavailable forecast, set `expected_returns` to null and
`decision_context.forecast_available` to Boolean `false`. The CLI preserves null
as `None` and permits a constrained risk-only fallback. A valid covariance,
current holdings, mandate, cash assumptions and return contract remain
necessary. The result does not report an invented expected return.

## Commands for a later authorized execution

The following are command templates, not commands executed for this delivery.
Replace the request paths with completed, reviewed request files.

```bash
frem predict --checkpoint CHECKPOINT_PATH --config CONFIG_PATH --input INPUT_PANEL_PATH --output NEW_FORECAST_PATH --device cpu
frem decide --request COMPLETED_DECISION_REQUEST_PATH --output NEW_DECISION_PATH --execute
frem allocate --request COMPLETED_SIMPLE_RETURN_REQUEST_PATH --output NEW_ALLOCATION_PATH --execute
frem compare-family --run COMPLETED_EXPERIMENT_MATRIX_PATH
```

`predict` explicitly defaults to CPU even if the checkpoint was trained on an
accelerator. `decide` and `allocate` require `--execute` because scenario
generation and optimization are computational runs. Neither command places
orders. Existing output files are not overwritten.

The complete decision artifact retains the original forecast parameters,
scenario returns, probabilities, simple moments, clipping/dependence assumptions,
allocation and all constraint diagnostics. File and content hashes identify the
request, forecast, mandate, master and generated scenario matrix. The terminal
prints a compact summary; the full scenario evidence remains in the JSON.

## Accounting and liability boundaries

This bridge handles **one synchronous hypothetical allocation**. Repeated
overlapping horizon forecasts cannot be compounded as sequential returns. The
existing `evaluation.build_execution_ledger` supports explicit, synchronized,
nonoverlapping, fully liquidated round trips from realized labeled forecasts
and supplied weights. Its assumptions include initially cash-funded intervals,
known per-side costs and aligned asset entry/exit times. The decision's retained
stable IDs must be explicitly mapped to that evaluation file's local asset
identifiers when preparing ledger weights.

A portfolio with carried positions, asynchronous exchange hours, unsettled
cash, taxes, partial fills or corporate-action handling during holdings requires
an appropriately detailed execution adapter. Those effects are not implemented
by labelling a one-step allocation as a backtest. Predictive scores, allocation
scenarios and realized net returns are separate evidence artifacts.

Likewise, `reserve_cash` is a specific immediate cash floor, not the full
definition of Laura's operating reserve. `mandate.py` provides annuity-due
arithmetic, dated cash-flow matching, scenario reserve requirements and a 2031
conditional facility-capacity interval. A funding study must connect those
assumptions and eligible reserve assets to all ten beginning-year liabilities.
An arbitrary cash allocation does not certify the case's required high degree
of funding certainty.

The default Laura mandate remains blocked until exact participant rules,
approved instruments and explicit discretionary risk/funding policy are
available. That does not prevent authoring the bridge; it prevents presenting
an unsupported client allocation as compliant.
