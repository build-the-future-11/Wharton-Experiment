# Data acquisition and information contract

**Status: source code authored; no dataset was downloaded, trained on, simulated or backtested in the authoring session.** This guide describes future actions for the operator. The code supports a strict tabular contract; it does not make arbitrary data scientifically interchangeable.

## 1. Choose a research universe before inspecting outcomes

For the first experiment, use one documented universe and one execution calendar. A fixed, genuinely predeclared set of liquid ETFs is a manageable engineering scope, provided the instrument choices were recorded before the evaluation period and distributions are accounted for. A present-day collection of attractive surviving tickers is not a point-in-time equity universe. A retrospective ETF study can be described as conditional on that fixed set; it must not be presented as unbiased performance over all investable instruments.

For broad equity research, obtain membership history, permanent identifiers, corporate actions and delisting outcomes from a licensed research source. CRSP's source documentation includes delisting-return information; consult the relevant product/version rather than assuming an ordinary price export already incorporates those outcomes. Institutional access may be required. [CRSP data documentation](https://www.crsp.org/wp-content/uploads/2023/10/crsp_us_stock.pdf) and [current provider product information](https://indexes.morningstar.com/research-data-products/crsp-us-stock-databases).

Record every source's license, retrieval time, version, economic meaning, currency, exchange calendar, adjustment method and limitations. Keep the unmodified source export. Use permanent instrument identifiers when tickers change or are reused. The manifest attests to this review; the validator cannot prove a vendor's completeness merely because a flag is set.

## 2. Files accepted by the implementation

`frem.data.load_panel(path, column_map=None)` reads CSV or parquet. A mapping has the form `{"vendor_ticker": "asset", "vendor_close": "close"}`: source name to canonical name. It does not guess which date represents a close, download anything, backfill missing prices, infer dividends or substitute yields for bond prices. Parquet requires the parquet dependency listed in the project installation instructions.

Place a JSON manifest at `<panel-path>.manifest.json`, for example `prices.parquet.manifest.json`. The loader preserves its fields in `DataFrame.attrs`, computes a file hash and computes a canonical data/provenance fingerprint. If `expected_file_sha256` is supplied, a mismatch is an error.

### Mandatory panel columns

| Column | Meaning and requirements |
|---|---|
| `asset` | Stable string instrument identifier; never silently reuse an identifier for another security. |
| `timestamp` | Market price/valuation event time with explicit timezone. It is converted to UTC nanoseconds. |
| `available_at` | Time that this observation was actually usable by the strategy, including feed/publication delay. Must not precede `timestamp`. |
| `close` | Finite positive actual traded/valuation price. For actual bond coupon accounting, this is the dirty price on a consistent face-value basis. |
| `currency` | Local quotation currency. Constant per instrument identifier. |
| `asset_type` | One of `equity`, `etf`, `bond`, `bond_etf`, `fund`, `commodity_etf`. |
| `price_kind` | One of `total_return_index`, `raw_with_actions`, `bond_dirty_with_coupons`. |
| `eligible` | Explicit boolean for known universe eligibility at this snapshot. |
| `eligibility_known_at` | Timestamp at which the eligibility classification became known; no later than publication. |
| `universe_source` | Evidence/source for the membership snapshot. |
| `executable` | Explicit boolean saying this observation is usable by the declared close-execution approximation. It does not guarantee a real fill. |
| `source` | Provider/export provenance. |

Duplicate `(asset, timestamp)` rows fail. Historical price revisions require a separate vintage-aware normalization before they enter this adapter; selecting the final revised row and assigning it an old availability time is invalid. Effective feature availability must increase strictly with observation time for each instrument. The adapter deliberately rejects out-of-order revision streams instead of pretending to handle them.

Daily dates such as `2024-06-03` are rejected as price timestamps. The operator must supply the actual exchange close/valuation time, timezone and publication lag under a documented calendar, including daylight-saving transitions, holidays and shortened sessions. Assigning every US close to midnight UTC is not a valid conversion. FX observations and external feature releases have their own availability fields.

### Return, FX and optional feature columns

| Condition | Additional required data |
|---|---|
| `price_kind=total_return_index` | Positive `total_return_index`; documented `corporate_action_source` and manifest return methodology. The index must correspond to the same security and economic interval as the price. |
| `price_kind=raw_with_actions` | `split_ratio`, `cash_distribution`, `corporate_actions_available_at`, `corporate_action_source`. Explicit no-event values are 1 and 0 respectively. |
| `price_kind=bond_dirty_with_coupons` | The same action columns, with `split_ratio=1`; `close` is dirty price and `cash_distribution` records coupons/redemption on the same face-value basis. |
| Currency differs from base currency | Positive `base_fx`, `fx_available_at`, `fx_source`; FX is base-currency units per one local-currency unit. |
| External `feature_*` columns | Numeric values, `feature_available_at` and manifest `feature_provenance`. Missing values are allowed; infinity is rejected. |
| Peer covariates | `market`, `sector`, `classification_known_at`; classification must be known at the decision. |
| Optional execution/market covariates | `open`, `high`, `low`, `volume`; OHLC ranges must be consistent and volume nonnegative. |
| Bounded eligibility | Optional `eligible_from` and `eligible_to`; when provided, the window builder respects them. Their provenance must reflect what was known at the time. |

For raw prices, `split_ratio_t` is new shares per old share, and `cash_distribution_t` is cash distributed **per unit held immediately before the interval**, expressed in the same local units as the prior price. The implemented wealth growth is

\[
G_t = \frac{P_t s_t + D_t}{P_{t-1}},\qquad W_t=W_{t-1}G_t.
\]

If a vendor reports dividends per post-split share, convert that convention explicitly. Do not put annualized dividend yield into `cash_distribution`. For a total-return index, the code uses the supplied index ratio. For foreign assets it multiplies local wealth by the timestamped FX conversion rate. Taxes, withholding and instrument-specific reinvestment conventions belong in the source methodology and later account-specific cash-flow ledger.

An actual bond yield is a discount-rate quotation, not a realizable bond return series. A bond ETF is an ETF whose observed price/NAV and distributions define its return. An actual individual bond needs a dirty-price/coupon or total-return history. The adapter does not construct a bond price from a government yield curve without security cash flows, accrued interest, recovery conventions and executable pricing.

**Known scope boundary:** the continuous log-return model requires strictly positive terminal wealth. A zero-recovery default or complete delisting loss cannot be represented by a finite log return. Such a record causes an error; do not remove the bankrupt instrument to make the data pass. A default point-mass model and a loss-aware execution ledger must be added before claiming coverage of those outcomes. The current implementation is not a universal defaultable-bond engine.

### Mandatory manifest fields

The following is an input template, not a completed source attestation:

```json
{
  "base_currency": "USD",
  "universe_status": "CHOOSE_point_in_time_complete_OR_fixed_ex_ante",
  "universe_frozen_at": "REQUIRED_VERIFIED_TIMESTAMP_IF_FIXED_EX_ANTE",
  "delistings_included": false,
  "availability_policy": "DOCUMENT_EXCHANGE_CALENDAR_AND_FEED_PUBLICATION_LAGS",
  "total_return_methodology": "DOCUMENT_DISTRIBUTIONS_SPLITS_DELISTINGS_AND_REINVESTMENT",
  "feature_provenance": {},
  "source_license": "DOCUMENT_LICENSE_AND_PERMITTED_RESEARCH_USE"
}
```

`point_in_time_complete` additionally requires `delistings_included=true` backed by an audit. `fixed_ex_ante` requires a timezone-qualified `universe_frozen_at`; observations before that actual declaration are excluded. A fabricated old freeze date does not turn a retrospectively selected universe into an ex-ante one.

## 3. Obtain price data and normalize it

Use a provider/export already covered by your access rights. A practical sequence is:

1. Export permanent instrument IDs, raw close/dirty prices, volume if available, distributions, split factors, listing/delisting events, currency and historical classifications for the declared period.
2. Export the provider's total-return series when its methodology is suitable, or normalize raw corporate actions to the formula above. Preserve the original action values as evidence.
3. Convert market dates using an exchange calendar, keeping price-event time distinct from information availability.
4. Attach historical eligibility and classifications from the frozen membership source. Include terminal outcomes; investigate missing delisting returns.
5. Attach FX in the correct quotation direction with explicit availability and record an as-of policy for asynchronous feeds. No future observation may be used to fill an earlier quote.
6. Write the canonical panel and its manifest, then run validation only when execution is authorized.

For exploratory access, Alpha Vantage documents separate raw daily and adjusted-daily endpoints, including split/dividend fields. Read the current endpoint access conditions; the artifact makes no claim about free entitlement or completeness. Its daily API fields still need the timestamp, availability, eligibility and provenance normalization above. [Official API documentation](https://www.alphavantage.co/documentation/). A current adjusted-close export by itself is not evidence of point-in-time universe construction or publication timing. No Alpha Vantage or Yahoo downloader is invoked automatically by FREM.

## 4. Macro covariates: vintages rather than final revised history

`point_in_time_macro(macro, decision_times, max_age_days=None)` accepts `series`, `observation_time`, `available_at`, `value`. At a decision it considers only released records, applies known revisions to their observation periods, then selects the newest observation period known at that time. A late revision of an older period does not replace a newer period. Before the first release, the feature remains missing; there is no backward fill.

The macro loader preserves a `series_universe` in the manifest/attributes. When a development cutoff leaves a declared series with no known observations, that series still supplies an all-missing feature column. This keeps the input schema stable across development and confirmation. Freeze the intended series list and explicit feature columns; adding a new macro series is a protocol change.

ALFRED distinguishes observation dates from vintage/real-time periods. FRED API version 1 supports observation retrieval with real-time bounds and pagination. These are the basis of the included opt-in adapter. [ALFRED vintage documentation](https://alfred.stlouisfed.org/help/downloaddata), [FRED observation endpoint](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

After later execution authorization, a caller can use:

```python
import getpass
import os
from frem.data_sources import FredVintageConfig, fetch_fred_vintages, load_macro

os.environ["FRED_API_KEY"] = getpass.getpass("Registered FRED API key: ")
request = FredVintageConfig(
    series_id="UNRATE",
    observation_start="1990-01-01",
    observation_end="2025-12-31",
)
manifest = fetch_fred_vintages(request, "data/raw/unrate_vintages.csv")
macro = load_macro("data/raw/unrate_vintages.csv")
```

These dates and this series demonstrate the API, not the frozen project protocol. The request preserves vintage fields and response hashes, paginates explicitly and redacts the API key from saved metadata and HTTP errors. It never obtains stock or bond return data.

Date-only vintage metadata does not establish an intraday release time. The adapter conservatively sets `available_at` to midnight UTC two calendar days after the vintage date, marks exact release times unverified, and rejects a smaller delay. For intraday work, import verified publisher release timestamps. This delay is a conservative research policy, not a claim about actual FRED publication latency.

## 5. Feature and label construction

`build_windows(frame, context, horizon, ...)` constructs causal return, rolling volatility, momentum, elapsed-time and optional volume/range features. All windows use trailing observations. `include_missingness=True` adds explicit `__missing` indicators. The scaler is fitted only on training `x`, never on `future_x`, validation or test. Median imputation, IQR scaling and optional clipping use the frozen training state. All-missing training features are recorded and assigned a zero imputation/one scale, while their missingness indicators remain binary.

For decision time `t`, entry is the first executable asset observation with `timestamp > t`. Exit is `H` executable intervals after that entry. The label is the cumulative base-currency log total return from entry to exit. Outcome maturity is no earlier than the exit and the availability of all required target information. Rows lacking future executable observations are counted and excluded from training; they are never assigned zero returns. `future_x` describes those future target observations and is training-only information.

`add_causal_peer_features` adds market/sector peer return means, market dispersion and peer counts. It uses the latest **already available** peer observations within a stated staleness bound and excludes the asset itself. Classifications must be historically known. Because different markets close at different times, these are asynchronous covariates, not synchronous portfolio returns or a learned graph.

`build_inference_windows` instead returns each currently eligible asset's latest context at a common `as_of` time. Unknown `y` and `future_x` are NaN, maturity is an explicit unknown sentinel and execution times are absent. It requires only past inputs. Use `align_asset_universe(data, trainer.asset_names)` for a narrower current panel; unseen assets cause a clear error rather than silently changing embedding IDs.

## 6. Protected-data handling

During development, remove raw observations whose market or required availability timestamps reach the protected boundary before building features or windows. Hash the complete source file as an opaque artifact without reporting protected-period summaries. Split construction also excludes earlier decisions whose outcomes mature inside the protected tail. Macro releases are joined as-of; contemporary revisions must not be assigned historical availability.

Changing source versions, universe definitions, return conventions, execution timing, macro release policies or feature schemas changes the experiment. Preserve the old manifest and failures, create a new development protocol, and do not silently reuse an already examined protected tail.
