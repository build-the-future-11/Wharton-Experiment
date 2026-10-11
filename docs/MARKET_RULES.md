# Market rules: provenance and meaning

**Rules inspected on 10 October 2026. This package generates research allocations and contains no broker integration.**

## Taiwan's 10% figure

The [TWSE guide](https://www.twse.com.tw/en/about/company/guide.html) describes daily price fluctuation bands, generally 10% relative to an opening reference for specified products, with exceptions. A daily price band constrains possible trading prices during a session; it does not state that a foreign investor may allocate only 10% of a portfolio, perform only 10% of trades, or own only 10% of every issuer.

The guide also describes product-specific units, odd-lot mechanisms, registration and settlement. Actual trading access can require separate verification. Its [foreign-investment legal-basis page](https://www.twse.com.tw/en/products/education/foreign/regulations/law.html) references the relevant foreign-investment regulations and TWSE operating rules. No universal 10% foreign-investor portfolio rule was established from these sources. This is not a claim that all issuer-specific ownership restrictions are absent.

| Rule type | Example input | Implemented treatment |
|---|---|---|
| Exchange daily price band | TWSE product-specific reference-price band | Informational `RuleRecord`; cannot be silently mapped to a portfolio weight cap. |
| Approved simulator universe | Organizer-provided list/version and availability date | Mandatory exact instrument-ID membership. |
| Foreign ownership headroom | Dated issuer/account headroom in base-currency value | Optional maximum buy-value constraint per security, if the real requirement is supplied. |
| Team concentration policy | Team decides a 10% Taiwan exposure cap | `with_market_policy_cap(..., kind='team_policy')`; clearly labeled discretionary policy. |
| Sector/country ETF exposure | Dated fund exposure fractions | Fractional look-through exposures counted against the same caps. |
| Tradability/lot size | Security-specific current master | No changes to non-tradable held instruments; separate lot rounding with a complete constraint recheck. |
| Cost schedule | Dated simulator/broker fees and tax treatment | Explicit per-side basis-point assumptions; cash is reduced before reserve checks. |

`Provenance` carries a source ID, title, location, verification timestamp, effective interval, revision, scope and notes. `RuleRecord` separates meaning from a numeric value. A source classified `unverified` or `proposed` cannot satisfy an allocation provenance gate. Metadata with future availability, stale observation times, or missing approval/exposure evidence excludes an instrument. The initial Laura configuration is intentionally incomplete only where external evidence or a team policy is missing.

## Competition constraints

[Wharton's official FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/) places the current participant case and trading requirements inside SurveyMonkey Apply. The public page is not a complete approved-security list. A stock trading somewhere in the world does not make it WInS-eligible, and a company being headquartered in Taiwan does not identify the exchange, settlement currency or economic FX exposure of every security it issues.

Before producing Laura-specific allocations, provide the current participant trading rules and approved universe, including any minimum-position, investment-type or market restrictions. The current optimizer supports long-only linear holdings, cash, concentration caps/floors, turnover, liquidity, dated eligibility and lot checks. If recovered rules require unsupported constraints such as an exact minimum number of distinct holdings or derivatives margin accounting, the integration must reject that rule set until the corresponding constraint implementation is added and verified. Do not mark such a rule set compatible by omitting its requirements.

## Data fields that must remain distinct

`market` means country/market exposure for concentration control. `currency` is the return series' local currency before base conversion. Optional `market_exposures`, `sector_exposures` and `currency_exposures` provide nonnegative fractions summing to one for funds or other approved mixed exposures. `exposure_verified` and `classification_source_id` are required. A plain listing symbol or exchange code does not establish look-through exposure.

All ADV, price, headroom, transaction values, NAV and absolute reserve amounts use the same base currency. Economic FX exposure can differ from quotation currency; populate the currency exposure mapping when needed. A yield is not a bond total-return series. Rate-shock stress calculations require dated duration, convexity and carry, while realized bond evaluation requires total returns or explicit coupons and prices.

## Market-data and execution limits

The package does not predict fill availability at price limits, model an order book, calculate issuer-wide foreign ownership from exchange records, or settle trades. `tradable`, ADV, lot size and verified headroom are supplied conditions for a hypothetical decision. A continuous target is not executable until lot rounding and all original constraints pass. Even then it remains a research target with declared cost and liquidity assumptions.
