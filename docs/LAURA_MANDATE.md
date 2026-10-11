# Laura Gao: recovered case, implementation and remaining inputs

**Status: case cash flows recovered; research allocations remain blocked until the participant universe/rules and the team's numerical policy are supplied. No portfolio, simulation, optimization or financial projection was executed in this authoring session.**

## Source and scope

The current user document, [ROI - Wharton Global Youth Investment Challenge Prep](https://docs.google.com/document/d/1kjLz0M7yy8jxIEbeUnsFML1dLQ2_elf8U9dqq0Gka-U/edit), contains a reproduced competition guide in the Case study tab, `t.ex0r35fa9gmc`. The guide was read directly on 10 October 2026. Its document identity and observed revision are preserved in `laura_case_evidence.json`; the configuration does not rely solely on an earlier assistant's summary.

This is a recovered user-document reproduction, not an authenticated current organizer publication. [Wharton's official FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/) directs registered teams to SurveyMonkey Apply for the current case, trading requirements and other participant materials. Check that material for amendments and the approved securities list before removing the allocation gate.

## What the reproduced case actually requires

| Date | Required treatment |
|---|---|
| Beginning of 2027 | Contribute $300,000. |
| Beginning of 2028 | Contribute an additional $150,000. |
| Before 2033 | No other contributions or withdrawals. Living expenses and short-term personal needs are funded elsewhere. |
| Beginning of 2031 | Communicate a credible range for the potential 2033 facility contribution, with a stated confidence level and assumptions. |
| Beginning of 2033 | Establish an operating reserve **before** making the first operating payment or a facility contribution. |
| Beginning of every year, 2033–2042 | Pay $50,000 from the portfolio: ten fixed nominal payments, without inflation adjustments. |
| After protecting the operating reserve | Determine a responsible facility contribution while preserving financial flexibility. |

All ten operating payments must be supported by the investment portfolio with a high degree of certainty. The model must not fill a shortfall with grants, fees, co-sponsors or unrelated personal income. Those sources may support other residency funding outside Laura's specified operating commitment.

The case does not prescribe a facility contribution or the facility's total cost. It asks the team to justify a funding-confidence definition, reserve composition, investment strategy, financial flexibility, and communication range. It does not require a separately sized contingency endowment. The `flexibility_buffer` variable is the team's explicit representation of capital it chooses to retain; it is a modeling choice, not an additional case obligation.

Long-term projections begin from the two contributions. WInS trading-period gains and losses do not alter that starting schedule. The guide waives personal-income/capital-gains tax calculations and the legal setup of the Taiwan residency. That last exemption concerns establishing the residency; it does not establish a blanket exemption from securities-market restrictions.

## What was not supplied by the case

The reproduced guide assigns no numerical maximum drawdown, country weight, sector cap, turnover limit, asset allocation, risk category or funding-confidence percentage. Entrepreneurial experience alone is not a quantified investment risk tolerance. The document's travel/recipe brainstorming mentions the United States, Taiwan, Japan, Europe and emerging markets. Those passages are presentation ideas, not a mandated eligible universe.

Likewise, the case text does not establish a 10% Taiwan foreign-trading cap. A competition-specific international-security limit may exist in the separate participant trading requirements; that must be checked there. See `MARKET_RULES.md` for the distinction between a price band, an ownership restriction and a portfolio policy.

## Implemented cash-flow mathematics

The beginning-2033 reserve is an **annuity due**, because the first payment happens immediately:

\[
R_{2033}(r)=\sum_{k=0}^{9}\frac{50{,}000}{(1+r)^k}.
\]

At a zero return assumption the nominal payment total is $500,000. A positive assumed rate reduces the arithmetic present value; it does not establish that an eligible, creditworthy portfolio can earn that rate or deliver each payment with high certainty. Negative rates are accepted when greater than minus 100%, and increase the required present value.

For a constant growth assumption `g`, the contribution-only value at the beginning of 2033 is

\[
P_{2033}=300{,}000(1+g)^6+150{,}000(1+g)^5.
\]

The implementation takes explicit year-by-year returns instead of requiring constant growth. It fails when a needed return assumption is missing. WInS gains are deliberately absent from this API.

### Reserve methods implemented

- `annuity_due_reserve`: deterministic constant-rate sensitivity.
- `discounted_cashflow_reserve`: liability-by-liability valuation using supplied date-matched discount factors; the immediate payment must have discount factor one.
- `required_reserve_by_path`: backward recursion under supplied between-payment return paths. With ten payments, exactly nine return intervals are needed.
- `reserve_path_analysis`: conditional coverage under supplied scenario probabilities, with every funding shortfall retained.
- `cashflow_matching_reserve`: nonnegative minimum-cost funding from supplied security cashflows and prices. A later coupon cannot fund an earlier liability; excess cash carries forward at zero interest.

The matching routine requires cash instruments or other immediately available funds to cover the first payment. It does not certify credit quality, real executable prices, instrument eligibility or absence of default. Those remain dated instrument inputs and must be evaluated separately.

## The 2031 communication range

`conditional_2031_analysis` starts from the **observed beginning-2031 balance** and takes two forward return columns, for 2031 and 2032. It projects beginning-2033 values and applies

\[
C_s=\max\{P_{2033,s}-R_{2033}-F,0\},
\]

where `F` is the declared financial-flexibility treatment. The result retains reserve shortfalls, the scenario weight assigned to funded outcomes, a requested interval and its actual mass under the supplied discrete distribution. `C_s` is a mechanical capacity after the chosen buffer. It is not an automatically recommended commitment.

A scenario interval is conditional on its paths and probability weights. It cannot be described as a calibrated real-world confidence interval without independent validation. The case's high-certainty requirement must be assessed on the operating reserve itself, not inferred from a favorable average ending balance.

## How to complete the pending configuration

Use `configs/mandates/laura_gao.pending.json` as the canonical starting point. Populate the participant-rules and approved-universe provenance from the actual current competition material. Record the team's numerical policy and its source. Add an eligible instrument master with current, base-currency prices/ADV, approval dates, exposure classifications, and genuine timestamps. Supply actual current holdings or explicitly declare an initial all-cash research portfolio.

Choose and document concentration, cash, turnover, liquidity and FX limits. Specify the funding-confidence target and financial-flexibility treatment. Remove each `missing_inputs` entry only when its requirement is met, then mark the mandate `reviewed_research`. This changes research allocation readiness; it does not authorize live trading.

The case provenance is recorded at `2026-10-10T12:56:02Z`. An as-of date before that recorded verification does not pass the current-source gate. Historical evaluation needs genuinely dated historical evidence rather than silently applying today's classifications to the past.

## Prior negative evidence is retained

The same document reports earlier synthetic negative results for APEN memory and a larger graph-memory model. The reports have been noted in the evidence record; their underlying run artifacts were not independently re-audited here. They are a reason to retain memory-off and simpler baselines. This new implementation is not evidence that those failures have been solved.
