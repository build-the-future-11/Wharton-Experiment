# Laura Gao: reserve instruments and proposed portfolio policy

Prepared 13 October 2026. Status: **sourced instrument research and a team-proposed policy; no orders, approved allocation, quoted ladder, or S6 investment forecast**. The proposal below separates the operating promise from the discretionary facility gift. It does not depend on an assumed successful trading model.

## Fixed case and timing

The user-supplied case provides USD 300,000 in January 2027 and USD 150,000 in January 2028, followed by ten nominal USD 50,000 operating payments in January 2033–2042. No external operating support is available. The operating reserve must be established before the first payment and facility spending. A useful facility range is needed in 2031. The facility amount is flexible; the case does not supply a numerical equity allocation, funding probability, or additional contingency endowment.

The existing repository convention is beginning-year contributions and payments. The exact banking deadline still needs to be distinguished from a date-only January 1 accounting convention. Proceeds must be available before the actual outgoing payment. Treasury's published payment rule moves a payment falling on a closed Federal Reserve business day to the next business day without extra interest. A redemption labelled January of the spending year therefore does not, by itself, establish timely liquidity. Use an earlier verified cash date and hold the proceeds through the spending date. [Treasury regulation, section 356.30](https://www.treasurydirect.gov/files/laws-and-regulations/auction-regulations-uoc/31-cfr-part-356.pdf)

The nominal operating sum is USD 500,000. At the start of 2033, the first USD 50,000 is immediate; only the remaining nine payments may be discounted. At a valuation date t, with dated zero-coupon discount factors D(t,T), the arithmetic liability value is

\[
R_t=\sum_{j=2033}^{2042}50{,}000D(t,T_j),\qquad
R_{2033}=50{,}000+\sum_{j=2034}^{2042}50{,}000D(2033,T_j).
\]

This equation is a pricing identity under supplied discount factors. It is not a trade quote or proof that eligible securities can be purchased at those factors. The realized reserve cost must include the appropriate cash dates, actual security cash flows, accrued interest where applicable, bid/ask prices, fees and lot sizes.

## Instruments suitable for investigation

| Instrument family | Sourced structure | Proposed role and material limitation |
| --- | --- | --- |
| Individually held nominal U.S. Treasury STRIPS | A stripped principal or coupon is a separate security paying once at its own maturity; access is through a financial institution or government-securities broker/dealer. | Investigate USD cash flows that arrive before each operating deadline. A security retained to payment removes the need to assume coupon reinvestment for that one promised amount. Availability, account eligibility, purchase price and settlement remain unverified. |
| Individually held nominal Treasury notes/bonds | Coupons arrive every six months and principal at maturity. | A dated cash-flow matcher can combine coupons and principal. Cash arriving early can be carried forward at a conservative zero return; cash arriving late cannot fund an earlier bill. Coupon amounts, accrued-interest purchase cost and residual holdings must be counted. |
| Individually held Treasury bills/cash | Bills mature within one year and pay face value at maturity. | Suitable candidates for a nearby spending deadline or settlement bridge. Rolling bills for the entire 2033–2042 schedule leaves future reinvestment rates unknown. |
| SGOV | iShares describes an index exposure to Treasury bills with remaining maturities of 0–3 months. | A candidate for an eligible short USD liquidity allocation. An ETF share does not promise a specified USD payment on each Laura liability date. |
| BIL | State Street describes a Treasury-bill index exposure with remaining maturities of 1–3 months and monthly index rebalancing. | Another short USD liquidity candidate, subject to ETF trading costs and future short-rate changes. It is not a ten-year cash-flow match. |
| VT | Vanguard describes global developed- and emerging-market equity exposure, including the United States, through the FTSE Global All Cap Index. | A candidate for the discretionary growth allocation, with global equity risk. It cannot certify the operating reserve. |
| VTI plus VXUS | Vanguard describes VTI as broad U.S. equity exposure and VXUS as developed/emerging equity exposure outside the United States. | An alternative two-fund growth construction if both are permitted and the team specifies geographic weights and rebalancing. A fixed arbitrary split is not automatically equivalent to VT. |

Instrument structure sources: [Treasury STRIPS](https://www.treasurydirect.gov/marketable-securities/strips/), [Treasury pricing and payments](https://www.treasurydirect.gov/marketable-securities/understanding-pricing/), [SGOV issuer](https://www.ishares.com/us/products/314116/ishares-0-3-month-treasury-bond-etf-sgov), [BIL issuer](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-bloomberg-1-3-month-t-bill-etf-bil), [VT issuer](https://advisors.vanguard.com/investments/products/vt/vanguard-total-world-stock-etfp), [VTI issuer](https://advisors.vanguard.com/investments/products/vti/vanguard-total-stock-market-etf), [VXUS issuer](https://advisors.vanguard.com/investments/products/vxus/vanguard-total-international-stock-etf?compositionTabBox=1).

The current VTI issuer page uses the name **Vanguard Morningstar Total Stock Market ETF**. Vanguard announced this benchmark-related renaming on 29 April 2026. The ticker is the stable instrument shorthand used here; older repository materials may use the earlier fund name. [Vanguard announcement](https://corporate.vanguard.com/content/corporatesite/us/en/corp/who-we-are/pressroom/press-release-vanguard-to-update-names-of-us-equity-index-funds-tracking-morningstar-indexes-042926.html)

These are instrument-role candidates, not a finding of competition eligibility. Issuer existence, a U.S. listing and an available webpage establish none of the following: the participant-approved universe, permission for direct Treasuries/STRIPS, account access, order availability, concentration compliance, or the team's mandate approval. Those must be checked against the authenticated participant materials and account. Do not replace an unavailable STRIP with a bond ETF while continuing to describe the result as a matched liability ladder.

A USD trading price does not eliminate the underlying foreign economic or currency exposures of a global stock fund. Look-through market exposure is distinct from listing currency. Conversely, Laura's stated operating cash flows are fixed nominal USD amounts, so a nominal USD liability match should be evaluated before introducing inflation-linked principal or FX assumptions. The case's tax simplification applies to the case calculation, not a claim about actual STRIPS taxation or real-world account permissions.

## Price, yield, duration and reinvestment

A yield is a quoted rate associated with price and cash-flow timing. It is not a realized annual total return. Treasury notes can trade above or below face value; the coupon percentage alone does not describe their cost or return. An intervening sale realizes the then-market price. [Treasury pricing explanation](https://www.treasurydirect.gov/marketable-securities/understanding-pricing/)

Long-dated fixed-payment securities can fluctuate substantially before maturity. The SEC explains that rising rates generally reduce fixed-rate bond prices and that longer maturities increase interest-rate sensitivity. Matching a liability's cash date can make the promised payment useful despite mark-to-market volatility, but it does not make a forced early sale immune to loss. [SEC interest-rate-risk bulletin, 26 June 2013](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-86)

A coupon bond's yield to maturity is a cash-flow internal rate of return. Interpreting it as a terminal compounded-wealth rate additionally requires that coupons received earlier earn the relevant reinvestment return. To see this directly, terminal wealth is principal plus each coupon multiplied by its actual reinvestment growth factor; changing those factors changes terminal wealth. A zero-coupon payment has no intermediate coupons to reinvest. A sequence of short-bill purchases still requires future purchases at unknown rates. These are mathematical timing distinctions, not assumed market forecasts.

SGOV/BIL provide continuously managed short-maturity exposure. Their current income or SEC-yield figures must not be copied into a fixed return assumption through 2042. Treasury-only bond funds can still lose market value; owning such a fund does not confer an individual bond's maturity payment on a particular ETF share. [SEC bond-fund explanation](https://www.investor.gov/introduction-investing/investing-basics/glossary/bond-funds-and-income-funds)

No yield, current price, historical return, historical volatility, or protected outcome has been used to size an investment in this memo. Search results and issuer pages may contain those fields; they were not input into forecasts, backtests, or the policy constants below.

## Concrete team-proposed policy candidate

The following parameters are an explicitly authored conservative candidate, not case facts, organizer rules, empirically optimal weights, or an already approved mandate. They can be compared with other prospectively declared policies without changing completed S6 findings.

1. **Protect operating payments first.** On each contribution date, evaluate the actual cost and feasibility of securing all remaining USD operating cash flows. Use only received investable assets. The January 2028 contribution is a future case cash flow, not January 2027 buying power or collateral. Until the full match is feasible, direct uncommitted received funds toward eligible matching securities or a short USD holding allocation and retain the measured unfunded gap. Do not conceal a gap with an assumed equity gain.
2. **Keep matching assets separate.** A completed reserve allocation is held for its scheduled cash flows. It is not collateral for leverage and is not sold to buy a falling stock market or increase the facility gift. Monitor cash-flow coverage and custody/settlement, not merely an unchanged historical purchase-cost label.
3. **Bound discretionary equity risk.** At initial funding, let V be received portfolio NAV, R the current executable cost of the complete operating match, and K the separately identified implementation/settlement costs. Let S=max(V-R-K,0). For subsequent decisions with matching securities already owned, compute S from unencumbered assets after the cost of filling only the remaining coverage gap, as specified below. Propose a growth target G=min(0.60S,0.40 max(V-K,0)), with an absolute permitted ceiling min(S,0.40 max(V-K,0)). Keep S-G in eligible short USD liquidity: a 60% equity/40% liquidity split of discretionary surplus when the total-NAV cap does not bind, and more liquidity when it does. Thus a complete loss of the long-only growth allocation cannot consume a separately secured reserve. This is an accounting protection conditional on segregation and no leverage; it is not a probability statement. Review quarterly and after contributions, with any rebalancing subject to actual costs and continued reserve coverage.
4. **Use a simple growth construction.** If eligible, evaluate VT as a single global-equity implementation, or VTI/VXUS as an explicitly weighted alternative. Do not stack all three by default, and do not use S6 outputs to override the reserve floor. Individual stocks, leverage, short sales and derivatives are excluded from this conservative candidate.
5. **Reduce facility uncertainty before communication.** By the 2031 decision, separate any intended minimum facility commitment into verified USD proceeds available by the facility-payment date. Keep uncertain equity-funded upside labelled optional. A later increase can follow realized gains and updated reserve/cost verification; a hoped-for return is not a commitment.

This policy intentionally pays for operating certainty with some foregone growth exposure. It is a defensible candidate under the no-outside-support constraint, not a claim of maximum attainable facility wealth. Partial matching before the second contribution remains vulnerable to changes in the cost of the unmatched liabilities and must be reported that way.

### Feasibility and accounting checks

**Use the reserve price at the decision date.** Under a purely illustrative constant annual discount rate q>-1 and the January 1 convention, the reserve prices are

\[
R_{2027}(q)=50{,}000\sum_{k=6}^{15}(1+q)^{-k},\quad
R_{2028}(q)=50{,}000\sum_{k=5}^{14}(1+q)^{-k},
\]
\[
R_{2031}(q)=50{,}000\sum_{k=2}^{11}(1+q)^{-k},\quad
R_{2033}(q)=50{,}000\sum_{k=0}^{9}(1+q)^{-k}.
\]

Subtracting the 2033 annuity price from 2027 wealth mixes valuation dates. Using a 2026 yield as q at every future purchase date additionally assumes an unchanged yield curve. Actual decisions instead need the then-executable dated cash-flow costs; q above is sensitivity notation only. For a constant *portfolio return* r, a separate illustrative wealth identity is V2033=300,000(1+r)^6+150,000(1+r)^5. Return r and discount rate q are different inputs; no relationship between them is assumed.

**Do not count the same reserve twice.** When a reserve is partially owned, let U_t be unencumbered net assets, excluding securities and cash already assigned to operating payments. Let C_gap,t be the current all-in purchase cost of additional eligible cash flows needed to complete coverage, respecting the existing holdings. Full funding is feasible from received assets only when U_t>=C_gap,t+K_t, where K_t includes only costs not already included in C_gap,t. Once the purchases settle, the discretionary surplus is S_t=U_t-C_gap,t-K_t. If the inequality fails, growth and facility allocations are zero and the shortfall is retained. A fully funded ladder has C_gap,t=0; a mark-to-market fall alone does not create a cash-flow deficit or require sale of the ladder. Conversely, an increase in its marked value is not a spendable facility windfall while its cash flows remain encumbered. Do not subtract the whole reserve cost again from U_t, or release its holdings merely because a separate rate-based annuity estimate is lower.

**Feasibility is conditional, including after 2028.** January 2027 purchasing power starts at USD 300,000. January 2028 adds USD 150,000 only upon receipt to the actual then-existing holdings and cash. At zero net investment returns and zero discount rates, the two contributions total USD 450,000 against USD 500,000 of operating payments: a USD 50,000 funding gap remains and facility capacity is zero. The policy cannot guarantee full matching solely from the contribution amounts, even in 2028. It must report an infeasible funding state if prices, costs and existing assets do not cover the obligations; positive assumed equity returns or future donations cannot repair that state by assertion.

**Compute the 80/20 split from liquid, unencumbered proceeds.** At 2033, if the operating ladder is already held, the immediate USD 50,000 must remain reserved in timely cash and the later nine payments remain encumbered. Apply the facility split to net proceeds outside that reserve after actual liquidation/settlement costs. Do not add the value of those holdings to V and then treat it as spendable surplus. If the reserve is instead being purchased then, V-R-K is the corresponding one-time funding calculation. The formulas are two representations of the same resource budget, not two separate reserve deductions.

## Facility flexibility and 2031 communication

At the 2033 allocation point, before the first operating payment, define realized investable wealth V, the fully specified operating reserve R (including the immediate payment), and implementation costs K. The facility gift F and retained discretionary flexibility B must satisfy

\[
F+B\le\max(V-R-K,0).
\]

The shared team proposal retains 20% of positive surplus as uncommitted flexibility and makes at most 80% available for the facility. With S=max(V-R-K,0), this gives

\[
F_{max}=0.80S,\qquad B=0.20S.
\]

The 20% is a team-selected fraction of positive surplus, not a new case-mandated operating endowment or a second prescribed portfolio. A 20%-of-surplus buffer is different from 20% of the gift; expressed relative to the maximum gift it is 25%. Report 0%, 20% and 30% retained-surplus sensitivity. A fixed-dollar buffer is an alternative policy, F=max(V-R-K-B,0), and must not be silently combined with the percentage policy. Any reserve deficit makes facility capacity zero and is recorded separately. Cost K and flexibility B must not double-count the same cash.

The 2031 communication should have three clearly distinguished amounts:

- **Cash-flow-supported minimum:** only the facility amount supported by already secured, eligible cash flows arriving by the planned gift date, after covering the operating reserve, costs and chosen flexibility. If no positive floor is secured, this number is zero.
- **Conditional planning range:** scenario-specific facility capacity derived from observed 2031 holdings/NAV, the remaining 2031–2032 path assumptions and liability costs. Report the assumptions, downside cases, and any zero-capacity or unfunded-reserve cases. In 2026 this can only be a sensitivity about a future 2031 decision, not an observed-2031 forecast.
- **Uncommitted upside:** additional gift capacity contingent on favorable realized investment outcomes; it is not promised to the facility project.

Use the actual 2031 communication timestamp. The existing two-calendar-year helper applies specifically to 1 January 2031 through 1 January 2033; a mid-2031 update has a shorter remaining horizon and must not reuse two full annual compounding periods. If future reserve purchases remain necessary, each planning scenario must pair its projected investable wealth with its own reserve purchase cost and costs, S_s=max(V_s-R_s-K_s,0), before computing F_s=0.80S_s. Taking an optimistic wealth quantile and subtracting an unrelated average reserve cost is not a joint downside assessment. For an already matched reserve, condition on the unencumbered assets and secured cash flows instead of repricing and spending the same reserve a second time.

The shared team proposal sets a **99% joint funding-certainty target for all ten operating payments**. This is a chosen policy objective, not an estimated or achieved probability. A 95% target can be displayed as sensitivity; neither target is satisfied merely because that percentage of a selected simulation set succeeds. Likewise, a central scenario interval or a minimum/maximum sensitivity table is not an empirically calibrated prediction interval. Recession odds, future FX paths and equity-return probabilities remain unassigned unless separately justified and validated. The operating cash-flow match is assessed separately as conditional contractual coverage from identified obligations and asset payments, rather than assigning it an invented 99% success probability.

## Existing code to reuse and its exact limits

The following source was inspected without importing scientific modules or executing new financial scenarios in this subtask.

| Existing function / contract | Reuse | Boundary |
| --- | --- | --- |
| `src/frem/mandate.py:520` — `annuity_due_reserve` | Flat-rate arithmetic for ten beginning-year payments, including immediate first payment | A constant rate is an explicit assumption, not a market term structure or secured return. |
| `src/frem/mandate.py:532` — `discounted_cashflow_reserve` | Apply one supplied discount factor per dated operating liability | It validates immediate discounting and currency consistency, but does not supply executable bonds or prices. |
| `src/frem/mandate.py:555` — `project_contributions` | Correctly compound 2027 and 2028 contributions through the year before the target | Uses supplied annual returns and excludes WInS gains; does not forecast them. |
| `src/frem/mandate.py:610` — `facility_contribution_analysis` | Preserve zero facility capacity and reserve deficits across supplied scenarios | Existing API uses a fixed buffer and common reserve amount; percentage flexibility and scenario-varying reserve costs need explicit preprocessing or an additive adapter. |
| `src/frem/mandate.py:653` — `conditional_2031_analysis` | Condition on beginning-2031 NAV and exactly two annual return columns | This applies a common reserve/buffer. It must not be represented as already observing 2031 NAV, or as a full joint model of wealth and liability prices. |
| `src/frem/mandate.py:682` / `:706` — `required_reserve_by_path` / `reserve_path_analysis` | Pay first, then grow; ten payments use exactly nine between-payment returns | Probabilities are assumed. The required-reserve routine rejects a -100% between-payment return rather than returning infinite required capital; do not silently discard such a stress case. |
| `src/frem/mandate.py:732` — `cashflow_matching_reserve` | Solve minimum-cost nonnegative holdings with cumulative asset receipts covering cumulative dated liabilities | SciPy linear program; allows zero-interest carryforward and preserves infeasibility. Quotes, dates, eligibility, costs, FX, lots and credit/custody remain external verification; no authenticated ladder has been priced here. |
| `src/frem/mandate.py:141` / `:295` — `Instrument` / `Mandate` | Validate dated prices, approval provenance, exposure classifications, explicit risk policies and cash flows | Public issuer descriptions do not fill private participant approval fields. |
| `src/frem/portfolio.py:270` — `optimize_portfolio` | Apply constraints, costs, holdings and a reserve-cash amount consistently | An immediate cash floor is not a ten-payment funding certificate. |
| `src/frem/decision.py:257` — `bounded_forecast_scenarios` | Retain forecast transformation and joint-scenario assumptions | A bounded model or new S6 forecast is not evidence that the reserve is funded or that the selected assets will outperform. |

## Source dates and completion boundary

All linked primary pages were checked on 13 October 2026. Treasury's educational pages and the SEC bond-fund glossary expose no publication date in the retrieved text; they are current source checks, not evidence about what a historical investor knew. The SEC interest-rate bulletin is dated 26 June 2013. The Treasury regulation PDF is a pre-2026 published source. The VTI naming announcement is dated 29 April 2026. Retrieved issuer pages carried visible data dates no later than 2 October 2026 (SGOV/VTI), 21 September 2026 (BIL), and September 2026 (VT/VXUS); only structural product descriptions were used, not those market-data values. Refresh the prospectus and actual participant/account eligibility before any implementation.

This memo completes the instrument-mechanics research and supplies a concrete policy candidate and reuse map. It does not establish executable ladder prices, approved investment permissions, future realized wealth, a calibrated 2031 interval, or a current model-based allocation.

Final review correction: after the 2031 communication, assets supporting its minimum are encumbered separately from free discretionary assets. The equity target is additionally capped at max(S-H,0), where H is independently secured surplus after operating liabilities and costs. Routine 60/40 rebalancing cannot spend H. The planning code records any H shortfall. The capped gift rule retains at least 20% surplus; favorable upside above the communicated cap remains uncommitted.
