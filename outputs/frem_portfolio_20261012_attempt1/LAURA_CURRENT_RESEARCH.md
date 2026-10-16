# Laura Gao — current investment direction, 12 October 2026

The evidence supports a diversified growth core and a separately priced USD
operating reserve. **The learned FREM allocation is not ready for deployment.**
Its newest S4 gate selected no memory in all six model cells, and the fixed
portfolio reanalysis missed the requested 6% uplift in all 32 primary cases.
Those tests concern public survivor histories, not a certified current forecast.

## Instrument research calls

| Role | Current research call | Why it fits the case | What must be resolved |
|---|---|---|---|
| Long-horizon growth core | Investigate **VT (Vanguard Total World Stock ETF)** as the diversified core | Its stated mandate covers developed and emerging global equities, including the US. A single diversified core reduces dependence on any one stock thesis. | Current WInS eligibility, accepted growth risk, executable price, actual holdings and sizing. The FREM backtest used a different 12-stock universe and does not establish VT outperformance. |
| Near-term liquidity | Investigate **SGOV**, or approved short Treasury bills, for cash awaiting investment and approaching payments | SGOV tracks US Treasury bills with 0–3 month maturities. The issuer reported a 3.72% 30-day SEC yield dated 8 October and a $100.51 closing price dated 9 October 2026. | These are dated issuer observations, not an executable quote or guaranteed future return. Short bills require reinvestment; they do not lock the 2033–2042 reserve. Fund fees and spread still matter. |
| Contractual operating payments | Price **individual USD Treasury cash flows or fixed-principal STRIPS** maturing before each required payment | Maturity cash flows can be matched to the nominal payment dates; each stripped principal/interest component is a separate security. | Exact CUSIPs, settlement, maturity before the beginning-year payment, current prices, access and WInS eligibility. STRIPS are held through financial institutions/brokers; access is not established here. |
| Model-driven stock selection | Keep the incremental FREM alpha sleeve at the research stage | There is no demonstrated 6%–36% ending-wealth advantage. The newest conservative memory choice is zero. | An identified external framework, admissible current inputs, stable out-of-time benefit after costs and an approved risk policy. |

Sources checked on 12 October 2026: [Vanguard VT mandate](https://advisors.vanguard.com/investments/products/new/vt/vanguard-total-world-stock-etf),
[iShares SGOV](https://www.ishares.com/us/products/314116/ishares-0-3-month-treasury-bond-etf),
and [TreasuryDirect STRIPS](https://www.treasurydirect.gov/marketable-securities/strips/).
These calls select candidates for the stated roles; no numerical portfolio
weights, trade sizes or price targets are certified from the current inputs.

## Funding comes before the facility contribution

The repository's saved case copy supplies $300,000 at the beginning of 2027 and
$150,000 at the beginning of 2028. Ten $50,000 operating payments begin in 2033
and end in 2042. At the start of 2033, reserve the operating commitment before
the first payment and facility spending. WInS returns do not alter the case's
two contributions. Official amendments remain an external input.

For six assumed annual growth rates `g_2027,...,g_2032`, apply each beginning-year
contribution before that year's return. A constant `g` gives
`W_2033 = 300000(1+g)^6 + 150000(1+g)^5`. The reserve under a hypothetical constant
discount rate `d` is `Σ(k=0..9) 50000/(1+d)^k`; the immediate payment is included.

At zero growth and zero discount, $450,000 does not cover $500,000 of payments:
there is a **$50,000 shortfall** and no facility surplus. At assumed 4% growth,
2033 wealth is about $562,094; that assumption is not a return forecast. A reserve
discount sensitivity does not establish executable bond prices. An SGOV yield
must not be substituted for the ten-year discount curve or called bond return.

Only after a reserve has been priced and the team chooses a flexibility buffer
can a conditional facility capacity be computed. The 2031 communication range
requires 2031 information and a validated forward distribution; no calibrated
confidence level has been invented today. The saved scenario grid preserves
negative surplus and returns `null` for unspecified confidence/facility capacity.

## Current allocation gate

Wharton's public [FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/)
and [rules](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/)
direct registered teams to SurveyMonkey Apply for this year's trading materials.
They do not establish that these specific instruments are approved. No ticker
is labeled eligible by this report.

The remaining concrete inputs are the current approved instrument list/rules,
current holdings and NAV, decision timestamp, reviewed cash/concentration/FX/
liquidity limits, a numerical funding-confidence policy and flexibility buffer,
and the exact source of the user's new quant/ML framework. Public instrument
facts and the ability to calculate a portfolio are insufficient to invent these.
