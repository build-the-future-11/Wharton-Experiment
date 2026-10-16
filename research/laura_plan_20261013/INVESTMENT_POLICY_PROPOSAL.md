# Laura Gao — proposed investment policy and funding plan

Prepared 13 October 2026 from the case supplied by the user. This is a concrete team proposal for review with the portfolio manager. It is not an advisor-approved mandate, a trade ticket, or an S6 investment forecast. The policy does not depend on the new model outperforming.

## Recommendation

Use a **liability-first portfolio**. Direct received contributions toward securing the ten operating payments. After the complete dated obligation and implementation costs are covered, target 60% of the discretionary surplus in diversified global equities and 40% in short USD liquidity, with equities capped at 40% of total portfolio value after costs. In 2033, make at most 80% of positive post-reserve surplus available for the facility and retain at least 20% as uncommitted flexibility. Establish a supported minimum and a firm gift cap before communicating a range in 2031.

The initial composition cannot responsibly be one fixed percentage of stocks and bonds: the dollars available for growth depend on the actual price of Laura's promised cash flows. No amount from the 2028 contribution becomes buying power in 2027.

| Decision | Proposed action |
|---|---|
| Beginning 2027: $300,000 received | Price the full 2033–2042 payment schedule. If it cannot be fully secured, use received funds for eligible matching cash flows or short USD liquidity, retain the measured funding gap, and allocate zero to discretionary equity. Do not assume next year's contribution has arrived. |
| Beginning 2028: $150,000 received | Reprice only the remaining cash-flow gaps, credit cash flows from assets already owned, and finish the match if feasible. Allocate only genuinely unencumbered surplus to the growth/liquidity rule. |
| Quarterly and after material changes | Verify liability coverage, costs, settlement, custody and surplus equity cap. Rebalance only unencumbered discretionary assets. From 2031, protect the assets supporting the communicated facility minimum as well as the operating allocation. A lower market price of a still-matched bond is not itself a missed-payment deficit. |
| During 2031 | Use dated actual holdings and remaining cash flows. Protect the communicated minimum; set a cap to retain upside and avoid overpromising. A beginning-year calculation requires beginning-year NAV and exactly the two remaining annual periods. |
| Beginning 2033 | Verify available proceeds and all ten payments, including the immediate first $50,000. Deduct incremental costs once, establish the reserve, determine the capped facility gift, and retain flexibility. |
| 2033–2042 | Pay each $50,000 from proceeds available before its banking deadline. Maintain the match for unpaid obligations. Do not replenish a falling equity allocation with protected operating assets. |

The growth/liquidity rule is a chosen balance between preserving the non-negotiable payments and retaining upside for the discretionary facility. It is not empirically optimal. If the mandate cannot fund the operating requirement even after the second contribution, the plan is underfunded; no facility pledge or assumed alpha repairs that fact.

## Investments and limits

For the operating allocation, investigate individually held nominal U.S. Treasury cash flows, including STRIPS where available, matched to the deadlines. A STRIP pays once at its maturity, so it avoids assuming reinvestment of intermediate coupons for that payment. Select proceeds dates before the actual spending dates and budget settlement and costs. Treasury access, quoted prices, lot sizes and competition eligibility still require verification. [Treasury STRIPS](https://www.treasurydirect.gov/marketable-securities/strips/).

For global equity surplus, the preferred simple candidate is **VT**, which holds developed and emerging market equities including the United States. It supplies multinational exposure; it does not secure fixed USD operating payments. For short liquidity, consider Treasury bills or **SGOV** if eligible. Its rolling short-bill exposure is not a fixed yield locked through 2042. [VT issuer](https://advisors.vanguard.com/investments/products/vt/vanguard-total-world-stock-etfp), [SGOV issuer](https://www.ishares.com/us/products/314116/ishares-0-3-month-treasury-bond-etf-sgov).

Proposed exclusions are leverage, short selling, derivatives and individual-stock selection. The model may be studied as a surplus-allocation research input, but cannot override reserve coverage. Global stock-fund listing currency is not the same as underlying FX exposure. Match the case's nominal USD obligation in USD; Taiwan's future facility costs and purchasing power remain uncertain, without inventing a TWD conversion or facility budget.

Current 2026–2027 instrument eligibility is not verified. The public ETF and Treasury lists located are explicitly **2025–2026**. The official FAQ directs current trading requirements and resources to the participant's SurveyMonkey Apply materials. This blocks certification of WInS instruments and quantities, not development of the strategic proposal. [Official FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/). Full search findings are in `OFFICIAL_RULES_SEARCH.md` and `SOURCE_ACCESS_RECEIPT.json`.

## Reserve size and growth arithmetic

Let g be a hypothetical constant annual net portfolio return. Wealth immediately before the beginning-2033 payment is

\[
W_{2033}=300000(1+g)^6+150000(1+g)^5.
\]

For a hypothetical flat reserve discount rate d at that same date,

\[
R_{2033}=50000\sum_{k=0}^{9}(1+d)^{-k}.
\]

The first payment is immediate and included. A yield quote is not an observed bond total return; these flat-rate sensitivities are not executable ladder prices. The corresponding reserve is $500,000 at 0%, $458,111.84 at 2%, or $421,766.58 at 4%. At 4%, its hypothetical *2027* value is instead $333,328.26: never deduct a 2033 value from a 2027 portfolio.

With incremental implementation costs K, the signed funding surplus is W−K−R. Any negative value is retained as a gap, not replaced with a forecast. Facility capacity is 0.80 max(W−K−R,0), before any communicated cap, and retained flexibility is at least 0.20 max(W−K−R,0).

The table holds the reserve at the deliberately simple **$500,000 nominal cash amount** and assumes zero incremental costs. Growth rates describe a sensitivity, not expected returns of the proposed sleeves or confidence-ranked outcomes.

| Hypothetical annual net growth | 2033 wealth | Operating gap | Maximum facility capacity | Retained flexibility |
|---|---:|---:|---:|---:|
| −5% | $336,594.71 | $163,405.29 | $0 | $0 |
| 0% | $450,000.00 | $50,000.00 | $0 | $0 |
| 2% | $503,460.85 | $0 | $2,768.68 | $692.17 |
| 4% | $562,093.64 | $0 | $49,674.91 | $12,418.73 |
| 6% | $626,289.57 | $0 | $101,031.66 | $25,257.91 |
| 8% | $696,461.51 | $0 | $157,169.21 | $39,292.30 |

About **1.876% constant annual net growth** is arithmetically required to reach $500,000 by 2033, with no facility contribution or incremental costs. This is a requirement, not a forecast. The full 18-cell cross-product, including all deficits, and cost/flexibility sensitivities are saved in `arithmetic_attempt3/scenarios.json` and `arithmetic_attempt3/SENSITIVITY_TABLE.md`.

A useful cost sensitivity is W=$600,000 and R=$500,000: with no incremental costs, capacity is $80,000; with $5,000 of incremental costs, it is $76,000 and flexibility is $19,000. Costs already deducted from net returns or reserve quotes must not be deducted again. Raising retained surplus from 20% to 30% reduces the zero-cost gift capacity to $70,000.

## Funding certainty and the 2031 range

Define the proposed high-certainty objective as **99% joint success across all ten payments**, not 99% separately for each year. No calibrated estimate establishing that target exists yet. Tests of arithmetic or a high success fraction in selected scenarios do not establish it. A verified cash-flow match instead supports conditional contractual coverage, subject to issuer payment, custody, dates, costs and reserve segregation. Do not assign these risks an invented probability.

For the 2031 range, use a policy bound rather than claiming a market prediction interval. Let H be independently secured, net surplus proceeds available by 2033, beyond all operating obligations and costs. Set the supported facility floor L=0.80H. From the communication date, encumber the assets supporting H: no rebalancing into equities, pledging them as collateral, or other discretionary spending. The equity target is additionally capped at max(S−H,0), and a shortfall in H must be reported. Choose and disclose a maximum gift U, with U≥L, and implement

\[
F_{2033}=\min\{0.80\max(S_{2033},0),U\}.
\]

If S2033≥H remains true, then L≤F2033≤U regardless of what the residual long-only equities do. Upside beyond the cap stays uncommitted. This proof is conditional on the secured floor; it is not a numerical probability of default or future wealth. If H cannot be secured, the supported floor is zero. Any breach must be disclosed and the communication revised.

**Illustration only:** suppose actual beginning-2031 wealth eventually equals $600,000, with $500,000 already protected as operating cash, $40,000 protected surplus cash, and $60,000 in long-only global equities. Assume all implementation costs are already covered and no interest on cash. Choose U=$80,000. Then communicate **$32,000–$80,000**, conditional on the protected cash remaining intact. A complete loss of the $60,000 equity sleeve leaves a $32,000 gift and $8,000 flexibility; a 40% two-year equity loss leaves a $60,800 gift; flat equities give $80,000; positive returns do not force a larger promise. These are not Laura's observed assets or an asserted 99% interval. The code also retains a counterexample in which protected surplus is breached.

**Draft fundraising language for that illustration, usable only if the stated 2031 assets are actually verified:**

> We plan to contribute between $32,000 and $80,000 toward the residency facility in 2033. This range is separate from the assets dedicated to ten annual $50,000 operating payments. Its lower end is supported by surplus cash reserved for the facility, and its upper end is a spending cap we have chosen to preserve financial flexibility. Market outcomes will influence our contribution within that range. The commitment remains conditional on the protected assets being available as planned; we will update co-sponsors if that condition changes.

The actual 2031 dollar range must be recomputed from actual holdings and verified proceeds. Today, the unconditional supported facility pledge is **$0**, because Laura's portfolio has not begun and no future assets are secured. The proposed formula and example are complete; claiming an observed 2031 balance now would be false.

## Evidence and competition delivery

The new planning module passed 28 bounded engineering tests. The separate S6 reservoir/engram/JEPA implementation previously passed 115 engineering tests; it has **zero market fits or backtests** in this delivery. Novelty, superior returns, causal effects and calibrated funding probabilities remain unestablished. The fresh resource check remains below the 512 MiB training admission floor. Admitted S6 data and a separate frozen development protocol are still absent.

The user's accepted definitions are recorded in `CASE_AND_POLICY.json`. The two functions are prediction and portfolio planning; mathematical invariants, delayed quantized updates and interacting branches do not themselves prove a return advantage.

The local current Trading Notes guide requires three notes corresponding to executed WInS trades, copied exactly, with a reflection of no more than 100 words each. It does not require finalized reserve projections at this stage. No transaction history was supplied or found, so this plan invents neither trades nor reflections. The proposed policy, calculations and conditional fundraising text support later IPS/report work. Source: `/Users/ryan/Downloads/2026_WGY_Trading Notes Analysis-FINAL.pdf`, both pages visually checked on 13 October 2026.

All prior failures and negative research remain preserved. This proposal is locally reviewable now; a complete ultimate model, calibrated funding certificate, current eligible trade allocation and final competition submission remain uncompleted.
