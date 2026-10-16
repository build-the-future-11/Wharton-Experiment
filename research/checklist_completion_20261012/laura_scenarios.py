"""Conditional cash-flow arithmetic, not forecast returns or an allocation."""
import json
from pathlib import Path
m=json.loads(Path('configs/mandates/laura_gao.pending.json').read_text())
contributions=[x for x in m['cashflows'] if x['kind']=='contribution']
liabilities=[x for x in m['cashflows'] if x['kind']=='operating']
rows=[]
for growth in [0,.02,.04,.06]:
 wealth=sum(x['amount']*(1+growth)**(2033-int(x['date'][:4])) for x in contributions)
 for discount in [0,.02,.04]:
  reserve=sum(x['amount']/(1+discount)**(int(x['date'][:4])-2033) for x in liabilities)
  rows.append(dict(assumed_growth=growth,assumed_reserve_discount=discount,wealth_before_2033_payment=wealth,reserve_before_2033_payment=reserve,net_surplus_before_buffer=wealth-reserve))
assert rows[0]['wealth_before_2033_payment']==450000
assert rows[0]['reserve_before_2033_payment']==500000
assert rows[0]['net_surplus_before_buffer']==-50000
out=Path('research/checklist_completion_20261012')
(out/'laura_scenarios.json').write_text(json.dumps({'scope':'Deterministic assumptions, not market predictions, yields or funding probabilities','case_source':m['case_source_id'],'rows':rows},indent=2)+'\n')
t='''# Laura Gao: funding analysis and allocation readiness, 12 October 2026

No current FREM return forecast or compliant stock allocation can be issued from the available artifacts. Historic exploratory checkpoints exist; they do not supply current admitted inputs, proven predictive utility or eligibility. The older watchlist is preserved as a dated opinion, not revalidated or endorsed here.

## Official website check

Wharton FAQ and the linked 2026-2027 SurveyMonkey Apply program page were checked today. The FAQ directs team leaders to the account for official case, trading requirements and deliverables. The public program page directs registered users to their application tasks; it does not provide the approved instrument list or team holdings. Public rules do not establish those missing values.

- https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/
- https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/
- https://wghsinvcomp.smapply.us/prog/2026-2027_wharton_global_high_school_investment_competition_/

## Conditional funding arithmetic

The saved case copy specifies $300,000 at the beginning of 2027, $150,000 at the beginning of 2028, and ten $50,000 beginning-year operating payments from 2033 through 2042. Case amendments remain unverified. WInS gains are not included in the long-term cash-flow projection.

At the beginning of 2033, before the first payment, wealth is 300000(1+g)^6 + 150000(1+g)^5. Reserve value is sum from k=0 to 9 of 50000/(1+d)^k. The k=0 term is immediate: this is an annuity due. The assumed rates below are sensitivities, not current yields, expected returns or a guarantee that reinvestment will be available.

| Assumed growth | Assumed reserve discount | 2033 wealth | Reserve | Surplus before flexibility buffer |
|---|---|---|---|---|
'''
for r in rows:t+=f"| {r['assumed_growth']:.0%} | {r['assumed_reserve_discount']:.0%} | ${r['wealth_before_2033_payment']:,.0f} | ${r['reserve_before_2033_payment']:,.0f} | ${r['net_surplus_before_buffer']:,.0f} |\n"
t+='''
At zero growth and zero discount, the operating obligations alone exceed contributions by $50,000; there is no facility surplus. A positive arithmetic surplus in another row is not a safe facility-spending limit. Volatile returns, sequence risk, transaction costs and the chosen funding-confidence/buffer policy need separate treatment.

## Investment direction supported by the case

Start by pricing a USD liability-matching reserve for the ten payments, then assess diversified growth exposure only against the residual funding capacity and reviewed risk limits. Treasury principal/interest STRIPS are a candidate to investigate for dated nominal liabilities, subject to access, exact maturity dates, executable prices and competition eligibility. TreasuryDirect describes these as separately tradable zero-coupon components; that product fact does not establish WInS eligibility. A generic bond fund is not an exact dated cash-flow match. No ticker weights or price targets are justified yet.

Source checked 12 October 2026: https://www.treasurydirect.gov/research-center/history-of-marketable-securities/strips/

## Exact outstanding inputs

1. Current official case amendments, trading requirements and approved instrument universe from the team application.
2. WInS current NAV, holdings and decision time; separately identify the long-run client contribution plan.
3. Team concentration, liquidity, turnover, cash and FX limits; numerical funding-confidence target and flexibility buffer.
4. Current admissible forecast/risk data, security terms/prices/costs, verified common execution horizon and dependence/scenario assumptions.
5. A justified forecast, or explicitly declared risk-only analysis with valid covariance and all other required inputs. Missing forecasts cannot be replaced by the old watchlist.
'''
(out/'LAURA_FUNDING_AND_BLOCKERS.md').write_text(t)
print('12 conditional funding scenarios computed; zero-rate boundary verified')
