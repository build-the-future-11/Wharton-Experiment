# Sixty hard judge questions and defensible answers

**Internal practice, not official questions or a claim of student experience.** Answers reflect the actual workspace on 2026-09-27. Students should replace conditional answers only with verified facts. Evidence IDs refer to EVIDENCE_APPENDIX.md.

## Client and suitability

### 01. Who exactly is the client?
The recovered Trading Notes instructions name Laura, but the full current case and identity are missing. We cannot substitute a prior-season client. This blocks a final recommendation. [S04, S06]

### 02. What is the client's investment objective?
The recovered instructions establish an operating commitment, a possible facility contribution and financial flexibility. Their amounts, dates and constraints are missing. The normalized stress inputs remain assumptions. [S05-S06]

### 03. How much wealth is available?
No sourced client balance is available. A starting value of one in the calculator is a unit convention. It must not be quoted as actual money or converted into an invented account balance.

### 04. How did you choose the horizon?
The calculator uses 120 months to illustrate cash-flow effects. It is a scenario assumption, not a sourced client horizon. The final IPS must use the actual horizon and payment dates.

### 05. How do you distinguish risk tolerance from risk capacity?
Tolerance concerns willingness to endure losses; capacity concerns whether losses jeopardize obligations. Neither can be filled from model output. Both require the case, and conflicts must be explained before allocating risk.

### 06. Why not put everything in cash?
Cash reduces exposure to asset losses but can lose purchasing power and forgo growth. Our zero-yield cash assumption makes that tradeoff explicit. Whether it is suitable depends on the client's constraints and available instruments.

### 07. What is the minimum reserve?
No client reserve has been approved. The calculator's candidate rule covers up to twelve modelled payments using a fixed planning inflation assumption. A real rule needs the actual schedule and liquidity constraints.

### 08. What happens if obligations exceed assets?
The calculator records unpaid obligations and the first shortfall instead of borrowing or adding hidden capital. A real plan would need an explicit reconsideration of spending, contributions, horizon or risk; returns cannot be guaranteed. [C06]

### 09. Which securities are eligible?
The current team-gated trading rules are missing. Hypothetical asset sleeves in the calculator do not establish any ticker's eligibility. No security has been recommended on that basis.

### 10. Did you interview the client?
No. No interview is claimed, and the public rules prohibit contacting the competition client. The official case is the proper source. [S02]

## Strategy and alternatives

### 11. What is actually distinctive about this approach?
Its intended strength is traceability from client fact to assumption to decision. We have not established commercial novelty or superiority over other teams. The preserved correction of an invalid result is evidence of transparency, not a competitive ranking.

### 12. Why use machine learning at all?
It is a research candidate, not a requirement. Tested baseline forecasts do not establish a reliable edge, so the current decision process does not depend on them. Any added model must justify its complexity with valid evidence. [C03]

### 13. Why not use a static diversified reference?
That is an important comparator. It may be adequate, but must still match the client's obligations, loss capacity and eligible universe. The calculator includes a simple fully invested reference without declaring it suitable.

### 14. Is a fixed cash buffer enough?
Possibly, but a fixed fraction can be too large or too small relative to dated payments. The sensitivity tool compares it with a payment-linked reserve. This is an illustrative comparison, not a proven optimal reserve policy.

### 15. Why twelve models if the proposal uses none for forecasts?
The models are research work. Their existence does not create evidence of investable skill. The final strategy should use only components the team can support and explain, not every available implementation. [C01-C03]

### 16. Has EWMA risk sizing been validated?
No allocation benefit is demonstrated here. EWMA is a transparent risk-monitoring candidate. Finite covariance output and code tests are insufficient to show improved funding or investment outcomes.

### 17. Why not use M11 for multi-period planning?
The frozen controller fails a basic accounting diagnostic and its legacy evaluation is invalid. We excluded it from the new calculator and retained the defect for a separately amended repair. [C05]

### 18. Why not use M12 as the integrated solution?
Its legacy runner evaluates part of the market fit in sample and integrates an unsupported controller. No defensible end-to-end benefit is established. Integration alone is not evidence of suitability.

### 19. What would make you change the strategy?
A verified change in client constraints, security eligibility, expected payment timing or a documented risk breach would trigger review. Numerical triggers cannot be finalized without the case. A short run of favourable returns is not enough.

### 20. What is the revenue model?
This is an investment competition package, not a venture. No fees, customers, revenue or market size are claimed. The relevant economics are costs, risk, inflation and the client's funding needs.

## Research integrity

### 21. What exactly invalidated the apparent forecasting win?
The synthetic features contained the contemporaneous driver of the label, so the model used information unavailable at prediction time. The resulting score is retained as a leakage demonstration, not forecasting evidence. [C02]

### 22. Why should we trust anything after that mistake?
Trust should depend on inspectable data paths, regression tests, retained adverse findings and reproducible artifacts. Same-machine replay increases confidence in repeatability, but does not replace an independent audit. [C04, C08]

### 23. Was the lockbox reopened to get a better result?
No lockbox run was performed in this pass. Frozen artifacts and model files were hash-checked against the starting snapshot. The existing repaired study is explicitly post-lockbox exploratory. [C07]

### 24. Is the ETF target definitely SPY?
No. The cached asset-0 target is inferred to be AGG from the historical download ordering and signatures. The manifest labels the mapping INFERRED. The filename alone is not proof of column identity.

### 25. Does no significant result prove no predictability exists?
No. It means the tested study did not establish the claimed advantage under its protocol and sample. Other methods, horizons or assets could differ; they require their own valid evidence. [C03]

### 26. Why is beating persistence insufficient?
For approximately unpredictable returns, yesterday's return can be a weak forecast. Shrinking toward zero may beat it without adding useful information. The historical mean and zero forecasts are stronger sanity checks in this setting.

### 27. What was the primary comparator?
The repaired study declares standardized ridge versus the historical mean as its primary contrast. Its broader challenger comparisons include zero and persistence alternatives, with multiplicity handling recorded in the result artifact. [C03]

### 28. Are folds and seeds independent evidence?
Not automatically. The legacy folds collapsed to the same split and ETF seeds read the same data. The repaired study uses disjoint test windows, but temporal dependence and shared training still limit independence. [C01, C03]

### 29. Why not report the best model from the old matrix?
Those scores were generated under leakage and degenerate splits, and some models measured different targets. Selecting the smallest pooled score would not establish a valid winner. [C01]

### 30. How many of the old multiplicity tests support a claim?
The legacy correction found zero of ten comparisons testable after duplicate handling. That is a data-design failure, not ten valid null discoveries or evidence that every model is equivalent.

### 31. Did the new replay change the findings?
It matched all 68 saved contrasts within the recorded numerical tolerance. It did not create a new independent sample or convert exploratory results into confirmation. [C04]

### 32. Why are some synthetic results near perfect?
The legacy-feature leakage demonstration is designed to exhibit the invalid information path. It must remain visibly separated from causal forecasts. Its high score is evidence of the defect. [C02]

### 33. Does the synthetic world demonstrate economic realism?
No. The repaired SIGNAL control has an i.i.d. latent, so lagged data are not expected to recover that contemporaneous signal. It is useful for catching spurious predictability, not validating market realism.

### 34. What would support a confirmatory claim?
A new fixed protocol with causal features, correct target identity, disjoint evaluation, strong comparators, costs where relevant, multiplicity control and genuinely untouched data. The spent lockbox cannot be repurposed.

### 35. Can another team reproduce the study from git alone?
Not the exact ETF numbers: the historical cache is ignored by git and has a recorded hash. Downloading again may change the data. Cache availability and redistribution terms remain unresolved limitations.

## Financial stress and accounting

### 36. Where did the return assumptions come from?
They are explicit modelling choices in the JSON config, not estimates fitted to market data. Each scenario is labelled assumption-only. They should be challenged rather than quoted as expected returns. [C06]

### 37. Why include an extreme joint crash?
It tests what simultaneous losses do to funding and wealth. Its chosen size is not a calibrated historical quantile, probability or prediction. We retain it alongside less adverse paths.

### 38. Why compare early and late losses?
Payment timing can make the order of returns matter. The two paths use the same shock and other return assumptions to expose that sensitivity. They do not establish a universal ranking of portfolio rules.

### 39. Does the reserve know future inflation?
It uses the fixed planning inflation assumption and the currently known payment amount. It does not see future realized returns or the realized future inflation path. High-inflation scenarios therefore test a planning mismatch.

### 40. Are obligations deducted more than once?
Not in the new calculator: each current obligation enters the due amount once, with prior unpaid arrears carried forward. Tests check wealth conservation and the single deduction. The frozen M11 defect is documented separately. [C05, C08]

### 41. How are trading costs charged?
Costs apply to absolute risky-asset purchases and sales, including proportional liquidation when cash cannot cover a payment. Rebalancing solves for post-cost wealth, preventing an implicit cash injection.

### 42. Are the costs actual WInS costs?
No. The five cost levels are sensitivity assumptions inherited from the research grid. Actual simulator rules and real-world execution costs must be verified separately.

### 43. Is the calculator self-financing?
Yes within its stated mechanics: it starts with normalized wealth, earns specified returns, pays costs and obligations, and never adds borrowing or contributions. Unit tests verify the conservation identity under zero returns. [C08]

### 44. What happens when assets are exhausted?
Wealth cannot become negative; unpaid obligations remain as arrears. The output retains the first shortfall and ending unpaid amount. Clipping portfolio value to zero does not erase the failure.

### 45. Why report real terminal wealth?
Nominal remaining capital can conceal lost purchasing power. The calculator deflates by the scenario's assumed inflation. That metric is conditional on the assumption, not a measured client outcome.

### 46. Does drawdown count planned withdrawals as losses?
The reported net investment drawdown excludes external payments through a unit-value calculation. Terminal wealth includes their impact. Both are shown because they answer different questions.

### 47. Is the reserve always the best policy?
No superiority claim is made. It changes market exposure, liquidity and cost. A fully funded path can still lose purchasing power, and no reserve rescues every infeasible spending plan.

### 48. Why not report confidence intervals?
The new stress paths are deterministic, deliberately chosen assumptions without estimated probabilities. Confidence intervals would imply a sampling model we have not justified. The forecasting study has separate statistical tests.

### 49. Are 75 combinations 75 experiments on independent data?
No. They are five assumed paths crossed with three policies and five cost settings. Correlated sensitivity cells do not provide independent replication. [C06]

### 50. What important costs or features are omitted?
Taxes, asset-specific expenses, cash yield, settlement delay, liquidity impact, external contributions and security eligibility are not modelled. The simplified calculator cannot substitute for a client-specific portfolio simulation.

## Competition and execution

### 51. Is there any real WInS performance in the package?
No verified WInS record is present. Hypothetical calculations and research backtests must never be described as executed competition performance.

### 52. Where are the original trade notes?
They must come from the actual WInS records, with the original text and execution evidence. No notes are backfilled or fabricated. The recovered instructions confirm three executed notes and a reflection of at most 100 words each. [S06]

### 53. Have you verified the exact final-report rubric?
Public criteria and recovered IPS/Trading Notes requirements are verified. The final-report instructions and numerical judging weights remain missing. [S01-S06]

### 54. Is the pitch five minutes because the rules require it?
No. Five minutes is an internal rehearsal plan. Official semifinal or final timing is unverified and must be taken from the team instructions.

### 55. Why is the package not submission-ready?
Complete client facts, final-report instructions, eligibility, executed trade evidence and student review are missing. Explicit submission authority is also absent. Completed local tools cannot satisfy those external requirements.

### 56. Does portfolio return determine the winner?
The public competition description emphasizes strategy quality, client fit, research and communication. We cannot infer a numerical judging formula from that description. [S01]

### 57. Was AI used?
Yes. Codex assisted with code review, calculation tools, source verification and these drafts. Students must verify and attribute retained material and cannot present it as their own independent work. [S02]

### 58. What did the students personally decide or learn?
No student decision or experience has been documented in this workspace. The final account must come from the students' actual process. This draft does not invent first-person teamwork stories.

### 59. What is the immediate next action?
Obtain the current registered-team Apply materials and extract source-backed facts. That enables a real client-specific policy and validation against the actual deliverable instructions.

### 60. What is the strongest honest claim today?
The repository has a reproducible exploratory baseline result, preserved negative findings, a tested assumption-only cash-flow calculator and a structured preparation package. It has not established an investment edge, client suitability, independent scientific validation or competition submission readiness.
