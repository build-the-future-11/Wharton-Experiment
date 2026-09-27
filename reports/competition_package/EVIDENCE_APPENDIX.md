# Evidence appendix and source register

**AI-assisted internal reference pack; reviewed 2026-09-27.** File references are repository-relative. Historical evidence is preserved; new calculator evidence is a separate generation.

## Public official sources

| ID | Source | Verified scope |
|---|---|---|
| S01 | [Wharton competition](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/) | 2026-27 dates and deliverable names; strategy, client alignment, research and communication guide evaluation |
| S02 | [Rules and roles](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/) | Team and advisor conditions; no client contact; student ownership and AI attribution; detailed instructions are team-gated |
| S03 | [Official FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/) | Competition runs September 28 through December 4, 2026; current materials are distributed through Apply |
| S04 | [Official Apply portal](https://wghsinvcomp.smapply.us/prog/2026-2027_wharton_global_high_school_investment_competition_/) | Location of private team materials, not a source for client facts available in this workspace |

Recovered PDFs now verify IPS and Trading Notes formatting and contents; see S05-S06 below. Numerical rubric weights, the complete client case, security eligibility and minimum overall trading activity remain unverified. Older-season pages and the prior client are not substitutes. Public notes are retrieval summaries, not archived copies of the private rulebook.

## Recovered local sources

S05: `data/source_documents/2026_WGY_Investment Policy-FINAL.pdf`, pages 1-3: objectives, strategy permanence, 50/500 word limits, title plus two strategy pages, Times New Roman 12-point double spacing, one-inch margins, no formal citations or graphics, at most 5 MB.

S06: `data/source_documents/2026_WGY_Trading Notes Analysis-FINAL.pdf`, pages 1-2: Laura, operating/facility context, three exact notes from executed trades and at most 100 words per reflection, submitted via Apply form. Both sources have hashes in `protocol/RECOVERED_DOCUMENTS.json`; they were not reauthenticated with Apply.

## Claim-to-artifact map

| Claim ID | Permitted statement | Evidence | Limit |
|---|---|---|---|
| C01 | Legacy matrix executed 1181 manifest cells | protocol/EXPERIMENT_MANIFEST.csv | Leakage-contaminated; folds/seeds do not establish independent replication |
| C02 | Synthetic lockbox forecasting claim withdrawn | RESEARCH_AUDIT.md; protocol/DECISIONS.md D-050 | Do not reuse it as an alpha claim |
| C03 | Repaired tested baselines show no statistically supported mean-benchmark edge | runs/repaired/REPAIRED_RESULTS.json | Exploratory; one-step asset-0 evaluation, not all models or assets |
| C04 | Same-machine replay matched 68 saved contrasts | audit/astra_2026-09-27/VERIFICATION.json | Repeatability, not independent validation |
| C05 | Frozen M11 fails a simple cash-path diagnostic | same verification receipt; D-059 | Not a repaired controller or validated allocation |
| C06 | 75 policy/path/cost combinations calculated | runs/competition_stress/RESULTS.json and SUMMARY.csv | Deterministic assumptions; not 75 independent observations |
| C07 | All historical protected files remained byte-identical at verification | audit/astra_2026-09-27/START_STATE.json and VERIFICATION.json | Source snapshot check only |
| C08 | Local tests pass | audit/astra_2026-09-27/TESTS.txt | Engineering check, not proof of forecasting or funding performance |
| C09 | Working package contains 60 judge questions | JUDGE_QA.md | Preparation content, not official judge questions |

Every calculator number is generated from the config and source hashes recorded inside RESULTS.json. Monthly ledger: wealth_start, obligation, due_with_arrears, paid, arrears, cash, wealth_end, fees, turnover, net_return and unit_value. Units are fractions of initial wealth. Monthly net drawdown excludes withdrawals; terminal wealth includes them. An overdue payment is not added twice to total contractual obligations.

## Assumption register

| Input | Value / location | Classification |
|---|---|---|
| Initial wealth | 1.0 | Normalization, not dollars or actual client wealth |
| Horizon | 120 months | Modelling choice, not client horizon |
| Base monthly payment | 0.002 of initial wealth | Modelling choice; inflation-linked within each assumed path |
| Steady annual sleeve returns | 4% / 2% | Assumptions; not estimated expected returns |
| Joint one-month shock | -40% / -15% | Stress assumption, applied early or late |
| Reserve horizon | 12 months, capped at remaining horizon | Policy choice, not optimized |
| Planning inflation | 2.5% | Fixed assumption available to the policy |
| Realized inflation | 2.5% or 8% | Scenario assumptions; not inflation forecasts |
| Trading cost | 0/5/10/25/50 bps per risky dollar traded | Sensitivity grid; no claim about WInS fees |
| Cash return, contributions, tax | Zero / none | Simplifications |
| Scenario probabilities | None assigned | No estimated failure probability or confidence interval |

## Evidence still required

The full official client case, final-report rubric/instructions, registration and eligibility, numerical client facts, current allowed-security list, verified executions and original trading notes are missing. IPS and Trading Notes instructions are recovered. No customer interviews, commercial revenue or partnerships are claimed; these are not applicable to the investment competition brief. The cached ETF's column mapping remains inferred, and redistribution rights are unresolved. A genuinely confirmatory forecasting result would need a new protocol and untouched data. No paper-ready or investment-ready status is asserted.

## AI assistance record

On 2026-09-27, Codex assisted with public-rule verification, code inspection, calculator development, local testing, and preparation of these working documents. Students must verify sources, make their own investment decisions and describe only their real work and experience. Material retained from this package must be acknowledged under the official AI policy. No student review or advisor approval has been recorded.
