# FREM S10 implementation and acceptance plan

15 October 2026. The deliverable at this stage is the complete reviewed design. The following source tree is planned, not implemented. No empty modules or passing-test claims stand in for the future work.

## 1 Architecture boundaries

All new implementation belongs in successors/frem_s10 and tests/s10. Preserve S6-S9 and the original protocol. Reuse audited mathematics only through an explicit version/hash dependency or a separately attributed copy; do not edit an earlier frozen module to accommodate S10.

| Planned module | Responsibility | Required output |
|---|---|---|
| contracts.py | Typed units, records, arrays, timestamps and status enums | Strict schema validation with reject reasons |
| data.py | Source admission, calendars, price/dividend targets and burn-in | Data/split/access manifest, exclusions, eligible counts |
| scenarios.py | Bounded shared scenario library and three experts | Joint atoms and normalized probabilities |
| scoring.py | ES, CRPS, variogram, quantiles, A/B coefficients | Reconstructible score values and coefficient checks |
| stacking.py | Global/local simplex optimization and shrinkage | Weights, KKT/active-face certificate, fallback reasons |
| stream.py | Issuance, maturity queue, deterministic online state | Append-only issued/matured forecast records |
| policy.py | Exact fees, finite grid, hold, nominal/robust utility | Decision and feasible-set/objective certificate |
| ledger.py | Shares, cash, dividends, marks and actual fees | Event and daily NAV ledger with attribution |
| benchmarks.py | Six fixed portfolio controls | Same-schema control ledgers |
| experiment.py | Complete core/ablation orchestration and counts | 192-ledger manifest or explicit incomplete failure |
| analysis.py | Paired-date statistics, uncertainty, diagnoses | Full comparison tables and terminal disposition |
| run.py | Resource/data/source/environment checks and one-use launcher | Nonzero failures or complete outcome receipt |
| report.py | Artifact-bound tables/figures and claim mapping | Methods/results package with actual evidence |

The exact filenames may change for engineering clarity before freeze; equations, experiment identities and failure behavior may not change silently.

## 2 Implementation sequence

1. Review the mathematical and chronology contract and resolve contradictions. Complete the S10 data admission and development-only precision preparation before calling the empirical study answerable.
2. Implement data/units, exact accounting and scenario invariants. Verify against hand calculations and independent references.
3. Implement proper scores, the active-face stacking solver and immutable delayed-feedback state. Check that global/equal/no-context controls are genuinely matched.
4. Implement policy/benchmarks and daily ledger. Verify actual charges versus hypothetical liquidation reserve, cash dividends and carried shares.
5. Implement the complete matrix, receipts, statistics and generated report. Require every planned identity, not just a successful primary.
6. Run constructed engineering integration, checkpoint/restart, corruption rejection and bounded profiling. Freeze only after all mandatory checks pass.
7. Execute a separately authorized, admitted, one-use development campaign. Audit every result and preserve the honest terminal disposition.
8. Only then decide whether a narrower supported candidate merits independent reproduction and a separately admitted confirmation study.

## 3 Acceptance evidence

| Property | Required check | Failure implication |
|---|---|---|
| No future information | Change future quotes/labels; earlier states, predictions and decisions must remain unchanged | Invalid chronology |
| Correct maturity | Target ending on decision day cannot enter before its availability; overlapping H4 labels purged | Invalid research timing |
| Shared information | All expert/mixture controls use identical atoms and cutoff identities | Invalid comparison |
| Correct distributions | Nonnegative normalized probabilities; known mixture scores including cross terms | Invalid estimator/metric |
| Proper optimization | Active-face solution matches dense reference and satisfies KKT; degenerate cases rejected | No certified mixture |
| Shrink/fallback | Sparse/OOD context yields the exact specified global fallback; no label-based gate | Invalid mechanism |
| Exact costs | Buy, sell, rebalance and hold examples reconcile; fees applied only to traded risky dollars | Invalid ledger |
| Dividends | Event ownership, cash credit, no reinvestment, no dividend liquidation fee | Invalid return accounting |
| Carried holdings | Two-block path differs correctly from forced liquidation; no fabricated intermediate exit fee | Invalid policy comparison |
| Risk and utility | Correlated/offsetting examples; zero-return/cash; within-support robust minimum identity | Invalid optimizer or misleading guarantee |
| Finite action optimum | Exhaustive independent tiny grid; tie handling and constraints | Invalid decision certificate |
| Missing/zero prices | Unsupported sources/events fail loudly rather than silently drop losses | Invalid admission |
| Complete outputs | Every core and ablation cell including zero/cash controls present | Incomplete scientific result |
| Bootstrap grouping | All assets/methods move together; fold weighting and both block families match reference | Invalid uncertainty |
| Checkpointing | Full restart equals uninterrupted execution; maturity/dividend events exactly once | Invalid reproducibility |
| Consumption | Blocked admission leaves freeze unconsumed; first scientific start consumes before fitting | Invalid one-use protection |
| Resource bounds | Measured profiling fits actual limits, with nonzero retained failures | Not run-ready |
| Report provenance | Every number regenerates from source-bound saved artifacts | Unsupported report |

All constructed fixtures are labeled engineering. No test count is preclaimed. No goal of making the tests pass may justify weakening an economic or scientific gate.

## 4 Review gates

| Gate | Current status | Evidence needed |
|---|---|---|
| S10 question and selected mechanism | SPECIFIED | Adversarial review and feasibility before implementation |
| Novel algorithm contribution | NOT_ESTABLISHED | Close-prior-art differentiation; otherwise retain empirical scope |
| Mathematical/model specification | REVIEW_DRAFT_COMPLETE | Independent derivation/numerical implementation review |
| Data admission | PENDING | Source/calendar/access/warm-up receipt |
| Precision feasibility | PENDING | Pre-2010-only preparation within a separate bounded allowance |
| Implementation and actual tests | NOT_STARTED | Complete source and meaningful runtime evidence |
| Run readiness and source freeze | NOT_READY | Admission, complete matrix, measured resources, immutable freeze |
| Untouched confirmation | NOT_READY | Admitted unused/prospective evidence and actual freeze date |
| Independent reproduction | NOT_STARTED | Clean environment and independent reviewer/implementation |
| Paper and research release | NOT_COMPLETE | Terminal conclusion and traceable evidence package |

S10's empirical design does not override the original user's IDEA READY standard. The narrow empirical reason to exist is explicit, but novelty and an answerable admitted experiment are not certified. No implementation starts as part of this design session. This is the user's requested mapping stage, not an added permission requirement.

## 5 Scientific artifacts and release

Every future output directory must be new and contain source/data/config/environment identities, all issue/update timestamps, score coefficients, weights, checkpoints, decisions, daily ledgers, control completeness, resource receipts and failures. Reports must distinguish implementation verification, exploratory outcome, independent reproduction and confirmation.

The manuscript should follow: problem and predecessor evidence; related work; full chronological model; accounting/proofs; frozen methodology; actual results; controls/ablations; calibration and risk; negative cases; data/fill/support limitations; reproducibility and terminal conclusion. Tables and figures must be generated from retained raw results.

A separate Laura adapter may later consume validated forecasts and dated mandate inputs. Its reserve/funding arithmetic must remain separate from the three-ETF research objective. No 99% funding probability, eligible current allocation or public submission is created by the S10 design.
