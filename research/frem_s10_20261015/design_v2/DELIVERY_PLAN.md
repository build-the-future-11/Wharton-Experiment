# S10 implementation backlog and definition of done

15 October 2026. This is an ordered future work contract. None of the implementation packages below is claimed complete. The design session produces specifications, review findings and static consistency evidence only.

## 1 Deliverable sequence and dependencies

Each work package has a named responsibility, prerequisite and reviewable exit artifact. The roles are engineering responsibilities, not claims that external people have been assigned. Implementers may combine small packages in one change if all criteria remain independently reviewable. No change may mix a scientific redesign with an unlabeled numerical optimization.

| ID | Deliverable | Depends on | Exit evidence |
|---|---|---|---|
| WP01 | Approve revision-2 equations, timestamps, inference and bounds; compile exact requirement matrix | Design package | Signed-off internal review record with each objection resolved or explicitly blocking |
| WP02 | Isolated package skeleton, strict config, exception catalog, metadata identities and offline CLI plan | WP01 | Import-side-effect test, no-root-dependency check, config rejection tests, complete non-executing plan |
| WP03 | Source/calendar/availability adapter, exact fold indices, action/dividend schema and burn-in | WP02 | S10-specific admission receipt or explicit rejection; no evaluation outcomes inspected for selection |
| WP04 | Numerical primitives: distances, proper scores, probability checks, fee root and utility | WP02 | Independent hand/high-precision references; tolerances and invalid-input behavior |
| WP05 | Experts, mixture solve, feedback queue and issued-forecast storage | WP03, WP04 | Pure-state chronology and metamorphic tests, KKT certificates, no-current-target leakage |
| WP06 | Precision preparation using the verified candidate and tested production bootstrap/artifact/consumption/resource subset | WP02-WP05; relevant constructed kernel and guard checks | One scoped pre-2010 receipt with all four planning comparisons, cold-start counts and durable one-use lifecycle |
| WP07 | Policy action registry, hold/cash, inverse-vol and minimum-variance controls | WP04 | 609/45 grid checks, projection certificate, exact tie behavior and correct support minimization |
| WP08 | Actual holdings/quote/event ledger, daily and liquidation marks | WP03, WP07 | Per-event accounting, ex-date ownership, no double exit fees, cash/negative-path examples |
| WP09 | Complete matrix and eight named ablations | WP05, WP08 | Generated end-to-end 192-cell inventory, comparison identities and matched inputs |
| WP10 | Bootstrap, H1/H2/H3, calibration and terminal scientific labels | WP09 | Independent index/statistic calculations, all controls and both blocks, no favorable-arm selection |
| WP11 | Immutable artifact store, checkpoint schema, hash journal and audit | WP02, WP05, WP08 | Corruption/fault-injection receipts and generated restart equivalence |
| WP12 | One-use launcher, owned-process supervisor, budget/cancellation behavior | WP09, WP11 | Preflight/consumption boundary tests, failure receipts, no market-resume path |
| WP13 | Scoped lint/type/coverage/reference CI and clean-wheel reproduction | WP02-WP12 | All mandatory engineering jobs with exact environment receipts |
| WP14 | Bounded profiling and equivalence-preserving performance improvements | WP09-WP13 | Time/RSS/disk forecast; scalar/vectorized/reference agreement; unchanged behavior identities |
| WP15 | Independent design/code review and run-ready audit | WP06 pass, WP10-WP14 | Required checks complete; exact data/source/environment/protocol freeze |
| WP16 | One authorized development campaign | WP15 and fresh resources | Full retained outcome or terminal failure; no unplanned retry |
| WP17 | Saved-output audit, paper tables, limitations and conclusion | WP16 | Claim -> raw artifact -> analysis -> table/figure mapping |
| WP18 | Separate independent reproduction and confirmation feasibility | WP17 | External/reproduction evidence where available; honest pending status otherwise |
| WP19 | Local research release and archival package | WP17-WP18 as claim scope requires | Allowlisted inspected artifact, environment, rights/provenance and terminal conclusion |

WP04 and static package/data work can proceed independently where dependencies allow. WP07/WP08 engineering work may run while precision is being prepared, but no full market campaign is admitted until WP06 passes. No worker is assigned overlapping file ownership without coordination. A second agent's source review is useful internal review, not external empirical replication.

WP06 includes the precision-specific subset of WP10 bootstrap/statistics, WP11 serialization/atomic storage and WP12 consumption/supervision/cancellation. These are the same production modules later extended for the full matrix, not provisional duplicate implementations. Before precision reads real values, run the applicable kernel tests plus constructed bootstrap/precision TEST-062–066/070, artifact TEST-072–076/079, interruption/resource TEST-077–080/082, and package/scope/evidence TEST-084–085/087–088/092–095. Cases referring to full campaigns are first exercised on the precision operation and later on the complete campaign; both evidence scopes are retained. No failed or missing required precision guard is waived because WP11/WP12 have later completion dates. Precision preflight requires its own reviewed declaration and test receipts; only the later full campaign requires the completed precision result.

## 2 Definition of done for each code change

A change is complete only when:

1. It names the exact requirements and planned tests it implements.
2. Public contracts specify units, shapes, mutability, timestamps, input rejection and outputs.
3. All new required paths contain real implementation; no placeholder or silent fallback stands in for behavior.
4. Relevant numerical/accounting/chronology tests pass under recorded environment and declared resource bounds.
5. Scope-limited lint, format and type checks pass; any narrowly scoped suppression has a rationale and test.
6. The diff preserves predecessor sources, receipts, locks, thresholds and protected-data boundaries.
7. Diagnostics, failure codes and artifact schemas are updated with any interface change.
8. Computational cost and storage complexity are stated; any optimization has equivalence evidence.
9. A reviewer can regenerate the evidence using complete commands and immutable inputs.
10. Current state records distinguish implemented, tested, scientifically run and independently verified.

Documentation checks alone do not satisfy items requiring runtime evidence. A required check that is blocked is reported as pending, never changed to optional to make the change appear complete.

## 3 Change classification and versioning

| Change kind | Examples | Required treatment |
|---|---|---|
| Documentation clarification with no behavioral difference | Explain existing units or correct a link | Review, scoped consistency check, retain older authoritative snapshots |
| Engineering-equivalent implementation | Vectorize a sum with declared numerical agreement | Reference/equivalence tests, resource evidence and versioned implementation identity |
| Numerical semantics | Change tolerance, reduction order affecting ties, RNG, solver policy | Numerical review; determine scientific impact before accepting |
| Scientific design change | Feature, horizon, expert, threshold, date, primary, score, update schedule | New design revision and record-decision; freeze anew before outcomes |
| Genuine validity repair after a run | Demonstrated leakage, incorrect accounting, broken declared baseline | Preserve failed/invalid run, record defect, determine affected scope and corrected freeze |
| Valid negative/inconclusive outcome | No useful memory or economic effect | Retain terminal result; no outcome-driven repair or repeat |

No public API or scientific schema is versioned by filename alone. A science identity includes the resolved configuration, model/inference semantics, data/folds and environment identities. An engineering release may contain a scientifically negative model.

## 4 Review checklists by role

The numerical reviewer checks units, score coefficients, simplex tangent geometry, PSD/risk identities, finite support, objective ties and residual tolerances. The chronology reviewer checks every feature/label availability, quote/fill ordering, dividends, delayed feedback and fold index. The software reviewer checks dependency direction, typing, pure functions, state ownership, exceptions, resources, artifacts and failure recovery. The empirical reviewer checks controls, precision, multiplicity, exposure, negative evidence and claim boundaries.

Each review yields actionable file/requirement references and closure evidence. Reviewer agreement does not replace an executable check or a market experiment. The independent empirical verifier must be distinguished from an internal collaborating agent.

## 5 Performance work without scientific drift

Measure separate stages: source decoding, target construction, pair distances, score coefficients, neighbor search, mixture solve, candidate evaluation, ledger IO, bootstrap and report generation. Include cold caches, peak live-array allocations and interpreter/library overhead.

Allowed optimizations preserve the exact declared calculation within its frozen numerical acceptance: share immutable forecasts across policies/costs, reuse verified pair distances, compress immutable numeric artifacts losslessly, stream output, preallocate bounded arrays, use exact active-face solve rather than repeated general optimization, and batch grid actions within measured memory.

Do not silently introduce float32, approximate neighbors, truncated atom support, sampled pair terms, altered seeds, changed bootstrap draws, parallel nondeterministic reductions or altered stopping. Each changes either the scientific calculation or its numerical behavior and needs review.

No speedup or memory-saving claim is made until measured against the same algorithm, inputs, environment and result tolerances. Passing a small constructed case does not prove full-study runtime feasibility.

## 6 Operational and research end states

Ready for engineering means the bounded question and implementation specification are coherent. Ready for precision means the minimal kernel and permitted source preparation are verified. Run-ready means all complete pipeline, precision, resource and freeze gates pass. Research supported/negative/inconclusive describes actual evidence. Independently verified and released require their own receipts.

S10 may finish as a high-quality implementation of an unsuccessful hypothesis. It may instead be stopped before a full run if data, power or resources cannot answer the question. Neither outcome permits relabeling a test fixture as a market result.

## 7 Explicitly deferred capabilities

Future extensions require a separate contract: broader equities/defaults/FX, point-in-time constituent data, vendor-verified availability, payment-date settlement, real order sizing/fills, taxes, sourced cash yield, independent confirmation, neural experts, a continuous density satisfying the old NLL criterion, Laura-specific calibrated funding and any live deployment.

These are not placeholder implementations hidden inside revision 2. Their boundaries are explicit so the bounded three-ETF research package can be reviewed without claiming a full trading platform.
