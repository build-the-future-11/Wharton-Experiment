# S10 revision 2: verification catalog

15 October 2026. **Specified tests, not authored or executed tests.** `TEST-001` through `TEST-096` are stable verification requirement IDs, not a promised pytest function count or a passing result. Each may need several hand cases/property examples. All current statuses are `PLANNED_NOT_IMPLEMENTED`.

The catalog applies the preserved [model specification](../MODEL_SPECIFICATION.md) and [experiment protocol](../EXPERIMENT_PROTOCOL.md), with the explicit overrides in [MODEL_CLARIFICATIONS.md](MODEL_CLARIFICATIONS.md). Package boundaries are in [ARCHITECTURE_AND_INTERFACES.md](ARCHITECTURE_AND_INTERFACES.md); execution/persistence rules are in [OPERATIONS_AND_ARTIFACTS.md](OPERATIONS_AND_ARTIFACTS.md); quality tiers are in [CODE_QUALITY_AND_CI.md](CODE_QUALITY_AND_CI.md). Where a normative detail remains unresolved, the associated test is blocked by that detail; an implementation cannot pick a favorable interpretation and declare a pass.

## 1. What counts as verification

Every test records the invariant it could falsify, its independent expected result, its rejection case and the applicable tolerance. Fixture generators use constructed values and clearly label `ENGINEERING_ONLY`; none is calibrated from a protected source, a passing market arm or a fabricated empirical result. Fixtures test timing and accounting mechanics, not the research hypothesis. Small known-signal examples only establish that the implementation can respond to an injected mechanism.

Tests must not derive their expected value by calling the implementation under test. For numerical kernels use hand calculations, a separately written scalar reference, or a high-precision reference. For orchestration use an independently enumerated identity set and event trace. For proof obligations retain both the derivation and its numerical counterexample tests. Mocks replace clocks, I/O faults and observers; they do not mock away the numerical algorithm that the test claims to verify.

The evidence record will map `requirement_id -> TEST-ID -> nodeid -> source/config/fixture hashes -> command/environment -> observed result/residual`. Retain failing inputs before a repair. When a new defect appears, add its smallest reproducer to the appropriate existing ID or allocate a new ID; do not relabel a failed test as a different requirement or overwrite its receipt.

## 2. Contracts and source admission

All cases in this section use generated metadata/rows, never actual retained market values. They prove adapter behavior only; genuine S10 data admission still requires a separately recorded source review.

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-001 / Q1 | Price gross, dividend-per-dollar, simple return, USD, shares and weights deliberately swapped at boundaries | Wrong units rejected; valid typed record preserved without an accidental conversion |
| TEST-002 / Q1 | Wrong asset order, feature order, rank, length, dtype, empty vector, bool-as-count, object/complex arrays | Named field error before numerical work; no positional broadcasting accepted |
| TEST-003 / Q1 | NaN, infinity, negative dividend, nonpositive price, negative shares/cash, invalid probability/weight | Explicit rejection. No clipping, row deletion or cash fallback |
| TEST-004 / Q1 | JSON duplicate key, unknown field, schema version, timestamp string without timezone, malformed manifest | Strict parser rejection with field/schema reason; unknown fields not silently ignored |
| TEST-005 / Q1 | Change an input array after record construction, then attempt mutation through a returned view | Immutable owned state/hash unchanged or write rejected; frozen dataclass alone is insufficient |
| TEST-006 / Q1 | Generated source sidecar with wrong hash, units, action convention, split claim or calendar identity | Admission fails; no consumed freeze or created result claim |
| TEST-007 / Q1 | Missing required daily mark, absent common-session anchor, duplicate quote and unsupported split/terminal event | Continuous ledger rejected, never silently shortened or forward filled |
| TEST-008 / Q1 | Source path traverses outside admitted roots, a symlink resolves to disallowed data, or an undeclared file is requested | Reader refuses access before content is opened; record only allowed identities |
| TEST-009 / Q1 | Generate 63/64 burn-in records; one becomes mature exactly at the cutoff; one feature has near-zero scale | Count and strict cutoff follow specification; exact declared scale floor and no later refit |
| TEST-010 / Q1 | Inject an extreme later observation into burn-in-independent fields | Earlier transform and feature hashes invariant; later data cannot change a saved transform |

## 3. Information clock, target construction and delayed state

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-011 / Q1 | Winter/summer and DST-boundary generated US sessions: quote at 09:30 `America/New_York`, decision one microsecond later, ideal same-price fill two microseconds later | Correct UTC conversion and strict order under revision 2; no fixed UTC open assumption; fill convention remains labeled idealized |
| TEST-012 / Q1 | Origin features supplied with availability just before, equal to and just after decision | Only strictly earlier information allowed. Current execution quote may mark existing holdings under its explicit contract but cannot enter forecast features |
| TEST-013 / Q1 | Thirteen irregular holiday-aware common-session weeks with a missing required interval | Features match an independent hand sequence for valid origins; missing history invalidates origin without inventing a holiday or filling a return |
| TEST-014 / Q1 | Four weekly intervals with two dividends and nonconstant prices | Direct endpoint `g`, cash-summed `d` and total return match hand calculations; no compounded dividend reinvestment |
| TEST-015 / Q1 | Dividend at start date, inside horizon and at endpoint; an endpoint trade changes ownership | Exactly `origin < ex_date <= endpoint` target convention; endpoint entitlement uses pre-trade shares; new endpoint buyers receive no earlier entitlement |
| TEST-016 / Q1 | Four-week target whose endpoint quote is observed but target availability is equal to/after the new decision | Maturity waits for the declared availability rule; no same-decision feedback leak |
| TEST-017 / Q1 | Future perturbation changes all later prices/dividends/labels while keeping earlier available prefix fixed | All earlier features, libraries, weights, forecasts, actions and canonical state hashes unchanged |
| TEST-018 / Q1 | Several forecasts mature together, byte-identical repeated event ingestion and cursor regression | Each issued forecast scored once from original atoms; deterministic order; every already-applied event ID and cursor regression rejected under CL-14; rereading an immutable artifact does not advance live state |
| TEST-019 / Q1 | Modify today's library before a past forecast matures | Its A/B score coefficients use the saved issued forecast, not rebuilt atoms; source forecast hash matches |
| TEST-020 / Q1 | Feed beyond library/feedback capacity with staggered maturities | Window eviction removes the declared oldest eligible identities only; 520/104 bounds and overlap metadata retained |
| TEST-021 / Q2 | Batch reference versus incremental weekly stream across warm-up, fold A end and fold B start | Canonical forecasts equal; fold B may use prior matured labels, while both wealth ledgers reset to their specified initial cash |
| TEST-022 / Q2 | Engineering checkpoint/restore at each transition boundary | Replay continues strictly after the saved cursor and equals uninterrupted canonical state/events; already-applied IDs remain rejected, with no duplicated maturity/dividend/trade. This does not authorize market resume |

## 4. Empirical scenarios and proper scores

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-023 / Q1 | U/R/C distributions on a tiny labeled library with exact age and distance ties | Shared atom identities; exact uniform/recent/neighbor weights; tie order by timestamp then ID; normalization tolerance honored |
| TEST-024 / Q1 | Neighbor boundary cases at 31/32/33 eligible records, threshold equality, and query beyond leave-one-out threshold | Exact eligibility/fallback branch; fixed linear percentile interpolation; labels never participate in distance |
| TEST-025 / Q1 | Reorder atoms then apply the same stable-ID canonicalization; duplicate an atom while dividing its mass | Same predictive law, ES, moments and policy objective within declared tolerance; duplicate identities themselves still rejected |
| TEST-026 / Q1 | Single deterministic atom equal to target, then shifted away | ES zero then Euclidean error; no artificial continuous likelihood or 0.01-nat claim |
| TEST-027 / Q1 | Two one-dimensional atoms at -1/+1 with equal mass and target zero, embedded in three dimensions | ES exactly 0.5 in unit scale, while either point expert scores 1.0; exposes omitted cross-expert terms and incorrect unbiased-pair substitution |
| TEST-028 / Q1 | Direct finite-law ES versus A/B mixture formula for vertices, edges and interior weights | Independent scalar enumeration and coefficient formula agree; B symmetry and zero distance diagonal checked |
| TEST-029 / Q1 | Common translation and orthogonal rotation of standardized score vectors; positive uniform rescaling | ES translation/rotation invariant and scales by the factor. Test occurs after fixed standardization; it does not license refitting asset scales |
| TEST-030 / Q1 | Perfectly aligned, offsetting and repeated joint atoms | Covariance symmetric/PSD to declared tolerance, expected returns in convex hull; dependence remains joint rather than separately resampled by asset |
| TEST-031 / Q1 | Discrete quantile with ties/mass exactly at requested probability, CRPS/MSE with one/two atoms, variogram with identical/opposite assets; loss probability exactly 0/1 and at each bin edge | Hand-derived values in CL-12's named units; smallest cumulative quantile, inclusive interval endpoints, per-asset averaging and separate basket Brier arithmetic correct; sparse bins/counts retained and no raw/standardized mixing |
| TEST-032 / Q1 | Near-identical large finite vectors, tiny distances and extreme but admitted score scales | Direct-distance reference agrees; no spurious negative distance/score hidden by clipping; nonfinite computation fails |

## 5. Convex stacking, contextual shrinkage and controls

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-033 / Q1 | Analytic optima at each vertex/edge/interior, plus large/small coefficient scales | All seven active faces considered; CL-14 scales H/a while retaining unit equality row/column; recover nu_original=s*nu_scaled before original-scale KKT checks; independent scalar/dense reference supplies objective bound |
| TEST-034 / Q1 | Deliberately clipped unconstrained solution versus correct boundary optimum | Implementation selects certified constrained optimum; a clip-and-renormalize substitute fails reference comparison |
| TEST-035 / Q1 | Three identical experts and symmetric objectives | Fixed regularization yields its declared unique reference solution; deterministic tie/certificate behavior |
| TEST-036 / Q1 | Singular solve, excessive negative tangent curvature, asymmetric B, scaled bordered-system condition around 1e12, and valid tangent convexity with indefinite full-space Hessian | Finite cond2 <= 1e12 required for each scaled bordered system and residuals checked at original scale; a bad face fails rather than being skipped; valid constrained problem needs no full-space Cholesky; no projection/jitter/substitute |
| TEST-037 / Q1 | Tiny negative feasible weight on either side of tolerance boundary | Only permitted rounding correction performed; normalization/objective/KKT rechecked; beyond-tolerance value rejected |
| TEST-038 / Q1 | Global records count 25/26; local counts/weight concentration near each eligibility boundary | Exact equal/global/local fallback and beta rule; effective weight count never described as independent sample size |
| TEST-039 / Q1 | Context weights permuted with matching records; labels perturbed without changing contexts | Neighbor choice unchanged by outcomes; only matured scores affect optimized weights; independent recency-weight scalar reference agrees |
| TEST-040 / Q1 | Equal, global, contextual and individual-score controls at the same issue origin | Same information/atom identities and update date. Individual-score softmax is stable and distinct from mixture-level scoring; no extra search or seeds |

## 6. Costs, portfolio decisions and realized accounting

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-041 / Q1 | $100 all cash, target risky weight 0.4, fee 0.01 | Fee root matches independent exact value `100/(1+0.004)`; cash+risky+fee=$100. This is a constructed fee test, not an allowed study cost |
| TEST-042 / Q1 | $40 risky/$60 cash sold completely at fee 0.01; mixed buy/sell targets; zero fee | $99.60 cash on full sale; Decimal/piecewise-linear reference matches other roots; correct turnover dollar basis |
| TEST-043 / Q1 | V/weights/fees spanning declared domain, root near bracket end and forced 100-iteration exhaustion | Monotone root bracket, accounting and bracket tolerances both satisfied; exhaustion fails instead of returning the last iterate as certified |
| TEST-044 / Q1 | Exact feasible hold and per-asset/total-cap drift at, just within and beyond 1e-12 weight tolerance | Feasible hold has zero trade/entry fee; within-tolerance residual recorded; materially infeasible hold excluded without clipping and mandatory rebalance recorded; no continuous-cap claim |
| TEST-045 / Q1 | Exhaustively generate weights as integer ticks, then convert to decimals | Exactly 609 grid points before optional distinct hold, 45 sum-.8 minimum-variance candidates; no floating-loop omissions, duplicates or invalid targets |
| TEST-046 / Q1 | Brute enumeration of reduced/full grids on tiny atoms, reordered actions and nontransitive approximate-tie example | Whole-set objective ties within 1e-10 then hold preference; whole-set minimum turnover ties within 1e-10*max(1,V) then lexicographic choice; result independent of iteration order; cash not penalized |
| TEST-047 / Q1 | Zero-return/zero-dividend atoms with zero and positive fees; all risky atoms strictly adverse | Zero-cost factor/utility identity; cash/hold ties handled correctly; model validly chooses cash when optimal, while a model exception is never cash success |
| TEST-048 / Q1 | CRRA factors near 1, very small positive values and invalid zero/negative values | Stable expression agrees with high-precision reference; invalid domains raise; gamma remains fixed |
| TEST-049 / Q1 | Two/three-atom utility vectors, including an adverse atom with zero nominal mixture probability, and contamination mass concentrated separately on each atom | Robust objective matches explicit minimum over all declared-support simplex vertices, including zero-weight atoms; rho=0 agrees with nominal; no out-of-support protection claim |
| TEST-050 / Q1 | Multi-block path with dividends and an unchanged holding, compared with forced liquidation | Actual ledger charges no intermediate hypothetical reserve; A4 does charge actual sales/re-entry. Final endpoint liquidation occurs exactly once |
| TEST-051 / Q1 | Independent share/cash event ledger with price change, dividend, trade and final liquidation | At each event NAV delta=price PnL+dividend+interest-fee within normalized tolerance; interest exactly zero; dividend is not charged a sale fee |
| TEST-052 / Q1 | Positive dollar rescaling of initial wealth and consistent holdings; fixed target sequence | Dimensionless decisions/objectives unchanged; dollar cash/fees/PnL scale. Counterfactual zero-cost replay uses saved targets and new fee roots without optimizing again |
| TEST-053 / Q1 | Daily-close trough between decisions, entry fees, final liquidation fees and missing mark | Drawdown includes initial capital and actual final-liquidation anchors, distinctly labeled; final fee once; trough caught; missing close rejected; no invented closes or intraday extrema |

## 7. Benchmarks, ablations and complete matrix

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-054 / Q1 | Cash, broad equal rebalance, matched 80% equal and buy-and-hold on a hand path | Exact exposure/dividend/final liquidation differences retained; 100%-risky broad controls not accidentally clipped to strategy caps |
| TEST-055 / Q1 | Inverse-volatility vector with one floored volatility and an over-cap unconstrained asset | Independent active-set projection satisfies sum .8 and .4 caps; causal inputs only; no arbitrary renormalization after clipping |
| TEST-056 / Q1 | Joint covariance and diagonal-shrink covariance where minimum-variance grid rankings differ | Fixed 50% shrinkage, 45-action enumeration and tie ordering correct; no Ledoit-Wolf label |
| TEST-057 / Q2 | Independent Cartesian identity list for eight forecasts, two policies, four costs, two folds, six benchmarks, eight ablations | Exactly 128+48+16=192 required identities, no duplicates. Metadata-only enumeration does not count as an executed ledger |
| TEST-058 / Q2 | Delete a primary, zero control, ablation or cost cell; duplicate a cell; truncate one event table | Matrix completeness fails before summary; no averaging surviving methods or calling the attempt complete |
| TEST-059 / Q1 | A1 direct-versus-repeated one-week atoms with nonconstant prices/dividends | Independent formulas differ as expected; A1 receives direct H4 maturity restriction and recomputed causal issued-score records |
| TEST-060 / Q2 | A2 frozen issued distribution independently initialized for each fold; A3 local/no-shrink; A4 liquidation; A5 no special hold | A2 atoms/probabilities unchanged even when query features change; only declared mechanism differs; A4/A5 forecasts identical to primary; A5 may coincide with a grid hold and logs that case |
| TEST-061 / Q2 | A6 window260 and A7/A8 neighbor16/64 with matched eligible/fallback reference cases | Each sensitivity has exact declared parameter propagation and distinct identity; no hidden retuning of another parameter |

## 8. Statistics, feasibility and terminal claims

Statistics tests use small deterministic arrays, exact enumeration where feasible and fixed generated inputs. They do not read market outcomes. A test of bootstrap code does not prove bootstrap validity for the actual adaptive financial history.

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-062 / Q1 | Labeled rows encoding date/asset/method/cost/fold identity through resampling | Every sampled date carries the complete paired family; folds sampled separately; no per-asset/per-method resampling |
| TEST-063 / Q1 | Small circular sequence at block lengths2/4, wraparound and remainder truncation | Independent block-index reference agrees; retained length exact; no favorable block-length choice |
| TEST-064 / Q1 | Unequal fold sizes, known score sums and target quantiles on a five-value array | Pooled weighting uses original date counts and fixed linear interpolation; not equal weighting of unequal folds |
| TEST-065 / Q1 | Identical scores, 5% boundary and control means just below/at/above 1e-12 in observed and resampled arrays | Ratio of date-weighted means, never mean datewise ratios; control mean <= 1e-12 undefined; any undefined required draw blocks claim without dropping/replacing it; strict primary margin retained |
| TEST-066 / Q1 | Deterministic campaign and pilot bootstrap replay, then run either operation first | Exactly 5,000 draws; CL-08 campaign SeedSequence(1901).spawn(4) and CL-09 pilot SeedSequence(1902).spawn(2) keep their descriptor order and independent PCG64 states; int64 starts/concatenation/truncation match reference; pilot controls share both pilot index arrays and cannot advance campaign streams |
| TEST-067 / Q1 | Known observed delta plus centered bootstrap samples; exact-zero A4/A5 forecast deltas | Null centering and plus-one p-value match scalar reference; policy-only forecast p=1, never invented prediction benefit |
| TEST-068 / Q1 | Sixteen ordered/tied p-values, including both block families and policy-only p=1 | Revision2 multiplicity family and Holm step-down correct, stable ID ties, first failure terminates rejection sequence; all entries reported |
| TEST-069 / Q1 | Saved path liquidatable-NAV increments with endpoint dividend, fee reserve, final actual liquidation and a resampled sequence whose concatenated holdings would be invalid | CL-12 log increments telescope to log(final wealth/initial cash); H3 equal-date mean block utility and separate terminal utility match hand values; hypothetical fees never debit cash; resampling not called an executable portfolio |
| TEST-070 / Q1 | Pre-2010-only precision inputs, projected-width/count boundaries, zero variability, no contextual/global difference, cold-start dates and a resample with control mean <= 1e-12 | CL-09 uses sample SD divisor4999 (ddof1), N=N_A+N_B eligible block counts and pilot n; scalar reference agrees; CL-14 undefined endpoint yields unidentifiable with no discarded draw; all four comparisons/cold-start dates retained; no threshold/window change |
| TEST-071 / Q2 | Decision-table cases for H1/H2 pass/fail/inconclusive/invalid, drawdown violation, incomplete families and partial controls | Exact terminal label and scope; no original NLL success, novelty, profitability guarantee or confirmatory label inferred from S10 ES/retrospective evidence |

## 9. Artifacts, consumption and fault injection

Tests create disposable engineering directories and use generated freezes. They do not touch the real S9 consumption marker, a real S10 freeze, protected data or prior scientific attempts.

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-072 / Q1 | Canonical JSON/array serialization twice, array-endianness case, differing field insertion order | Defined canonical payload/hash stable, schema/units preserved; elapsed time and paths excluded only from numerical equality payload, still retained as metadata |
| TEST-073 / Q1 | Corrupt source/config/data/environment/hash, checkpoint array shape and schema version | Integrity rejection before use; error identifies mismatched identity without dumping protected values |
| TEST-074 / Q2 | Inject write failure before publish, after temp write and at journal boundary | Existing artifact bytes unchanged; partial temp output never interpreted as completed state; failure remains visible if possible |
| TEST-075 / Q2 | Existing output path, pre-existing consumption marker and two concurrent start attempts | At most one scientific-start claim; second request rejected; no overwrite, marker deletion or retry identity reuse |
| TEST-076 / Q2 | Precision/full-campaign preflights with missing operation-specific prerequisite, failed admission and immediate post-consumption crash | Precision needs declaration plus tested kernel/bootstrap/guard receipts, not its own unproduced result; campaign requires precision pass; identities distinct. Prestart rejection unconsumed, poststart failure consumed/incomplete; ordering durable across crash |
| TEST-077 / Q2 | SIGINT/termination at forecast, ledger and analysis boundaries in an engineering subprocess | Terminal interruption with last durable checkpoint/start record; no successful completion and no automatic resume |
| TEST-078 / Q2 | Attempt market `resume`, delete-marker suggestion and checkpoint reuse under different source/config | Unsupported/rejected in v2. Checkpoint is available for audit/engineering reproduction only; any scientific continuation needs a separate protocol |
| TEST-079 / Q2 | Disk-full/permission-denied simulation, failure writing the failure receipt | Nonzero process result and surviving durable artifacts; stderr fallback does not claim the absent receipt exists |
| TEST-080 / Q2 | Resource observer unavailable, budget exceeded and mocked clock reversal in both precision and full-campaign operation | Same verified production supervision/resource/cancellation subset used before real pilot; unknown admission fails closed, monotonic clock and each operation's budget enforced; no duplicate unguarded pilot runner or fabricated RSS |
| TEST-081 / Q2 | Regenerate report from generated saved artifacts, then alter a raw row/table provenance key | Every reported value traces to a verified source; alteration or missing control prevents a valid report; analysis does not import a fitting path |
| TEST-082 / Q2 | Warning, expected typed exception and unexpected exception with chained cause | Required status, nonzero exit, traceback/cause and record IDs retained; no broad except-and-pass or exception-as-cash |
| TEST-083 / Q2 | Generated complete end-to-end engineering campaign and an injected defect near its end | All 192 small ledgers/artifact families present in valid case; failing case remains incomplete with successful earlier cells retained |

## 10. Package, quality, resources and regression protections

| ID / tier | Requirement and discriminating case | Acceptance or required rejection |
|---|---|---|
| TEST-084 / Q0 | Parse/import graph and inspect package modules in a restricted temporary environment | No predecessor/heavy runtime imports, package cycles, import-time data access, network, thread start, directory creation or freeze mutation |
| TEST-085 / Q0 | Strict type/lint/format scope inventory, invalid fixture schema and intentionally misspelled pytest marker | Complete new-source scope with zero unexplained violations; strict markers/config checks fail on typos; no whole-module exclusions |
| TEST-086 / Q3 | Build sdist/wheel, inspect archive members and install wheel outside checkout | Correct metadata/entry point/`py.typed`; no datasets/secrets/predecessor code; imports resolve from installed wheel; unavailable offline dependencies reported |
| TEST-087 / Q3 | Compare declared lock to metadata, installed distributions and environment receipt | Exact compatible identity or explicit mismatch; no silent re-resolution/upgrade; runtime imports require only declared NumPy plus standard library |
| TEST-088 / Q1 | Same fixture twice in locked process settings, then different hash seed/input dictionary order | Same canonical numerical results/decisions; mutable global RNG/order dependence exposed; full timing receipts allowed to differ |
| TEST-089 / Q3 | Same constructed suite on claimed Python3.12/Linux and Python3.12/macOS targets | Residual/decision comparison under frozen tolerances, drift disclosed; unavailable platform remains unverified rather than marked pass |
| TEST-090 / Q4 | Fixed generated shape ladder up to declared library/action bounds, one worker/thread | Measured worker and aggregate owned-process-tree RSS/time/work counters saved; no unbounded history growth beyond declared storage; proposal limits actually assessed |
| TEST-091 / Q4 | Vectorized/cache/chunk optimization versus independent reference before/after | Same certified objectives/actions and canonical semantics; report allocations/time with fixed input; no scientific advantage inferred from faster code |
| TEST-092 / Q1 | Regression identity inventory includes S6-S9 and revision1 design bytes | Unchanged predecessor hashes; a mismatch is a review blocker, never repaired by replacing old expected hashes |
| TEST-093 / Q2 | CI invocation under denied network, no data mounts and empty protected fixture path | Generated suite still complete; attempted external access fails visibly; no market collection or execution by default |
| TEST-094 / Q0 | Requirement-to-test registry with missing/orphaned ID, unjustified skip and expired quality exception | Readiness fails with exact gap; counts/coverage cannot substitute for missing critical evidence |
| TEST-095 / Q2 | Claims ledger points to absent evidence or labels static parse as runtime/independent empirical evidence | Report validation rejects unsupported claim; engineering/internal review remains separately named |
| TEST-096 / Q4 | Targeted semantic mutants in chronology, costs, cross-expert score and consumption order | A relevant test detects every non-equivalent critical mutant; survivors reviewed with concrete counterexample or equivalence argument |

## 11. Property and metamorphic campaigns

Use bounded property generation only where it explores a meaningful domain: simplex points, monotone timestamps with controlled invalid perturbations, self-financing portfolios, atom permutations, money-unit scaling and event-prefix replay. Examples should concentrate on boundaries, ties, degenerate arrays and dynamic ranges rather than many uniformly easy cases. Each generated input respects unrelated assumptions so a failure isolates the intended property.

Initial proposal: at most 100 generated examples per ordinary property, <=20 for expensive solver/end-to-end properties, a fixed CI profile and no unbounded data-dependent loops. Shrinking failures must be retained as explicit regression examples with the tool version and minimized values. A seed/example database is a reproducibility aid, not a stable cross-version guarantee. Hypothesis documents persisted examples and failure replay mechanisms; use those to recover a failure, then promote the minimized case into a hand-readable permanent fixture. [Hypothesis failure replay](https://hypothesis.readthedocs.io/en/latest/tutorial/replaying-failures.html), retrieved 15 October 2026.

Required metamorphic checks are TEST-017 future-prefix invariance, TEST-025 law-preserving atom transformations, TEST-029 geometric score invariance, TEST-039 no-label neighbor selection, TEST-052 dollar scaling, TEST-088 deterministic order invariance and TEST-091 reference equivalence. Metamorphic relations must state their assumptions; arbitrary asset permutation is not valid unless all scales, ordering labels, constraints and arrays are permuted together.

Do not test universal market dominance. Rising broad benchmarks and jointly adverse risky atoms are deliberate counterexamples to that impossible expectation. A perfectly predicted injected fixture does not meet S10 H1/H2; it verifies only the fixture's specified response.

## 12. Targeted mutation checks

Mutation work occurs only after meaningful critical tests exist, in a disposable copy of the new S10 source. It never edits frozen/predecessor files. Use at most 20 reviewed semantic mutants per verification campaign and a fixed <=10-minute engineering budget. No tool-specific dependency is assumed yet; hand-applied temporary mutants are sufficient and easier to audit than a whole-repository mutation campaign.

The minimum targeted faults are: replace `< decision` with `<= decision`; score past forecasts using the current library; remove an ES cross term; clip/renormalize an infeasible solver output; credit endpoint dividends to post-trade buyers; charge the hypothetical exit fee every block; use pre-fee target dollars; omit a matrix control; consume only after first result; and treat interruption as resumable market success. Record which TEST IDs kill each fault. A survivor is a specific test gap or a documented equivalent mutant, not a reason to inflate test count. A 100% mutation score over these faults does not prove absence of other defects.

## 13. Verification sequence and completion rule

1. **Static design and metadata admission:** resolve the normative clock, dimensions, matrix identities, feasibility formula and explicit source boundary. No numerical market content is needed.
2. **Minimal precision kernel and production guard implementation:** after the research-question gate, WP06 implements the necessary contracts, data transformations, stream/scoring/stacking and precision statistic, plus the required production bootstrap/artifact/exclusive-consumption/supervision/resource/cancellation subset. Require relevant Q0/Q1 and guard-focused Q2 tests before real pre-2010 preparation. Later work extends these same modules rather than creating a duplicate unguarded precision runner. Candidate precision cannot be measured by an unimplemented or untested model.
3. **Separate admitted pre-2010 preparation:** preflight needs the reviewed declaration and kernel/guard evidence, but cannot require its own unproduced precision result. Apply its distinct consumption identity, explicit bounded protocol, separate pilot RNG schedule and current resource admission; retain the complete result and enforce quantitative adequacy. Full-campaign preflight later requires a passing precision receipt. This is scientific preparation outside CI and this planning session; later evaluation values cannot select settings.
4. **Full system implementation:** policy/ledger, benchmarks, all fixed ablations, matrix, artifacts, statistics, reports and operations. Run affected Q1 plus Q2; finish packaging/targeted stress before readiness. Retain every failure and justified correction.
5. **Freeze/readiness review:** every required ID has actual evidence or an explicit applicable/not-applicable justification accepted before freeze. Missing chronology/accounting/integrity/control evidence is blocking. No success-by-skip.
6. **Scientific attempt:** use the exact frozen study only after separate execution scope and admission. Tests remain engineering evidence. An interrupted or failed consumed attempt is terminal under v2; no automatic rerun or checkpoint continuation.
7. **Artifact audit and release:** independently re-derive numbers from saved artifacts, preserve honest scientific disposition, and separate internal verification from later independent reproduction.

The catalog is complete enough to guide implementation, review and evidence collection. It does not claim all possible bugs have been anticipated, an environment has been validated, any test has passed, or the model has empirical merit. Newly discovered hazards extend the catalog and decision history rather than disappear behind a green summary.
