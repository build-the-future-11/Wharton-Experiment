# S10 architecture and interface contract

15 October 2026. Planning extension 2. **Proposed interfaces, not implemented code.**

This document expands the engineering design of [model revision 1](../MODEL_SPECIFICATION.md) and its [192-ledger protocol](../EXPERIMENT_PROTOCOL.md), subject to the explicitly versioned [model clarifications](MODEL_CLARIFICATIONS.md). It does not change their numerical model parameters, scientific endpoints, folds, controls or budgets. The addendum explicitly changes the previously ambiguous decision/fill timing and resolves the frozen-state ablation; these changes must be recorded before implementation. Section 13 maps the architectural consequences and remaining implementation-freeze details.

## 1 Package and dependency boundary

Build an isolated distribution at `successors/frem_s10`, import name `frem_s10`. Install and verify that distribution independently. The repository root currently declares Torch, pandas, SciPy, scikit-learn and Arrow dependencies; S10 must not require installing the root distribution or import `frem` to load its CLI. Earlier frozen successor modules remain unchanged.

The planned interpreter is isolated Python 3.12 and the production numerical dependency is NumPy. Standard-library modules provide records, parsing, hashing, files and CLI handling. Dependency groups named `quality` and `build` provide the chosen verification and packaging tools, matching the quality document's commands. Optional plotting and documentation dependencies remain separate from runtime. Exact compatible dependency versions and hashes are selected and locked during implementation readiness; this document does not claim an untested lock is reproducible. Avoid introducing a general optimizer when the three-expert active-face solve and finite action enumeration are the specified algorithms.

```text
successors/frem_s10/
  pyproject.toml
  README.md
  CHANGELOG.md
  src/frem_s10/
    __init__.py                 # version only; no IO or eager numerical imports
    __main__.py                 # delegates to cli.main
    cli.py                     # stdlib entry point and explicit command routing
    errors.py                  # stable error codes and structured exceptions
    metadata.py                # IDs, UTC times, units, enums, manifest envelopes
    configuration.py           # strict config parsing and manifest compilation
    identities.py              # canonical metadata and content identities
    arrays.py                  # checked immutable numerical record constructors
    domain.py                  # array-backed scientific records
    data/
      source_reader.py         # allowlisted source IO and scope enforcement
      admission.py             # source/calendar/action/availability validation
      calendar.py              # common weekly anchors and fold membership
      targets.py               # distinct weekly and direct-H4 (g,d) targets
      features.py              # 13-week inputs and fixed burn-in transform
    math/
      distances.py             # deterministic pair/context distances
      probabilities.py         # simplex checks and stable normalization
      scores.py                # ES, A/B, CRPS, variogram and discrete quantiles
      simplex.py               # active-face solution and KKT certification
      fees.py                  # self-financing root and residual certification
      utility.py               # stable CRRA and finite-support robust objective
    model/
      library.py               # matured, bounded scenario membership
      experts.py               # U/R/C and extrapolation fallback
      stacking.py              # G, local, shrinkage and standalone-loss control
      issuance.py              # immutable forecasts from approved visible state
      feedback.py              # one-time scoring of the exact issued forecast
      transition.py            # pure chronological forecast-state transition
    portfolio/
      actions.py               # integer-grid construction and exact-hold action
      policy.py                # complete nominal/robust action comparison
      execution.py             # approved decision-to-fill contract
      ledger.py                # shares/cash/fees/dividends/marks transitions
      benchmarks.py            # the six declared portfolio controls
    experiment/
      matrix.py                # explicit cell registry and completeness checks
      ablations.py             # eight named, validated mechanism differences
      engine.py                # coordinates pure transitions; no source fetching
    checkpoint.py            # explicit state serialization/audit reconciliation
    analysis/
      metrics.py               # saved forecast/ledger diagnostics
      bootstrap.py             # paired fold-wise shared bootstrap indices
      decisions.py             # unchanged H1/H2/H3 and terminal disposition
    runtime/
      admission.py             # stdlib resource/scope/environment admission
      runner.py                # one-use campaign state machine
      events.py                # append-only event and failure writer
      storage.py               # exclusive paths, atomic objects, manifests
      cache.py                 # verified engineering-equivalent intermediates
    reporting/
      tables.py                # values from validated analysis records
      render.py                # optional human-readable report generation
      claims.py                # claim-to-input-to-artifact mapping
  tests/
    unit/
    property/                  # bounded generated invariants and metamorphic cases
    contracts/
    chronology/
    integration/
    references/                # independently expressed mathematical oracles
    fault/                     # corruption, interruption and failure preservation
    slow_engineering/          # separately bounded scaling and stress cases
    fixtures/                  # tiny generated cases, clearly engineering-only
  schemas/                     # versioned JSON schemas and enum catalogs
  examples/                    # complete generated-fixture invocation only
```

`successors/frem_s10/tests` is the sole canonical S10 test location. Existing root tests remain untouched; no duplicate root discovery adapter is created. The nested module tree above supersedes revision 1's preliminary filename list for engineering organization only. Each production file has one responsibility. A new dependency, layer bypass or shared predecessor helper needs a documented reason and source identity before freeze. Shared code is either a separately versioned dependency or an attributed, reviewed copy with equivalence evidence; no runtime import of a mutable old study is allowed.

## 2 Import DAG and ownership

Arrows below mean “may import.” The DAG is enforced by an architecture check, not convention alone.

| Layer | May import inside S10 | Must not import |
|---|---|---|
| `errors`, `metadata`, `identities` | Each other only where acyclic | NumPy, filesystem adapters, model, policy, runner |
| `configuration` | Metadata, errors, identities | Scientific execution modules |
| `arrays`, `domain` | Metadata, errors, checked NumPy helpers | Data files, runner, reports |
| Pure `math` | Domain/arrays/metadata/errors; lower math helpers | Data adapters, experiment, runtime |
| Pure data transforms | Domain and math | Model, policy, analysis, runner |
| `model` | Domain, math, pure data transforms | Portfolio, results, analysis, source readers |
| `portfolio` | Domain and math | Feedback, future labels, analysis, source readers |
| `experiment` | Model, portfolio, pure data, configuration | CLI, report rendering; no network clients |
| `analysis` | Domain, math, validated saved-result schemas | Model training, policy replay, source downloads |
| `runtime` | Configuration/identity helpers and explicit service interfaces | Reporting at import time |
| `reporting` | Validated analysis/artifact records | Model/portfolio state mutation |
| `cli` | Stdlib admission/configuration first; later explicit imports | Numerical modules before resource admission |

The runner may import the engine after admission. The engine receives stores and event sinks as explicit interfaces; it does not import the concrete runner, so the dependency is one-way. Computational functions receive configuration values, never read environment variables, inspect global paths or use the wall clock. Time, budget accounting and IO belong to the caller. Only the report command imports optional plotting packages.

The research engine has one owner of forecast state and one owner of each ledger. It shares immutable issued forecasts among policy/cost arms. No ledger can mutate a forecast, feedback queue, or another ledger. Decisions and learned mixtures cannot import evaluation scores for their own immature targets.

## 3 Units, axes and time vocabulary

All computational arrays use finite `float64`, owned contiguous storage and explicit shape validation. Precision is not inferred from Python annotations. Every boundary checks axis labels and content identity as well as shape. Runtime constructors distinguish these records even when their physical array representation is identical.

| Symbol/type | Shape | Meaning and validation |
|---|---|---|
| `AssetOrder` | Three labels | Exactly `(SPY, TLT, GLD)`, not alphabetically reordered |
| `PriceGross` | `[N, 3]` or `[3]` | Dimensionless `g > 0`; terminal-zero assets unsupported here |
| `DividendPerStartDollar` | `[N, 3]` or `[3]` | Dimensionless `d >= 0`, cash per starting risky dollar |
| `TotalSimpleReturn` | `[N, 3]` or `[3]` | `r = g + d - 1`; never accepted in place of price gross |
| `WeeklyLogHistory` | `[13, 3]` | Oldest to newest `log(g_week+d_week)` |
| `RawContext`, `ScaledContext` | `[12]` | Asset-major: last, sum4, sum13, population-SD13 per asset |
| `FeatureMean`, `FeatureScale` | `[12]` each | Fixed pre-2008 burn-in transform; scales positive |
| `ScoreScale` | `[3]` | Fixed burn-in SD of H4 simple returns, floor `1e-4` |
| `StandardizedReturn` | `[N, 3]` or `[3]` | `r/b`; dimensionless score coordinates |
| `AtomProbability` | `[N]` | Nonnegative, sum one within specified tolerance |
| `ExpertProbability` | `[3, N]` | Ordered `(U, R, C)` over exactly the same atoms |
| `MixtureWeight` | `[3]` | Ordered `(U, R, C)`, checked simplex |
| `ScoreA`, `ScoreB` | `[3]`, `[3, 3]` | Issued-forecast ES coefficients; B symmetry checked |
| `ContextBatch`, `FeedbackA`, `FeedbackB` | `[M,12]`, `[M,3]`, `[M,3,3]` | M aligned matured records, `M <= 104` |
| `USDVector`, `ShareVector`, `PriceUSD` | `[3]` | Distinct dollars, share quantities, dollars/share |
| `TargetWeight` | `[3]` | Dimensionless, nonnegative; grid or benchmark-specific caps; integer grid has 609 points, at most 610 with hold |
| `TerminalFactor`, `Utility` | `[N]` each | Positive wealth factor; finite dimensionless CRRA value |
| `PairDistance` | `[N,N]` | Symmetric standardized-return Euclidean distances |
| `DateScorePanel` | `[T, methods]` | Complete aligned decision dates within named fold |
| `BootstrapIndices` | `[5000, T_fold]` per block length/fold | Integer row indices, shared by all comparisons |

`N <= 520` for normal arms; zero control has `N=1`; the library minimum is 64. A zero control is an explicit distinct distribution type, not an exception that weakens scenario-library admission. Exact expert/minimum/window changes for ablations are checked against the arm registry. USD floats carry no assertion of cent-level broker settlement; this research specification uses ideal fractional shares and continuous proportional fees.

Use timezone-aware UTC times and canonical signed integer microseconds since the Unix epoch in machine records. JSON also supplies a UTC ISO-8601 display string derived from that integer. Reject naive datetimes, mixed time bases and contradictory display strings. `SessionDate` is a calendar label, not an observation/availability timestamp. Required roles are `observed_at`, `available_at`, `decision_at`, `fill_at`, `origin_at`, `endpoint_at`; they are never interchangeable.

Availability is strict `< decision_at`. All weekly inputs and labels carry their own maximum contributing availability. Date boundaries such as `2017-01-01` compile to explicit UTC boundaries once and are retained in the resolved manifest. A hash of a file proves bytes, not authorization to expose its values. An opaque whole-file hash may verify the retained source; source date filtering must occur before any excluded outcome is exposed to a model, policy, diagnostic or log.

Under clarification v2, a session's scheduled opening quote is observed at 09:30 `America/New_York`, converted with the admitted dated timezone/calendar rule. The decision timestamp is that opening instant plus one microsecond; the ideal target-weight fill is plus two microseconds at the same quote. This is an explicit zero-latency/zero-price-move research assumption. Current risky dollars are shares times that observed quote. Dividends due on that session are credited once using entering shares before the trades. Forecast context remains based on the last fully mature history; the opening quote does not become an extra return feature. H4 label availability remains midnight UTC after the ending session, so a label ending at this opening instant remains unavailable for this decision.

## 4 Records and immutable state

Use frozen, slotted records for scalar metadata and wrappers with private owned array storage for numerical payloads. Constructors copy externally supplied arrays, validate them, and mark internal buffers read-only. Public array access returns read-only views or explicit copies. A frozen dataclass alone does not make an array immutable; mutation through aliases must be prevented and exercised by tests. Content hashes are rechecked at persistence boundaries. This is a programming contract, not a sandbox against malicious code.

Every persisted record has `schema_name`, `schema_version`, `record_id`, `semantic_hash`, `science_id`, `source_set_id`, `config_id`, and an explicit units/axis declaration appropriate to its payload. Records with time carry the roles described above. Optional fields use explicit null plus a reason enum; absence must not mean zero, cash, success or unobserved exposure.

| Record | Required content beyond the common envelope |
|---|---|
| `SourceIdentity` | Relative logical path, raw-byte SHA-256, byte count, source/acquisition manifest identity, declared access scope |
| `Observation` | Asset/session, prices/dividend action, role timestamps, source row ID, event type and supported-action validation |
| `OriginRecord` | Origin/endpoint IDs, 13 weekly input IDs, raw context, separate H4 g/d/r, feature availability and target maturity |
| `Transform` | Mean/scale/b, burn-in record IDs in order, fitting cutoff, count and fallback dimensions |
| `ScenarioLibrary` | Ordered mature origin IDs, joint g/d, contexts, availability, window rule and transform ID |
| `ExpertSet` | Library ID, U/R/C probabilities, current context, distances, fallback flags and threshold evidence |
| `IssuedForecast` | Origin/endpoint/maturity, pre-issue state hash, expert set ID, arm ID, alpha, final probabilities, optimizer/fallback certificates |
| `MaturedFeedback` | Issued forecast hash, exactly matched outcome ID, score A/B, maturity event ID, raw/scaled issuing context |
| `ForecastState` | Transform, library membership, pending issued IDs, mature feedback queue, cursor, consumed event IDs and previous state hash |
| `Valuation` | Shares, cash, visible marks and mark-availability IDs, NAV, decision time, valuation-contract ID |
| `Decision` | Cell/origin/forecast IDs, full feasible-action-set identity, selected action, hold/cash alternatives, objectives, fees and tie/constraint certificate |
| `Fill` | Decision ID, actual execution marks, shares/cash before and after, turnover, fee, self-financing residual, fill-contract ID |
| `LedgerState` | Cell ID, shares/cash, current valid marks, cumulative attribution, last event and state hash |
| `LedgerEvent` | Event type/ID/order, state-before/after hashes, cash/share/mark deltas, source or decision ID, conservation residual |
| `NumericalCertificate` | Method/version, scale/tolerances, measured residuals, selected active face/action, pass/failure reason |
| `AnalysisResult` | Complete source result IDs, endpoint/aggregation/bootstrap identity, defined/undefined status, values and claim limits; CL-12 standardized versus raw diagnostic units remain separate |

`ForecastState` is independent of fees, policy, current holdings and realized portfolio performance. No feedback update accepts a ledger or a chosen trade. `LedgerState` is independent of learned model weights except through its already saved decision. Keep forecasting evidence and trading evidence separate while linking both to the same issued distribution. Actual daily marked NAV and hypothetical liquidatable boundary NAV are distinct fields; the latter does not generate fee cash events. CL-12 defines their exact use in economic statistics.

Pending forecasts retain enough exact atom/probability information to score their original distributions later. Once mature, A/B may replace that payload in active working memory, but the immutable issued artifact remains available to audit; reconstructing a historical forecast using today's library is prohibited.

## 5 Public Python API contract

The following signatures define interfaces to implement. They are intentionally in documentation; no stubs or executable placeholder files are supplied. `FrozenArray` denotes the validated numerical wrapper, not an unchecked `Any`. Keyword-only configuration prevents ambiguous numeric positional arguments. Domain constructors perform shape/unit checks; pure functions reject mismatched identities.

```python
def admit_sources(
    sources: tuple[SourceIdentity, ...], *, reader: ScopedSourceReader,
    contract: DataContract,
) -> AdmissionReport: ...  # IO boundary; report includes exclusions and rejection reasons

def build_origin_records(
    admitted: AdmittedObservations, *, calendar: CalendarContract,
    target: TargetContract,
) -> tuple[OriginRecord, ...]: ...

def fit_burnin_transform(
    records: tuple[OriginRecord, ...], *, config: FeatureConfig,
) -> Transform: ...

def update_library(
    previous: ScenarioLibrary, matured: tuple[OriginRecord, ...], *,
    decision_at: UTCInstant, config: LibraryConfig,
) -> ScenarioLibrary: ...

def issue_experts(
    library: ScenarioLibrary, context: ScaledContext, *,
    decision_at: UTCInstant, config: ExpertConfig,
) -> ExpertSet: ...

def score_coefficients(
    issued: IssuedForecast, outcome: MatureOutcome, *, transform: Transform,
) -> ScoreCoefficients: ...  # A[3], B[3,3], exact issued expert distributions

def solve_simplex(
    mean_a: ScoreA, mean_b: ScoreB, anchor: MixtureWeight, *,
    l2_coefficient: float, tolerances: SolverTolerances,
) -> CertifiedMixture: ...

def fit_mixtures(
    feedback: tuple[MaturedFeedback, ...], context: ScaledContext, *,
    decision_at: UTCInstant, config: StackingConfig,
) -> MixtureBundle: ...  # G, L or fallback, beta, final alpha, all certificates

def issue_forecasts(
    visible: VisibleForecastState, request: IssueRequest, *,
    config: ForecastConfig,
) -> IssuanceBatch: ...  # eight core identities; no current outcome argument

def advance_forecast_state(
    previous: ForecastState, event: ForecastEvent, *, config: ForecastConfig,
) -> ForecastTransition: ...  # next immutable state plus append-only artifacts

def enumerate_actions(
    valuation: Valuation, *, config: ActionConfig,
) -> ActionSet: ...

def solve_rebalance(
    holdings: USDVector, cash: USD, target: TargetWeight, *,
    one_way_cost: CostFraction, tolerances: FeeTolerances,
) -> CertifiedRebalance: ...

def choose_action(
    forecast: IssuedForecast, valuation: Valuation, actions: ActionSet, *,
    config: PolicyConfig, cost: CostFraction,
) -> Decision: ...

def execute_decision(
    decision: Decision, ledger: LedgerState, fill: AdmittedFillEvent, *,
    contract: ExecutionContract,
) -> LedgerTransition: ...

def advance_ledger(
    previous: LedgerState, event: LedgerInputEvent, *,
    config: LedgerConfig,
) -> LedgerTransition: ...

def compile_matrix(config: ResolvedScienceConfig) -> ExperimentMatrix: ...

def compute_analysis(
    records: CompleteValidatedResults, *, config: StatisticalConfig,
    bootstrap_indices: BootstrapIndexFamily,
) -> AnalysisBundle: ...

def run_campaign(
    manifest: AdmittedRunManifest, *, store: ArtifactStore,
    clock: MonotonicClock, budget: BudgetMonitor, events: EventSink,
) -> CampaignReceipt: ...  # orchestration boundary; consumes before scientific work
```

The `...` above means “signature specified in the plan,” not permission to leave production bodies empty. Public APIs return structured values and certificates rather than printing results. Exceptions represent invalid inputs or unsuccessful computations. A failed forecaster cannot return a normal cash decision.

`VisibleForecastState` is a capability-style record containing only approved past features, mature library records and mature feedback. It omits current/future targets. Portfolio APIs receive the issued forecast plus explicitly visible valuation information. Source readers, source paths and raw full-history tables are not arguments to forecast or policy functions. Tests must attempt to break these boundaries.

## 6 Pure kernels and IO services

| Pure computation | Explicit external service |
|---|---|
| Calendar construction and exclusion decisions on admitted metadata | Read source bytes and audit their approved access scope |
| Feature/target transformation, scoring, experts and mixtures | Acquire source manifests; no network acquisition in a frozen run |
| Action enumeration, costs, utility and ledger transitions | Open/create artifacts, append logs, fsync and atomic commit |
| Matrix compilation and result completeness | Resource measurement, monotonic timing and signal handling |
| Bootstrap arithmetic on explicit indices | Construct a versioned seeded index family in its recorded environment |
| Report table values and disposition logic | Render documents/figures and write report files |

Pure functions never mutate their inputs. Each state transition returns a next state and a tuple of deterministic domain events. The orchestration layer writes those events and the new state atomically according to the runtime contract. A report can be regenerated from retained verified outputs without importing the forecaster or executing policies.

## 7 Configuration authority and versioning

The scientific sources of truth are the recorded model/protocol revision and its reviewed resolved scientific configuration. The current `design.json` is a design manifest, not a run-ready configuration. During implementation, the config compiler must reconcile every scientific field with the reviewed revision and fail any mismatch. Unknown fields, duplicate JSON keys, nonfinite values, wrong types, implicit units and unsupported enum values are errors.

Use three separate inputs:

1. `ScienceConfig`: all assets, horizons, transforms, library/expert/stacking rules, tolerances, policies, costs, folds, benchmarks, ablations, endpoints, statistics and stopping ceilings. Every field is explicit. A field omitted from the file must not inherit a scientific default from code.
2. `RunConfig`: fresh output path, input artifact locations, bounded logging settings, environment identity and thread/resource settings. It cannot override scientific values. Resource settings may tighten a ceiling but cannot weaken admission or enlarge a budget.
3. `ReportConfig`: output format and cosmetic presentation. It cannot hide cells, select a favorable interval family or change a statistical result.

Merge no arbitrary dictionaries. The validated compiler creates one immutable `ResolvedRunManifest` and its identity before scientific work. CLI options are limited to explicit configuration locations and declared operational controls; there are no silent environment-variable overrides for fees, windows, thresholds, dates or primary arms. A complete resolved copy is retained even if the run fails admission.

Version axes are distinct: package version, scientific revision, config schema version, artifact schema version, numerical-kernel version, report schema version and study/freeze identity. A code fix is not automatically scientifically equivalent. Changing a kernel or schema after a freeze requires an impact review; if behavior can change, retain the old result and create a corrected version/freeze. Do not migrate an old receipt in place.

Readers accept only documented compatible versions. Required scientific field changes require a new schema version. Unknown major versions fail closed. Migration tools write new artifacts with old/new identities, the exact migration implementation, reason and equivalence evidence. Migration does not reset a consumption record or turn failed evidence into success.

## 8 Identity and serialization rules

Maintain both **raw-byte artifact digests** and **semantic numerical identities**. They answer different questions and are never substituted for one another.

- Source files: SHA-256 of exact original bytes, byte count and acquisition/provenance IDs. Do not normalize or rewrite originals.
- Metadata identity: UTF-8 canonical JSON with sorted object keys, fixed separators, no duplicate keys, no NaN/Infinity, explicit enum strings, normalized UTC integers, and arrays in their declared order. Scientific float scalars are encoded by their exact finite binary64 hexadecimal representation for semantic hashing; human display decimals are separately derived.
- Numerical payload identity: include schema/units/axes, rank and dimensions, declared little-endian float64 or integer dtype, then C-order bytes. Define signed-zero treatment once; preserve its bits by default. Hashes cannot rely on platform-native endian or dictionary iteration order.
- Stored arrays: a documented non-object format with pickle disabled and strict dtype/shape/size checks before allocation. Record the file SHA-256 as well as the semantic payload identity. Decoders reject object arrays and unknown dtype codes.
- Source tree identity: relative-path-sorted file manifest of all relevant source, schemas, lockfiles, config, protocols and tests used for readiness. Record Git commit and dirty status as context; Git HEAD alone cannot identify a dirty working tree.
- Environment identity: exact Python/dependency/build/platform/numerical-library versions and thread settings. Resource receipts are evidence objects, not ingredients of the numerical state hash.
- State identity: deterministic scientific payload and parent state/event IDs. Exclude wall-clock log timestamps, absolute workspace paths, measured RSS and temporary filenames from the numerical identity; retain them in the run envelope.

Record IDs are derived from type/version plus canonical scientific identity and upstream IDs. An absolute local path is a locator, not a dataset identity. Two studies with matching arrays but different access/freeze scopes remain distinct studies. A hash cache never grants permission to read protected data.

Checkpoints are explicit schema objects and typed arrays; do not pickle arbitrary Python classes. A checkpoint records the next event cursor, pending maturity records, processed event IDs, cumulative accounting, RNG/index-family state where applicable, input/source/config/kernel identities and all active ledger identities. Loading checks every identity before creating mutable runtime state. Under clarification v2, actual market failure is terminal for the consumed attempt: there is no market resume command. Checkpoint continuation is exercised only in generated engineering equivalence tests, while saved market checkpoints support audit. Their presence never authorizes a retry.

## 9 Ordering and determinism

Declare every ordering that can affect arithmetic or state:

| Object | Required order |
|---|---|
| Assets/features/experts | SPY/TLT/GLD; four declared features per asset; U/R/C |
| Source observations | Session/observation time, asset index, action kind, stable source row ID |
| Library and feedback | Origin time then stable record ID; retain the last allowed records after maturity filtering |
| Equal-distance neighbors | Squared distance, origin timestamp, record ID |
| Maturity updates | Availability, original forecast origin, issued forecast ID; once per outcome/forecast relation |
| Opening events | Observe admitted opening quote; credit same-session dividend entitlement using entering shares; decide at open+1us; fill at open+2us; retain the distinct dated availability of all labels |
| Simplex faces | Fixed bitmask order of the seven nonempty expert subsets |
| Forecast arms | The eight identifiers in protocol order; no hash-set iteration |
| Action grid | Integer ticks for .05 increments; lexicographic integer tuples; hold is explicitly tagged |
| Policy ties | Hold within specified objective tolerance, otherwise lower turnover, then lexicographic weights |
| Cells | Family, fold, forecast/benchmark/ablation ID, policy ID, ascending cost |
| Bootstrap | Fixed stream descriptors `(A,2), (B,2), (A,4), (B,4)`; shared per-draw index family |
| Artifact manifests | Logical relative-path order; reject duplicate logical paths |

Compute grid weights from integer ticks, not repeated floating-point addition. Keep all 609 main grid points and all 45 fixed-sum minimum-variance benchmark points; do not prune actions using a heuristic. Deduplicate a hold vector that exactly equals a grid point while retaining its special tie identity. Determine the maximum objective before collecting the full tolerance-tied set; do not use order-dependent pairwise approximate comparisons. Global reductions and pair distances use a specified stable iteration/chunk order; parallel completion order must not change sums. Single-worker/single-numerical-thread requirements remain in force.

For CL-08, create `SeedSequence(1901)` and spawn exactly four children in the fixed descriptor order above. Each child initializes `Generator(PCG64(child))`. For each of 5,000 draws, sample `ceil(N_fold / L)` independent uniform integer starts in `[0, N_fold)`, with int64 output; expand each into L consecutive indices modulo `N_fold`, concatenate in sampled-start order and truncate to `N_fold`. Retain all four index matrices and their identities. A report cannot create or consume a new stream. Freeze the exact NumPy/environment versions and the resulting index-artifact identity. The 16 secondary tests are ordered for Holm by raw p-value, ablation ID and block length, including exact p=1 policy-only comparisons as CL-08 requires.

Byte-identical numerical replay is required within the frozen environment for uninterrupted versus checkpointed/offline generated engineering execution. Market checkpoint continuation remains disabled. Cross-platform bitwise identity is not assumed merely from using float64; portability verification records semantic tolerances, control/tie choices and any divergence. A difference that changes a selected action, fallback or gate requires review even if aggregate metrics look close.

## 10 Cache contract

Caches are optional performance aids. A cache hit and a fresh calculation must give the same canonical scientific result. Disabling every cache must leave forecasts, decisions, ledgers and analysis identical within the declared environment.

| Cache | Complete key |
|---|---|
| Parsed admitted records | Source raw hashes, scope/access contract, adapter/schema/calendar versions |
| Features/targets | Ordered admitted-record IDs, target/feature config and transform identity |
| Standardized contexts | Ordered origin IDs and fixed transform identity |
| Pair distances | Ordered atom IDs, score-scale identity, distance-kernel version/dtype/order |
| Expert coefficients | Issued expert probabilities, exact outcome ID, transform and score-kernel version |
| Action grid | Grid/cap config, version; hold never belongs to the shared static grid |
| Policy utilities | Forecast ID, actual valuation/holdings ID, cost, action-set ID, policy and utility-kernel identity |
| Bootstrap indices | Statistical config, ordered fold/date IDs, generator/environment identity |

Price changes, source scope, event order, outcomes, holdings and costs cannot be omitted from keys where they affect the value. A pair-distance matrix may be reused across mixtures sharing atoms; a decision may not be reused across different ledger states merely because the forecast is the same.

Caches are namespaced by scientific revision, kernel version and source scope. Read caches only after admission. Reject symlinks/path escapes and unexpected object types. Writes use exclusive temporary objects and atomic rename, followed by integrity verification. An interrupted partial object is never a valid hit. Eviction removes only the disposable cache copy; immutable issued forecasts, results, receipts and consumption records are not caches.

A corrupt cache is a retained integrity failure under [the operations contract](OPERATIONS_AND_ARTIFACTS.md), not a normal cache miss. A missing optional cache may be recomputed from exact admitted inputs within the same remaining budget. A corrupt source, forecast, checkpoint or committed result is likewise a retained failure. Caches never bypass the one-use registry, resource gate or protected-value boundary.

## 11 Validation and error model

Validation is layered: parse/schema, source/time/access, shape/unit/identity, numerical feasibility, state transition, matrix completeness, then analysis/report provenance. Validate early, before expensive allocation or scientific imports where possible. Every public constructor and IO boundary is checked; internal kernels may rely only on validated domain objects.

| Error family | Examples | Required behavior |
|---|---|---|
| `CONFIG_CONTRACT` | Unknown field, conflicting revision, non-explicit scientific value | Nonzero preflight failure; no consumption |
| `ACCESS_SCOPE` | Forbidden date/value exposure, missing scope identity | Reject before returning a scientific record |
| `SOURCE_INTEGRITY` | Wrong SHA, source-row conflict, split/terminal event unsupported | Retain admission failure; no dropping the event |
| `CHRONOLOGY` | `available_at >= decision_at`, out-of-order cursor, duplicate maturity | Stop, retain implicated IDs; no fallback forecast |
| `SHAPE_UNIT` | Wrong asset axis, generic returns at fee/ledger boundary | Reject the call |
| `NUMERICAL` | Nonfinite result, invalid probability, curvature/KKT/fee residual failure | Stop the attempt and retain measured certificate |
| `STATE_CONFLICT` | Replayed event changes payload, mismatched checkpoint/source | Reject; no silent last-write-wins behavior |
| `MATRIX_INCOMPLETE` | Missing zero control, duplicate cell, partial fold | Incomplete failed result; no primary success claim |
| `RESOURCE_BUDGET` | Admission failure, wall/RSS/origin limit exceeded | Retain blocked/failed receipt at the actual phase |
| `STORAGE_INTEGRITY` | Existing output, partial manifest, path escape, damaged evidence | Fail safely without overwriting prior artifacts |
| `SPECIFICATION_UNRESOLVED` | Missing approved mark/fill or ablation timing contract | Implementation/readiness blocked at named requirement |

Use stable error codes, structured `stage`, `record_ids`, expected/observed metadata, numerical residuals and exception cause. These error families are descriptive causes, not lifecycle terminal states or CLI exit codes. The operations contract is authoritative for exact persisted lifecycle names and exit-code mapping; the CLI maps the error plus phase to that contract rather than inventing aliases. Do not log a forbidden label as an error detail. Sensitive machine paths may be redacted in portable reports while their logical artifact IDs remain resolvable. No broad `except Exception: return cash`, dropped rows, changed tolerances or warning-only integrity defects.

Distinguish expected scientific fallbacks from errors. Sparse/OOD context returning U or G is a valid, logged rule; uncertified optimization, missing prices or malformed input is an error. Duplicate input with identical identity is handled only by the explicit replay/idempotency protocol; a scientific duplicate presented as a new record is rejected.

## 12 Integration and delivery acceptance

The architecture is accepted only after evidence shows:

1. A clean isolated installation imports the CLI without the root package or Torch; startup can produce a resource/admission failure before numerical imports.
2. Architecture checks reject forbidden imports and cyclic dependencies. Public API constructors reject wrong shapes, units, identities and times.
3. Golden hand calculations and independent references validate the mathematical interfaces; input mutation cannot corrupt retained state.
4. The eight forecasts share the correct information, all policy/cost arms reuse immutable predictions, and feedback uses exact saved forecasts rather than regenerated ones.
5. The complete manifest compiles to 128 strategy, 48 benchmark and 16 ablation identities with no omissions, duplicates or hidden extra trials.
6. Cached/uncached, chunked/reference and interrupted/uninterrupted engineering traces match their declared equivalence contracts, including state, action and event identities.
7. Report regeneration cannot run a model, policy or source download and cannot silently omit failed cells.
8. Package metadata, configs, schemas, dependency locks, command help and error catalog describe the implemented behavior; no placeholder body or unfinished critical path remains.

These are future engineering gates. No implementation, tests, model run or installation is performed by this planning document.

## 13 Clarification register and implementation obligations

The architecture review surfaced the following interface questions. [The v2 model addendum](MODEL_CLARIFICATIONS.md) is normative for the resolved scientific cases. Revision 1 remains preserved; there is no silent reinterpretation of its pre-open statement.

| ID | Review finding | Binding treatment |
|---|---|---|
| AR-01 | Pre-open decisions and exact opening-mark holdings/costs were incompatible without a mark/fill rule. | V2 observes the scheduled opening quote, decides at open+1us and fills at open+2us at that quote. The ideal no-move assumption is reported. No pre-open access to the upcoming open is permitted. |
| AR-02 | A2's “freeze state” statement left query-dependent C/local retrieval unclear. | V2 freezes the primary issued atoms and final probabilities at the first evaluation decision of each fold. Later forecasts reuse that exact distribution; current context and new feedback cannot change it. Keep per-origin issuance/outcome IDs distinct and continue scoring. |
| AR-03 | Same-session dividends and trades needed ownership ordering. | V2 credits entitlement once from entering shares before trades. Bounds remain `origin < ex_date <= endpoint`; cash is separate from price and ex-date entitlement remains the declared research approximation. |
| AR-04 | A2 needs a causal initial snapshot per fold; repeated preparation could hide compute. | Record logical unique origins and actual issue/update/solve counts separately. Each executed issue/update origin, including repeated preparation, consumes the per-arm origin allowance. Reuse an exactly matching immutable state only with a complete identity match; only actual avoided computation avoids a second count. The static preparation DAG must fit the unchanged ceilings before freeze. Use CL-09's minimal-kernel/pilot order and the operations lifecycle. |
| AR-05 | A seed alone does not fully specify bootstrap index construction. | CL-08 and section 9 now bind PCG64, stream spawning, draw construction, truncation, pairing and emitted index artifacts. Keep seed 1901, 5,000 draws, both block lengths and the full 16-test Holm family. No seed search. |
| AR-06 | Checkpoint equivalence could be mistaken for permission to restart a consumed market campaign. | V2 allows checkpoint continuation only for generated engineering verification; actual market failure is terminal and no market resume command exists. |

The implementation gate order is static source/scope admission, tested minimal precision kernel, separately admitted pre-2010 precision preparation, completion of the remaining pipeline, then run readiness. The pilot retains the declared cold-start/fallback behavior; it does not require that all local machinery is warmed before scoring begins or change its window to obtain a favorable precision estimate. Precision preparation is not silently expanded to the full evaluation. Both pilot and campaign obey the operations contract's stricter aggregate 256 MiB owned-process-tree ceiling. A comprehensive architecture plan does not establish a perfect model, an answerable experiment or a passing scientific gate.
