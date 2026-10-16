# S10 revision 2: code quality and continuous verification

15 October 2026. **Planning contract only.** No S10 package, tool configuration, lock, CI workflow or runtime result is created by this document. Every command below is a future command, dependent on implementation, reviewed dependencies and the applicable resource gate. There is no present claim that these commands work in this checkout.

This plan extends the preserved [revision 1 implementation plan](../IMPLEMENTATION_PLAN.md). [MODEL_CLARIFICATIONS.md](MODEL_CLARIFICATIONS.md) takes precedence where it resolves revision 1 ambiguity; package ownership follows [ARCHITECTURE_AND_INTERFACES.md](ARCHITECTURE_AND_INTERFACES.md). Quality checks cannot change a scientific threshold, drop a control or turn an incomplete attempt into a successful result.

## 1. Verified starting point and proposed target

Read directly on 15 October 2026: the root `pyproject.toml` declares Python `>=3.11`, Ruff `py311`, setuptools/wheel builds, and NumPy, pandas, SciPy, scikit-learn, PyTorch and PyArrow runtime dependencies. Its development extra contains pytest, Ruff and build; it does not declare mypy, Hypothesis or coverage. The existing root `uv.lock` belongs to that project. This inspection is metadata evidence, not a fresh environment or build check.

S10 will be an isolated package at `successors/frem_s10/src/frem_s10`, with its own `pyproject.toml`, `uv.lock`, README, license metadata, `py.typed`, and quality configuration. Canonical tests live in `successors/frem_s10/tests`; revision 2 supersedes revision 1's root test location as an engineering packaging clarification. Root pytest collection must not implicitly execute S10. Package isolation avoids importing predecessor code or changing its dependency resolution. Proposed first supported interpreter is **CPython 3.12**, with `requires-python = ">=3.12,<3.13"` until another minor version is deliberately verified. The lock records the tested patch version and platform identities; this narrow support statement makes no claim about every Python 3.12 patch or platform.

The runtime dependency proposal is **standard library plus NumPy only**. No GPU framework, pandas, SciPy, plotting library, online provider SDK or optimizer service is required for forecasts, accounting, audit or the core runner. Standard-library CSV/JSON and non-object NumPy arrays provide the initial storage boundary. Plots, property testing, type checking, package building and optional audits belong in explicit development/report dependency groups. The exact versions will be resolved once in a later engineering session, then pinned and verified; this design does not guess a compatible lock or alter the root lock.

The runtime deliberately uses a three-expert active-face solver and a finite portfolio grid. A new dependency needs a documented capability gap, size/license review, an exact resolved version, a meaningful verification case, and evidence that it does not change the frozen mathematics. Convenience alone does not justify loading the legacy scientific stack.

## 2. Dependency direction and ownership

| Boundary | Allowed responsibility | Prohibited coupling |
|---|---|---|
| `metadata`, `errors`, `identities`, `configuration`, `arrays`, `domain` | Shapes, units, timestamps, identities, immutable records, named failures | Reading files, choosing a model, printing, numerical execution at import |
| Pure `math` and `model` | Pure array calculations, delayed transitions and certificates | Accessing filesystem, wall clock, environment, protected sources or global mutable state |
| `portfolio` | Pure decision/accounting transitions | Choosing data exclusions; using an actual future quote in forecast features; hiding numerical failure as cash |
| `data.source_reader`, `runtime.storage`, `checkpoint` | Explicit validated read/write boundaries | Fetching data implicitly; trusting object arrays/pickle; interpreting data metadata as executable code |
| `experiment` | Fixed matrix orchestration through explicit services | Modifying scientific settings after results; automatic scientific restart |
| `analysis`, `reporting` | Artifact-derived diagnostics and rendering | Re-fitting models, changing primary arms or modifying saved evidence |
| `runtime`, `cli` | Admission, dispatch, resource observer, terminal receipt and human diagnostics | Mathematical policy hidden in argument handling |

This is a responsibility summary, not an alternate import graph; the exact allowed DAG is in ARCHITECTURE_AND_INTERFACES. Neither the numerical core nor report generation imports the CLI. There are no imports from root `frem`, `successors.frem_s9` or other predecessors. An audited formula may be reimplemented with attribution and reference tests; a source dependency must be explicit and frozen, never resolved by a mutable search path.

An AST/import inventory check will report package cycles, unexpected external imports and prohibited predecessor imports. It will inspect all new source rather than only files reached by passing tests. Package import must neither open a source nor create a run directory, consume a freeze, change logging settings, set NumPy global error policy or start threads.

Each module has one named responsibility, an owning reviewer role, and a contract-to-test mapping. Common mathematical kernels have one implementation, but independent test references intentionally do not import that kernel. Avoid generic plugin frameworks, deep class hierarchies and a universal `utils.py`. Split a module by a coherent boundary, not to meet a line count.

## 3. Types, units and API contracts

Use strict mypy on the whole new package, supporting scripts and S10 tests. The checker is a development dependency, never imported at runtime. `strict = true`, `warn_unused_ignores = true`, `warn_unreachable = true`, and `show_error_codes = true` form the initial policy. Pin the mypy version: its strict-mode option set may change. Local suppressions must name the error code and explain the unavoidable boundary; global `ignore_errors`, `ignore_missing_imports`, `follow_imports = "skip"`, blanket `Any` escape hatches and unchecked casts are forbidden. [mypy configuration](https://mypy.readthedocs.io/en/stable/config_file.html), retrieved 15 October 2026.

Use `numpy.typing.NDArray[np.float64]` for numerical arrays and runtime checks for rank, dimension and finite values. Type hints do not prove shape, units, positivity or chronological admissibility. Separate wrappers/records distinguish price gross returns, dividend cash per initial risky dollar, simple returns, standardized scores, shares, USD cash, target weights and scenario probabilities. A plain array named `returns` cannot cross the policy/accounting boundary. Conversion functions state the formula, units and information cutoff.

Frozen dataclasses with slots are the default public record. Array inputs are defensively copied when crossing a persistence/state ownership boundary and stored read-only; `frozen=True` alone does not prevent array mutation. Shared read-only arrays may be reused internally when ownership is documented. Hashes use canonical data bytes, not object identity or Python's randomized `hash()`.

Public functions use explicit keyword-only configuration where multiple same-typed arguments could be confused. Return named records rather than positional tuples of similar arrays. Mutable default arguments, implicit global generators, implicit local-time datetimes and truth-value coercion of arrays are prohibited. UTC instants are timezone-aware; market local time is converted using the admitted `America/New_York` calendar/timezone rule before storage. Booleans must not pass as integer counts merely because Python permits that subtype relation.

Public docstrings include purpose, equation/specification reference, shape/dtype, units, observation/availability constraints, return invariants, failure behavior and determinism limitations. Private nontrivial kernels document assumptions and derivation. Comments explain why, especially tolerance and event-order choices; they do not paraphrase assignment statements. README examples must use generated engineering inputs and state their scope.

## 4. Explicit errors and observable status

The proposal is a small typed exception hierarchy rooted in `S10Error`. The CLI maps these to stable nonzero exit/status values in the operations contract; the core never calls `sys.exit`.

| Error family | Examples | Required disposition |
|---|---|---|
| `ContractError` | Wrong units/shape, duplicate JSON key, nonfinite value, unknown schema | Reject input, name field and expected invariant |
| `ChronologyError` | Immature label, duplicate maturity, availability equal to decision, wrong event order | Invalid timing; retained failing record identity |
| `DataAdmissionError` | Unsupported split, zero price, missing mark, unverified source/calendar | Block before scientific start where detectable |
| `NumericalError` | Uncertified simplex solve, nonpositive utility argument, bad fee residual | Terminal invalid attempt if started; no substitute action |
| `IntegrityError` | Wrong source/config/hash, malformed checkpoint, missing matrix cell | No analysis success; preserve evidence |
| `ResourceLimitError` | Admission floor, wall/RSS/process limit | Blocked before consumption or terminal interrupted after start |
| `ArtifactIOError` | Disk full, permission failure, partial checkpoint | Nonzero exit; preserve prior durable artifacts and record failure if writable |
| `ExecutionStateError` | Consumed freeze, existing output, prohibited resume | Reject action without touching prior attempt |

Catch only errors that the layer can explain or handle. A single outer runner boundary may catch unexpected `Exception` to write an `UNEXPECTED_FAILURE` receipt and re-raise/exit nonzero. It must retain exception type, chain and traceback; it must not catch-and-pass, turn failure into a cash forecast, retry market work, or overwrite a prior receipt. `KeyboardInterrupt` and termination signals require explicit interruption handling; do not blanket-catch `BaseException` and continue. A hard kill can prevent a final receipt, so the prior consumption/start journal is itself durable evidence of an incomplete attempt.

Validation uses ordinary exceptions, not `assert`, because Python optimization can remove assertions. Assertions remain useful in tests. A warning is never the only response to a violated scientific invariant. Diagnostics contain record IDs, paths within the declared artifact scope, residuals and limits, not credentials or protected values.

## 5. Numerical implementation discipline

1. Use float64 for all scientific state. Reject object, complex and nonfinite inputs; do not silently downcast float32 checkpoints. Accepting an input conversion must be explicit and recorded at intake.
2. Keep denominators, clipping, floors and tie rules centralized with their specification references. A floor declared for burn-in scales does not authorize clipping returns, distances, probabilities or wealth elsewhere.
3. Use scoped `np.errstate` around risky arithmetic, then explicit finite/domain checks. Underflow may be permitted only where analytically benign and tested; overflow, invalid arithmetic and division by zero cannot disappear under a broad ignore setting. [NumPy error handling](https://numpy.org/doc/stable/reference/generated/numpy.errstate.html), retrieved 15 October 2026.
4. Reuse exact pair-distance terms, but verify direct norms before adopting a faster dot-product distance identity: cancellation near equal large vectors must not be hidden by unconditional clipping. Chunking/caching changes require reference equivalence and the same stable ordering.
5. Preserve the specified sum-to-one, symmetry, curvature, KKT and fee-root tolerances. CL-14 additionally binds scaled bordered-system conditioning, whole-set action ties, weight feasibility and the relative-score denominator. Each residual uses the named coefficient/monetary scale; recover nu_original=s*nu_scaled before checking original-scale KKT residuals. Reports save raw and normalized residuals, not just `PASS`. Recheck after permitted tiny-negative-weight cleanup; no failing face or undefined bootstrap draw is silently dropped.
6. Stable utility uses the specified log/expm1 form. The tests cover values near one, small positive wealth and invalid zero/negative wealth. No policy cap, penalty or parameter changes to make numerical tests easier.
7. Use explicit stable sorting and IDs for atom, expert, event and action order. Reductions occur in a fixed order. Near ties obey declared objective/turnover/lexicographic rules; no nondeterministic map/set traversal controls a choice.
8. Use explicit NumPy generators and the exact separate campaign/pilot descriptor/`SeedSequence`/PCG64 child-stream schedules in MODEL_CLARIFICATIONS. Pilot seed1902 streams cannot advance campaign seed1901 streams; engineering fixtures have separate declared generators. Record NumPy and bit-generator versions, draw shapes and ordering. Identical seeds do not guarantee identical results across every NumPy build or call pattern. [NumPy random compatibility](https://numpy.org/doc/stable/reference/random/compatibility.html), retrieved 15 October 2026.
9. Separate canonical numerical payloads from wall time, RSS, paths and other run metadata. Same locked environment plus same inputs must reproduce canonical state/results exactly; full receipt bytes need not match when timing differs. Across supported platforms, compare scientific arrays and decisions under predeclared tolerances and report drift rather than manufacturing byte equality.
10. High-precision test references use `Decimal` from string literals and local contexts, or exact `Fraction` where appropriate. These references are deliberately small and do not replace the float64 production specification. [Python Decimal](https://docs.python.org/3.12/library/decimal.html), retrieved 15 October 2026.

## 6. Scoped lint and format policy

The following is a **proposed fragment**, not a file installed by this planning session. It applies only to new S10 source/tests. It does not reformat predecessors or inherit root exceptions accidentally.

```toml
[tool.ruff]
target-version = "py312"
line-length = 110
preview = false

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "UP", "B", "SIM", "C4", "PIE", "RUF", "C90", "D"]

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.per-file-ignores]
"successors/frem_s10/tests/**" = ["D"]
```

All official checks pass an explicit config and source/test paths from the repository root, so relative per-file rules use that invocation context. Inspect `ruff check --show-files` during setup to prove the intended files are included. Ruff supports scoped configs and explicit `--config`; nearest configs are not implicitly merged. [Ruff configuration](https://docs.astral.sh/ruff/configuration/), retrieved 15 October 2026.

CI uses `ruff check` without mutation and `ruff format --check`. Review formatting diffs before freeze. Do not use `--unsafe-fixes`, blanket `noqa`, automatic suppression insertion or a broad exclusion to make a gate pass. A necessary exception identifies its rule, reason, reviewer and review date in the quality exception register. The formatter owns style; mathematical readability may justify a documented narrow lint exception.

## 7. Measurable maintainability without score gaming

| Check | Proposed acceptance | Anti-gaming interpretation |
|---|---|---|
| Strict typing | Zero reported errors in in-scope code; every suppression reviewed | Casting to `Any` or excluding a module is not a fix |
| Lint/format | Zero unexplained violations; no preview rules | Prior code is outside this change; do not waive new defects to preserve a global badge |
| Dependency graph | No runtime cycles, forbidden imports or import-time I/O | Moving a forbidden operation into a helper does not satisfy the boundary |
| Critical requirements | Every chronology, accounting, solver, integrity and consumption requirement has positive and rejecting evidence | Parameterizing the same easy case many ways does not increase demonstrated coverage |
| Complexity | McCabe >10 requires simplification or specific reviewed exception | Prefer readable event logic to arbitrary helper fragmentation |
| Size | Functions >80 nonblank lines or modules >500 trigger review | These are review triggers, not correctness claims or automatic bans |
| Public contracts | Each public entry point documents units, shape, time and failure | Docstring presence alone is insufficient |
| Duplication | One production implementation per mathematical rule; aliases explicit | Independent references must remain independently derived |
| Missing implementation | No silent `pass`, TODO-return, fabricated result or swallowed error on required paths | An explicit unsupported capability that raises is acceptable and reported |
| Suppressions/debt | Every waiver has owner, reason, affected tests and revisit condition | No quota of allowed bugs; scientific-critical debt blocks readiness |
| Tests | Every failure retained; unjustified skip/xfail blocks required coverage | Test count and coverage percentage are not targets to optimize |
| Review | Zero unresolved correctness/integrity blockers | Cosmetic suggestions do not become evidence of scientific quality |

Coverage is a **gap detector**, with source and branch reports retained. Initially propose >=90% branch coverage overall and 100% exercised feasible branches for maturity, fees, integrity and freeze consumption; each missed branch must receive a rationale and test plan. The exact percentage is subordinate to requirement coverage: a trivial test can execute a branch without checking it. Do not increase coverage by dropping files, excluding failure branches or broad `pragma: no cover`. An unreachable defense can be documented with a proof/review note rather than a false runtime claim. Coverage.py distinguishes branch opportunities from line execution. [Branch coverage](https://coverage.readthedocs.io/en/latest/branch.html), retrieved 15 October 2026.

## 8. Environment, lock, package and artifact policy

Separate groups contain `quality` tools (Ruff, mypy, pytest, Hypothesis, coverage), `build` tools, and optional report tools. Record exact package versions, hashes, interpreter, platform, architecture, libc/BLAS where applicable and timezone database identity. A resolved lock is necessary but does not itself certify reproducibility. No automatic upgrade happens during an experiment.

Use the S10-local lock only. Provisioning a development environment is a separately recorded engineering operation with resource admission. After a reviewed cache/wheelhouse is populated, verification runs offline; missing cached packages cause a clear blocked step, not an implicit network download. `uv sync --locked` checks lock/metadata consistency; `--frozen` alone skips that freshness check and is insufficient for this gate. `uv run --no-sync` avoids changing the prepared environment. [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/), retrieved 15 October 2026.

Build source and wheel distributions with the declared backend in an environment already containing its pinned build requirements. An offline `python -m build --no-isolation` is permitted only after that environment has been checked; missing requirements fail. Preserve metadata, wheel/sdist member list and artifact hashes. Inspect archives for accidental datasets, predecessor source, credentials, absolute workspace paths and execution receipts. Build CLI syntax is documented by [PyPA build](https://build.pypa.io/en/stable/), retrieved 15 October 2026.

Install the built wheel into a disposable clean environment using cached dependencies and run its smoke check outside the checkout, with `PYTHONPATH` unset. Verify `frem_s10.__file__` resolves into that environment and forbidden heavy dependencies are absent. Editable-install tests alone cannot certify the wheel. A lock upgrade or dependency patch creates a new environment receipt and focused numerical regression; it never silently rewrites an already consumed freeze.

Frozen artifacts bind source/config/data/environment hashes and schema versions. Source diffs may be dirty during development; freeze must bind exact bytes and disclose dirty state instead of pretending HEAD captures them. Text/array serialization has a documented canonical form, byte order and schema. Do not load pickle, object arrays or arbitrary class names from a checkpoint. Atomic writes use a temporary file on the same filesystem, validated content and a final publish step; incomplete temporary files cannot be mistaken for successful artifacts.

## 9. CI tiers and precise scope

These are local/offline workflow definitions first. A hosted CI workflow, if later added, runs only generated fixtures with read-only repository permissions and no secrets or market-data mounts. There is no scheduled research, market retraining, provider access or freeze consumption in CI. A marker is not a security boundary: collection paths and fixture loaders also exclude market data.

| Tier | Trigger and scope | Proposed initial resource ceiling | Required evidence |
|---|---|---|---|
| Q0 static | Each S10 code/doc change: syntax/schema, contracts inventory, lint, format, types, no forbidden imports | One process at a time; 120 s each tool, 512 MiB; no numerical imports for plain schema check | Commands, versions, scope, exit codes and unmodified predecessor identities |
| Q1 core | Relevant change: hand/reference, bounded property, chronology/accounting and rejection tests | One process/thread, 180 s total, 256 MiB | Test IDs, failure/minimized inputs, numerical residuals |
| Q2 integration | Before readiness and affected orchestration/persistence changes | Generated complete matrix and corruption/interruption tests; 600 s, 256 MiB | Complete identities, report regeneration, no network/data access |
| Q3 packaging/portability | Before first freeze, after dependency changes, and release | One job at a time; 600 s, 512 MiB per platform | Offline wheel/source build and fresh-environment smoke; Python 3.12 Linux/macOS checks when resources exist |
| Q4 targeted stress | Before first freeze and after affected performance work | One thread; 600 s, 256 MiB numerical worker; small fixed size schedule | Time/RSS/operation counts, regression ratios, exact/reference equivalence |
| Scientific work | Explicit separate preparation/evaluation directive, never CI | Its own protocol limits and >=512 MiB host floor | Data admission, actual freeze, consumed-attempt lifecycle and scientific receipt |

The ceilings are proposals to measure, not measurements. WP06 must pass the relevant Q2 artifact/consumption/resource/cancellation tests for its required production subset before any real precision preparation; those guards do not wait until the full experiment implementation. Precision preflight requires its declaration and guard/kernel verification, not its own future result; full-campaign preflight additionally requires the precision pass. Type/build tools may use more memory than the numerical runtime, hence their separate engineering envelope. If a tool cannot meet its proposed bound, preserve that failure and revise the engineering budget before another bounded check; never lower the scientific resource guard. Required tests that cannot run remain pending. No automatic rerun of a failed test job merely to obtain green: retain failure, diagnose, and rerun only after a justified correction or documented infrastructure recovery.

Tests register `unit`, `property`, `reference`, `integration`, `fault`, `slow_engineering` and `market` markers. Every CI command names only the appropriate S10 test directories and excludes market; market tests are not in ordinary collection and cannot be activated by changing `-m` alone. Strict markers reject misspelled categories. [pytest markers](https://docs.pytest.org/en/stable/how-to/mark.html), retrieved 15 October 2026.

## 10. Future command contract

Run from the repository root only after the planned project/test files and resolved environment exist. `uv` and all tools must have verified installed versions; these examples do not provision them today. A future runner must capture raw exit codes and stdout/stderr in fresh receipt paths.

```bash
rtk proxy uv lock --project successors/frem_s10 --check --offline
rtk proxy uv sync --project successors/frem_s10 --locked --offline --group quality --group build
rtk proxy uv run --project successors/frem_s10 --locked --offline --no-sync ruff check --config successors/frem_s10/pyproject.toml successors/frem_s10/src/frem_s10 successors/frem_s10/tests
rtk proxy uv run --project successors/frem_s10 --locked --offline --no-sync ruff format --check --config successors/frem_s10/pyproject.toml successors/frem_s10/src/frem_s10 successors/frem_s10/tests
rtk proxy uv run --project successors/frem_s10 --locked --offline --no-sync mypy --config-file successors/frem_s10/pyproject.toml successors/frem_s10/src/frem_s10 successors/frem_s10/tests
rtk proxy uv run --project successors/frem_s10 --locked --offline --no-sync python -m pytest -c successors/frem_s10/pyproject.toml --strict-markers successors/frem_s10/tests/unit successors/frem_s10/tests/contracts successors/frem_s10/tests/chronology successors/frem_s10/tests/property successors/frem_s10/tests/references
rtk proxy uv run --project successors/frem_s10 --locked --offline --no-sync python -m coverage run --branch --source=frem_s10 -m pytest -c successors/frem_s10/pyproject.toml --strict-markers successors/frem_s10/tests/unit successors/frem_s10/tests/contracts successors/frem_s10/tests/chronology successors/frem_s10/tests/property successors/frem_s10/tests/references successors/frem_s10/tests/integration successors/frem_s10/tests/fault
rtk proxy uv run --project successors/frem_s10 --locked --offline --no-sync python -m coverage report --show-missing
rtk proxy uv run --project successors/frem_s10 --locked --offline --no-sync python -m build --no-isolation successors/frem_s10
```

The second command mutates only the future isolated environment and belongs to provisioning, not every CI check. The actual implementation must set per-attempt coverage output/build directories rather than overwrite verification receipts. Tool flags have been checked against current primary documentation; availability and compatibility of any selected locked release still require later validation. [uv CLI](https://docs.astral.sh/uv/reference/cli/), retrieved 15 October 2026. No command in this section authorizes market execution or imports an S10 package during this design session.

## 11. Review and change control

Each implementation change includes: the requirement/equation changed, exact behavior, affected TEST IDs, negative/edge cases, resource impact, evidence locations and remaining limits. Review in small coherent stages: contracts and clock; scores/solver; policy/ledger; orchestration and integrity; analysis/report; package/operations. Avoid a single unreviewable implementation dump.

The author cannot self-certify an independent empirical reproduction. A second agent or colleague can perform an internal adversarial source/math review and must be labeled as such. Independent implementation or empirical reproduction is a later separately recorded activity. Reviewers explicitly try a simpler alternative, search for future access, challenge units and accounting, and inspect failure paths before approval.

Scientifically significant changes require the repository `record-decision` workflow and a new corrected version/freeze where applicable. Formatting, speedups and refactors still need behavioral equivalence evidence. A valid negative market result does not create a bug-fix justification. Runtime failure or interruption after market consumption is terminal for that attempt in revision 2; checkpoints support engineering replay and audit only. Any later scientific continuation requires a separately recorded recovery protocol and cannot be enabled by a hidden resume flag.

Readiness requires the complete [verification catalog](VERIFICATION_CATALOG.md), all critical review findings resolved, tested data/precision kernel plus production bootstrap/artifact/consumption/resource/cancellation subset before admitted pre-2010 preparation, full-system verification after precision feasibility, measured resources, and artifact-bound reproducibility instructions. A clean lint/type/coverage report supports engineering quality; it does not prove that S10 predicts markets, earns the 6% margin, satisfies the original density hypothesis or is perfect.
