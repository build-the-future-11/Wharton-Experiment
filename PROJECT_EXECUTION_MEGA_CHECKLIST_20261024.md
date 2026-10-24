# FREM project execution mega checklist

**Prepared:** 24 October 2026
**Canonical frontier:** FREM-S14 release 0.2.4
**Current scientific status:** `NOT_EVALUATED`
**Immediate state:** static release verified; local numerical admission failed; existing private Kaggle owner profile unresolved; 0/19 qualification arms and 0/171 historical cells completed for 0.2.4.

This is the ordered checklist for finishing the current project honestly. It complements the broader capability backlog in `RESEARCH_FEATURE_MEGA_CHECKLIST.md`. Work through the gates in order. A checked item must have the named evidence; source code, a green UI, a static parse, or a paper draft does not substitute for runtime or scientific evidence.

## Non-negotiable rules

- [x] Preserve every prior negative, inconclusive, invalid, failed, interrupted, superseded, and unrun artifact.
- [x] Preserve frozen S2-S8 outcomes and do not rerun them to seek a positive result.
- [x] Preserve S14 Versions 1-8 and releases 0.2.0-0.2.3 under their existing identities.
- [x] Treat Version 8 as a terminal semantic-verification failure; do not reinterpret its retained metrics.
- [x] Keep protected outcomes unopened until a separately frozen confirmation is authorized and ready.
- [ ] Never change the cohort, endpoint, threshold, arms, seeds, folds, bootstrap family, or failure denominator after viewing a valid result.
- [ ] Never drop a failed seed or cell from a declared family.
- [ ] Never overwrite a prior output directory, receipt, archive, notebook, decision, or state snapshot.
- [ ] Never call adjusted-close survivor-ETF evidence point-in-time, investable, confirmatory, or profitable evidence.
- [ ] Record every scientifically meaningful deviation with `record-decision` before affected execution.
- [ ] Update `RESEARCH_STATE.json`, `docs/CODEX_HANDOFF.md`, and `docs/STATIC_VERIFICATION.md` after each substantive gate.

## Gate 0 — Current truth and repository preservation

- [x] Confirm branch `codex/frem-research-execution-20261020`.
- [x] Confirm current local commit `b41711a636e1210c20fb394faf46c414d104dcf8`.
- [x] Confirm the branch is one commit ahead of its remote.
- [x] Confirm the worktree contains retained untracked October 21 evidence.
- [x] Inventory every untracked path by provenance, owner run, date, sensitivity, and intended disposition in `research/frem_s14_execution_20261024/GATE0_UNTRACKED_INVENTORY.json`.
- [ ] Verify every untracked Version 5-8 receipt and downloaded artifact against its recorded hashes.
- [x] Add only the reviewed checklist, state, documentation, inventory tooling, and session receipts; retained run directories remain unstaged.
- [x] Run a high-confidence secret-value scan over the exact proposed tracked set.
- [x] Verify no Kaggle token, browser cookie, private credential, or restricted raw input is present in the proposed tracked set.
- [x] Verify `git diff --cached --check` for the exact intended change set.
- [ ] Decide whether today’s 0.2.4 commit and separately reviewed evidence should be pushed; verify the remote SHA if pushed.
- [x] Reconcile canonical aliases: `current_design_revision` now resolves to exact FREM-S14 0.2.4.
- [x] Make the S14 active-candidate field unambiguous while retaining the historical 0.1.9 object.
- [x] Preserve the original `research/protocol.json` as FREM-P0 and explicitly point to the active S14 specification, historical-suite configuration, and release record rather than rewriting history.

**Gate 0 acceptance:** reviewed inventory, no secret findings, canonical aliases resolve to exact 0.2.4, intended repository state is reproducible, and no historical evidence is lost.

## Gate 1 — Exact 0.2.4 release identity

- [x] Preserve unrun 0.2.0 after module/wheel and packaged-budget disagreement.
- [x] Preserve unrun 0.2.1 after release-scope quality failures.
- [x] Preserve unrun 0.2.2 after corrections were absent from the immutable archive.
- [x] Preserve unrun 0.2.3 after invocation-dependent lint and non-fail-fast build behavior.
- [x] Bind 0.2.4 source identity `cba3c5c4bb9d0f4ab3ae12d85b465baa13d100c8b36fe15c2c91331844c706e2`.
- [x] Bind wheel SHA-256 `57ae8e68b83eef921d83d58f2976f040a13281f3c505dfe6c98c7c4eb9d96486`.
- [x] Bind archive SHA-256 `ddc04e93b37a086684624a2ad8d03de9269df1fa7f23ed14b9fcc8b2943f7c21`.
- [x] Bind notebook SHA-256 `e8fb05b52ee26f6a3045280e92bde247f85b381b7702bbc2fce4f4d97755670f`.
- [x] Verify module and wheel version equality.
- [x] Verify all 15 installed modules and 17 archive payload hashes.
- [x] Verify packaged/runtime fold ceiling is 4,000 seconds and notebook ceiling is 13,500 seconds.
- [x] Verify private visibility, GPU setting, Internet-off setting, notebook syntax, and embedded release equality.
- [x] Pass 59 dependency-free source tests with zero failures/errors.
- [x] Pass the same 59 tests against the extracted clean wheel.
- [x] Pass checkout and extracted-payload Ruff lint and format checks.
- [x] Rebuild the wheel at a fixed epoch with byte-identical SHA-256.
- [ ] Re-read the immutable release files immediately before upload and refuse any hash drift.

**Gate 1 acceptance:** `RELEASE_VERIFICATION.json` and `ENGINEERING_RECEIPT.json` both identify the same immutable 0.2.4 artifacts and every pre-upload check remains current.

## Gate 2 — Execution access and resource admission

### Local path

- [x] Record the failed local samples: 330.234375, 358.359375, and 331.6875 MiB available against the 512 MiB floor.
- [x] Record rising pageouts: 342665, 342757, and 342774.
- [x] Stop before importing NumPy, SciPy, Torch, or XGBoost.
- [ ] If local execution is reconsidered, obtain three fresh samples at or above 512 MiB with stable pageouts.
- [ ] Bind any passing local receipt to the exact 0.2.4 source, interpreter, environment, and command.
- [ ] Continue monitoring the resource floor throughout any admitted local numerical work.

### Kaggle path — immediate blocker

- [ ] User opens the browser profile already authenticated as Kaggle owner `ryangomez010`.
- [ ] Confirm the visible Kaggle account identity before any upload.
- [ ] Confirm the target notebook is private.
- [ ] Confirm Internet is disabled.
- [ ] Confirm the intended GPU accelerator and available quota.
- [ ] Confirm the fixed input source is attached under the expected read-only path.
- [ ] Confirm no equivalent 0.2.4 run is already queued, running, or complete.
- [ ] Confirm no temporary token is needed; if one becomes necessary, use the existing explicit token lifecycle and retain revocation evidence.
- [ ] Record an access/readiness receipt before transmission.

**Gate 2 acceptance:** either a fresh exact-source local admission passes or the existing private Kaggle owner, quota, accelerator, Internet, input mount, and duplicate-run state are verified.

## Gate 3 — Upload and saved-source readback

- [ ] Upload only `frem_s14_kaggle_bundle-0.2.4.zip` and the exact 0.2.4 notebook/metadata.
- [ ] Record upload timestamps, Kaggle notebook slug, saved-version number, and script version ID.
- [ ] Verify the saved notebook remains private.
- [ ] Download or use the supported API to read back the saved source.
- [ ] Compare saved notebook cells with the local notebook byte-for-byte or by the declared normalized-cell method.
- [ ] Verify the saved embedded archive equals the local archive.
- [ ] Verify wheel, configuration, release record, and input expectations from saved source.
- [ ] Confirm no Kaggle editor mutation, stale draft, or alternate account changed the payload.
- [ ] Record a saved-identity receipt before starting numerical execution.
- [ ] Refuse launch if any identity differs; preserve the mismatched saved version as failed evidence.

**Gate 3 acceptance:** private saved-version identity independently matches all bound 0.2.4 hashes.

## Gate 4 — Target runtime and numerical qualification

- [ ] Record exact Python, NumPy, SciPy, Torch, CUDA, cuDNN, XGBoost, BLAS, driver, GPU, CPU, RAM, and filesystem facts.
- [ ] Confirm every version satisfies the frozen 0.2.4 runtime contract.
- [ ] Run the complete warning-strict target test suite.
- [ ] Retain JUnit/test logs, per-test outcomes, elapsed time, peak memory, and failures.
- [ ] Run finite-grid diffusion matrix exponential and moment references.
- [ ] Run reservoir contraction and spectral-feature references.
- [ ] Run mature-only engram and graph-causality tests.
- [ ] Run JEPA mask, covariance/variance, and gradient tests.
- [ ] Run Student-mixture NLL, Brier, coverage, moment, and ordered-aggregation tests.
- [ ] Run train-only deterministic center/MAD/target-scale tests.
- [ ] Run CUDA/AMP equivalence tests at frozen tolerances.
- [ ] Run checkpoint interruption/resume and epoch-boundary equivalence tests.
- [ ] Run retrieval-provenance and semantic-readback numerical tests.
- [ ] Complete all 19 generated qualification arms.
- [ ] Preserve arm/seed status for all 19 arms, including failures and not-started arms.
- [ ] Run the in-target qualification verifier.
- [ ] Download and independently verify the qualification bundle.
- [ ] Reject market execution if any required test, arm, identity, or verifier fails.

**Gate 4 acceptance:** all required target tests pass, 19/19 arms complete, producer and independent qualification verification pass, and the exact bundle is retained.

## Gate 5 — Historical-proxy intake

- [ ] Verify exact retained source bytes and hashes before parsing.
- [ ] Verify the prospective seven-ETF cohort; do not reintroduce QQQ or intersect calendars silently.
- [ ] Verify every source file maps to exactly one canonical ETF path.
- [ ] Reject duplicate basename, stock/ETF taxonomy, missing file, extra file, and foreign mount conditions.
- [ ] Verify all seven calendars are exactly compatible under the frozen contract.
- [ ] Verify OHLCV schema, timestamps, finite values, open interest, and hostile-value checks.
- [ ] Verify no row, date, price, return, or terminal outcome was filled or invented.
- [ ] Verify source rights/provenance and document the adjusted-close survivor-proxy limitation.
- [ ] Freeze the admitted input inventory and parsing receipt before fitting.
- [ ] Reconstruct chronological prefixes, purges, embargoes, target maturity, and folds independently.
- [ ] Verify all preprocessing is train-only and deterministic across target/runtime implementations.
- [ ] Verify baseline and ablation information parity.
- [ ] Refuse execution on any unresolved intake, calendar, split, maturity, or provenance defect.

**Gate 5 acceptance:** immutable admitted-input archive and independent intake/split/preprocessing receipts pass for all seven assets.

## Gate 6 — Fixed 171-cell historical-proxy suite

- [ ] Compile the exact registry: 19 arms × 3 seeds × 3 folds = 171 cells.
- [ ] Verify registry hash, configuration hash, source hash, input hash, and environment hash are embedded in every cell.
- [ ] Verify all cell output paths are fresh and non-overwriting.
- [ ] Start only after Gates 0-5 pass.
- [ ] Enforce the 4,000-second per-fold and 13,500-second notebook ceilings.
- [ ] Enforce the declared global resource and storage budgets.
- [ ] Preserve every checkpoint, log, warning, exception, timeout, interruption, and recovery action.
- [ ] Permit only exact checkpoint resume; reject restart with altered identity.
- [ ] Record each cell as `COMPLETE`, `FAILED`, or `NOT_STARTED`.
- [ ] Never remove a failed cell from the denominator.
- [ ] Retain raw predictive laws, score vectors, targets, fold metadata, preprocessing facts, trace evidence, and model/resource metadata.
- [ ] Retain forecast-policy ledgers and cost accounting separately from predictive scores.
- [ ] Complete all declared seeds before family comparison.
- [ ] Stop and classify the suite as incomplete if the notebook-wide deadline prevents required completion.
- [ ] Do not rank or interpret arms before the complete-family and verifier gates pass.

**Gate 6 acceptance:** immutable inventory accounts for all 171 cells and the complete declared families required by the protocol are validly complete.

## Gate 7 — Semantic and statistical verification

- [ ] Run the target-runtime semantic verifier on every completed cell.
- [ ] Run the separately controlled independent verifier from downloaded artifacts.
- [ ] Reconstruct train-only center, MAD scale, target scale, labels, folds, and preprocessing.
- [ ] Recompute per-observation NLL, directional Brier score, coverage, and other declared scores.
- [ ] Recompute scalar summaries from persisted vectors using ordered `math.fsum` reductions.
- [ ] Verify every retained forecast law is finite, normalized, correctly shaped, and bound to the intended target.
- [ ] Reconstruct allocation, costs, drift, liquidation, wealth, drawdown, turnover, and certainty-equivalent arithmetic.
- [ ] Verify repeated seeds are aggregated before fold/date inference as frozen.
- [ ] Recompute paired moving-block intervals, centered-bootstrap null tests, and Holm correction.
- [ ] Verify the practical threshold remains 0.01 nats per observation.
- [ ] Deliberately corrupt one copied producer artifact and verify the independent verifier rejects it.
- [ ] Produce a discrepancy table for every tolerance-bound comparison.
- [ ] Mark the entire affected evidence invalid if preprocessing or summary reconstruction does not meet its frozen tolerance.
- [ ] Never loosen tolerance in response to an observed discrepancy.

**Gate 7 acceptance:** both semantic verifiers reproduce the admitted suite within the frozen laws and tolerances.

## Gate 8 — Result classification and failure analysis

- [ ] Lock the verified raw evidence before reading comparative summaries.
- [ ] Calculate the primary control-NLL-minus-candidate-NLL effect and paired interval.
- [ ] Apply the frozen 0.01-nat practical threshold.
- [ ] Require complete seeds and valid integrity evidence.
- [ ] Classify the primary result exactly as `SUPPORTED`, `MIXED`, `NEGATIVE`, `INCONCLUSIVE`, or `INVALID`.
- [ ] Report every baseline and ablation, including those outperforming FREM.
- [ ] Report fold and seed heterogeneity without selecting favorable subsets.
- [ ] Report calibration, tail, coverage, runtime, memory, and parameter evidence.
- [ ] Report proxy portfolio evidence separately from forecast evidence.
- [ ] Compare proxy wealth with cash, equal-weight assets, risk-only controls, and declared costs.
- [ ] Analyze mechanism traces without treating behavior as proof of efficacy.
- [ ] Document simpler explanations: compute, parameters, data access, optimization, control weakness, and survivor bias.
- [ ] Produce an explicit limitations and invalid-inference list.
- [ ] Preserve a valid negative or inconclusive result as terminal for 0.2.4.
- [ ] If invalid, identify the exact validity defect, record a decision, and create a new version before affected rerun.

**Gate 8 acceptance:** one evidence-backed terminal disposition with no outcome-driven protocol change.

## Gate 9 — Current manuscript and evidence products

- [ ] Build a claim ledger: claim → experiment → raw artifact → analysis → figure/table.
- [ ] Generate the primary comparison table directly from verified arrays.
- [ ] Generate complete arm/ablation tables with uncertainty and failure counts.
- [ ] Generate calibration, interval, tail, and fold/seed heterogeneity figures.
- [ ] Generate runtime, peak-memory, parameter, and throughput tables.
- [ ] Generate retrieval, router, reservoir, neurogenesis, and diffusion behavior figures from traces.
- [ ] Generate the proxy economic table with adjusted-close and idealized-execution limitations attached.
- [ ] Generate a negative-results and failed-execution appendix from receipts.
- [ ] Update abstract, results, discussion, limitations, conclusion, and reproducibility statement.
- [ ] Remove or weaken every claim that exceeds the result disposition.
- [ ] Keep novelty at candidate/application level unless the literature audit supports more.
- [ ] Verify every numerical value in the manuscript against machine-readable evidence.
- [ ] Verify every citation against the statement it supports.
- [ ] Compile the paper from clean sources.
- [ ] Render every page and inspect tables, figures, references, overfull boxes, clipping, and legibility.
- [ ] Produce a paper-build and visual-inspection receipt.

**Gate 9 acceptance:** a compiled and visually inspected manuscript whose claims and numbers trace to verified evidence.

## Gate 10 — Point-in-time scientific confirmation

The S14 ETF suite is exploratory historical-proxy work even if it passes. It cannot by itself complete the original scientific claim.

- [ ] Specify the exact scientific population and intended scope of inference.
- [ ] Acquire rights/provenance-verified point-in-time identifiers and membership.
- [ ] Include inactive securities, delistings, mergers, identifier changes, and terminal outcomes.
- [ ] Reconcile splits, distributions, corporate actions, executable prices, spreads, impact, and FX where relevant.
- [ ] Retain original publication and revision timestamps for macro, filing, or event features.
- [ ] Prohibit current-membership and current-vintage backfilling.
- [ ] Build and test a ragged-panel lifecycle adapter.
- [ ] Freeze training, development, validation, OOD, and unused confirmation boundaries before outcomes.
- [ ] Freeze the primary endpoint, practical threshold, baselines, seeds, repetitions, compute budget, and failure policy.
- [ ] Establish a contamination ledger proving the confirmation source is unused for redesign.
- [ ] Run development only within its fixed budget.
- [ ] Freeze the final candidate and hashes before confirmation access.
- [ ] Execute the untouched confirmation once.
- [ ] Preserve and classify the confirmation regardless of sign.
- [ ] Do not substitute proxy, engineering, or same-data replay evidence for confirmation.

**Gate 10 acceptance:** a valid untouched confirmation tied to point-in-time data and a prospectively frozen protocol.

## Gate 11 — Independent reproduction

- [ ] Create a clean-host reproduction package with exact source, wheel, environment, configuration, sanitized inputs, and verifier.
- [ ] Ensure the reproducer has no mutable access to the producer run directory.
- [ ] Rebuild or install in the declared clean environment.
- [ ] Reconstruct raw-to-analysis evidence without copying producer summaries as truth.
- [ ] Regenerate tables and figures.
- [ ] Reconcile every discrepancy explicitly.
- [ ] Retain the independent operator/environment receipt.
- [ ] Bind the reproduction outcome to the same terminal scientific disposition.

**Gate 11 acceptance:** independent evidence reconstruction succeeds or produces a bounded, disclosed discrepancy that limits the claim.

## Gate 12 — Release and publication readiness

- [ ] Freeze the final scientific source, protocol, data instructions, environment, results, paper, supplement, figures, and tables.
- [ ] Generate complete artifact manifests and SHA-256 inventories.
- [ ] Verify licenses and redistribution rights for code, data, figures, and third-party assets.
- [ ] Remove credentials, private data, transient caches, and nonredistributable artifacts.
- [ ] Verify authorship, acknowledgments, funding, conflicts, ethics, and data-use statements from real records.
- [ ] Audit current venue rules, page limits, formatting, anonymization, supplementary-material, and artifact requirements.
- [ ] Build and inspect the venue package.
- [ ] Build and inspect the arXiv package if separately authorized.
- [ ] Run final claim, citation, equation, table, figure, statistic, and reproducibility audits.
- [ ] Freeze the release commit/tag only after exact remote verification.
- [ ] Record the final scope and disposition in canonical state.
- [ ] Obtain separate authorization before public submission or external delivery.

**Gate 12 acceptance:** exact, independently verifiable release package with a defensible terminal conclusion and current venue compliance.

## Gate 13 — Improvements and new feature backlog

These additions are not permission to mutate the frozen 0.2.4 release. Read-only external verifiers and viewers may be developed against copied artifacts if they do not alter execution. Anything that changes source, model behavior, data, runtime, statistics, budgets, or outputs requires a new immutable version; anything that changes the scientific hypothesis or evaluation requires a separately frozen successor.

Priority meanings:

- `P0`: evidence integrity, execution safety, or reproducibility work needed before broad claims or release.
- `P1`: high-value capability to implement only after the current 0.2.4 disposition is locked.
- `P2`: exploratory successor research requiring a new hypothesis, falsifier, protocol, budget, and version.

### A. Evidence integrity and campaign reliability

- [ ] **NF-A01 — Artifact transparency log.** `P0` Append every source, input, cell, checkpoint, verifier, analysis, figure, and paper artifact to a hash-chained, append-only ledger; acceptance is a verifier that detects deletion, insertion, reordering, and mutation.
- [ ] **NF-A02 — Signed terminal receipts.** `P0` Sign final campaign and reproduction manifests with a project release key while retaining ordinary hashes; acceptance is offline signature verification with documented key custody and rotation.
- [ ] **NF-A03 — Two-phase artifact commit.** `P0` Write outputs to a staging directory, verify completeness and hashes, then atomically publish a terminal manifest; acceptance is fault-injection proving partial outputs cannot appear complete.
- [ ] **NF-A04 — Campaign state machine model.** `P0` Define legal transitions for prepared, admitted, running, interrupted, resumed, failed, complete, verified, invalid, and released states; acceptance is rejection of impossible or backward transitions.
- [ ] **NF-A05 — Evidence garbage-collection guard.** `P0` Add a dry-run inventory that refuses deletion of any referenced artifact, receipt, freeze, or failure; acceptance is an orphan/reference report with no destructive default.
- [ ] **NF-A06 — Foreign-artifact quarantine.** `P0` Move unmatched outputs into a read-only quarantine index rather than silently ignoring them; acceptance is provenance classification and explicit promotion criteria.
- [ ] **NF-A07 — Receipt schema versioning.** `P0` Give every receipt a schema ID, validator, migration policy, and unknown-field behavior; acceptance is validation of every retained S10-S14 receipt or a documented legacy exception.
- [ ] **NF-A08 — Clock and timezone integrity.** `P0` Record monotonic duration plus UTC wall time and detect clock jumps; acceptance is a synthetic time-warp test that cannot falsify deadlines or ordering.
- [ ] **NF-A09 — Storage exhaustion preflight.** `P0` project worst-case checkpoints, logs, laws, traces, and downloads before launch; acceptance is fail-closed refusal when durable free space is insufficient.
- [ ] **NF-A10 — Duplicate-run fingerprint service.** `P0` Compare candidate, account, notebook, input, registry, and execution identities before launch; acceptance is detection of queued, running, terminal, renamed, and forked equivalents.

### B. Data quality, provenance, and point-in-time support

- [ ] **NF-B01 — Identifier history graph.** `P1` Track ticker, exchange, CUSIP/ISIN or provider ID, company, merger, spin-off, rename, and delisting intervals; acceptance is a dated graph with no current-identity backfill.
- [ ] **NF-B02 — Point-in-time universe engine.** `P1` Reconstruct eligible assets at each decision time from effective-dated membership evidence; acceptance is a membership replay with inactive and later-listed controls.
- [ ] **NF-B03 — Corporate-action reconciliation engine.** `P1` Reconcile splits, reverse splits, cash/stock dividends, rights, mergers, spin-offs, tender events, and delisting proceeds; acceptance is provider-level accounting with unresolved events rejected.
- [ ] **NF-B04 — Executable-price layer.** `P1` Separate observed close, adjusted close, bid, ask, volume, market impact, and execution timestamp; acceptance is a ledger that never labels adjusted-close changes as executable trades.
- [ ] **NF-B05 — Multi-currency total-return accounting.** `P1` Add dated FX sources, conversion timing, cash balances, withholding, and currency attribution; acceptance is independently reconstructed base-currency wealth.
- [ ] **NF-B06 — Vintage-aware macro store.** `P1` Preserve first release, revision sequence, publication timestamp, and observation period; acceptance is as-of queries that never return a future revision.
- [ ] **NF-B07 — Filing availability pipeline.** `P1` Retain accession, original acceptance time, amended filing links, issuer mapping, and parser provenance; acceptance is an as-of join audit with late/amended examples.
- [ ] **NF-B08 — News/event source contract.** `P2` Admit news only with original publication time, edit history, entity mapping, rights, and missing-event policy; acceptance is a leakage test and source-specific license receipt.
- [ ] **NF-B09 — Missingness taxonomy.** `P1` Distinguish not listed, market closed, provider gap, suspended, halted, stale, unavailable, structurally absent, and parse failed; acceptance is explicit masks and no generic imputation.
- [ ] **NF-B10 — Calendar federation.** `P1` Model exchange sessions, half-days, holidays, time zones, daylight saving, auction times, and cross-market asynchrony; acceptance is adversarial multi-market calendar tests.
- [ ] **NF-B11 — Data drift and correction monitor.** `P1` Compare new provider snapshots with prior bytes without overwriting them; acceptance is a dated row-level correction report and downstream impact map.
- [ ] **NF-B12 — Dataset rights and redistribution matrix.** `P0` Bind each dataset and derivative to license, access date, allowed uses, retention, redistribution, and publication limits; acceptance is release refusal for unresolved rights.

### C. Model and mechanism successor ideas

- [ ] **NF-C01 — Confidence-aware memory write gate.** `P2` Store episodes only when delayed evidence exceeds a train-only reliability threshold; acceptance is a capacity-matched FIFO/uniform/full-memory comparison under a new freeze.
- [ ] **NF-C02 — Regime-conditional memory banks.** `P2` Separate retrieval banks using an input-only latent regime posterior; acceptance is no-label-leakage tests and a single-bank matched control.
- [ ] **NF-C03 — Hierarchical memory timescales.** `P2` Allocate short-, medium-, and long-horizon banks with fixed total capacity; acceptance is a capacity-matched flat-bank comparison and horizon-specific retrieval audit.
- [ ] **NF-C04 — Learned memory eviction with delayed credit.** `P2` Train eviction from matured counterfactual score contribution rather than recent outcome magnitude; acceptance is a prospective credit definition and exact replay.
- [ ] **NF-C05 — Retrieval diversity constraint.** `P2` Penalize near-duplicate retrieved episodes while preserving causal eligibility; acceptance is diversity/coverage/score trade-off against unconstrained retrieval.
- [ ] **NF-C06 — Uncertainty-aware graph edges.** `P2` Weight graph transitions by similarity uncertainty and support, with a no-edge fallback; acceptance is calibrated edge-confidence tests and graph ablations.
- [ ] **NF-C07 — Causal graph lag experiment.** `P1` Shift or permute neighbor information to distinguish genuine lagged cross-asset signal from contemporaneous leakage; acceptance is a frozen lag/permutation family.
- [ ] **NF-C08 — Sparse interpretable router.** `P2` Compare capped soft routing with sparsemax/entmax or monotone rule-constrained routing; acceptance is matched compute, stability, calibration, and expert-use evidence.
- [ ] **NF-C09 — Bayesian or ensemble router uncertainty.** `P2` Propagate uncertainty in expert weights rather than using a single point gate; acceptance is proper-score and calibration comparison without extra test tuning.
- [ ] **NF-C10 — Joint multi-horizon scenario law.** `P2` Generate coherent one-, five-, and twenty-step paths whose marginals agree; acceptance is marginal-consistency, energy/variogram, and horizon-purge tests.
- [ ] **NF-C11 — Copula or normalizing-flow residual law.** `P2` Compare the finite Student mixture with a joint tail-dependent law under a fixed parameter/compute ceiling; acceptance is tail calibration and strong simple-law controls.
- [ ] **NF-C12 — Extreme-loss support correction.** `P2` Use a return representation and predictive support compatible with total loss and terminal events; acceptance is hand-verifiable bankruptcy/delisting cases.
- [ ] **NF-C13 — Adaptive spectral bank under regularization.** `P2` Learn frequencies with anti-collapse, bandwidth, and causality constraints; acceptance is fixed-Fourier, random-feature, and no-spectral controls.
- [ ] **NF-C14 — Reservoir stability certificate.** `P1` Produce per-checkpoint contraction/spectral-radius bounds and state-norm alarms; acceptance is a certificate that rejects unstable weights before inference.
- [ ] **NF-C15 — Neurogenesis budget controller.** `P2` Allocate neuron recruitment under a fixed global capacity and compute budget; acceptance is matched fixed-width and heuristic-recruitment controls.
- [ ] **NF-C16 — Mechanism distillation.** `P2` Distill the full model into a simpler law while retaining traceable teacher/student errors; acceptance is proof that any gain is not achievable by the simpler student under equal information.

### D. Training, optimization, and numerical robustness

- [ ] **NF-D01 — Deterministic kernel audit.** `P0` Inventory nondeterministic CUDA/BLAS operations and either replace, bound, or record them; acceptance is repeated-run tolerance evidence on the target.
- [ ] **NF-D02 — Gradient-health dashboard.** `P1` Record per-module norm, clipping rate, nonfinite rate, update-to-weight ratio, and saturation; acceptance is generated solely from retained training traces.
- [ ] **NF-D03 — Optimization failure classifier.** `P1` Classify divergence, collapse, dead routing, inactive memory, overfit, underfit, numerical overflow, and timeout without using protected metrics; acceptance is labeled synthetic fixtures.
- [ ] **NF-D04 — Compute-matched early stopping.** `P1` Allocate equal validation access and wall/step budgets across neural controls; acceptance is a parity ledger that rejects hidden extra tuning.
- [ ] **NF-D05 — Precision ladder.** `P1` Compare float64 reference, float32, TF32, bfloat16, and mixed precision at frozen tolerances; acceptance is per-operation and end-to-end error/resource tables.
- [ ] **NF-D06 — Memory-efficient checkpointing.** `P1` Add activation checkpointing or recomputation only after equivalence tests; acceptance is identical scientific outputs within tolerance and measured savings.
- [ ] **NF-D07 — Streaming feature construction.** `P1` Process bounded chronological chunks without changing split or preprocessing semantics; acceptance is byte/tolerance equivalence with the nonstreaming reference.
- [ ] **NF-D08 — Adaptive batch sizing behind a fixed rule.** `P2` Select batch size from pre-run resource facts rather than outcomes; acceptance is a deterministic policy and matched effective optimization budget.
- [ ] **NF-D09 — Robust optimizer comparison.** `P2` Compare AdamW with one or two predeclared alternatives under identical search budgets; acceptance is complete seeds and no optimizer cherry-picking.
- [ ] **NF-D10 — Numerical condition monitoring.** `P1` Log covariance condition number, diffusion eigenspectrum, scale floors, mixture degeneracy, and solver residuals; acceptance is fail-closed thresholds justified before execution.
- [ ] **NF-D11 — Exact resume fuzzing.** `P0` Interrupt at randomized safe boundaries across training, scoring, export, and verification; acceptance is no duplicate charge and final equivalence.
- [ ] **NF-D12 — Resource-aware sharding planner.** `P1` Produce a deterministic shard plan from measured qualification time, memory, storage, and quota; acceptance is no cross-shard duplicate or missing registry cell.

### E. Evaluation, statistics, and falsification

- [ ] **NF-E01 — Negative-control labels.** `P1` Add time-shuffled, asset-permuted, and impossible-future controls that must show no valid advantage; acceptance is predeclared failure if they appear predictive.
- [ ] **NF-E02 — Leakage canary features.** `P0` Inject controlled future-only canaries in tests and prove admission rejects or performance audit detects them; acceptance is a retained adversarial receipt.
- [ ] **NF-E03 — Placebo decision times.** `P1` Evaluate the same pipeline on predeclared non-event/placebo dates; acceptance is a frozen comparison that cannot select dates from outcomes.
- [ ] **NF-E04 — Block-length adequacy study.** `P1` Estimate dependence diagnostics using development data and freeze the allowed inference family before confirmation; acceptance is a sensitivity report without choosing favorable intervals.
- [ ] **NF-E05 — Hierarchical uncertainty decomposition.** `P1` Separate date, asset, fold, seed, and model variation while keeping the decision date as the primary unit; acceptance is a declared model and simulation calibration.
- [ ] **NF-E06 — Sequential-testing guard.** `P0` Track every viewed comparison and prevent optional stopping or repeated protected looks; acceptance is a consumption ledger tied to analysis identities.
- [ ] **NF-E07 — Practical-equivalence region.** `P1` Report evidence for meaningful superiority, inferiority, and practical equivalence around the frozen threshold; acceptance is defined before new outcomes.
- [ ] **NF-E08 — Baseline dominance audit.** `P0` Require comparison with the strongest simple and learned control, not only no-memory; acceptance is a complete baseline ranking with parity failures exposed.
- [ ] **NF-E09 — Robustness budget compiler.** `P1` Turn declared perturbations into a finite registry with multiplicity handling and cost projection; acceptance is immutable enumeration before execution.
- [ ] **NF-E10 — Cross-policy forecast attribution.** `P1` Cross forecast laws and allocation policies to separate forecast skill from policy luck; acceptance is a complete law-by-policy ledger matrix.
- [ ] **NF-E11 — Economic break-even analysis.** `P1` Calculate spread, fee, impact, turnover, leverage, and tax break-even points without claiming actual executability; acceptance is input-sensitivity tables and explicit assumptions.
- [ ] **NF-E12 — Worst-case missing-cell bounds.** `P1` Bound conclusions under failed or unavailable cells rather than dropping them; acceptance is conservative sensitivity with the full denominator.
- [ ] **NF-E13 — Model confidence calibration.** `P1` Test whether declared confidence tracks proper-score and coverage performance out of time; acceptance is reliability evidence and an abstention threshold frozen on development data.
- [ ] **NF-E14 — External benchmark reproduction.** `P1` Reproduce published comparator results only where data, code, units, and split identities are available; acceptance is a discrepancy report rather than assumed literature numbers.

### F. Performance, platform, and cost engineering

- [ ] **NF-F01 — Stage-level profiler.** `P0` Measure intake, features, fit, inference, scoring, verification, compression, and export separately; acceptance is per-stage wall, CPU, GPU, RAM, VRAM, and I/O evidence.
- [ ] **NF-F02 — Cost estimator with uncertainty.** `P1` Predict runtime, quota, and storage from qualification repeats using conservative intervals; acceptance is calibration against completed campaigns.
- [ ] **NF-F03 — Kernel fusion audit.** `P1` Fuse only proven hotspots and retain a scalar/reference path; acceptance is numerical equivalence and measured target speedup.
- [ ] **NF-F04 — Vectorization correctness harness.** `P0` Compare each vectorized scoring, label, and policy routine with a small scalar implementation; acceptance is property-based adversarial testing.
- [ ] **NF-F05 — I/O and archive compression benchmark.** `P1` Evaluate chunking/compression without losing raw evidence or random access; acceptance is byte-preserving decompression and measured savings.
- [ ] **NF-F06 — Multi-host portability matrix.** `P1` Verify supported CPU/GPU operating systems and explicitly reject unsupported combinations; acceptance is environment-bound receipts, not assumed portability.
- [ ] **NF-F07 — Quota-aware campaign scheduler.** `P1` Schedule immutable shards within declared account limits without silently shrinking the matrix; acceptance is complete registry accounting and no duplicate charge.
- [ ] **NF-F08 — Preemption-safe cloud adapter.** `P1` Detect impending termination, checkpoint once, publish a durable receipt, and resume exactly; acceptance is forced-preemption testing.
- [ ] **NF-F09 — Energy/carbon measurement.** `P2` Record available device energy and region assumptions as an efficiency metric; acceptance is transparent uncertainty and no precision beyond measured support.
- [ ] **NF-F10 — Minimal reference implementation.** `P1` Maintain a slow, simple, auditable implementation for small cases; acceptance is equality with optimized paths and use by the independent verifier.

### G. Research interface, inspection, and collaboration

- [ ] **NF-G01 — Read-only campaign dashboard.** `P1` Display identities, cell states, budgets, failures, and verification without rerun controls; acceptance is generation solely from immutable manifests.
- [ ] **NF-G02 — Split and maturity timeline explorer.** `P1` Visualize training, validation, test, embargo, label maturity, and holdings intervals; acceptance is equality with machine-audited boundaries.
- [ ] **NF-G03 — Prediction-law inspector.** `P1` Show expert distributions, mixture moments, scenarios, controls, and outcome authorization for one decision; acceptance is artifact-only rendering.
- [ ] **NF-G04 — Engram provenance explorer.** `P1` Trace retrieved episodes, maturity, asset, similarity, age, weights, graph paths, and counterfactual credit; acceptance is rejection of future or unknown episodes.
- [ ] **NF-G05 — Failure browser.** `P1` Index every intake, test, qualification, model, checkpoint, timeout, verifier, and release failure; acceptance is no default filter that hides negative evidence.
- [ ] **NF-G06 — Claim-to-evidence browser.** `P1` Navigate from manuscript sentence to table/figure, analysis, raw arrays, cell, environment, source, and input; acceptance is detection of orphan claims.
- [ ] **NF-G07 — Protocol diff viewer.** `P1` Render scientific versus engineering differences between versions; acceptance is automatic flagging of cohort, endpoint, threshold, seed, fold, model, data, and budget changes.
- [ ] **NF-G08 — Reproduction wizard.** `P1` Generate exact clean-host steps from a release manifest while refusing unsupported environments; acceptance is an independent dry run.
- [ ] **NF-G09 — Reviewer audit bundle.** `P1` Package the protocol, flaw ledger, negative results, artifact map, statistics, and minimal replay for adversarial review; acceptance is no dependency on mutable local state.
- [ ] **NF-G10 — Session handoff generator.** `P0` Generate a concise handoff from canonical state and receipts, separating facts, claims, blockers, and next gate; acceptance is schema validation and no inferred results.

### H. Security, privacy, and release hardening

- [ ] **NF-H01 — Secret scanning at source and archive boundaries.** `P0` Scan staged files, notebooks, logs, ZIP members, wheel metadata, and generated papers; acceptance is fail-closed release on any unresolved finding.
- [ ] **NF-H02 — Credential lifecycle receipts.** `P0` Record creation, scope, use, revocation, and local removal for temporary tokens without storing the secret; acceptance is post-use invalidation evidence.
- [ ] **NF-H03 — Dependency provenance and SBOM.** `P0` Record package names, versions, hashes, licenses, source indexes, and transitive dependencies; acceptance is offline installation from approved artifacts.
- [ ] **NF-H04 — Vulnerability audit with scientific-impact triage.** `P1` Separate exploitable release risks from irrelevant build findings and record remediation effects; acceptance is an auditable disposition.
- [ ] **NF-H05 — Archive path and symlink defense.** `P0` Reject traversal, absolute paths, links, special files, duplicate names, and decompression bombs; acceptance is adversarial archive fixtures.
- [ ] **NF-H06 — Private-data egress test.** `P0` Prove the execution path cannot transmit input bytes or credentials when Internet is declared off; acceptance is target-observed denial and output inventory review.
- [ ] **NF-H07 — Reproducible container or environment image.** `P1` Bind an immutable environment image where the host permits it; acceptance is source-to-image provenance and clean execution parity.
- [ ] **NF-H08 — Release permission matrix.** `P0` Label artifacts public, private, redistributable, derived-only, credential-sensitive, or prohibited; acceptance is automated export filtering plus manual review.

### I. Paper, visualization, and scientific communication

- [ ] **NF-I01 — Auto-generated methods appendix.** `P1` Render dimensions, hyperparameters, budgets, arms, seeds, folds, tolerances, and failure policy directly from frozen configs; acceptance is no hand-copied values.
- [ ] **NF-I02 — Machine-checked result prose.** `P1` Parameterize core result sentences from the claim ledger and terminal disposition; acceptance is build failure on unsupported wording or stale numbers.
- [ ] **NF-I03 — Architecture diagram from schema.** `P1` Generate model and evidence-flow diagrams from declared components and artifact edges; acceptance is consistency with source/config inspection.
- [ ] **NF-I04 — Negative-evidence timeline.** `P1` Visualize S2-S14 failures, invalidities, repairs, and closed conclusions without implying linear improvement; acceptance is receipt-backed dates and identities.
- [ ] **NF-I05 — Uncertainty-first figure style.** `P1` Standardize intervals, raw fold points, seed summaries, thresholds, missing cells, and failure counts; acceptance is visual QA and accessible color/label checks.
- [ ] **NF-I06 — Reproducibility badge self-audit.** `P1` Evaluate artifact availability, data restrictions, compute, commands, hashes, and independent replay against the target venue’s current criteria; acceptance is a dated checklist with evidence.
- [ ] **NF-I07 — Limitation-to-claim linter.** `P1` Flag profitability, causality, point-in-time, SOTA, novelty, safety, and generalization language when required evidence is absent; acceptance is adversarial manuscript fixtures.
- [ ] **NF-I08 — Public technical report package.** `P2` Prepare a sanitized report only after authorization, preserving negative findings and data limits; acceptance is a reviewed package with no private inputs or inflated claims.

### J. Portfolio and decision-science extensions

- [ ] **NF-J01 — Forecast-versus-policy factorial design.** `P1` Cross each forecast law with every fixed policy and each policy with fixed control forecasts; acceptance is attribution of gains to forecasts versus allocation.
- [ ] **NF-J02 — Liability-aware utility specification.** `P2` Model dated contributions, withdrawals, obligations, and shortfall separately from competition PnL; acceptance is a prospectively frozen mandate and unit tests.
- [ ] **NF-J03 — Capacity and market-impact stress.** `P1` Evaluate wealth across asset size, volume participation, spreads, and impact assumptions; acceptance is scenario tables without claiming real capacity.
- [ ] **NF-J04 — Turnover source decomposition.** `P1` Attribute turnover to forecast changes, covariance changes, constraints, rebalancing schedule, and forced lifecycle events; acceptance is independent ledger reconstruction.
- [ ] **NF-J05 — Risk-budget attribution.** `P1` Decompose exposure, volatility, drawdown, tail loss, concentration, and cash usage by asset and policy rule; acceptance is conservation checks.
- [ ] **NF-J06 — No-trade and simple-rule dominance tests.** `P0` Require candidate policies to beat cash, buy-and-hold, equal weight, and risk-only rules under identical dates and costs; acceptance is complete matched ledgers.
- [ ] **NF-J07 — Decision abstention policy.** `P2` Define a train-only rule for remaining in cash under unsupported forecasts; acceptance is coverage/utility curves and no outcome-tuned threshold.
- [ ] **NF-J08 — Mandate compatibility engine.** `P2` Validate eligibility, concentration, leverage, liquidity, horizon, tax, and funding constraints from an explicit mandate; acceptance is rejection of missing or ambiguous rules.

### K. Improvement backlog acceptance and pruning

- [ ] Assign each improvement an owner, dependency, expected evidence, compute/storage estimate, and target version before implementation.
- [ ] Mark each item `CURRENT_EXTERNAL_TOOL`, `NEXT_RELEASE_ENGINEERING`, `SUCCESSOR_SCIENCE`, `PARKED`, or `REJECTED_WITH_REASON`.
- [ ] Create a dependency graph and identify the smallest P0 path to trustworthy evidence.
- [ ] Estimate the total cost of the selected backlog before authorizing execution.
- [ ] Reject features that add complexity without a discriminating experiment or measurable acceptance criterion.
- [ ] Reject redundant features already covered by a stronger existing control.
- [ ] Require a scientific decision record for any item that changes the hypothesis, mechanism, data, endpoint, statistics, or interpretation.
- [ ] Re-rank the backlog only from engineering need, validity risk, or the locked terminal result; never from protected outcomes.

**Gate 13 acceptance:** every selected improvement has an explicit boundary, version, evidence requirement, dependency, and budget; frozen 0.2.4 remains byte-identical and its terminal result remains preserved.

## Parked successor work — do not mix into the S14 execution chain

### FREM-S11

- [ ] Keep S11 `BLOCKED_UNCONSUMED_NOT_EVALUATED` until exact-source host admission passes.
- [ ] Acquire and admit its declared archive only under its own protocol.
- [ ] Execute its one-use development consumption separately.
- [ ] Do not transfer S14 evidence or outcomes into S11.

### FREM-S12

- [ ] Complete the external saved-version ledger before scientific consumption.
- [ ] Upload and qualify exact P0v8 only under its own CPU/Internet-off contract.
- [ ] Keep search at 0/144 and fixed work at 0/120 until qualification passes.
- [ ] Do not inherit S10/S14 host qualification.

### FREM-S13

- [ ] Keep S13 `NOT_EVALUATED` until its exact CUDA package passes numerical verification.
- [ ] Preserve S13 as a separate candidate; do not treat S14 execution as S13 evidence.

### Any new S15 or other mechanism

- [ ] Start only after the current result or validity defect is classified.
- [ ] Write: previous finding → remaining problem → new hypothesis → material difference → falsifier.
- [ ] Create a new protocol, version, budget, output root, and protected boundary.
- [ ] Do not use a new successor to erase or relabel an unfavorable S14 result.

## Definition of project completion

- [ ] The active scientific question has a valid terminal disposition.
- [ ] The disposition is supported by admitted data and complete evidence.
- [ ] Protected confirmation is preserved and executed prospectively where the claim requires it.
- [ ] Independent reproduction has been completed or its limitation is explicit.
- [ ] Every claim maps to raw evidence and stays within scope.
- [ ] The paper and artifact package compile, verify, and pass visual inspection.
- [ ] The repository and remote identify the exact release.
- [ ] Negative and failed evidence remain accessible.
- [ ] No live trade, broker order, third-party message, or public submission occurred without separate authorization.
- [ ] Canonical state records `SUPPORTED`, `MIXED`, `NEGATIVE`, `INCONCLUSIVE`, or `INVALID` with the exact scope of the conclusion.

The project is complete when the question has a scientifically honest terminal answer and the evidence can be reconstructed. It is not complete merely because a model, package, notebook, test suite, GPU run, manuscript, or positive point estimate exists.
