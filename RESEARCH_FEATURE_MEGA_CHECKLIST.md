# FREM research feature mega checklist

**Prepared:** 20 October 2026  
**Current scientific status:** `NOT_EVALUATED`  
**Active implementation frontier:** FREM-S14 v0.1.3 saved version `359955736` is preserved as an external dataset-mount failure before notebook execution. The separately packaged v0.1.5 trace candidate adds readback provenance only. Exact GPU numerical qualification, the declared 171-cell historical-proxy suite, saved-output readback, and any scientific interpretation remain pending.

## What the project is trying to contribute

FREM's narrow, falsifiable contribution is a causal forecasting system that retains and retrieves matured, useful market episodes rather than treating every past observation as equally relevant. The original FREM hypothesis asks whether delayed-feedback, utility-weighted causal spectral memory improves out-of-time probabilistic forecasts over an otherwise identical memory-off model under matched information and compute.

S14 is the most capable current implementation. It combines a contractive multiscale reservoir, residual-guided bounded neurogenesis, mature-neighbor graph engrams, a finite-dimensional stochastic diffusion latent transition, masked multi-offset JEPA, and three stability-constrained mixture routers. Its output is a heavy-tailed predictive law, with a deliberately separate, cost-aware proxy portfolio ledger. The current 19-arm, three-seed, three-fold design is a useful engineering and exploratory proxy benchmark; it is not a point-in-time-valid, investable, protected, or confirmatory study.

The decisive missing capability is therefore not another untested architectural label. It is a source-bound experimental system that can establish when a mechanism helps, where it fails, how much compute it requires, and whether its evidence is independently reconstructable. Completed S2--S8 public results and generated observations remain mixed, negative, or inconclusive; they must remain available as controls and failure cases.

## Checklist rules

- A checked item requires the acceptance evidence stated beside it. Source review, static parsing, packaging, or a rendered figure alone does not check an item.
- `P0` items are gates for trustworthy execution or interpretation. `P1` items are high-value research capabilities. `P2` items follow a completed core matrix or require a separately frozen successor.
- Items that alter S14's architecture, dataset, primary endpoint, inference, or compute family require a new prospective protocol version and a recorded decision before any affected outcomes are viewed. They must not reopen a valid closed result.
- The current eight-ETF adjusted-close cohort may support only the declared historical-proxy evidence. Point-in-time membership, delistings, corporate actions, executable prices, spreads, impact, FX, and funding inputs are separate admission work.
- No item authorizes live trading, third-party communication, public submission, protected-tail access, or a metric-driven retry.

## 1. Core model and mechanism capabilities

- [ ] **M01 — Complete the exact S14 numerical reference qualification.** `P0` Run the finite-grid diffusion, graph-enrollment, reservoir-contraction, JEPA, mixture-moment, gradient, and resume tests on the exact released source and declared numerical runtime. **Acceptance evidence:** warning-strict test receipt, source/runtime/hardware hashes, and retained failures if any; no market cell may start on a failed qualification.

- [ ] **M02 — Establish CUDA/AMP semantic equivalence.** `P0` Compare CPU reference, CUDA, and AMP forward predictions, losses, gradients, graph retrieval masks, mixture moments, and one resumed epoch at frozen tolerances. **Acceptance evidence:** per-operation maximum-error table, deterministic seed receipt, and a documented tolerance justified from the reference implementation.

- [ ] **M03 — Measure full-configuration memory and throughput.** `P0` Profile the full 19-arm qualification and representative historical fold by stage: intake, feature construction, model fit, inference, scoring, and readback. **Acceptance evidence:** reproducible GPU/CPU peak-memory, wall-time, parameter-count, and energy/compute table; failures and OOMs remain in the denominator.

- [ ] **M04 — Validate deterministic checkpoint continuation under failure.** `P0` Force interruption at each declared epoch boundary and at the delayed-feedback release boundary, then resume once from every checkpoint. **Acceptance evidence:** bitwise or frozen-tolerance equivalence for all retained laws, router state, engram bank, neurogenesis statistics, optimizer state, wall-time accounting, and final manifest.

- [ ] **M05 — Build an explicit mechanism-state trace.** `P1` Retain a per-decision trace linking reservoir state, active-neuron mask, neurogenesis trigger, engram candidates, graph edge weights, latent diffusion moments, expert laws, and router gates. **Acceptance evidence:** schema-validated trace archive and a reader that reconstructs one prediction without re-running training.

  **20 Oct execution evidence:** v0.1.5 now retains and verifies the full-arm engram retrieval provenance (episode decision/maturity times, asset, slot, similarity, age, graph/base/final weights, values, coverage, and reconstructed memory moments). The candidate has a clean-wheel CLI receipt in [research/frem_s14_trace_20261020](research/frem_s14_trace_20261020), but M05 remains open because the numerical trace test and CUDA qualification have not run.

- [ ] **M06 — Add counterfactual memory-credit accounting.** `P1` For each matured retrieved episode, quantify its marginal effect on the chosen law and score by masking it after maturity without changing future information. **Acceptance evidence:** causal replay test, per-episode credit ledger, and comparisons against FIFO, uniform, no-graph, and no-memory controls.

- [ ] **M07 — Test retention-policy identifiability.** `P1` Implement matched FIFO, age-decay, random, score-only, and utility-credit retention policies using the same capacity, retrieval budget, and delayed labels. **Acceptance evidence:** a frozen ablation family whose complete result table reports occupancy, age, retrieval diversity, score, cost, and uncertainty.

- [ ] **M08 — Add a stable out-of-distribution abstention gate.** `P1` Derive an input-only novelty score from train-only feature support, reservoir state distance, and retrieval support; route unsupported decisions to a declared conservative law. **Acceptance evidence:** synthetic and historical shift assays showing the gate cannot inspect labels, a coverage-versus-score curve, and explicit abstention frequency.

- [ ] **M09 — Enforce multi-horizon joint-law coherence in a successor.** `P2` Replace independently fitted horizon laws with a finite joint scenario construction whose one-, five-, and twenty-step marginals agree by construction. **Acceptance evidence:** pre-data successor freeze, marginal-consistency tests, joint energy/variogram reconstruction, and a matched independent-horizon control.

- [ ] **M10 — Evaluate learned spectral adaptation only as a bounded successor.** `P2` Compare the fixed Fourier map with a spectrally regularized learned frequency selector that preserves causal inputs and a fixed parameter/compute cap. **Acceptance evidence:** new freeze, spectral-collapse diagnostics, matched no-spectral control, and full seed/fold accounting.

- [ ] **M11 — Add tail-support-aware forecast laws.** `P2` Compare the current Student mixture with a bounded-support or extreme-value-tail law suitable for returns below -100% only when the return representation supports it. **Acceptance evidence:** likelihood and calibration checks, tail-quantile scoring, and a documented statement of how the law maps to portfolio scenarios.

- [ ] **M12 — Separate cross-asset signal from graph leakage.** `P1` Provide an asset-permutation and time-lagged-neighbor experiment that preserves graph dimensions but destroys contemporaneous peer information. **Acceptance evidence:** causal perturbation tests and a complete no-graph, shuffled-graph, and same-asset-only comparison family.

## 2. Training, evaluation, and experiment execution

- [ ] **E01 — Run the one-command S14 qualification on the exact Kaggle GPU release.** `P0` Execute the immutable notebook with the declared GPU, fixed dependencies, no existing output directory, and no CPU substitution. **Acceptance evidence:** downloaded saved-version bundle whose source, wheel, notebook, GPU, tests, qualification results, logs, and manifest verify locally.

- [ ] **E02 — Execute the complete 171-cell proxy suite once qualified.** `P0` Run all 19 arms × 3 seeds × 3 expanding folds with the frozen assets, horizon, epochs, early stopping, deadlines, and failure policy. **Acceptance evidence:** all cells are `COMPLETE`, `FAILED`, or `NOT_STARTED` in the immutable inventory; any incomplete family receives an incomplete disposition rather than a ranked result.

- [ ] **E03 — Add an experiment-plan compiler.** `P1` Compile a human-readable protocol JSON into an immutable cell registry, shard map, expected artifacts, budgets, and analysis family before execution. **Acceptance evidence:** registry hash is embedded in every cell, duplicate/missing/foreign cells are rejected, and the compiled plan can be reconstructed from a saved bundle.

- [ ] **E04 — Add fault-injected campaign recovery.** `P1` Test process kill, disk-full simulation, corrupt checkpoint, stale lock, failed baseline, partial download, and expired accelerator cases. **Acceptance evidence:** each fault retains a diagnostic receipt, never overwrites evidence, and either resumes exactly or terminates the declared cell safely.

- [ ] **E05 — Make baseline parity auditable.** `P0` Emit an input, split, target, tuning-label, parameter-count, optimizer-step, device, and wall-time ledger for every baseline and ablation. **Acceptance evidence:** a machine-readable parity report flags unequal information, incomplete seeds, unequal calendar rows, or extra tuning access before comparative inference.

- [ ] **E06 — Add a compute-matched neural baseline family.** `P1` Configure at least one GRU/temporal-convolution/transformer-style sequence control under a declared parameter, tuning, and wall-time budget comparable with S14. **Acceptance evidence:** frozen hyperparameter registry, exact runtime ledger, and no silent fallback if a dependency or device is unavailable.

- [ ] **E07 — Add a classical probabilistic baseline family.** `P1` Include regularized VAR, dynamic-factor, and volatility-aware distributional controls that emit the same finite-law interface as FREM. **Acceptance evidence:** score reconstruction from common scenarios and an explicit record of each model's information set and calibration labels.

- [ ] **E08 — Run data-fraction learning curves.** `P1` Train every compared method on identical chronological 10%, 25%, 50%, and 100% prefixes while keeping validation/test fixed. **Acceptance evidence:** frozen fractions, complete methods/seeds, uncertainty intervals, and resource curves; no data-efficiency claim from a single fraction.

- [ ] **E09 — Add calibration diagnostics to the terminal analysis.** `P1` Compute PIT histograms, reliability for directional and downside events, interval coverage/width, CRPS/energy score, tail quantile exceedances, and sharpness by fold. **Acceptance evidence:** retained per-date predictions and a report that distinguishes forecast calibration from portfolio outcomes.

- [ ] **E10 — Test statistical sensitivity without outcome selection.** `P1` Recompute the fixed analysis for predeclared block lengths, nonoverlap rules, seed aggregation, and score weighting. **Acceptance evidence:** one primary analysis plus clearly labeled sensitivity panels, all derived from the same complete raw forecast archive.

- [ ] **E11 — Add temporal overlap and maturity audit.** `P0` Validate that no train, validation, test, checkpoint-selection, delayed-feedback, or portfolio holding interval shares forbidden target observations. **Acceptance evidence:** machine-generated interval graph, failing adversarial examples, and a per-fold signed audit receipt.

- [ ] **E12 — Add cost-aware allocation cross-over evaluation.** `P1` Cross every retained forecast law with cash, equal-weight, risk-only, and model policies under 1x/2x/4x costs without using realized test returns in targets. **Acceptance evidence:** complete crossed ledger matrix, policy metadata, turnover, drawdown, certainty-equivalent, funding-shortfall, and nonoverlap verification.

- [ ] **E13 — Introduce a valid early-stop fairness protocol.** `P1` Predeclare the metric, patience, maximum epochs, and label access used for every neural method, then preserve all checkpoints and selected checkpoint identity. **Acceptance evidence:** matched early-stop ledger and an audit that rejects hidden test-driven checkpoint selection.

- [ ] **E14 — Add independent score reconstruction in a second implementation.** `P0` Recompute Student-mixture likelihoods, energy/variogram scores, bootstrap samples, Holm corrections, and portfolio arithmetic with a small independent verifier. **Acceptance evidence:** byte-addressed raw inputs, equality or explained tolerance report, and a test that catches a deliberately corrupted producer summary.

## 3. Research extensions and robustness studies

- [ ] **R01 — Freeze and run a real missingness assay.** `P1` Apply date/asset/channel missingness patterns derived from an admitted provider, rather than only independent synthetic masking. **Acceptance evidence:** source-bound corruption map, unchanged labels, paired candidate/control cells, and missingness-stratified uncertainty.

- [ ] **R02 — Run delayed-feature and revision-vintage stress tests.** `P1` Delay macro, filing, and alternative-data availability according to observed publication times and compare against accidental current-vintage joins. **Acceptance evidence:** raw vintage manifests, as-of join audit, and evidence that no future revision was used at any decision time.

- [ ] **R03 — Test regime-shift generalization.** `P1` Predeclare crisis, recovery, inflation, and quiet-market intervals using external dated criteria, then measure performance and calibration without selecting windows from outcomes. **Acceptance evidence:** frozen regime ledger, complete per-regime tables, and a test for no post hoc date changes.

- [ ] **R04 — Test cross-market transfer.** `P1` Hold out one market or asset-class family with genuine point-in-time identifiers, currency, and calendar handling, train on the remainder, and evaluate without target leakage. **Acceptance evidence:** admission receipt, feature-availability audit, per-market score/calibration report, and an explicit nontransferable fallback for unsupported observations.

- [ ] **R05 — Build a ragged-panel successor adapter.** `P1` Support listing, delisting, mergers, identifier changes, calendar differences, terminal losses, and time-varying eligibility without intersection selection or survivor-only backfill. **Acceptance evidence:** adversarial synthetic lifecycle fixtures plus a rights/provenance-verified real-source admission receipt.

- [ ] **R06 — Add corporate-action and total-return reconciliation.** `P0` Reconcile prices, splits, cash distributions, delisting proceeds, FX, and return conventions into a dated total-return contract. **Acceptance evidence:** provider-level reconciliation table, unreconciled-row refusal, and a fresh dataset version rather than alteration of prior proxy artifacts.

- [ ] **R07 — Quantify retrieval robustness under adversarial near-neighbors.** `P1` Inject stale, duplicated, regime-mismatched, and extreme-loss episodes that pass superficial similarity, while preserving maturity constraints. **Acceptance evidence:** predeclared adversarial cases, retrieval trace comparison, and explicit failure behavior.

- [ ] **R08 — Test representation robustness to feature scale and unit shifts.** `P1` Vary only train-derived scaling, currency/unit transforms, and monotone feature transforms to test whether spectral/reservoir retrieval is invariant where it should be. **Acceptance evidence:** transformation contract, invariant/noninvariant expected outputs, and completed paired cells.

- [ ] **R09 — Add a causal feature ablation atlas.** `P1` Evaluate each feature group—prices, volume, macro vintages, filings, and text events—alone and in frozen nested combinations. **Acceptance evidence:** availability-bound data dictionary, exact feature masks per cell, and group-level incremental uncertainty rather than a single importance ranking.

- [ ] **R10 — Develop timestamped event and filing ingestion.** `P2` Build adapters for SEC filings and/or news only after event-to-asset linkage, original publication times, rights, revisions, and missing-event policy are auditable. **Acceptance evidence:** source-specific manifest, as-of test suite, and a held-out event-time experiment.

- [ ] **R11 — Test generalization across forecast horizons.** `P1` Evaluate the same frozen mechanism at one, five, and twenty observations with horizon-specific purges and nonoverlapping tests. **Acceptance evidence:** complete horizon grid, joint or explicitly separate-law designation, and no reuse of a favorable horizon as the primary result.

- [ ] **R12 — Add a fixed-budget alternative-training comparison.** `P2` Compare supervised-only, masked-JEPA, contrastive temporal, and diffusion-transition objectives under equal source data, epochs, and compute ceilings. **Acceptance evidence:** a new protocol with a complete objective ablation matrix and retained optimization failures.

## 4. Experiment interface and model inspection

- [ ] **I01 — Provide a protocol-safe experiment CLI.** `P1` Offer commands to compile, validate, shard, start, resume, inspect, verify, analyze, and export a declared experiment without editing code. **Acceptance evidence:** help text, JSON schemas, no-overwrite behavior, and end-to-end generated-fixture tests for every command.

- [ ] **I02 — Build a read-only run ledger dashboard.** `P1` Render cell state, source/config/environment identities, budgets, device metrics, failure traces, and verification status from immutable manifests. **Acceptance evidence:** static local dashboard generated solely from a bundle, with no action that can rerun or modify a completed campaign.

- [ ] **I03 — Add a fold and split timeline viewer.** `P1` Visualize train/validation/test intervals, embargoes, delayed-label maturity, checkpoint-selection boundaries, and portfolio holding periods. **Acceptance evidence:** viewer data reconstructed from the saved configuration and an automated assertion that every displayed interval agrees with the split audit.

- [ ] **I04 — Add a prediction-law inspector.** `P1` For a chosen decision, show component weights, means, scales, degrees of freedom, finite scenarios, moments, realized outcome when authorized, and matched control laws. **Acceptance evidence:** values loaded from raw saved artifacts; viewer labels protected/unavailable outcomes and refuses to synthesize absent values.

- [ ] **I05 — Add an engram provenance explorer.** `P1` Display each retrieved episode's decision time, maturity time, asset, rank, age, eligibility, graph transition weight, retention score, and counterfactual credit. **Acceptance evidence:** links to M05 trace records and a test proving future-matured episodes are never renderable as eligible.

- [ ] **I06 — Add reservoir, neurogenesis, and latent-field diagnostics.** `P1` Plot active-neuron count, recruitment triggers, spectral norm bound, reservoir state norms, diffusion coefficients, latent uncertainty, and JEPA losses by epoch/fold. **Acceptance evidence:** arrays retained during execution, train/validation separation, and figure-generation tests from recorded traces.

- [ ] **I07 — Add a router behavior report.** `P1` Plot expert weights, entropy, cap hits, regret inputs, downside/tail objectives, and expert errors through time and by regime. **Acceptance evidence:** causal input timestamps, a no-label-leakage audit, and side-by-side full/uniform-router/no-risk comparisons.

- [ ] **I08 — Add a failure and evidence browser.** `P1` Index failed intake, qualification, cell, checkpoint, and verifier receipts with their exact terminal state and remediation boundary. **Acceptance evidence:** failures remain visible alongside completed cells and are not hidden by default filters or summary rankings.

## 5. Reproducibility, provenance, and artifact management

- [ ] **P01 — Produce a hermetic release lock.** `P0` Freeze wheel/source, Python, CUDA, PyTorch, XGBoost, BLAS, driver, notebook, input, and configuration identities for a campaign. **Acceptance evidence:** clean install on the declared target rebuilds the qualification bundle and rejects identity drift.

- [ ] **P02 — Implement one-command evidence replay.** `P1` From a downloaded bundle, reconstruct intake checks, splits, targets, preprocessing, scores, inference, figures, tables, and claim ledger without training. **Acceptance evidence:** replay receipt with a complete inventory and byte/tolerance comparison to producer artifacts.

- [ ] **P03 — Preserve immutable raw input and supplier manifests.** `P0` Retain exact admitted input bytes, provider metadata, license/rights statement, retrieval time, checksum, and schema mapping before numerical parsing. **Acceptance evidence:** archive verifier rejects a bundle with substituted, truncated, or unrecorded input.

- [ ] **P04 — Maintain a campaign-wide claim ledger.** `P1` Bind every textual result or capability claim to raw artifact hashes, analysis settings, uncertainty, scope, limitations, and disposition. **Acceptance evidence:** automated paper/dashboard check flags any numerical or scientific claim lacking a completed evidence row.

- [ ] **P05 — Add retention and storage budgeting.** `P1` Estimate and enforce storage for raw laws, traces, checkpoints, source panels, ZIPs, and figures before launch; make lossless versus summarized artifacts explicit. **Acceptance evidence:** budget receipt, artifact-size inventory, and refusal before a campaign exceeds declared storage capacity.

- [ ] **P06 — Add external-saved-version inventory reconciliation.** `P0` Audit every private Kaggle saved version, notebook revision, output bundle, and consumption attempt for the exact campaign identity. **Acceptance evidence:** complete ledger with no unexplained fork, duplicate execution, or missing saved output before market cells are admitted.

- [ ] **P07 — Add an independent clean-host reproduction protocol.** `P2` Give a second operator an immutable release, sanitized admitted inputs, and verification command with no access to the original run directory. **Acceptance evidence:** independently generated receipt that either reconstructs the result or records a bounded discrepancy; it does not overwrite the original evidence.

- [ ] **P08 — Make resource admission continuous.** `P0` Recheck the 512 MiB/no-rising-pageout guard before numerical import and monitor the declared floor throughout long runs. **Acceptance evidence:** periodic resource receipt, source/runtime binding, and a fail-closed interrupted state when the guard is breached.

- [ ] **P09 — Create a provenance graph export.** `P1` Export a machine-readable DAG from raw source to intake, fold, cell, checkpoint, analysis, figure, and claim. **Acceptance evidence:** graph validation rejects orphan figures, foreign cells, cycles, and references to mutable files.

## 6. Paper-ready evidence products

- [ ] **W01 — Generate the primary comparison table from raw arrays.** `P0` Report complete cell counts, score differences, paired intervals, p-values, Holm adjustments, practical threshold, calibration, resources, and disposition for every declared arm. **Acceptance evidence:** scripted table source and input hashes; no typed numerical values.

- [ ] **W02 — Generate a mechanism ablation figure.** `P1` Show full S14 against no-memory, no-graph, no-neurogenesis, no-SPDE, no-JEPA, no-spectral, uniform-router, and appropriate baseline controls by fold. **Acceptance evidence:** predeclared arm order, shared axes/uncertainty, complete-family requirement, and generated caption naming the evidence scope.

- [ ] **W03 — Generate calibration and tail-risk figures.** `P1` Produce reliability, PIT, interval coverage, tail exceedance, sharpness, and score-versus-coverage plots for every completed fold. **Acceptance evidence:** each point traces to retained forecast laws and date-level outcomes with protected data excluded when required.

- [ ] **W04 — Generate memory and routing behavior figures.** `P1` Create episode-age/utility distributions, retrieval diversity, graph contribution, gate entropy, neurogenesis timeline, and diffusion uncertainty plots. **Acceptance evidence:** direct trace-to-figure pipeline and explicit language that behavior is not efficacy.

- [ ] **W05 — Generate an economic-proxy table with boundaries.** `P1` Report wealth, turnover, fees, drawdown, certainty equivalent, and cost stresses for every crossed forecast-policy pair, while labeling the adjusted-close proxy and idealized execution assumptions. **Acceptance evidence:** independent ledger reconstruction and a required limitations footnote in the exported table.

- [ ] **W06 — Publish a negative and failure appendix from the evidence ledger.** `P0` Include failed qualification attempts, unavailable cells, prior negative/inconclusive FREM studies, and reason-coded exclusions. **Acceptance evidence:** appendix is generated from retained receipts and reconciles exactly to the campaign inventory.

- [ ] **W07 — Create a claim-scope and limitations matrix.** `P0` Map each allowed claim—engineering, proxy forecasting, robustness, portfolio simulation, confirmatory evidence, reproducibility—to its required and achieved evidence. **Acceptance evidence:** manuscript build fails when a claim exceeds its recorded scope.

## 7. Public demonstration and researcher handoff

- [ ] **D01 — Provide a synthetic, no-market interactive demonstration.** `P1` Let a user create controlled recurring-regime, delayed-feedback, and near-neighbor scenarios and compare FREM against memory-off/FIFO/uniform controls. **Acceptance evidence:** fixed-seed notebook or local app, visible causal timeline, and a banner stating that generated behavior is not financial validation.

- [ ] **D02 — Provide an artifact-backed inference reader.** `P1` Accept only an already-trained, verified bundle and display predictions, laws, traces, and controls without fitting, downloading data, or generating investment recommendations. **Acceptance evidence:** bundle-identity verification and a demonstration using a generated fixture plus one completed admissible research bundle when available.

- [ ] **D03 — Release a reproducible tutorial notebook.** `P1` Walk through intake contracts, split/maturity rules, a generated qualification, evidence replay, and interpretation of a negative result. **Acceptance evidence:** clean-environment execution receipt, fixed outputs, and links to the exact protocol sections governing each step.

- [ ] **D04 — Add a research-result export package.** `P1` Export a signed or hash-indexed archive containing README, protocol, environment, manifests, raw arrays, verifier, figures, tables, and limitations. **Acceptance evidence:** archive opens on a clean host, verifies before rendering results, and contains no credentials, private raw data, or unlicensed redistribution.

- [ ] **D05 — Build a formal confirmation handoff.** `P2` If and only if a completed development campaign satisfies its frozen decision rule, generate a separate confirmation packet with unused-source requirements, contamination ledger, budget, falsifier, and independent-operator instructions. **Acceptance evidence:** new protocol identity and explicit statement that the development results neither consume nor substitute for confirmation.

## Recommended implementation order

1. Close the P0 execution chain: M01--M04, E01--E02, E05, E11, E14, P01, P03, P06, and P08.
2. Make completed evidence inspectable and replayable: M05, I01--I08, P02, P04--P05, P09, and W01--W07.
3. Use complete proxy evidence to decide whether a new extension is justified; then freeze and budget only the relevant P1/P2 successor work rather than launching all variants.
4. Prioritize source validity before scaling: R05--R06 and R10 are prerequisites for any broad-market or profitability language. R01--R04 and R07--R12 test generalization only after the data and core execution gates pass.
5. Build D01--D04 from generated or already-admitted immutable evidence. D05 is conditional on the frozen development rule, not a default next step.

This checklist deliberately treats a correct negative result, failed qualification, incomplete matrix, or data-admission refusal as an informative terminal artifact. The project becomes more scientifically useful by making those outcomes diagnosable and reproducible, not by rerunning them until a favorable score appears.
