# FREM memory diagnosis and completion checklist — 12 October 2026

The immediate S3 failure is that validation-selected residual corrections do not transfer consistently to the next public window. Retrieval executes, the gate is nonzero, and score arithmetic checks out. The available evidence does not identify a single causal failure or demonstrate a change that will make memory outperform the base. A defensible completion can be negative or inconclusive; no checklist can guarantee perfect predictions or profitability.

This document answers the request for causes, proposed remedies and a complete research-readiness checklist. It is a diagnosis and proposed successor work, not a new frozen experiment, execution authorization, performance repair or permission to reopen completed outcomes. S2/S3 remain closed. Protected 2025+ remains untouched.

## Evidence inspected and verified today

- Current canonical state, original workflow, protocol, handoff, static-verification boundaries and S3 specification/source.
- Preserved synthetic, S2 and S3 diagnostic reports; these concern distinct lineages and are not interchangeable causal evidence.
- New standard-library artifact audit: **49 frozen source hashes matched; 11,880 saved prediction rows reconstructed; maximum NLL discrepancy 2.22e-15**. All six S3 FREM seed/window cells, all 14 arms and raw references were included. This is not independent retraining.
- Receipt: [`diagnosis_v2.json`](../research/frem_memory_diagnosis_20261012/diagnosis_v2.json). Reproducible script: [`diagnose.py`](../research/frem_memory_diagnosis_20261012/diagnose.py). Run from the repository root with a new output path; the script refuses overwrite. The first successful audit, its exact script and an initial lint failure remain preserved; the formatted script passes Ruff and reproduces the same diagnostic cells. No scientific model imports or fits were executed.

## What the evidence explains

| Finding | What is established | What remains a hypothesis | Proposed discriminating work |
|---|---|---|---|
| Tiny validation advantage | All six combined-memory choices improve validation by only **0.000925–0.006061 nats**, on **22 distinct validation months**. Twelve weight/power combinations are searched for combined memory. Selection accepts any improvement over zero. | Selection noise and regime changes may explain why corrections fail to transfer. They are not separated by current evidence. | Nested chronological development splits; regularize toward zero and require stable incremental benefit across held-forward blocks. Compare to unchanged selection using a fixed search budget. |
| Old fitted base and calibration | The 2023 base gradient-fit partition ends **May 2011** and the 2024 one **November 2011**. Checkpoint selection extends to 2014; calibration ends **December 2016 / September 2017**. Later data supplies residuals and gate selection, not base gradients. Dates are research-coordinate label timestamps. | The base may be stale; memory may be correcting base drift rather than learning transferable recurring errors. This chronology alone does not prove harm. | A rolling-origin successor with recent, disjoint fitting/selection/calibration/residual blocks and identical controls. Freeze each base before constructing its residual bank. |
| Nearly flat similarity weighting | Primary keys are unit-normalized and **103-dimensional**. `retrieve` uses the mean squared coordinate distance. For unit keys, this distance is at most **4/103 ≈ 0.038835**, so `exp(-distance)` is always at least **0.961909**. Neighbor ordering still depends on distance; the within-neighbor similarity weighting and similarity confidence factor barely discriminate. | A better distance scale might help relevance weighting, or amplify noise. This is a concrete scale-design concern, not proof of a bug that invalidates S3. | Test dimension-independent cosine/squared-Euclidean distance with a bounded train-only temperature against the exact old rule. Require better predictive relevance than matched random and recent retrieval. |
| Sparse conditional evidence | Same-asset retrieval selects four records; final primary banks hold **10–11 records per stock**. Mean effective neighbor counts are about **3.61–3.69**. Mean selected-neighbor ages are **125–140 days**, depending on cell. | Four correlated/noisy residuals may be inadequate for conditional mean and variance estimates. Bigger capacity can retain stale or irrelevant data too. | Log candidate availability at every decision, distance separation, dependency-aware support and neighbor diversity; test a bounded pooling/retention change only if relevance diagnostics justify it. |
| Memory is active | Mean effective mean-correction gates are roughly **0.1565 in 2023** and **0.0736 in 2024**; all rows retrieve four neighbors. S3 includes causal online maturation. | A weak gate is not established as the limiting factor. | Preserve zero fallback. Increasing weights is not a justified repair; the no-confidence arm already performed worse in 2023. |
| Relevance is unproven | Combined beats age/asset-matched random by **−0.003511 nats in 2023** and only **+0.000166 in 2024**, as descriptive point differences. Learned-key, larger-bank, dual-bank and cross-asset variants did not improve both windows. | Similar past inputs may not imply similar future residual distributions. No stable residual signal may exist at this target/horizon. | Check out-of-time residual predictability and nearest-versus-matched-random relevance before adding more memory machinery. Retire mean memory if it fails. |
| Extra calibration failed to transfer | Calibrated memory-off NLL is worse than raw FREM by **0.009818 / 0.018856 nats**. Five of six cells selected mean shrinkage of 0.5; one selected 0; all selected scale multiplier 1. | The calibration vintage and finite-sample selection may be responsible. Raw winning on spent outcomes does not justify retroactively changing the primary. | Compare raw, identity-shrunk calibration and recent chronological calibration as predeclared successor controls. |
| Synthetic success is easier than market success | The generated recurrence test directly supplies the planted regime state in its retrieval keys. Real primary keys are fixed trailing summaries, reservoir states, spectra and n-grams; they are not trained to identify the latent planted state. | Recovering useful regime information from real observations may be the unresolved part. | Add noisy/partially observed regime fixtures; test key construction end to end. Do not treat an oracle-state retrieval success as proof that real keys discover financial recurrence. |

Source anchors: [`s3.py`](../src/frem/s3.py), [`s3_forecaster.py`](../src/frem/s3_forecaster.py), [`model.py`](../src/frem/model.py), [`run_frem_s3.py`](../scripts/run_frem_s3.py), [`S3 report`](../research/frem_s3_20261011/REPORT.md). New numerical summaries are derived from the saved audit, not new experiments.

Earlier S2 is different: both weights were zero because all offered nonzero corrections worsened validation. Its same-asset bank had only 5–6 candidates and was 394–547 days old at the first evaluation. S3 already addressed frozen updates and absolute-age attenuation. Recommending those again as an unimplemented cure would be misleading. The original two-epoch synthetic model's large mean bias and evolving training residuals likewise must not be presented as measured causes of current S3's failure.

## What would plausibly fix the problem

The highest-priority candidate is a simpler successor with a recent frozen base, recent disjoint calibration/residual data, correctly scaled retrieval similarity and a conservative evidence-based memory gate. Its benefit is unproven. Test changes separately before a bounded combination; otherwise a positive result would not identify which change helped.

Memory must estimate information the base missed. A useful diagnostic is whether causal keys predict future standardized residual moments out of time, and whether similarity adds information beyond recency, asset identity and unconditional volatility. If that conditional structure is absent, memory-off is the correct result. More capacity, epochs or architecture cannot guarantee signal.

For probability forecasts, mean and uncertainty corrections must be evaluated separately. The scale update uses a second moment around the base mean; a successor should derive whether the corrected mean changes the appropriate residual second moment and test that derivation on known distributions. This is a candidate mathematical review, not an asserted demonstrated S3 bug. A volatility-only successor would change the claim and needs its own protocol; S3's risk-only arm did not win both windows.

Selection overfitting is a documented general risk when optimizing a noisy finite-sample validation criterion; it is a plausible explanation here, not a diagnosis proved by that literature. See [Cawley and Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html). This was a targeted methodological source check, not a completed novelty review.

## Checklist with completion criteria

Checked items describe existing evidence or today's completed inspection. Unchecked items require work or an explicit scope decision. A model change is complete only after its declared experiment and evidence review, not after code exists.

### 1. Preserve evidence and define the successor

- [x] Retain valid negative, mixed and inconclusive predecessor results, failures, freezes and protected-use boundaries.
- [x] Confirm S3 retrieval is connected and nonzero; verify saved primary arithmetic and frozen sources.
- [x] Separate measured facts, plausible causes and proposed interventions in this diagnosis.
- [ ] Define one successor question, causal mechanism, assumptions, falsifier and intended contribution; do not simply rename S3.
- [ ] Review nearest retrieval, residual-learning, online-adaptation and probabilistic-forecasting prior art; document novelty collisions and simpler alternatives.
- [ ] Declare the complete trial/compute budget, fixed seeds, stopping rules, failure handling and decision authority before new fits.
- [ ] Register all previously inspected assets/windows and keep them development-only; identify genuinely untouched evidence for final assessment.
- [ ] Define acceptable completion as a defensible supported, mixed, negative or inconclusive result, with precise claim scope.

### 2. Data and information boundaries

- [ ] Acquire the canonical panel and sidecar, or explicitly limit the successor to public exploratory research. An exploratory study cannot satisfy canonical confirmation.
- [ ] Verify provenance, access rights, source versions, actual availability timestamps, identifiers, corporate actions, delistings/terminal outcomes, historical eligibility and FX where required.
- [ ] Specify return target, horizon, units, timing, currency, missingness and execution assumptions; never treat yield as bond return or reconstruct historical eligibility from current survivors.
- [ ] Purge label-maturity overlap at every fit, selection, calibration, residual, development and confirmation boundary.
- [ ] Include enough independent dates/regimes to resolve the declared 0.01-nat effect; estimate precision/power using development evidence without inspecting protected outcomes.
- [ ] Keep seeds, correlated assets and overlapping windows distinct from independent financial histories; do not multiply the effective sample size by their counts.
- [ ] Predeclare missing/zero-quote rules and exclusions; preserve original rejected records and fail closed on unavailable provenance.

### 3. Base predictor and probability calibration

- [ ] Audit convergence, selected checkpoints, clipping, mean bias and train-only output units under a bounded budget; no indefinite epoch search.
- [ ] Specify recent rolling-origin fitting with clean subsequent calibration/residual blocks; keep base and memory-off controls identical.
- [ ] Ensure each stored residual references the exact frozen base/scaler/calibrator version; invalidate or rebuild incompatible banks at a declared base-version transition.
- [ ] Test calibration against identity/raw forecasts under the same chronological selection budget; penalize unnecessary movement from identity.
- [ ] Report NLL, MAE/RMSE, directional Brier, interval coverage/width and PIT diagnostics, with date-level uncertainty and declared conditioning groups.
- [ ] Compare JEPA on/off under matched data and compute; retain it only if the claimed benefit survives its control, or document its limited role.

### 4. Retrieval, memory and gate

- [ ] Measure per-query candidate counts, ages, norms/distances, neighbor concentration, dependency-aware support, correction size and realized delayed NLL contribution.
- [ ] Test dimension-independent distance scaling as one isolated successor intervention, preserving the old metric as a control. Do not alter the frozen predecessor.
- [ ] Establish incremental nearest-neighbor relevance over age/asset-matched random retrieval, recent averaging, unconditional residual averages and simple volatility adaptation.
- [ ] If warranted, test cross-fitted task-relevant keys on future residual distributions. S3 already tried one ridge projection; a new proposal must explain its material difference.
- [ ] Specify mean and risk correction estimands separately, including Student-t scale versus standard deviation and centering after a mean shift.
- [ ] Shrink unsupported corrections to zero; calibrate support from matured out-of-sample evidence rather than geometric proximity alone. Test a null nonharm criterion.
- [ ] Select gate/calibration choices across nested chronological development blocks with fixed complexity penalties or a preregistered conservative rule.
- [ ] Test capacity/pooling/retention only under matched evidence and compute; bigger, dual and cross-asset banks already lack consistent S3 benefit.
- [ ] Audit delayed utility credit: its objective must match deployed NLL, and shared credit to all retrieved episodes must not be assumed to identify each episode's causal value.
- [ ] Test missing/delayed labels, long gaps, abrupt shifts, no available neighbors, stale banks and serialization/resume equivalence. Predictions must be invariant to unavailable future labels.
- [ ] Test generated recurrence with noisy/hidden state and realistic key extraction, plus null and misspecified-key controls; distinguish supplied oracle information from learned relevance.

### 5. Discriminating development experiments

- [ ] Freeze a small primary sequence: unchanged baseline; recent base/calibration; revised distance only; conservative gate only; then a justified combination. Fix the exact matrix before outcomes.
- [ ] Include zero, historical mean, Ridge, XGBoost, no-JEPA and appropriate simple volatility/adaptation controls. Add strong task-relevant forecasting baselines within a declared fair budget.
- [ ] Hold information, base checkpoint, calibration opportunities and budgets fixed for the primary memory-on/off attribution.
- [ ] Give online comparison methods the same matured-label access; compare added memory cost against spending equivalent resources on the base.
- [ ] Retain every seed, failed cell and declared comparison; run source-bound arithmetic/completeness checks before interpreting results.
- [ ] Report per-window effects, paired uncertainty, predeclared multiplicity handling and block-length sensitivity. No favorable subgroup can replace the primary claim.
- [ ] Stop if meaningful residual predictability or relevance is absent, if confidence remains unresolved within budget, or if simple controls explain the gain. Record a valid negative result.

### 6. Engineering and reproducibility

- [ ] Consolidate effective configuration: the S3 public JSON retains stale five-epoch/single-seed prose while the frozen specification and worker use 12 epochs/three seeds. Preserve old files and emit an unambiguous resolved successor config.
- [ ] Build a clean reproducible environment; execute affected integrity tests and the full required regression suite, preserving failures. The historical 217-test receipt is not a fresh check of future edits.
- [ ] Test mathematical operators, finite/shape invariants, gradients where applicable, delayed timing, version mismatch rejection and deterministic checkpoint roundtrips.
- [ ] Establish resource admission and bounded wall/RSS budgets; profile real latency, memory, storage and scaling. Prior slow-mode exceptions do not automatically extend to new experiments.
- [ ] Complete a bounded end-to-end dry run including failure recovery, full manifests, raw predictions, metric reconstruction and reproducible tables/figures.
- [ ] Preserve dirty working-tree evidence, locks and all predecessor source/data hashes; use fresh output directories and no destructive cleanup.
- [ ] Obtain independent reproduction of the main training/evaluation result. Recomputing saved likelihoods is not independent training reproduction.

### 7. Confirmation and investment scope

- [ ] Only after a mature supported development hypothesis exists, freeze code, data, splits, baselines, seeds, endpoint, 0.01-nat decision rule, statistics, exclusions, stopping and environment.
- [ ] Evaluate the untouched admitted confirmation once, preserve its consumption receipt and classify the result honestly. Repair/re-freeze only for a documented validity defect, never an unfavorable valid result.
- [ ] If making an investment claim, implement and validate executable holdings/cash/settlement accounting, costs, turnover, liquidity, defaults, FX and eligible-universe rules.
- [ ] For that separate scope, compare net returns, certainty equivalent, Sharpe under declared assumptions, drawdown and liability shortfall using synchronized nonoverlapping decisions and the same starting holdings.
- [ ] For competition allocation, obtain official participant universe/rules, current holdings, dated instrument facts and the team's numerical risk/funding policy. Research NLL alone cannot certify an allocation.

### 8. Paper, audit and release

- [ ] Update the manuscript with S2/S3 and any successor results directly from verified artifacts; the existing 23-page draft predates the public results.
- [ ] Audit the full equation-to-code contract, assumptions, citations and novelty boundaries; include negative controls, failed attempts and unresolved limitations.
- [ ] Complete a claim → experiment → raw artifact → analysis → table/figure map, with dates, uncertainty and scope for every numerical claim.
- [ ] Independently regenerate tables/figures and inspect the compiled paper visually; verify that the abstract and conclusion do not overstate the evidence.
- [ ] Complete environment lock, reproduction commands, data-access instructions, redistribution/license checks, authorship, bibliography, supplement and source/artifact hashes.
- [ ] Freeze a release identity and update canonical state/history with the defensible terminal disposition. Public submission, trading and third-party messages require separate scope.

The order matters: source-bound diagnosis → a small successor protocol → admitted fresh development evidence → bounded discriminating experiments → confirmation only if justified → evidence-complete paper/release. A completed checklist certifies the stated research process and artifact, not perfect forecasts.
