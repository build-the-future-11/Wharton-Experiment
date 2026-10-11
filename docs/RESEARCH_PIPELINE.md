# Research process, checkpoint map and evidence rules

The complete original pipeline is preserved verbatim in `research/original_user_protocol.txt`. This file maps it onto the supplied implementation and the work still requiring actual evidence. The user's no-run instruction takes precedence over later execution steps in that general pipeline.

## Precise research direction

**Problem:** financial observations can contain recurring contexts mixed with noise, changing regimes and delayed outcomes. A memory system may retain salient episodes that do not actually help prediction.

**Candidate mechanism:** fixed causal descriptors identify comparable contexts; bounded memory retains episodes according to improvement observed only after the relevant outcomes mature. Spectral surprise affects selection, but surprise alone is not treated as predictive value.

**Primary question:** does utility-retained memory improve out-of-time Student-t negative log likelihood over the same predictor with memory disabled, under matched information and declared resource limits?

**Falsifier and precision:** a valid paired interval at or below zero supports a negative directional result; a confidence interval entirely above the proposed 0.01-nat threshold supports the practical advantage within the tested scope. Crossing that threshold without sufficient precision is inconclusive. Leakage invalidates an experiment; it does not scientifically falsify the mechanism. A benefit that disappears against FIFO/uniform retention does not support special utility-based retention.

## Alternatives considered

| Alternative | Why retain or reject it for this version |
|---|---|
| Simple ridge, reservoir ridge or boosted trees | Retained as controls. They may explain the task adequately with less complexity. |
| Stable-key episodic residual retrieval | Chosen candidate. Delayed utility and retention have explicit counterfactual-free rules that can be disabled or replaced. |
| Fully learned key store with unrestricted online neural changes | Not chosen. Encoder drift and protected-label reuse would complicate identification; future versions need their own protocol. |
| A completely new Fourier transform | No mathematical need established. Standard DFT plus adaptive descriptors specifies the desired behavior without claiming an unsupported transform theorem. |
| A learned graph policy implemented by XGBoost | Rejected as a category mistake. Trees provide prediction; dated peer covariates provide the current cross-asset context. |
| Physics-informed pricing PDE | No valid governing equation or boundary data supplied. The current model uses explicit stochastic assumptions and funding identities, without implying markets obey a known physical law. |

## Checkpoints

| User checkpoint | Supplied artifact | Status of this release |
|---|---|---|
| Research state | `RESEARCH_STATE.json`, original protocol, source evidence and history | Created and preserved. |
| 1 — Idea ready | Paper scope, related work, alternatives, adversarial objections | Defensible candidate hypothesis; exhaustive novelty and benefit remain unproven. |
| 2 — Model specification ready | Paper equations, `MODEL_SPEC.md`, `INTERFACE_CONTRACT.md` | Written and compared against source. |
| 3 — Methodology ready | `research/protocol.json`, data and evaluation guides, experiment presets | Prospective design; actual dataset selection and final freeze still required. |
| 4 — Model implemented | Source modules, model/memory/training tests | Source implemented and statically inspected; runtime correctness unverified. |
| 5 — Experiment system implemented | CLI, baselines, splits, metrics, robustness, manifests and reports | Source implemented; end-to-end execution unverified. |
| 6 — Run ready | Future clean install, test outputs and bounded first matrix | Not performed. Do not label run-ready until these pass. |
| 7 — Confirmatory frozen | Future freeze identity, environment and one-use receipt | Not created; candidate configuration is not a freeze. |
| 8 — Evidence locked | Future claim/artifact audit and full seed families | No new empirical evidence exists. |
| 9 — Verified | Future independent environment and evidence regeneration | Not performed. Independent source review is a narrower completed activity. |
| 10 — Research complete | Future terminal scientific result and final empirical manuscript | Not reached; source package is the current deliverable. |

## Experiment accounting

The initial development plan is eight methods and three seeds. The optional robustness suite is a separate candidate budget, not an automatic extension to obtain better-looking results. Select a justified subset before execution. Include candidate and mechanism-off controls wherever a robustness conclusion about memory is intended; a one-model perturbation cell can describe sensitivity but cannot establish a relative advantage.

The full confirmation candidate includes the primary comparison, the complete prespecified secondary mechanism family and all baseline families. Baseline classes share allowed data but do not all have identical selection procedures: their scale calibration uses an internal purged training tail, while FREM uses external tuning. Hybrid XGBoost changes the use of validation labels. Report these differences rather than calling every result compute- or information-matched.

Independent repetitions are optimization seeds. The primary statistical unit is a UTC decision date, with repeated observations averaged within asset before equal asset/date weighting. Do not count thousands of correlated instrument windows as independent market histories.

Specific held-out markets, sectors, crisis periods and execution-cost assumptions must be declared from external criteria. The code provides selection/perturbation/statistical operators; no particular historical subgroup has been validated or silently chosen here.

## Strongest adversarial experiment

Give full FREM, no-memory FREM, FIFO and uniform-retention FREM the same causal panel, chronological splits, seeds, hyperparameters and resource ceilings. Freeze model and analysis choices. Evaluate once on an untouched chronological tail. Preserve the complete observation/seed grid, compare fitted distributional scores, and inspect memory credit and costs.

The memory claim fails to gain support if its interval cannot clear the practical threshold. The proposed retention explanation weakens if equally bounded simple retention performs just as well. A small predictive win accompanied by excessive runtime or trading costs does not establish efficient investment value. A later portfolio analysis must use a validated common execution horizon and the same dated constraints for candidate and controls.

## Change and failure procedure

For every revision record: observation, proposed cause, change, expected consequence, discriminating experiment, and result. For unrun source fixes record the static defect and the future test intended to establish the fix; do not invent an outcome.

Keep unique experiment directories. Their manifests include source/data/provenance/config identities, environment, start/end state, resource summaries and tracebacks. Confirmatory receipts survive failures to discourage accidental re-exposure. An invalid run may motivate a corrected version only with an explicit defect record.

Maintain a claim ledger with fields `claim`, `scope`, `run_directory`, `raw_artifact_sha256`, `analysis_configuration`, `figure_or_table`, `uncertainty`, `limitations`, and `disposition`. No empirical claim can be marked supported until these resolve to actual retained files.

The prior document's reported APEN and graph-memory failures remain historical evidence with verification status attached. They may motivate a successor mechanism, but must not be silently reclassified as implementation failures or deleted because the current idea sounds more promising.

## Manuscript completion rule

The current paper is a methods and prospective evaluation manuscript. Add empirical results only from complete saved predictions and traceable analyses. Report negative, mixed and inconclusive findings as such. Regenerate plots from evidence rather than typing values into figures. Complete an independent citation, equation, claims, data-rights and authorship review before any external submission.
