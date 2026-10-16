13 October morning review: independent accounting reverified 2,276 ledgers, 34,276 periods and 1,424 comparisons. The 6% target remains unmet in all 32 primary cases, and memory increment remains zero. Frozen source/prediction identities and all 155 S4 predecessor files are unchanged. No new framework source, model fit or protected access. The overnight heartbeat is PAUSED. Read [morning review](../research/frem_portfolio_20261012/MORNING_REVIEW.md) and its linked audit receipt.

Latest portfolio update (12 October 2026): the isolated [portfolio evidence layer](../successors/frem_portfolio_v1/README.md) passed 31 targeted tests and completed 1,424 exploratory comparisons. All 32 primary S4 cases missed 6% ending-wealth uplift; range -2.9566% to +0.3478%, with zero increment over memory-off. Independent arithmetic verified 2,276 ledgers. Read [portfolio handoff](../research/frem_portfolio_20261012/HANDOFF.md), [results](../outputs/frem_portfolio_20261012_attempt1/REPORT.md), and [Laura current research](../research/frem_portfolio_20261012/LAURA_CURRENT_RESEARCH.md). No new model fit in this portfolio task or protected access. Preserve completed results; external framework source remains unspecified.

12 October S4 update: completed 6 fits/24 outputs after 247 tests under renewed slow-mode authorization. All non-off arms selected zero; no memory gain. Previously inspected 2023/2024 windows are development only. See research/frem_s4_20261012/REPORT.md and PUBLIC_RULES_AND_LAURA.md. Paper revised to include S4.

Latest update (12 October 2026): paper/main.tex and paper/main.pdf now include actual S2, slow-study and S3 results. See research/checklist_completion_20261012 for the fresh 133,844-row audit, table provenance, 57-item checklist disposition, successor proposal and Laura funding blockers. No new fits or protected access.

# Codex handoff: source delivery to valid research execution

Current evidence index: [research completion status](RESEARCH_COMPLETION_STATUS.md). The fixed FREM-S3 campaign is complete; [its report](../research/frem_s3_20261011/REPORT.md) and [receipt](../research/frem_s3_20261011/completion_receipt.json) record 217 regression passes, 15 generated mechanism cells, 24 learned public fits and 12 constant cells. Both primary memory intervals include zero. Preserve the completed protocol without tuning or retries.

Local maintenance added 25 release/source-archive regression cases and ran 42 lightweight tests. Read [release inventory instructions](RELEASE_INVENTORY.md) before regenerating manifests or re-auditing the old synthetic run. Historical root manifests are retained; fresh manifests must use a new directory and do not certify tests or science. The old synthetic audit now uses an explicit archive with the original hashes instead of mismatched live successor source.

Latest public campaign: [slow six-window completion receipt](../outputs/public_equity_slow_20261011_attempt1/completion_receipt.json). All 24 declared cells completed after 194 tests under explicitly authorized one-worker, one-thread, nice-10, batch-32, 768 MiB sampled RSS limits. Peak observed model RSS was 386.66 MiB. The separate campaign extends the recorded whole-month all-zero-quote quarantine; strict original adapter and raw quotes remain unchanged. FREM point NLL improved over no-memory in four windows and worsened in 2020/2022. These are exploratory survivor-sample, single-seed, unadjusted comparisons, not confirmatory evidence. Source hashes and 8,684 prediction NLLs were verified; no protected outcomes accessed. Preserve this completed matrix without metric-driven reruns. See `research/decisions/public_slow_execution_20261011.json` and `public_slow_result_20261011.json`.

Latest execution: [FREM-S2 completion receipt](../research/frem_rebuild_20261011/completion_receipt_v2.json). The additive successor and training-scaled head passed 194 tests and completed 22 exploratory public-history fits / 24 prediction cells. Across 23 stocks, 19 crisis months and seeds 11/29, NLL was -0.490295 for legacy FREM, -0.559094 for scaled FREM, -0.603182 for the frozen-base successor, and -0.709526 for the best zero-return control (lower is better). Both calibrated memory weights were zero. Optimization improved; memory benefit and SOTA were not established. Do not retune this completed fold to manufacture a win.

The user authorized slow execution despite the normal memory-floor failure: one thread, one low-priority process at a time, 32-row neural minibatches, 768 MiB process ceiling; observed sampled peak was 404.2 MiB. Preserve both prefit failures and the exploratory-only ENB.TO May 2003 missing-month amendment. Raw all-zero quotes remain unchanged; no price was filled and no historical holiday or terminal-loss ruling was made. The canonical strict adapter still rejects those quotes.

The [model specification](../research/frem_rebuild_20261011/specification.json), [complete results](../outputs/frem_s2_development_20261011_attempt3/summary.json), checkpoints, diagnostics and failed attempts are retained. `scripts/run_frem_rebuild.py` is the explicit-execution launcher; it refuses existing output directories, verifies tests before fits, bounds each process, and requires the complete seed/model family before summarizing. Independent reproduction must use a fresh output directory and preserve the original declaration and outcomes.

The archived 23-page manuscript predates the public successor runs; the current 27-page revision includes them; the current [research completion status](RESEARCH_COMPLETION_STATUS.md) links the later evidence. Submission readiness remains blocked. No canonical admitted-market or protected evaluation occurred.

Loss diagnosis: [retained FREM diagnosis](../research/frem_loss_diagnosis_20261011/REPORT.md) identifies forecast bias as the dominant numerical failure. The saved memory correction helps slightly; a separately trained memory-off model is a different checkpoint. Source-supported training/memory concerns remain hypotheses pending controlled development work. Thirteen standard-library diagnostic/audit tests passed; no new model fit or protected evaluation occurred.

## Current state

The source package and methods/protocol manuscript are authored. Subsequent engineering verification passed 163 FREM tests, 67 RE-JEPA v0.3 tests, 45 RENG-JEPA v0.1 tests, full-scope Ruff, static validation, and all three package builds. The bounded first FREM market run stopped before model execution because `data/raw/panel.parquet` is absent. **Canonical admitted-market execution and protected confirmation remain blocked. Public exploratory fits, saved forecasts and research-ledger receipts exist within their stated limits.** Read `RESEARCH_STATE.json`, `research/repository_cleanup_verification_2026-10-11.json`, and `docs/STATIC_VERIFICATION.md` first; do not translate engineering compatibility into scientific readiness.

The public exploratory matrices are completed. The execution sequence below applies to a future admitted canonical dataset; it is not an instruction to rerun completed public protocols. Recheck resource admission before scientific imports or new work. Keep the canonical state and original negative evidence throughout.

## 1. Establish a clean environment

Use Python 3.11 or 3.12. The authoring toolchain is not a tested scientific execution environment. CPU is the default. A CUDA environment is an explicit later choice; MPS with the spectral branch is rejected because support is not assumed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
python scripts/validate_source.py
python -m pytest -q
```

These are reproduction commands; the current Python 3.12 environment and committed lock passed them. The command and resource-reporting environment is Linux/macOS; use a Linux environment on Windows. Save the actual resolved environment after a successful install:

```bash
mkdir -p research/environment
python -m pip freeze > research/environment/requirements.resolved.txt
python --version > research/environment/python.txt
```

Use the resolved lock to reproduce that environment. The broad dependency ranges in `pyproject.toml` were not empirically certified. Record operating system, device, BLAS/CUDA settings where relevant, and actual elapsed resources. Do not silently skip unavailable XGBoost or replace a baseline with a different estimator.

The tests include small constructed cases and some tiny neural fits; running them is execution. Their current passing receipt is engineering evidence only. Keep future failing output, make the minimum justified fix, and update the state and paper if mathematical behavior changes.

## 2. Acquire the inputs without inventing provenance

### Securities and returns

First obtain the actual competition participant approved-security list and trading requirements. A copied case study and the public FAQ do not establish the eligible universe. Obtain date-specific instrument metadata for any historical policy experiment.

For market data, use a provider or university export that grants the required rights. `docs/DATA_GUIDE.md` covers CRSP-style point-in-time equity data, Alpha Vantage-style price/action exports, and total-return versus individual-bond inputs. A vendor-specific export is normalized into the strict canonical schema; the package does not assume a universally valid stock downloader.

Keep raw prices, distributions, split conventions, permanent identifiers, membership history and delisting outcomes. Supply actual timezone-qualified close/valuation times and availability. Attach base-currency FX with its own availability. Include the sidecar manifest at `data/raw/panel.parquet.manifest.json`. An adjusted-close series alone does not certify these properties.

Never invent a historical fixed-universe declaration. `fixed_ex_ante` excludes observations before the real declaration date. A retrospectively selected current ETF set can be used for a clearly labeled engineering exercise, but cannot be described as a historically predeclared universe. Point-in-time claims require the appropriate evidence and terminal outcomes.

### Macro inputs

Macro covariates are optional. If enabled, use vintages rather than the latest revised series as though it had been known historically. The FRED adapter is explicitly opt-in and requires a registered API key in `FRED_API_KEY`; keep secrets out of config files and committed artifacts.

```bash
frem acquire-fred --request configs/data/fred_request.example.json --output data/raw/unrate_vintages.csv --execute
```

Review the example dates and series before this future download. The adapter's two-day availability delay is a conservative date-vintage policy, not a measured release timestamp. One series file can be used directly as `data.macro_path`. For multiple series concatenate their canonical rows, retain all source hashes and licenses, and create an audited combined manifest with `series_universe`. The adapter does not acquire security prices.

### Liability and policy inputs

The cash-flow facts are already in `configs/mandates/laura_gao.pending.json`. Resolve its `missing_inputs` from real participant materials and a stated team policy. Set funding confidence, financial flexibility, concentration/cash/turnover/FX/liquidity limits and actual current holdings. An all-zero risky holding vector means explicitly all cash; omitting holdings means unknown, and is blocked.

Provide dated instrument prices, tradability, approvals, rule provenance and look-through exposure when needed. A contemporary rule cannot be applied to an earlier date merely by changing its timestamp. Do not turn the Taiwan 10% daily price band into a portfolio cap.

## 3. Validate only the development data

The supplied presets point to `data/raw/panel.parquet`, use USD, and propose `2025-01-01T00:00:00Z` as the protected boundary. No data was inspected to validate that choice. Confirm a useful unseen tail exists and record any amendment before viewing its outcomes.

```bash
frem inspect-data --path data/raw/panel.parquet --before 2025-01-01T00:00:00Z
frem plan --config configs/first_execution.json
```

For a vendor column mapping, create a JSON source-to-canonical map and pass `--column-map` to inspection, with the same mapping in the experiment config. The first compatibility dataset must still satisfy the economic contract. Do not relax publication, delisting or split checks to make a test pass.

No full-tail price summary should be printed during development. The code hashes the full file as opaque identity but constructs development features only from allowed observations. A protected-data hash is not permission to inspect protected outcomes.

## 4. Perform the first bounded run

Only after tests and data validation succeed:

```bash
frem run --config configs/first_execution.json --phase development --execute
```

This preset declares four methods (`frem`, `no_memory`, `ridge`, `xgboost`), one seed, two neural epochs and a 900-second matrix ceiling. Its purpose is to expose integration defects and measure actual costs. It is not a full research result and cannot be used to claim data efficiency or profitability.

Keep the printed run directory. Inspect its manifest, split indices, fitting cutoff, feature/asset schema, predictions, scale/df values, training history, parameter/resource counts and any failure trace. Check that inference occurs after every label used for fitting or checkpoint selection. An incomplete matrix is not compared by silently dropping failed methods.

Then run `frem compare --run <printed-run-directory>` and `frem report --run <printed-run-directory>` on complete evidence. A short compatibility panel may lack enough independent blocks for valid interval estimation; that should fail clearly, not be worked around by calling windows independent. Use larger pre-protected data for the statistical gate.

## 5. Development within a declared budget

`configs/development.json` contains 24 candidate cells and a 12-hour matrix ceiling. `configs/robustness_suite.json` is a separate 36-cell optional suite with its own cap. Select the justified experiment subset before starting; do not automatically run all available variants or restart to obtain a good seed.

```bash
frem run --config configs/development.json --phase development --execute
frem suite --config configs/development.json --suite configs/robustness_suite.json --execute
```

The second command is an independent execution decision. For a claim that memory is more robust, declare matching memory-off/FIFO controls within the relevant suite and adjust the total budget explicitly. A single-method sensitivity cannot answer that relative question.

Budget enforcement is cooperative. Python training checks time and the matrix records overrun; a native tree fit or optimizer call may not return immediately. Use a scheduler-level timeout if there is a hard machine deadline, and preserve interruption artifacts. Existing configurations are candidate ceilings, not hardware benchmarks.

Record every important change using an event JSON and:

```bash
frem record-decision --event research/my_decision.json
```

An event should contain the observation, hypothesized cause, change, predicted effect, discriminating experiment and result/status. Retain all earlier artifacts. If simpler methods win, accept and explain that outcome.

## 6. Freeze before confirmation

Complete the dataset, source manifests, final configuration, exact baseline family, seeds, statistical settings, exclusion/stopping rules and environment. The candidate primary comparison is FREM versus no-memory FREM, with a 0.01-nat minimum effect. Date-block length must respect actual execution/outcome maturity overlap.

Do not mutate the central question to match the development winner. A materially new mechanism becomes a successor study. Record the development-selection rationale and create a uniquely named freeze:

```bash
frem freeze --config configs/confirmatory.json --output research/freeze_v1.json --rationale "Development audit complete; the documented hypothesis, data, model family and analysis are fixed before protected evaluation."
frem run --config configs/confirmatory.json --phase confirmatory --freeze research/freeze_v1.json --execute
```

The command checks code, config, raw data, provenance manifests and environment against the freeze. A consumption receipt prevents accidental reuse. The exact analysis protocol is archived and hashed, so later statistical options cannot silently change confirmation.

If a protected attempt fails, preserve its receipt, raw outputs and reason. A genuine validity defect may justify a corrected protocol; deleting the receipt and retrying an unchanged valid experiment is not the recovery path. No file mechanism can turn previously inspected data into a fresh untouched test set.

## 7. Analyze complete evidence

```bash
frem compare --run <complete-confirmatory-run>
frem compare-family --run <complete-confirmatory-run>
frem report --run <complete-confirmatory-run>
```

The paired score matches exact observations, outcomes and availability/execution metadata, preserves complete seeds, and uses asset/date balancing. The secondary family applies Holm correction to all prescribed controls. Statistical significance, practical effect, and net investment value are separate conclusions.

A support label remains conditional on a leakage/baseline/fairness audit. Preserve negative, mixed and inconclusive outcomes. Do not select favorable plots or seeds and discard contradictory evidence.

## 8. Forecast and apply the investment contract

After a fitted checkpoint exists, frozen unlabeled inference can be requested with:

```bash
frem predict --checkpoint <model-checkpoint-directory> --config configs/development.json --input <current-canonical-panel> --as-of <UTC-decision-time> --device cpu --output artifacts/current_forecasts.csv
```

The current panel must match the trained feature semantics and contain only already available observations. The asset mapping is preserved when predicting a narrower universe. An unknown asset is rejected, not assigned an existing asset's identifier.

Read `docs/FORECAST_TO_PORTFOLIO.md` before preparing a `decide` or `allocate` request. Require the exact common forecast/execution horizon, currency, dated correlation or an explicit independence assumption, bounded log-return scenarios, current holdings and a reviewed mandate. Run each command with `--execute` only after those inputs are verified. These commands produce research decisions; they place no orders.

For future portfolio evidence, use `build_execution_ledger` only on synchronized nonoverlapping horizons. Compare policy alternatives under the same current holdings, costs, reserve and instrument rules. Retain shortfalls and infeasible allocations. A scenario range is conditional on its scenario assumptions, not an automatically calibrated confidence statement.

## 9. Finish the paper from evidence

Populate empirical tables from retained analyses; no manual invented numbers. The report generator supplies descriptive summaries and figures, while paired analyses establish the specified uncertainty. Build the current manuscript with:

```bash
python scripts/build_paper.py
```

That is document rendering, not research execution. For submission, review the final citations, equation/code correspondence, source licenses, case confidentiality, authorship, venue requirements and every claim-to-artifact link. The provided paper is currently a methods/protocol manuscript, not a submission containing demonstrated results.

## Suggested instruction to paste into the next Codex session

> Read AGENTS.md, RESEARCH_STATE.json, docs/STATIC_VERIFICATION.md and docs/CODEX_HANDOFF.md. Preserve prior reported negative results and the protected tail. Set up a clean environment and run the authored integrity tests. Fix concrete failures and record each change. Help normalize my licensed dataset and verify its provenance. When tests and data validation pass, run only the first_execution development preset, keep every failure, and report actual runtime, integration defects and data/split integrity. Do not run the full development suite or consume the confirmatory freeze as part of this initial compatibility task. Do not trade or publish. Update the canonical research state before finishing.


## 2026-10-11 FREM-S3 execution update

S3 is implemented and its fixed exploratory campaign is complete. Read `docs/FREM_S3.md`, `research/frem_s3_20261011/REPORT.md` and `research/frem_s3_20261011/completion_receipt.json`. Full 217 tests passed; all 15 mechanism cells, 24 learned public fits and 12 constant cells completed. Combined memory worsened 2023 and improved January-October2024, but both primary intervals include zero and no memory arm improved both windows. Extra calibration also worsened both versus raw base forecasts. Preserve this unsupported/inconclusive primary result. Do not tune or rerun these completed windows to manufacture support. Public sample is new to this campaign, not canonical admitted market data. Protected2025+ remains untouched; no portfolio or confirmation claim.


## 13 October 2026 S6 implementation update

The additive [S6 successor](../successors/frem_s6/README.md) passed115bounded numerical integrity checks (59MiBpeak) and changed-file Ruff. See [report](../research/frem_s6_20261013/REPORT.md) and [receipt](../research/frem_s6_20261013/RECEIPT.json). Actual market-training admission failed before scientific imports:512MiBfloor/pageouts, admitted window data and a separate frozen development protocol. Zero S6market fits/backtests or protected access. All111S5frozen identities remain unchanged; closed negative results stay closed. Tiny integration-test fits are engineering evidence only. Optional XGBoost runtime, full24-cell development, independent retraining, global default-aware data support and Laura-specific allocations are unverified or blocked. Do not claim complete ultimate model, returns, novelty or calibrated funding probabilities.


## 13 October 2026 Laura case clarification and planning

The user supplied the complete case, accepted the proposed mathematical terms, and defined dual functionality as prediction plus portfolio planning. The new `successors/laura_planning_v1` module passed 28 standard-library tests at 25.55 MiB peak. It calculates contributions, dated reserves, explicit costs, shortfalls, equity targets, retained flexibility and conditional 2031 facility bounds. All 18 growth/reserve sensitivities are saved, including deficits; none is a backtest or market forecast.

Read `research/laura_plan_20261013/INVESTMENT_POLICY_PROPOSAL.md`, `RECEIPT.json` and `successors/laura_planning_v1/MATHEMATICS.md`. The liability-first policy is now a concrete team proposal. Current WInS holdings are needed for transaction-specific work, not for the strategic 2027 contribution plan. Current instrument eligibility remains unverified; the public lists found are 2025–2026. The 99% joint funding target is uncalibrated. After the 2031 communication, protected facility-minimum assets cannot be spent through rebalancing.

All 43 S6 delivery identities remain unchanged. Fresh memory samples of 162.61, 159.86 and 165.75 MiB fail the 512 MiB training floor. No S6 market fits, backtests or protected access occurred; admitted data and a separate frozen protocol remain absent. Historical negatives, receipts and freezes remain intact. Do not claim complete ultimate-model delivery or demonstrated investment advantage.


## 2026-10-14: S6 real-data intake and frozen execution package

Real SPY/TLT/GLD prices acquired; 1770 rows reconciled; six new intake/resource tests passed. Frozen 24-cell protocol includes XGBoost. Zero market fits: local 512 MiB admission failed, SSH timed out, GCP credentials expired, HF Jobs unavailable. See research/frem_s6_real_20261014/REPORT.md. No protected outcomes consumed.


## 14 October 2026: completed S6 and fresh S7 development

The user explicitly waived RAM admission/training RSS guards; the original guarded runner remains unchanged. S6 completed all 24 frozen ETF fits with 115 integrity tests. Fixed expert averaging had better predictive NLL; memory advantage unsupported. See research/frem_s6_real_20261014/completed_ram_override/REPORT.md.

The subsequent S6 diagnosis isolated complementary-expert downweighting and omitted holiday intervals, gross allocation losses and costs. Additive S7 was frozen before acquiring 2019, used continuous common-session weeks and explicit dividend accounting, and completed all three seeds and forecast/policy/cost controls. 125 tests and saved-output audit passed. All three selection strengths were zero; S7 matched equal expert averaging and lost predictive score to S6 memory. At 10 bps its cost-aware policy remained cash and missed benchmark gains. This successor did not establish improvement. Preserve the completed 2019 protocol without tuning/retesting it. See research/frem_s7_20261014/REPORT.md, AUDIT.json and ALL_BASELINES.csv. No 2025+ protected outcomes accessed; no canonical confirmation or funding certification.


## 14 October 2026: S8 successor closed

S7 diagnosis and S8 development are complete. Read [the final report](../research/frem_s8_20261014/FINAL_REPORT.md), [receipt](../research/frem_s8_20261014/RECEIPT.json), and [audit](../research/frem_s8_20261014/AUDIT.json). S7 rejected positive 2018 half-gains because both were below its fixed 0.01-nat threshold; its 2019 single-week, large-allocation candidates never cleared the round-trip fee hurdle. No 2019 tuning or new policy replay occurred.

S8 used delayed soft density gating and continuous four-interval committed allocation, frozen before 2020-2022 acquisition with one intake-only corrected source freeze before fitting. Three base fits/seeds, 153 weekly intervals, 15 forecast variants, four policies/four costs completed. NLL gains were +0.0013675 versus equal mixing and +0.0006838 versus S6 memory; both fixed interval families include zero. At 10bps, same-policy H4 wealth was S8 $98,745.27, equal forecasts $96,211.62, S6 memory $97,134.00. Cash was $100,000 and equal-weight assets $104,927.25. Prediction is inconclusive; the economic gate failed. No favorable secondary arm replaces the primary.

125 unchanged S6/S7 tests and 18 S8 checks passed in retained runs; 34,560 NLL values, 1,230 soft-gate decisions, 2,640 labels and 252 ledgers reconciled. All 319 predecessor file hashes are unchanged. Tests, DNS/intake/analysis failures, source freezes, checkpoints and ledgers remain preserved. Legacy SPY/GLD feature-cache exposure means newly acquired/held out from S6/S7 is not globally untouched confirmation. No protected 2025+ access. Do not tune or rerun the closed 2019 or 2020-2022 evaluation; a further successor requires a new hypothesis and freeze.


## 15 October 2026 loss diagnosis and improvement checklist

The [13-item checklist](FREM_LOSS_REPAIR_CHECKLIST_20261015.md) records measured causes, source-confirmed approximations and unproven explanations separately, with severity, proposed changes and required evidence. It prioritizes retained-output attribution, horizon alignment, calibrated decision inputs, simple expert controls, joint risk and fresh-data feasibility. Successor implementation and execution remain pending; no budget or new freeze was assigned. S6-S8 outcomes and thresholds remain unchanged. See `research/loss_repair_checklist_20261015/RECEIPT.json` and the appended `record-decision` event.


## 15 October 2026 evidence-bound restart

Read [the restart report](../research/restart_20261015/REPORT.md), [mathematical contracts](../research/restart_20261015/MATHEMATICAL_CONTRACT.md), [prior-art review](../research/restart_20261015/PRIOR_ART_AND_IDEA_REVIEW.md) and [checkpoint record](../research/restart_20261015/CHECKPOINTS.json). C01 loss attribution is complete; the other checklist items are partial or proposed. The separate standard-library auditor passes 40 tests and lint, checks 240 saved strategy ledgers and 36,720 intervals with maximum dollar discrepancy 8.30e-10, and verifies 49 selected frozen identities. It performs no policy replay or training. Primary asset contributions are SPY +$6,988.60, TLT -$5,511.30, GLD -$1,921.78; fees $810.24; ending wealth $98,745.27. Both saved predictive interval upper bounds against equal mixing are below 0.01 nats, with both intervals including zero.

All audit failures and source versions are retained. Audit schema repairs do not invalidate or reopen S8. Reference compounding, horizon variance, finite joint scenario moments and exact self-financing costs have proofs and constructed tests; they are not a fitted market model or performance guarantee. No new model fit, policy replay, scientific freeze or protected outcome access occurred. The canonical panel remains absent and unused evaluation admission is unresolved; normal resource admission fails. The proposed next ceiling is 12 fits/60 training minutes/one worker, not frozen or launched. Successor Checkpoint 1 remains NOT_READY under the supplied user pipeline. No distinct novel mechanism, independent reproduction, full successor paper/release or investment success is established. Preserve all prior scientific dispositions and 2025+ boundary. See the appended record-decision event and research/restart_20261015/RECEIPT.json.


## 15 October 2026 S9 exploratory implementation and blocked execution

Read [the S9 report](../research/frem_s9_20261015/REPORT.md), [specification](../research/frem_s9_20261015/SPECIFICATION.md) and [current checklist](../research/frem_s9_20261015/CHECKLIST.md). This separate empirical diagnostic resolves a narrow exploratory question using existing pre-2017 training-era data; it does not claim a novel algorithm, untouched confirmation or original memory success. The full deterministic scenario forecaster, horizon/context/risk controls, exact cash/dividend fee accounting, complete 96-ledger matrix, checkpoints, descriptive intervals and guarded one-use runner are implemented. 37 engineering tests and lint pass. Data admission gives243/399training examples and38/51evaluation blocks. Freeze1 binds20source/data/spec identities and environment.

Actual attempt1 exited3 before any fit: host free-plus-speculative memory74.81/89.84/120.94MiB below512MiB with rising pageouts. Market fits and new market ledgers are zero; the freeze is unconsumed. Do not call constructed tests market results. Continue with the exact README command in a new output directory once resource admission passes; do not weaken guards or remove consumption receipts. Keep the canonical 2025+ boundary closed. Existing S6-S8 negative/inconclusive findings, original0.01-nat criterion and6%objective remain unchanged. No independent reproduction, empirically completed successor paper/release or profitable system established.


## 15 October 2026 S10 design completed before implementation

The user requested a fully mapped S10. Read [the S10 overview](FREM_S10_DESIGN.md), [model specification](../research/frem_s10_20261015/MODEL_SPECIFICATION.md), [proposed protocol](../research/frem_s10_20261015/EXPERIMENT_PROTOCOL.md), [failure map](../research/frem_s10_20261015/FAILURE_REPAIR_MAP.md), and [implementation plan](../research/frem_s10_20261015/IMPLEMENTATION_PLAN.md). The design combines direct joint price/dividend scenarios, three empirical experts, delayed mixture-level proper scoring with contextual shrinkage, and holdings-aware nominal/robust utility with exact fees. The proposed fixed matrix contains 128 strategy, 48 benchmark and 16 ablation ledgers.

Status: DESIGN_COMPLETE_REVIEW_AND_FEASIBILITY_PENDING. No S10 implementation or model tests, no market fits/ledgers, no new scientific freeze or protected access. Algorithm novelty, S10 data admission, precision, runtime feasibility and confirmation remain unresolved. The new ES criterion is a separate hypothesis and cannot satisfy the original 0.01-nat density criterion. The 6% wealth objective is retained. S9 remains unrun/unconsumed; S6-S8 remain closed. Prior mutable state/handoff/journal bytes are saved under research/frem_s10_20261015/before; historical receipts are not rewritten. See the design receipt and appended record-decision event.


## 2026-10-15 — S10 revision 2 comprehensive implementation plan

User asked to extend the design/planning through everything, including code quality. Start with [the master blueprint](FREM_S10_IMPLEMENTATION_BLUEPRINT.md), then the versioned `research/frem_s10_20261015/design_v2/blueprint.json`. This was planning only: no S10 source/tests, dependency environment, market fits/ledgers, protected access or scientific freeze. Scientific status remains NOT_EVALUATED.

The extension has 14 normative clarifications, 60 module/work-package/test-mapped requirements, 96 planned verification IDs, 19 delivery packages, a strict quality/CI contract, explicit artifact/process operations and internal adversarial review. It retains the 192-ledger matrix, original scientific thresholds and original S10 design bytes. Tests listed in the catalog are obligations, not authored or passing tests.

Read MODEL_CLARIFICATIONS.md before implementation. It resolves ideal quote/decision/fill timing, dividend ownership, exact calendar/fold indices, full-law A2 freezing, neighbor propagation, action enumeration/projection, constrained solver/KKT certification, exact campaign/pilot RNG schedules, a 16-test secondary family, diagnostic units/NAV and fixed numerical/error boundaries. Actual market attempts have no resume route.

WP06 must implement and verify the relevant production bootstrap, artifact, consumption, resource and cancellation subset before real precision preparation. Its preflight does not require its own future result; full-campaign preflight requires the passing precision result. Shared modules are extended later, not duplicated in a temporary runner. Precision projects to N_A+N_B using its separate fixed pilot streams and can fail or be unidentifiable.

Next is internal requirement sign-off and the isolated `successors/frem_s10` package/configuration/metadata work. Canonical tests belong under that package. Exact environment locks, implementation, runtime checks, precision, profiling, source freeze, execution, confirmation and independent reproduction remain pending. No planning statement guarantees market advantage or novelty.

The separate decision is `research/decisions/frem_s10_design_v2_20261015.json`, appended through record-decision. This session's before snapshots preserve prior root state/handoff/journal. The final `design_v2/RECEIPT.json` records static consistency and identity checks. S6-S8 negative evidence, S9's frozen sources and unconsumed state, original S10 documents and earlier receipts remain unchanged.


## 2026-10-15 S10 integrated repair (current)

Read `research/frem_s10_repair_20261015/REPORT.md`, `research/frem_s10_repair_20261015/RECEIPT.json`, `research/frem_s10_repair_20261015/FLAW_CHECKLIST.md` and `research/frem_s10_repair_20261015/VERIFICATION_FINAL.json` before resuming. Current phase is generated integration verified, scientific gates pending; older planning entries above are historical snapshots.

274 generated tests passed with warnings as errors (42.95s), full source/test strict typing/lint/format passed, and the final source wheel installed offline into a fresh environment with hash-pinned dependencies. The exact192-cell generated policy/ledger replay has only2blocks/fold and constructed3-atom forecasts. Separate tests execute all6real kernel variants, a scoped generated pilot, and24/25-date constructed retained-row statistical analysis. Do not merge these into a nonexistent full empirical run. Saved artifacts round-trip through a whitelisted typed reader; all16 secondary comparisons and both required primary block/control families are retained.

Design4 includes prospective CL15 A4 opening-sale ordering and CL16 minimum-variance tie units. Original score/wealth thresholds and completed S6/S7/S8 outcomes remain unchanged. S10 ES identifies only the total-return law, not separate price/dividend decomposition. S9 remains unconsumed. The selected48historical identities and old journal prefix were rechecked.

Remaining internal work: operation-specific admission/artifact producers and fixed worker/supervisor/freeze join, weekly certificate/state archives, full catalog closure (18bounded VERIFIED/66PARTIAL/12PENDING), complete independent saved-output scientific audit and maximum-support/length profiling. Source/calendar/close-clock admission, rights, real precision, frozen development, unused confirmation and independent empirical reproduction remain separate gates. Latest host admission failed at82.36/92.70/84.02MiB against512MiB with increasingpageouts. Never extend an earlier RAM waiver or infer scientific authorization from helper success.

No market S10 fit, policy replay, protected evaluation, research release or public submission occurred. Same-host agent review and a clean wheel installation are engineering evidence only. Reproduction commands are in `successors/frem_s10/README.md`; exact command logs and retained generated fixtures are in `research/frem_s10_repair_20261015/integrated_verification_attempt2`.


## 16 October 2026 S10 production integration (current)

Read `research/frem_s10_production_20261016/REPORT.md`, `FLAW_CHECKLIST.md`, `RECEIPT.json`, `CANDIDATE_IDENTITY.json` and `EXECUTION_PLAN.json`. This section supersedes older readiness descriptions above without rewriting their historical evidence. The exact plan is `852e85d6e4889687d7ae555afdadb21b1a563a8392c206a17ca75ebb3a32eba2`; code identity is `11a0e935a9e0c8ba208b520704f3c6b75180273a32ca069b0f9d817a9b960696`. A complete source snapshot is retained on the disclosed dirty checkout; the old Git base does not contain this candidate.

Production source/calendar admission, exact verification/catalog receipts, prospective declarations, consumption, workers, weekly archives, saved numerical analysis, generated qualification and persistent continuation are integrated. Small scoped tests/static checks and the actual network/data/output/research denial probe passed. All96 obligations are mapped:89 authored bounded-complete,6partial,1Linux-unimplemented; these are coverage assessments, not current full-suite passes. Current full-suite/build/eleven-mutant execution is still pending. Preserve every failed scoped attempt. Eleven fixed mutants cover all ten normative faults plus omitted exit cost. TEST093 has no waiver. TEST091 optimization timing remains open; no speed improvement is claimed.

Source integrity admission covers3021 SPY/TLT/GLD sessions in2005–2016 and25 early closes, with separately pinned community calendar bytes. It is retrospective current-vintage evidence: no PIT certification, observed fills, vendor latency, payment-date cash settlement or redistribution rights are established. Pilot values stay before2010, campaign before2017, and protected confirmation remains closed. The ES claim concerns total-return law only, not separate price/dividend identification.

The actual controller attempt exited3 before verification:101.25/138.234375/136.5MiB and rising pageouts, against the unchanged512MiB three-sample rule. No S10 scientific freeze, consumed attempt, market fit, market ledger or protected access occurred. The local heartbeat `frem-s10-guarded-research-continuation` is ACTIVE at this handoff and checks every30minutes using only the exact plan. It remains quiet while blocked and pauses on nonretryable failure, source/environment drift or terminal result. The app and host must remain running. Do not weaken guards, kill unrelated work, acquire new data or rerun a valid negative study.

Resume with `rtk proxy successors/frem_s10/.venv/bin/python -B -m frem_s10 advance --manifest research/frem_s10_production_20261016/EXECUTION_PLAN.json --execute`. Its fixed order is complete verification, one precision pilot, generated full-scale qualification, then one development campaign. Nested sandbox and /bin/ps require the supported execution context. If mutation preflight loses headroom after earlier engineering commands, the retained failure stops for review; automatic retry is deliberately conservative. Independent replication, unused confirmation and empirical paper/release remain incomplete.
