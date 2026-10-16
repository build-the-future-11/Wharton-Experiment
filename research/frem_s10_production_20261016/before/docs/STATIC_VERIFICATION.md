# Release verification: scope and limits

**Historical authoring-time record: source was authored and statically reviewed; scientific status remains NOT EVALUATED.**

This document preserves the original no-runtime authoring boundary. The historical cleanup passed 163 FREM tests, 67 RE-JEPA v0.3 tests, 45 RENG-JEPA v0.1 tests, full-scope Ruff, static validation, and three package builds; see `research/repository_cleanup_verification_2026-10-11.json`. Later S2, public slow-study, and S3 receipts contain completed exploratory public runs. S3 passed 217 regression tests. Current status and exact receipts are in [RESEARCH_COMPLETION_STATUS.md](RESEARCH_COMPLETION_STATUS.md). No canonical admitted-market or protected confirmation has occurred.

The user explicitly asked for the full implementation and research package without model runs or experiments. This verification respects that boundary. The machine-readable inventory in `STATIC_VERIFICATION.json` contains the exact final Python/JSON counts, authored test-function inventory and syntax findings.

## Performed

- Parsed every packaged Python source, test and helper using Python's abstract syntax tree parser, then compiled the syntax tree to an in-memory code object **without executing it**.
- Checked that referenced relative Python modules exist, and parsed project JSON and TOML syntax.
- Independently reviewed model/data/evaluation/portfolio interfaces and compared the manuscript's equations and stated behavior with the implementation.
- Reviewed primary-source terminology, temporal JEPA/Fourier-reservoir prior art, the Taiwan price-band distinction and the provenance of Laura's copied case.
- Built the methods paper with LaTeX and BibTeX, checked resolved citations and layout diagnostics, rendered every page, and visually inspected the contents, architecture, equations, liability section, tables and references.
- Inspected the release archive and verified its file hashes and archive integrity as file operations. These checks do not execute the research package.

## Concrete corrections made during static review

| Issue | Correction in the delivered source/specification |
|---|---|
| Learned uncertainty lost at scoring | Evaluators require saved per-observation Student-t degrees of freedom rather than silently using a fixed value. |
| Misleading observation/seed comparisons | Exact outcomes, execution/availability times, full configured cells and the complete seed-observation grid are checked. |
| Unequal mixed-frequency weighting | Repeated observations are averaged within asset/UTC date before equal asset and date weighting; tuning, scoring and reporting share that hierarchy. |
| Too-short bootstrap blocks | Actual execution/outcome-maturity overlap constrains block length, beyond a nominal horizon alone. |
| Mutable confirmatory analysis | Exact protocol bytes and analysis-source hashes are retained; confirmation settings and primary/sensitivity status are enforced. |
| Missing multiplicity family | Complete prescribed secondary comparisons receive Holm adjustment; partial families cannot establish support. |
| Unfrozen provenance | Raw panel/macro files and their sidecar source manifests are hashed. |
| Repeated protected use | A persistent consumption receipt prevents accidental repeat use of the same freeze, including after an interrupted attempt. |
| Suite settings bypass validation | Every merged variant passes the same validation, including duplicate/configuration and budget checks. |
| Unequal or incomplete time limits | Baselines share the declared per-fit ceiling, and matrix/suite timers include preparation and completion; native calls remain cooperative. |
| All-missing perturbation scale | Nonfinite severity is rejected; explicit zero-median/unit-width fallbacks are recorded. |
| Changed macro schema across cutoff | The declared series universe survives pre-protected filtering, including all-missing series. |
| Log returns passed as simple expectations | Dated return/horizon contracts and the bounded scenario bridge make the distribution change explicit. |
| Null forecast blocks intended fallback | Null remains missing and the risk-only path requires an explicit unavailable-forecast declaration. |
| Incorrect entry-fee basis | The round-trip ledger and allocation layer use consistent post-fee self-financing accounting. |
| Incomplete selected checkpoint | Fitting that cannot complete a requested validation fails instead of presenting an unselected checkpoint as valid. |
| Mutable online checkpoint | Frozen checkpoint state is retained separately from any explicit adaptive-evaluation state. |
| Paper/code discrepancies | JEPA loss normalization, residual-tree clipping order, retention-policy comparison, constraint scope, statuses and liability-module mapping were reconciled. |

`INDEPENDENT_REVIEW.md` retains the original observations and its final source-review addendum. Keeping the earlier finding makes the review auditable; it does not mean the final code still has that visible defect.

## Not performed during the original authoring session

At authoring time, no project modules were imported, no environment/dependency installation was tested, and no authored test was executed. Those are historical statements. Later receipts establish public-data acquisition, exploratory neural and baseline fitting, saved predictions, and the stated bounded research-ledger results. They do not establish canonical market admission, protected confirmation, executable global holdings accounting, or a realized funding outcome.

Passing tests and syntax checks do not prove complete numerical stability, scientific validity, API behavior across every dependency version, provider compatibility, or absence of every bug.

## Remaining execution gates

1. Reproduce the recorded environment independently when resources admit it; earlier installation/build receipts remain historical.
2. Execute affected integrity tests for changes and retain failures; do not interpret test counts as scientific evidence.
3. Acquire and validate licensed data with truthful time, action, FX, eligibility and terminal-outcome provenance.
4. Perform the bounded first development exercise and verify artifacts, fitting cutoffs, comparison completeness and actual compute.
5. Select development work within its declared budget, then freeze before protected confirmation.
6. Resolve the official participant universe/rules and the team's numerical policy before Laura-specific allocation.
7. Complete empirical calibration, portfolio evidence, claim-to-artifact audit and final manuscript results.

The implementation's stated scope is also material. The portfolio bridge is one synchronous hypothetical allocation, and the supplied realized-return ledger accepts synchronized nonoverlapping intervals. It is not a full asynchronous settlement/holdings simulator. A positive immediate cash floor does not itself certify all ten operating liabilities. Zero-recovery defaults require a point-mass outcome/wealth treatment rather than deletion from the study.

## Repeating this static check

```bash
rtk proxy .venv/bin/python scripts/validate_source.py --output docs/STATIC_VERIFICATION_NEW.json
```

Use a new output filename: the helper refuses to overwrite an earlier report. It checks package presence, rejects duplicate JSON keys and nonfinite numeric values, and uses the standard library without importing `frem`, pytest, PyTorch or any scientific model. After a code edit, recheck affected runtime behavior; a static PASS does not substitute for that evidence.


## 13 October 2026 S6 implementation update

The additive [S6 successor](../successors/frem_s6/README.md) passed115bounded numerical integrity checks (59MiBpeak) and changed-file Ruff. See [report](../research/frem_s6_20261013/REPORT.md) and [receipt](../research/frem_s6_20261013/RECEIPT.json). Actual market-training admission failed before scientific imports:512MiBfloor/pageouts, admitted window data and a separate frozen development protocol. Zero S6market fits/backtests or protected access. All111S5frozen identities remain unchanged; closed negative results stay closed. Tiny integration-test fits are engineering evidence only. Optional XGBoost runtime, full24-cell development, independent retraining, global default-aware data support and Laura-specific allocations are unverified or blocked. Do not claim complete ultimate model, returns, novelty or calibrated funding probabilities.


## 13 October 2026 Laura case clarification and planning

The user supplied the complete case, accepted the proposed mathematical terms, and defined dual functionality as prediction plus portfolio planning. The new `successors/laura_planning_v1` module passed 28 standard-library tests at 25.55 MiB peak. It calculates contributions, dated reserves, explicit costs, shortfalls, equity targets, retained flexibility and conditional 2031 facility bounds. All 18 growth/reserve sensitivities are saved, including deficits; none is a backtest or market forecast.

Read `research/laura_plan_20261013/INVESTMENT_POLICY_PROPOSAL.md`, `RECEIPT.json` and `successors/laura_planning_v1/MATHEMATICS.md`. The liability-first policy is now a concrete team proposal. Current WInS holdings are needed for transaction-specific work, not for the strategic 2027 contribution plan. Current instrument eligibility remains unverified; the public lists found are 2025–2026. The 99% joint funding target is uncalibrated. After the 2031 communication, protected facility-minimum assets cannot be spent through rebalancing.

All 43 S6 delivery identities remain unchanged. Fresh memory samples of 162.61, 159.86 and 165.75 MiB fail the 512 MiB training floor. No S6 market fits, backtests or protected access occurred; admitted data and a separate frozen protocol remain absent. Historical negatives, receipts and freezes remain intact. Do not claim complete ultimate-model delivery or demonstrated investment advantage.


## 2026-10-14: S6 real-data intake and frozen execution package

Real SPY/TLT/GLD prices acquired; 1770 rows reconciled; six new intake/resource tests passed. Frozen 24-cell protocol includes XGBoost. Zero market fits: local 512 MiB admission failed, SSH timed out, GCP credentials expired, HF Jobs unavailable. See research/frem_s6_real_20261014/REPORT.md. No protected outcomes consumed.


## 14 October 2026: S8 successor closed

S7 diagnosis and S8 development are complete. Read [the final report](../research/frem_s8_20261014/FINAL_REPORT.md), [receipt](../research/frem_s8_20261014/RECEIPT.json), and [audit](../research/frem_s8_20261014/AUDIT.json). S7 rejected positive 2018 half-gains because both were below its fixed 0.01-nat threshold; its 2019 single-week, large-allocation candidates never cleared the round-trip fee hurdle. No 2019 tuning or new policy replay occurred.

S8 used delayed soft density gating and continuous four-interval committed allocation, frozen before 2020-2022 acquisition with one intake-only corrected source freeze before fitting. Three base fits/seeds, 153 weekly intervals, 15 forecast variants, four policies/four costs completed. NLL gains were +0.0013675 versus equal mixing and +0.0006838 versus S6 memory; both fixed interval families include zero. At 10bps, same-policy H4 wealth was S8 $98,745.27, equal forecasts $96,211.62, S6 memory $97,134.00. Cash was $100,000 and equal-weight assets $104,927.25. Prediction is inconclusive; the economic gate failed. No favorable secondary arm replaces the primary.

125 unchanged S6/S7 tests and 18 S8 checks passed in retained runs; 34,560 NLL values, 1,230 soft-gate decisions, 2,640 labels and 252 ledgers reconciled. All 319 predecessor file hashes are unchanged. Tests, DNS/intake/analysis failures, source freezes, checkpoints and ledgers remain preserved. Legacy SPY/GLD feature-cache exposure means newly acquired/held out from S6/S7 is not globally untouched confirmation. No protected 2025+ access. Do not tune or rerun the closed 2019 or 2020-2022 evaluation; a further successor requires a new hypothesis and freeze.


## 2026-10-15 S10 integrated repair (current)

Read `research/frem_s10_repair_20261015/REPORT.md`, `research/frem_s10_repair_20261015/RECEIPT.json`, `research/frem_s10_repair_20261015/FLAW_CHECKLIST.md` and `research/frem_s10_repair_20261015/VERIFICATION_FINAL.json` before resuming. Current phase is generated integration verified, scientific gates pending; older planning entries above are historical snapshots.

274 generated tests passed with warnings as errors (42.95s), full source/test strict typing/lint/format passed, and the final source wheel installed offline into a fresh environment with hash-pinned dependencies. The exact192-cell generated policy/ledger replay has only2blocks/fold and constructed3-atom forecasts. Separate tests execute all6real kernel variants, a scoped generated pilot, and24/25-date constructed retained-row statistical analysis. Do not merge these into a nonexistent full empirical run. Saved artifacts round-trip through a whitelisted typed reader; all16 secondary comparisons and both required primary block/control families are retained.

Design4 includes prospective CL15 A4 opening-sale ordering and CL16 minimum-variance tie units. Original score/wealth thresholds and completed S6/S7/S8 outcomes remain unchanged. S10 ES identifies only the total-return law, not separate price/dividend decomposition. S9 remains unconsumed. The selected48historical identities and old journal prefix were rechecked.

Remaining internal work: operation-specific admission/artifact producers and fixed worker/supervisor/freeze join, weekly certificate/state archives, full catalog closure (18bounded VERIFIED/66PARTIAL/12PENDING), complete independent saved-output scientific audit and maximum-support/length profiling. Source/calendar/close-clock admission, rights, real precision, frozen development, unused confirmation and independent empirical reproduction remain separate gates. Latest host admission failed at82.36/92.70/84.02MiB against512MiB with increasingpageouts. Never extend an earlier RAM waiver or infer scientific authorization from helper success.

No market S10 fit, policy replay, protected evaluation, research release or public submission occurred. Same-host agent review and a clean wheel installation are engineering evidence only. Reproduction commands are in `successors/frem_s10/README.md`; exact command logs and retained generated fixtures are in `research/frem_s10_repair_20261015/integrated_verification_attempt2`.
