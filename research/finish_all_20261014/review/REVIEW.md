# Same-host adversarial evidence review, 14 October 2026

The retained S8 result survives this bounded review: prediction is **INCONCLUSIVE**, its frozen economic point-win gate failed, and the requested 6% minimum / 36% stretch terminal-wealth target was not reached. No arithmetic, chronology, frozen-source identity, or equation/implementation defect that would justify reopening the completed S8 experiment was found. This is a separate AI-agent review on the same host, not human review, a clean environment, independent retraining, or provider-data certification.

## Concrete completion gaps found

1. **P2 — The paper supplied at session start omits S6–S8.** The preserved `../before/paper/main.tex` line 26 identifies version 0.4 dated 12 October and its abstract/conclusion mainly describe S2–S4. It cannot be the final report of the latest work. The parent agent is preparing an artifact-derived successor section and updated conclusions. Preserve the earlier manuscript, and check the new section against this review's numerical receipt.
2. **P2 — Historical resource blocks are still presented as current subordinate state.** In `../before/RESEARCH_STATE.json`, `s6_development_20261013.status`, `market_fits`, and `xgboost_runtime_exercised`, plus `s6_real_20261014.latest_requested_full_run`, still contain blocked/zero values, while the later S6/S7/S8 completion entries show successful exploratory execution. Keep the blocked receipts but label these entries as historical and make current status resolve explicitly to completion receipts.
3. **P2 — The root wheel is not an S8 reproduction package.** `pyproject.toml:26–27` discovers packages only under `src`; S6/S7/S8 and the saved-output auditors live outside that tree. A final artifact must include the successor source dependency closure, exact source/data manifests, saved outputs, environment lock, audit/analysis helpers, and root-relative instructions. A successful root-wheel build would not establish successor installability or reproduction. A source/evidence bundle can address this without modifying frozen scientific source.
4. **Scientific-scope boundary — S8 did not freeze the user's 6% wealth minimum.** `research/frem_s8_20261014/SPECIFICATION.md:32` requires positive differences against four economic controls. The requested 6%/36% thresholds are stronger. They can be reported retrospectively as target arithmetic, but cannot be retroactively renamed the frozen S8 gate. This does not change the failure verdict: the primary fails even the weaker frozen gate.

The first three findings concern finishing and delivery, not evidence that the market experiment was invalid. None licenses outcome-driven fitting, policy selection, or reopening the 2019/2020–2022 studies.

## Fresh checks and their limits

- The unchanged standard-library saved-output auditor passed again: 35 frozen files, 319 predecessor identities, 2,640 raw labels, 34,560 NLL values, 1,230 delayed-gate decisions and 252 ledgers. Maximum NLL error was 4.44e-15; accounting residual was 4.37e-11 dollars. Result: `s8_saved_audit.json`.
- A separately authored standard-library review verified 22 S6, 16 S7 and 35 S8 source identities; independently recalculated S8 mean score differences, benchmark uplifts and wealth targets; and checked original/corrected freeze chronology. Result: `claims_and_identities.json`; reusable helper: `verify_saved_claims.py`.
- The original freeze was at 14:51:05 UTC, before the retained downloads. Corrected freeze2 was at 14:54:29 UTC after acquisition and before the 14:55:21 UTC execution receipt. The recorded intake amendment and original freeze identity remain intact. A source freeze before acquisition and a corrected freeze before fitting are distinct claims.
- All 18 current S8 tests passed in 33.52 seconds. They include the small generated integration fit, which is engineering evidence only. Exact command and log are in `RECEIPT.json` and `s8_tests.log`. The unchanged historical 125 S6/S7 tests were not rerun. Test memory/CPU peak was not measured during this review.
- Bootstrap endpoint values were inspected against frozen thresholds, not independently resampled by the new standard-library claim checker. The saved-output auditor reconstructs densities and ledgers, not independent retraining. S6/S7 raw-score audits were read as historical receipts; only their source identity checks were repeated here.

No market fit, policy replay, new forecast, protected 2025+ access, data download, external communication, or publication occurred in this review. S8 immutable sources, prior outputs and freezes were not edited.

## Checked mathematical correspondence

`successors/frem_s8/router.py:19–37` implements the stated delayed fixed-share density strategy update. Mature feedback requires availability strictly before issue. For each released loss vector, it uses normalized `q * exp(-loss)`, then adds a share of 1/52 toward the two-strategy uniform prior. The saved-output reconstruction checks 1,230 decisions from saved loss histories.

For the concave policy surrogate, fix budget multiplier λ ≥ 0 and write `b_i = H μ_i - c - λ`. The stationary candidates above and below the current weight are `(b_i-c)/(8 H v_i)` and `(b_i+c)/(8 H v_i)`. If the subgradient straddles zero the current weight is optimal; clipping imposes the box. The summed weight is monotone in λ, so the implemented multiplier search imposes the 0.8 budget. This matches `policy.py:15–58` and the specification. The grid, zero-cost analytic, permutation and holding-accounting tests pass.

The correct solution of that surrogate does not make it exact expected simple-return utility. The inputs are Student-t log-return mixture moments; exponential moments do not exist, diagonal risk omits dependence, and H=4 repeats one-week coefficients. The implementation/specification already acknowledges those limits. Future horizon, joint-risk, information, calibration and refitting experiments are new scientific work, not bug repairs demonstrated by this review.

## Verified numerical claims

S8 10-bps H4 terminal wealth is **$98,745.273682**, versus $96,211.618095 for equal forecast H4, $97,133.997143 for S6 memory H4, $100,000 cash, and $104,927.246746 equal-weight assets. Its relative uplifts are **+2.633420%, +1.658818%, −1.254726%, and −5.891676%** respectively. All fail 6%; all fail 36%.

The equal-weight-assets target corresponds to $111,222.881551–$142,701.055575, leaving a $12,477.607869–$43,955.781893 gap. This requires 12.636157%–44.514315% more than S8's current ending wealth. The zero-cost candidate ends at $99,990.333156; differences across cost settings are not pure fee attribution because policy decisions also change.

The independently recalculated prediction gains are +0.001367499685 nats versus equal mixing and +0.000683787961 versus S6 memory. Both retained 6- and 12-date interval families include zero; none has a lower bound above 0.01. Fixed-half gain is +0.000061525192 with intervals spanning zero. The locality gain is only +0.000025527154, but both retained intervals are strictly positive (approximately 3e-6 to 5e-5); wording that says every secondary interval includes zero would be wrong. This tiny descriptive locality result does not satisfy practical support, and the global control ends slightly wealthier.

The point-selected ridge/H1 $122,905 outcome cannot replace the frozen S8 primary. The three seeds are optimization repeats on one ETF history. Conditional date-block portfolio intervals omit refitting and procedure selection uncertainty. Legacy SPY/GLD feature-cache exposure makes the evaluation held out from S6/S7 fitting, not globally untouched. These limits are material in the final paper and release statement.

## Reproduction and release requirements

For local reproduction from saved evidence, retain the full data/output/source identities needed by `audit_saved.py`, the S6/S7 dependency files, original and corrected S8 freezes, predecessor manifest, consumption receipt, test logs, raw predictions/checkpoints, all forecast/policy/cost ledgers, and analysis/table/figure helpers. Preserve root-relative directory layout. Running analysis or audit into a new destination is allowed; do not delete the study-consumption receipt to rerun the closed experiment.

The bundle should label data and case redistribution status explicitly and remain local until authorization and rights are resolved. Citation/source review, human authorship/affiliation confirmation, venue processing, clean-environment reproduction and a new confirmatory hypothesis are separate gates; this review does not fulfill them. The defensible release claim is a completed bounded development report with inconclusive predictive evidence and failed economics, not completion of the broader canonical or worldwide program.
