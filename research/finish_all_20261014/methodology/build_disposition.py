"""Build an evidence-linked disposition without importing scientific packages.

This writes only the additive methodology directory. It reads checklist text,
source files and metadata/report artifacts; never raw market data or protected
outcomes. A disposition is not an execution receipt or a passing research gate.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CHECKLIST = Path('/Users/ryan/.codex/attachments/cc40b72b-495c-400e-ad01-876b6d3bb818/Pasted text.txt')

EVIDENCE = {
    's8': 'research/frem_s8_20261014/FINAL_REPORT.md',
    's8audit': 'research/frem_s8_20261014/AUDIT.json',
    's8spec': 'research/frem_s8_20261014/SPECIFICATION.md',
    's8freeze': 'research/frem_s8_20261014/freeze2.json',
    's8fresh': 'research/frem_s8_20261014/FRESHNESS_AUDIT.json',
    's8policy': 'successors/frem_s8/policy.py',
    's8router': 'successors/frem_s8/router.py',
    's7': 'research/frem_s7_20261014/REPORT.md',
    's7freeze': 'research/frem_s7_20261014/freeze2.json',
    's6': 'research/frem_s6_real_20261014/completed_ram_override/REPORT.md',
    's6model': 'successors/frem_s6/model.py',
    's6policy': 'successors/frem_s6/policy.py',
    's6kernel': 'successors/frem_s6/kernel.py',
    's6manifest': 'data/s6_real_etf_20261014/windows.json.manifest.json',
    's3': 'research/frem_s3_20261011/REPORT.md',
    's3diagnosis': 'research/frem_memory_diagnosis_20261012/diagnosis_v2.json',
    's4': 'research/frem_s4_20261012/REPORT.md',
    'ledger': 'successors/frem_portfolio_v1/engine.py',
    'laura': 'research/laura_plan_20261013/INVESTMENT_POLICY_PROPOSAL.md',
    'laurareceipt': 'research/laura_plan_20261013/RECEIPT.json',
    'legacy_metadata': 'legacy/wharton_lab_v0/data/manifests/DATA_MANIFEST.yaml',
    'regional_metadata': 'data/public_regional_factors_20261011/manifest.json',
    'protocol': 'research/protocol.json',
    'handoff': 'docs/CODEX_HANDOFF.md',
    'static': 'docs/STATIC_VERIFICATION.md',
    'state_before': 'research/finish_all_20261014/before/RESEARCH_STATE.json',
    'proposal': 'research/finish_all_20261014/methodology/SUCCESSOR_RECOMMENDATION.json',
    'review': 'research/finish_all_20261014/methodology/PRIOR_ART_AND_ADVERSARIAL_REVIEW.md',
    'exposure': 'research/finish_all_20261014/methodology/EXPOSURE_INVENTORY.json',
}

# id, status, actual finding, scope decision, exact remaining completion criterion,
# dependencies by checklist id, evidence ids. Completion means the user request,
# not merely writing this audit, has been satisfied in the specified bounded scope.
ROWS = [
 (1,'PARTIAL_EMPIRICAL_TARGET_NOT_MET','S8 zero-cost terminal wealth is 99,990.33 from 100,000. Existing year/fee summaries establish weak gross decisions, not profitable gross signals.','Complete retained-decision attribution; do not retune closed prices.','A new frozen candidate achieves positive gross utility and the declared net uplift on genuinely subsequent evidence; attribution must reconcile asset, exposure, schedule and cost effects.',[2,4,7,40,42],['s8','s8policy']),
 (2,'PARTIAL_CROSSED_CONTROLS_COMPLETE','S8 already crosses all fifteen forecasts with four policies and four costs. Density ranking and ending wealth disagree.','Preserve crossing; add separate mean/rank/scale diagnostics to retained evidence.','Mean/rank/uncertainty diagnostics and complete crossed controls explain the tradeoff, then the whole selected procedure passes its new economic contract.',[1,3,9,40,45],['s8','s8spec']),
 (3,'SPECIFIED_NOT_VALIDATED','S8 is explicitly a log-moment surrogate. Existing S6 simple-return scenarios are a separate, unintegrated decision model. Student-t log-return exponential moments diverge.','Use a direct-simple-return or fully declared finite joint scenario law in a new version.','Equation/code agreement and numerical integration/optimizer tests pass; the same forecasts are compared under both policies on an unconsumed outer window.',[4,20,25,45],['s8policy','s6policy','review','proposal']),
 (4,'SPECIFIED_NOT_VALIDATED','H4 holds weekly coefficients constant; H1 is only a declared ablation. No direct H4 target or calibrated path law was evaluated.','Directly forecast actual holding endpoints; no retrospective selection of H1.','Issue, execution, label maturity and feedback availability align for each horizon, with matched-target calibration and a frozen future comparison against unchanged approximation.',[3,5,42,45],['s8spec','s8policy','proposal']),
 (5,'PARTIAL_DIAGNOSIS_NO_VALIDATED_REPAIR','Daily trend/rms feed weekly targets; actual weekly duration varies. A fixed square-root-five scale change was not established as beneficial.','Replace a constant conversion guess with actual-horizon target calibration in a new version.','Forecast scale and mean calibrate at the same target horizon; all unchanged and duration-aware controls are evaluated without using closed outcomes for selection.',[4,9,45],['s6model','s7freeze','proposal']),
 (6,'UNTESTED_SUCCESSOR_HYPOTHESIS','S8 bases stop before 2017; only routing adapts during 2020-2022. Staleness is real, causal contribution unisolated.','Predeclare expanding quarterly refresh against frozen bases; reset versioned residual bank.','Matched refreshed/frozen procedure executes on new outer data with identical information/search budget, mature-label purge and complete model-transition receipts.',[17,42,45],['s8spec','s6model','proposal']),
 (7,'UNTESTED_SUCCESSOR_HYPOTHESIS','S6/S8 use one channel of eight daily close returns; no longer-context group has established incremental value in this ETF lineage.','One fixed additional historical-context group first; reject unhelpful groups.','Added-feature and unchanged-feature models have identical chronology, capacity/search allowance and policy; the group improves predefined subsequent predictive and economic endpoints.',[4,8,42,45],['s6model','s8spec','proposal']),
 (8,'UNTESTED_INCREMENTAL_SPECIALIZATION','Six experts exist including ridge, trend, volatility, reservoir-JEPA and XGBoost; useful specialist complementarity is not established by their names.','Improve one source of information and measure complete-mixture contribution before expanding router.','Leave-one-expert and information-matched controls quantify each expert contribution in the complete mixture on a new frozen outer window.',[7,9,11,45],['s6model','s8']),
 (9,'PARTIAL_DIAGNOSTICS_REQUIRED','Density losses and annual economics are retained; S8 scale calibration uses 2017 and does not establish economically calibrated means.','Report bias, opportunity magnitude, ranking, coverage and tails independently using retained outputs.','Complete issued forecast-to-realized mean/scale diagnostic tables with uncertainty and unchanged-forecast controls; any calibrated successor must pass chronological outer checks.',[2,4,10],['s6model','s8','s3diagnosis']),
 (10,'PARTIAL_NEGATIVE_CONTROL_EVIDENCE','S3 additional calibration worsened both public windows relative to raw forecasts. Raw and identity controls are necessary.','Identity calibration remains incumbent; any extra calibration is an inner-selection hypothesis.','Raw, identity-shrunk and selected calibrators are retained and chosen strictly within chronological inner data, then evaluated untouched; no automatic calibrated-win claim.',[9,45],['s3','s3diagnosis','proposal']),
 (11,'EMPIRICAL_REQUIREMENT_NOT_MET','S8 improves fixed-half blend by only 0.000061525 nats, with both interval families spanning zero.','Keep fixed blending as incumbent and do not add adaptive complexity.','A separate memory candidate exceeds fixed blend by a predeclared practical lower confidence bound on unconsumed data under matched budgets.',[12,14,42,45],['s8','s8spec']),
 (12,'EMPIRICAL_REQUIREMENT_NOT_MET','Local-versus-global gain is 0.000025527 nats; global S8 ends 12.84 dollars wealthier. Tiny positive conditional score intervals do not establish useful context.','Test context predicting future expert-loss differences beyond recent/global loss, not just retrieval activity.','Delayed context model beats global/recent/shuffled controls by the frozen practical threshold on new outer dates; report calibration and wealth separately.',[11,14,15,45],['s8','s8router']),
 (13,'DEFERRED_CONDITIONAL','Routing uses date-level averaged asset features and shared expert weights. Asset-specific shrinkage is a new capacity increase.','Defer until context usefulness is established; then compare one fixed hierarchical shrinkage rule.','A global/asset/shrunk comparison with identical information, delayed labels and regularization budget passes outer evaluation.',[12,15,42,45],['s6model','s8router','proposal']),
 (14,'PARTIAL_PRIOR_CONTROLS_INCONCLUSIVE','S3 compared real, random, recent and other memory arms; matched random was competitive and no arm improved both windows.','Retain those negatives; age-matched and unconditional controls belong to a separately justified memory test.','All real/shuffled/age-matched/recent/unconditional controls use equal support and delayed availability; meaningful held-out relevance and contribution is demonstrated.',[12,15,16,45],['s3','s3diagnosis','s4']),
 (15,'PARTIAL_SUPPORT_MEASURED','S3 diagnosis records neighbor age, effective count, occupancy and gate behavior. Support is small and not independent merely because rows are stored.','Do not increase capacity; add support/contribution diagnostics before a new memory hypothesis.','Every retrieval reports normalized weight concentration, effective support, age and regime coverage, plus delayed realized incremental contribution; limits and correlations explicit.',[14,16],['s3diagnosis','s6kernel']),
 (16,'PARTIAL_TESTED_NO_ESTABLISHED_BENEFIT','S4 altered distance but all corrections selected zero; deployed equality cannot identify whether distance itself helps.','Do not repeat S4 or tune its gates; only one theory-motivated retention/kernel change in a new memory version.','Equal-capacity global/recent/old-kernel controls and the new rule are fixed before a new window and show meaningful incremental benefit.',[12,14,15,45],['s4','s6kernel']),
 (17,'SPECIFIED_FOR_FUTURE_REFITS','Current frozen-base banks avoid transition mixing, but a rolling-refit bank compatibility contract is absent.','Version model, feature scaler, calibration, horizon and target definition on every residual row; rebuild with out-of-fit predictions.','Transition tests reject incompatible residuals; a new-model bank has documented strictly historical out-of-fit residual construction and no future-label access.',[6],['s6model','proposal']),
 (18,'PARTIAL_ABLATIONS_NO_USEFUL_EFFECT','S6 completed no-JEPA and immediate/unquantized arms across seeds; effects were nearly identical and did not establish useful benefit. Full simpler-branch/compute-matched attribution remains open.','Preserve negative ablations; exclude extra components from minimal successor.','Complete matched supervised/simple-branch/JEPA/update controls with actual resources and useful outer effect, or retain simpler incumbent and make no component-benefit claim.',[7,8,42],['s6','s6model','proposal']),
 (19,'COMPLETE_BOUNDED_SCOPE','Zero correction is accepted in S3/S4; equal forecasts and fixed-half blend are evaluated incumbents in S8. No compulsory memory activation.','Preserve the simple fallback as a legitimate result.','Already satisfied for completed exploratory comparisons; future versions must keep the incumbent and cannot force memory to pass a cosmetic activity test.',[],['s3','s4','s8']),
 (20,'PARTIAL_IMPLEMENTED_NOT_INTEGRATED','S6 provides shrinkage covariance and joint heavy-tail/simple-return/CVaR machinery; S8 primary remains diagonal.','Integrate one joint scenario decision rule under an additive source freeze, not into closed S8.','Covariance chronology/PSD and actual scenario moment/tail checks pass; common forecasts cross diagonal/joint policy under identical horizon, costs and constraints.',[3,4,25,45],['s6policy','s8policy','proposal']),
 (21,'UNTESTED_SIGNAL_DECAY_POLICY','H1/H4 and costs were compared, but signal-persistence estimation, partial adjustment and no-trade regions were not jointly validated.','Estimate decay only inside permitted development and predeclare one trade-speed rule.','Decay estimate uncertainty and identical-forecast hold/partial/no-trade controls are retained; new net utility improves beyond simply reducing exposure.',[4,6,20,45],['s8','s8policy','ledger','review']),
 (22,'PARTIAL_DESCRIPTIVE_ATTRIBUTION','S8 2020 path gains 0.990 percent versus equal-weight 19.929 percent; later downside reduction does not recover that deficit.','Complete saved-decision asset/exposure/holding-duration attribution; never add a hindsight 2020 regime rule.','Attribution reconciles all interval contributions, costs and terminal wealth and separates forecast misses from policy/risk-budget effects.',[1,2,23,26],['s8','s8policy']),
 (23,'NOT_YET_MATCHED_CONTROL','Original equal-weight asset benchmark has different risky exposure than S8 capped at 80 percent.','Retain original benchmark and add 80/20 cash/equal-weight plus same-constraint passive diagnostics under a new diagnostic label.','New benchmark accounting passes exact same dates, costs, review schedule, caps and liquidation checks; original primary result is not overwritten or promoted.',[25,26,40],['s8spec','s8policy','ledger']),
 (24,'UNTESTED_ROBUST_SIZING','Density calibration and restrained exposures do not establish that uncertainty-aware mean sizing improves net utility.','Use one predeclared shrinkage control in the simple successor.','Point/shrunk sizing use identical forecasts, support, joint-risk/cost constraints and outer data; utility or dated shortfall improves with exposure reduction separately reported.',[9,20,23,45],['s6policy','s8policy','proposal']),
 (25,'BLOCKED_REALISTIC_INPUTS','S8 assumes zero cash yield, ex-date cash dividends, weekly reinvestment, no payment delay/taxes and simplified costs/fills.','Maintain this narrow exploratory label; acquire dated cash/execution/action inputs before executable claims.','Cash returns, spread/slippage/impact, settlement/payment timing and actual availability are sourced and matched across candidate/benchmarks, with sensitivity and accounting tests.',[43,49],['s8spec','s6manifest']),
 (26,'PARTIAL_FEE_BILL_DISTINGUISHED','S8 report correctly states zero-cost versus 10-bps paths differ in decisions and the 810.24 dollar fee bill is not their terminal-wealth difference.','Add fixed-decision fee and policy-response bridge without new policy selection.','A fixed-weight/cost counterfactual reconciles cash-flow timing and compounding, with direct fees, fee opportunity cost and changed-policy response separately reported.',[1,23],['s8','s8policy','ledger']),
 (27,'PARTIAL_IMPLEMENTED_NOT_VALIDATED','S6 includes multi-period scenario machinery; no validated wealth advantage follows from optimizer correctness.','Keep a simple rule and unchanged surrogate under identical forecasts/constraints.','End-to-end crossed one-/multi-period/simple policy comparison and solver audits complete on a new window; claim limited to measured decision-layer increment.',[3,4,20,25],['s6policy','review','proposal']),
 (28,'PARTIAL_STRATEGIC_PLAN_COMPLETE','Liability-first 2027/2028 contribution plan, dated reserves, 2031 protected minimum and conditional facility arithmetic are implemented; 99 percent joint funding is uncalibrated.','Preserve strategic plan; do not reinterpret ETF terminal wealth as Laura funding success.','Actual mandate/holdings/prices and dated cash flows drive a verified joint-shortfall simulation or conditional cash-flow match; no probability certification from arithmetic tests.',[25,49],['laura','laurareceipt']),
 (29,'DEFERRED_CONDITIONAL','Current experts have a single weekly target; distinct direct multi-horizon heads are not validated.','Allow only H1 primary and H4 diagnostic in the proposed bounded horizon study; no architecture expansion.','Each head has actual-horizon targets, purged chronology, calibration and frozen metric; new outer results justify retaining each head.',[4,5,7,45],['s6model','proposal']),
 (30,'DEFERRED_CONDITIONAL','Within-date relative rankings are plausible but unproved, especially with only three ETFs.','Do retained rank diagnostics first; relative-return model is a separate later hypothesis.','Relative and absolute models share data/search budget; held-out rank and economic increments beat simple equal/risk controls.',[2,7,9,42,45],['s8','proposal']),
 (31,'BLOCKED_TIMESTAMPED_INPUTS','No verified real-time macro/rate vintage pipeline is attached to the S8 ETF forecasts. Existing regional factor data are current-vintage aggregates.','Exclude from first successor; do not backfill current releases.','Exact publication/vintage timestamps and transformations pass availability checks; a single added feature group shows incremental subsequent value.',[7,43,45],['regional_metadata','handoff','proposal']),
 (32,'BLOCKED_DATA_AND_UNIVERSE','Three ETFs cannot supply company valuation/earnings/breadth with historical universe eligibility.','Exclude until point-in-time equity source and universe are supplied.','Dated fundamentals, reporting lags, identifier/membership history and inactive/terminal securities support an appropriately frozen universe and incremental group test.',[31,43],['s6manifest','handoff','proposal']),
 (33,'DEFERRED_CONDITIONAL','S8 soft adaptation is delayed but does not establish a separately calibrated regime detector.','No hindsight regime labels; global/no-regime incumbents remain.','A time-available detector and fixed transition rule outperform global/no-regime controls on new dates with identical information/tuning budget.',[12,21,42,45],['s8router','proposal']),
 (34,'DEFERRED_CONDITIONAL','Asset-specific hierarchical sharing has not been established; existing averaging shares information across contemporaneous assets.','Defer until there is enough cross-asset evidence and context signal.','All pooled/unpooled/shrunk models respect common availability, report effective independent support and show an outer increment.',[13,42,43,45],['s6model','proposal']),
 (35,'PARTIAL_IMPLEMENTATION_SCOPE_LIMITED','Finite log return cannot encode total loss; portfolio engine accepts simple return -1 and S6 scenario generator has floor/default behavior. Historical inactive-security coverage is absent.','Preserve loss handling and do not invent default frequency; broaden only with appropriate data.','Zero-recovery/default accounting and actual post-floor distribution calibrate against terminal-outcome data; no disappearing default rows or log-transform loss deletion.',[3,20,43],['ledger','s6policy','s6manifest','review']),
 (36,'DEFERRED_NO_CAPACITY_JUSTIFICATION','No evidence shows useful signals are capacity-limited; no-JEPA/immediate controls are almost identical.','Do not add larger sequence models or learned retrieval now.','Simpler models first demonstrate useful signal and capacity diagnostics predict a specific failure; a compute/information/search-matched expansion then wins on a new outer window.',[7,8,18,42],['s6','s3','proposal']),
 (37,'BLOCKED_PROVENANCE_AND_CONTAMINATION','No timestamped news corpus or model-training contamination audit was supplied.','Exclude from first successor; modern model historical knowledge cannot be treated as contemporaneously available.','Publication timestamps, revision history, model knowledge-cutoff/exposure assessment and matched text-off controls support a valid new study.',[31,43,45],['proposal']),
 (38,'DEFERRED_SIMULATOR_NOT_VALIDATED_FOR_RL','Simple execution assumptions and absent admitted new evidence make learned allocation premature.','Do not deploy RL; first validate simulator and simple policies.','Simulator/accounting/cost/timing adversarial tests and held-out behavior pass; a fixed-budget RL study beats simple policies without exploiting simulator artifacts.',[3,20,25,27,45],['s8spec','ledger','proposal']),
 (39,'BLOCKED_UNSPECIFIED_SOURCE','External forecast adapter exists; the exact new framework repository/model/source was not supplied.','Do not invent a substitute or arbitrary framework.','Receive exact source/version and forecast semantics, then implement adapter, audit horizon/availability/asset/currency contract and run a fair frozen comparison.',[4,43,45],['ledger','handoff']),
 (40,'SPECIFIED_NOT_EMPIRICALLY_PASSED','S8 original gate only required positive differences; the requested 6 percent minimum and 36 percent stretch were stronger. New proposal states full benchmark/cost/risk/cash-flow contract.','Preserve S8 original gate and separately report whether the requested wealth margins are met.','Freeze exact candidate, required benchmarks, outer horizon, cash flows, costs, liquidation, constraints and point/reliability thresholds before new outcome access.',[23,25,42,45],['s8spec','s8','proposal']),
 (41,'COMPLETE_BOUNDED_SCOPE','S8 prediction and economic gates are separately reported. Gains 0.0013675 and 0.0006838 do not meet 0.01-nat lower-bound requirement.','Retain original predictive threshold and INCONCLUSIVE disposition; no weakening to obtain support.','Already satisfied as evidence separation; a future performance claim requires the preserved predictive gate under the correct comparable density target.',[],['protocol','s8','s8spec','proposal']),
 (42,'NOT_ENOUGH_INDEPENDENT_EVIDENCE','153 common weekly dates from three ETFs and three optimization seeds do not establish broad independent-market reliability.','Perform procedure-level precision planning before choosing any new outer study.','Document serial/cross-asset dependence, effective blocks, expected interval width/power and actual data availability; reject a design unable to resolve the practical claim within budget.',[40,43,44,47],['s8spec','s8','proposal','review']),
 (43,'BLOCKED_CANONICAL_DATA','Public quotes are current-vintage survivor histories; vendor publication timing, inactive securities, historical membership, FX and terminal completeness are not certified.','Keep retrospective sample scope; do not relabel it point-in-time global data.','Provider/rights/provenance/action/availability/universe/terminal/FX contracts pass for the specific intended claim before canonical execution.',[25,31,32,35],['s6manifest','handoff','regional_metadata']),
 (44,'COMPLETE_BOUNDED_METADATA_INVENTORY','Known S6/S7/S8 windows, S3/S4 windows and legacy ETF-cache coverage are recorded; current new-download freshness is not global freshness.','Maintain metadata inventory with scope caveat; no protected raw reads.','This bounded inventory is complete for checked authoritative manifests and freezes; any new source requires a fresh identity/exposure admission before being called untouched.',[],['s8fresh','legacy_metadata','s8freeze','s7freeze','s3','s4','exposure']),
 (45,'SPECIFIED_NOT_EXECUTED','Earlier calibration/selection uses chronological partitions, but no complete new nested feature/model/calibration/policy selection procedure has been executed.','Specify all selection inside inner time folds; no secondary winner promotion.','An implemented outer-temporal harness repeats every selected operation using only earlier data and preserves complete selection history under a fixed budget.',[42,44,47],['s7freeze','s8spec','proposal','review']),
 (46,'COMPLETE_BOUNDED_SCOPE','S8 retains all declared seeds, fifteen forecasts, four policies, four costs, interval families and failures; audit verifies 319 predecessor identities.','Keep append-only freeze/result history; do not drop arms or overwrite repaired runs.','Satisfied for completed S8 artifacts subject to continuing preservation; new work must retain the same completeness and source identity discipline.',[],['s8audit','s8','s8freeze']),
 (47,'NOT_PROCEDURE_LEVEL_UNCERTAINTY','S8 intervals are conditional on saved forecasts/decisions and omit retraining/model-selection uncertainty.','State this limit and design procedure-level outer evaluation; no bootstrap relabeling.','Every resampled/outer evaluation used for the procedure claim includes fitting and inner selection with justified time dependence; coverage assumptions and precision limitations are explicit.',[42,45],['s8spec','s8','review','proposal']),
 (48,'NOT_INDEPENDENTLY_RETRAINED','Same-host arithmetic reconstruction verifies values and ledgers; independent clean-environment retraining is not recorded.','Prepare reproducibility package and separate engineering review from retraining.','An independent environment/runner executes unchanged frozen source/data/protocol in new output directories and reconciles major predictions, analyses, figures and resource receipts.',[43,46],['s8audit','s8','static']),
 (49,'BLOCKED_CURRENT_MANDATE_INPUTS','Current eligibility, executable prices, holdings and instrument-specific costs are not verified; strategic contribution plan exists.','Complete strategic local plan; require actual participant materials only for executable allocation.','Current dated approved universe/rules, holdings/NAV, prices, costs, settlement, risk/liquidity/FX constraints and team mandate supplied and validated.',[25,28],['laura','laurareceipt','handoff']),
 (50,'PENDING_CANONICAL_RECONCILIATION','Canonical state and older summaries retain historical zero-fit/blocker fields even though S6/S7/S8 completed. Latest explicit sections supersede them.','Add unambiguous latest-evidence pointers and mark historical fields as historical; preserve failures.','Canonical current status resolves to S6 24 fits, S7 3 fits and S8 3 fits and their terminal outcomes; historical blocked attempts are retained with dates and supersession links.',[],['state_before','handoff','static','s6','s7','s8']),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exposure():
    sources = [
        'research/frem_s8_20261014/FRESHNESS_AUDIT.json',
        'research/frem_s8_20261014/freeze2.json',
        'research/frem_s7_20261014/freeze2.json',
        'research/frem_s6_real_20261014/freeze2/protocol.json',
        'data/s6_real_etf_20261014/windows.json.manifest.json',
        'research/frem_s3_20261011/freshness_inventory.json',
        'research/frem_s3_20261011/freshness_output_check.json',
        'research/frem_s3_20261011/REPORT.md',
        'research/frem_s4_20261012/REPORT.md',
        'configs/frem_s3_public.json',
        'data/public_equity_crises_v1/manifest.json',
        'data/public_regional_factors_20261011/manifest.json',
        'legacy/wharton_lab_v0/data/manifests/DATA_MANIFEST.yaml',
    ]
    inventory = {
        'status': 'BOUNDED_REPOSITORY_METADATA_INVENTORY_COMPLETE',
        'as_of': '2026-10-14',
        'method': 'Read source/manifest/freeze metadata and completed reports; no raw quote/cache arrays or protected outcomes opened.',
        'protected_outcomes_read_by_this_audit': False,
        'claim_limit': 'Authoritative known lineage metadata only; absence of recorded exposure is not proof of global ignorance or independent financial histories.',
        'windows': [
            {'scope': 'SPY/TLT/GLD', 'window': '2005-2016', 'status': 'EXPOSED_TRAINING', 'evidence': ['research/frem_s8_20261014/freeze2.json'], 'eligible_as_untouched_outer_test': False},
            {'scope': 'SPY/TLT/GLD', 'window': '2017-2018', 'status': 'EXPOSED_CALIBRATION_SELECTION_AND_S6_EVALUATION', 'evidence': ['research/frem_s6_real_20261014/freeze2/protocol.json','research/frem_s7_20261014/freeze2.json'], 'eligible_as_untouched_outer_test': False},
            {'scope': 'SPY/TLT/GLD', 'window': '2019', 'status': 'CLOSED_S7_EVALUATION_NO_RETUNING', 'evidence': ['research/frem_s7_20261014/freeze2.json','docs/CODEX_HANDOFF.md'], 'eligible_as_untouched_outer_test': False},
            {'scope': 'SPY/TLT/GLD', 'window': '2020-2022', 'status': 'CLOSED_S8_EVALUATION_NO_RETUNING', 'evidence': ['research/frem_s8_20261014/freeze2.json','research/frem_s8_20261014/FRESHNESS_AUDIT.json'], 'eligible_as_untouched_outer_test': False},
            {'scope': 'SPY/GLD legacy feature cache', 'window': '2020 onward through historical download day', 'status': 'LEGACY_CACHE_EXPOSURE_DECLARED_IN_METADATA', 'evidence': ['legacy/wharton_lab_v0/data/manifests/DATA_MANIFEST.yaml'], 'eligible_as_untouched_outer_test': False, 'note': 'Metadata mtime 2026-09-26; no cache values opened. Legacy presence does not authorize access to original 2025+ protected outcomes.'},
            {'scope': 'Twelve current-survivor US equities', 'window': '2023-October2024', 'status': 'CLOSED_S3_AND_EXPLICIT_REUSED_S4_DEVELOPMENT', 'evidence': ['research/frem_s3_20261011/REPORT.md','research/frem_s4_20261012/REPORT.md'], 'eligible_as_untouched_outer_test': False},
            {'scope': 'Public 24-stock survivor crisis sample', 'window': 'Requested 1998-2024 histories; six historical crisis/recovery windows studied', 'status': 'ACQUIRED_AND_PARTLY_EVALUATED_CURRENT_VINTAGE_DATA', 'evidence': ['data/public_equity_crises_v1/manifest.json','docs/CODEX_HANDOFF.md'], 'eligible_as_untouched_outer_test': False, 'note': 'No remaining segment is certified untouched merely because it lacks a named experiment here.'},
            {'scope': 'Seven regional factor research portfolios', 'window': '1926/1990-December2024 normalized metadata coverage', 'status': 'ACQUIRED_NUMERICALLY_PARSED_ZERO_MODEL_CELLS_IN_MANIFEST', 'evidence': ['data/public_regional_factors_20261011/manifest.json'], 'eligible_as_untouched_outer_test': None, 'note': 'Potential separately scoped research data, not a certified unseen tradable-wealth test. Current-vintage revisions, overlapping regions and unverified historical availability remain. Raw archives may contain 2025+ and remain opaque.'},
            {'scope': 'All original protected data', 'window': '2025 onward', 'status': 'PROTECTED_DO_NOT_ACCESS', 'evidence': ['research/protocol.json','docs/CODEX_HANDOFF.md'], 'eligible_as_untouched_outer_test': False, 'note': 'No mature supported candidate, full confirmatory freeze or precision justification has established a reason to spend it.'},
        ],
        'uninspected_development_window_identified': False,
        'run_ready': False,
        'decision': 'No new ETF evaluation window is certified available. Close current studies honestly and require new data/exposure/precision admission for a successor.',
        'source_identities': [{'path':p,'sha256':digest(ROOT/p)} for p in sources],
    }
    (HERE/'EXPOSURE_INVENTORY.json').write_text(json.dumps(inventory, indent=2)+'\n')


def main():
    write_exposure()
    text = CHECKLIST.read_text()
    requests = {int(n): request for n,request in re.findall(r'^- \[ \] (\d+)\. (.+)$',text,re.MULTILINE)}
    assert set(requests) == set(range(1,51))
    assert [r[0] for r in ROWS] == list(range(1,51))
    items=[]
    for number,status,finding,decision,criterion,deps,keys in ROWS:
        paths=[EVIDENCE[k] for k in keys]
        for p in paths:
            assert (ROOT/p).is_file(), p
        items.append({
            'id':number,'request':requests[number],'status':status,
            'assessment_complete':True,
            'scientific_requirement_completed':status.startswith('COMPLETE_'),
            'finding':finding,'scope_decision':decision,
            'exact_completion_criterion':criterion,'dependencies':deps,
            'evidence':[{'path':p,'sha256':digest(ROOT/p)} for p in paths],
        })
    payload={
        'schema_version':1,'date':'2026-10-14',
        'title':'All 50 requested failure/fix items: evidence-linked disposition',
        'scope':'Current-source and retained-evidence audit. Proposed or deferred work is never counted as executed scientific completion.',
        'checklist_attachment_sha256':digest(CHECKLIST),
        'overall':'CURRENT_RESEARCH_INCONCLUSIVE_PREDICTIVE_TARGET_NOT_MET_SUCCESSOR_NOT_RUN_READY',
        'new_market_fits':0,'protected_outcomes_accessed':False,
        'assessment_count':len(items),
        'scientific_requirement_completed_count':sum(i['scientific_requirement_completed'] for i in items),
        'status_counts':dict(Counter(i['status'] for i in items)),
        'current_session_addenda':[],
        'items':items,
    }
    (HERE/'CHECKLIST_50_DISPOSITION.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Disposition of all 50 requested items','',
           'All 50 items were assessed against source and retained evidence. **Assessment is complete; the 50 scientific requirements are not all complete.** Deferred options were not implemented merely to fill a checklist. The original predictive result remains inconclusive, the wealth target remains unmet, and a new successor is not run-ready.','',
           'The JSON companion includes the exact original request, evidence hashes, dependencies, scope decision and completion criterion for every item. True completion flags are limited to the stated bounded scope.','',
           '| Item | Status | Finding and next scientific requirement |','|---:|---|---|']
    for i in items:
        title=i['request'].split('. ')[0]
        lines.append(f"| {i['id']}. {title} | {i['status']} | {i['finding']} **Criterion:** {i['exact_completion_criterion']} |")
    lines+=['','These are audit dispositions as of this build. Any subsequent same-session diagnostic or canonical reconciliation is recorded as an explicit addendum, not silently inferred from a proposal.','']
    (HERE/'CHECKLIST_50_DISPOSITION.md').write_text('\n'.join(lines))
    print(json.dumps({'items':len(items),'scientific_requirement_completed':payload['scientific_requirement_completed_count'],'status':'AUDIT_WRITTEN_NOT_SCIENTIFIC_EXECUTION'}))


if __name__ == '__main__':
    main()
