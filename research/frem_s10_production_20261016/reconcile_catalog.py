"""Static evidence reconciliation; imports no project or numerical code."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
BASE = ROOT / "research/frem_s10_repair_20261015/VERIFICATION_FINAL.json"
TESTS = ROOT / "successors/frem_s10/tests"
baseline = json.loads(BASE.read_text())
collection = (DEST / "catalog_collection_attempt1.log").read_text().splitlines()
collected = [line for line in collection if line.startswith("tests/") and "::" in line]
assert len(collected) == len(set(collected)) == 360

nodes = {}
queued = {}
for path in sorted(TESTS.glob("test_catalog_*_audit.py")):
    tree = ast.parse(path.read_text())
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or not function.name.startswith("test_"):
            continue
        doc = ast.get_docstring(function) or ""
        match = re.search(r"TEST-(\d{3}(?:/\d{3})*)", doc)
        if match is None:
            continue
        family = f"tests/{path.name}::{function.name}"
        exact = [node for node in collected if node == family or node.startswith(family + "[")]
        for number in match.group(1).split("/"):
            identity = "TEST-" + number
            if path.name == "test_catalog_boundaries_audit.py":
                queued.setdefault(identity, []).append(family)
            else:
                assert exact, family
                nodes.setdefault(identity, []).extend(exact)

def families(identity, *prefixes):
    for prefix in prefixes:
        matches = [node for node in collected if node == prefix or node.startswith(prefix + "[")]
        assert matches, prefix
        nodes.setdefault(identity, []).extend(matches)

for identity in ("TEST-018", "TEST-019", "TEST-021", "TEST-022", "TEST-039", "TEST-040", "TEST-072", "TEST-073"):
    families(identity, "tests/test_model_archive_audit.py::test_all_weekly_generations_restore_exact_maturities_and_issued_laws")
for identity in ("TEST-035", "TEST-037", "TEST-073"):
    families(identity, "tests/test_model_archive_audit.py::test_solver_certificates_survive_compact_exact_roundtrip")
for identity in ("TEST-074", "TEST-077", "TEST-079"):
    families(identity, "tests/test_model_archive_audit.py::test_failed_generation_retains_prefix_and_forbids_automatic_continuation")
families("TEST-073", "tests/test_model_archive_audit.py::test_rejects_mismatched_saved_state_mixture_and_maturity")
families("TEST-021", "tests/test_model_archive_audit.py::test_callback_retains_nonselected_origins_and_failure_stops_next_issue")
families("TEST-054", "tests/test_policy_benchmarks_audit.py::test_fixed_benchmarks_and_entry_fees", "tests/test_policy_benchmarks_audit.py::test_buy_hold_preserves_dividends_and_drift_above_policy_caps")
families("TEST-058", "tests/test_readback_audit.py::test_resigned_partial_cannot_be_claimed_complete", "tests/test_readback_audit.py::test_fully_resigned_tape_envelope_cannot_admit_duplicate_variant")
families("TEST-060", "tests/test_experiment_audit.py::test_frozen_state_freezes_entire_law_separately_per_fold_and_controls_share_laws")
families("TEST-078", "tests/test_production_runtime.py::test_cli_requires_explicit_execution_and_never_accepts_resume")
families("TEST-081", "tests/test_readback_audit.py::test_full192_readonly_roundtrip_preserves_arrays_accounting_and_scope")
families("TEST-083", "tests/test_experiment_audit.py::test_full_192_generated_matrix_completes_and_each_fold_starts_from_cash", "tests/test_experiment_audit.py::test_late_cell_failure_retains_all_previous_cell_evidence")
for prefix in ("test_complete_constructed_registry_is_not_scientific_authorization", "test_catalog_gaps_never_disappear_behind_passing_counts", "test_outcome_for_uncollected_test_is_rejected"):
    families("TEST-094", "tests/test_catalog_readiness_audit.py::" + prefix)
mutants = json.loads((DEST / "semantic_mutants_attempt2/receipt.json").read_text())
for mutant in mutants["mutants"]:
    families("TEST-096", mutant["test"])

# Each upgrade addresses the specific previously recorded missing case. These
# are bounded semantic evidence reviews, not a final-source execution receipt.
upgrades = {1, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 37, 38, 40, 42,
            48, 49, 52, 53, 54, 58, 59, 64, 65, 66, 69, 71, 78, 83, 94, 96}
notes = {
1: "All declared Unit pairs are checked with identical numeric shape: mismatched units reject without conversion.",
2: "Added rank/length/empty/bool/object/complex/nonfinite numeric boundaries; full semantic asset/feature-order rejection remains separate.",
3: "Added NaN/infinity at the shared array boundary; exhaustive negative shares/cash/weight/probability domain permutations remain incomplete.",
4: "Timezone-free timestamps now reject; retain strict JSON/unknown-schema coverage. Full malformed-manifest matrix remains partial.",
6: "Source producer has exact raw hash/calendar comparison, current-vintage scope, and durable admission failure tests; action/sidecar conventions and integrated no-consumption matrix still require complete coverage.",
7: "Generated source tests cover early close and protected numeric sentinels; unsupported terminal-event and complete continuous-ledger corruption cases remain missing.",
9: "New scale-floor boundary test was corrected after a floating-reduction oracle mistake; pending resource-gated rerun. 63/64/cutoff tests retained.",
12: "Added exact equal/future feature-availability rejection; quote-valuation versus forecast-input integration remains a separate boundary obligation.",
13: "Added irregular thirteen-interval feature-order hand calculation; it passed before the final test-file correction. Final candidate rerun pending.",
17: "Existing full-kernel label perturbation checks protect current issued laws. Complete later-price/dividend/action/prefix-hash perturbation is not yet demonstrated.",
18: "Simultaneous-maturity ordering passed in the last boundary batch; weekly archive retains exact checkpoints and feedback. Final candidate rerun pending.",
20: "Explicit 521-to-520/260 library retention and 105-to-104 feedback eviction passed. This is membership construction, not maximum-scale fitting/profile evidence; final candidate rerun pending.",
21: "Every weekly variant transition is archived, including origins omitted from evaluation frames. No independent complete warm-up/A/B reference event trace or maximum-scale receipt yet.",
22: "Seven-generation model checkpoint roundtrip and saved-ledger roundtrip pass separately; atomic combined model/dividend/trade restoration at every boundary remains incomplete.",
23: "U/R/C probabilities are checked against scalar ages, exact equal distances and deterministic stable-ID ties, with labels perturbed independently.",
24: "31/32/33 leave-one-out thresholds match an independent linear order statistic; existing threshold-equality/OOD and new no-label-routing checks retained.",
25: "Added covariance/PSD and energy-score atom-split invariance. Joint canonical-ID plus moments plus complete portfolio-objective invariance remains partial.",
26: "One-atom exact target and shifted target have independent norm, CRPS and MSE references.",
28: "Arbitrary interior mixture scores match the independent finite-law pairwise sum; B symmetry and diagonal behavior are explicit.",
29: "Orthogonal permutation/sign rotations and positive score scaling now supplement translation invariance.",
30: "Aligned/offsetting/repeated joint atoms have independently computed covariance, nonnegative eigenvalues and dependence checks.",
31: "Hand references now include CRPS/MSE/variogram, exact quantile mass, all fixed bin edges, sparse threshold and basket/per-asset event conventions.",
32: "Near-identical large finite vectors are checked against Decimal differences; orthogonal/scaling metamorphic checks retained.",
33: "All coordinate orientations are exercised; small coefficient scales and original multiplier recovery near condition boundary remain incomplete.",
34: "A constructed boundary optimum strictly beats naive clipping/renormalization, with certified feasibility.",
35: "Identical experts recover the regularization anchor and certificates survive exact archive serialization.",
36: "Added asymmetric rejection and valid tangent convexity with indefinite full Hessian; singular solve and condition-number threshold cases remain missing.",
37: "Tiny negative face solutions on both sides of the allowed correction threshold are discriminated, including certified post-correction solution.",
38: "Exact global25/26, local32/33 and effective-count16 boundary versus concentrated fallback are checked.",
39: "Labels cannot alter neighbor routing and scalar recency is checked. Matching record/weight permutation of the complete local fit still needs an explicit case.",
40: "Nonconstant recency-weighted individual control is independently derived; common immutable atoms/contexts and archive issue identities are checked.",
42: "The exact40-risky/60-cash full-sale root99.6 now supplements mixed and zero-fee reference cases.",
43: "Root near both brackets, independent accounting tolerance and forced100-iteration exhaustion remain a specific unclosed numerical test gap.",
44: "Both per-asset and total-cap cases at/inside/outside1e-12 passed; final corrected test-file rerun pending.",
47: "All-adverse law selects exact cash under nominal/robust including zero nominal mass; passed last boundary batch. Final candidate rerun pending.",
48: "Near-one and tiny-positive CRRA factors match Decimal arithmetic; existing invalid-factor rejection retained.",
49: "Every finite-support contamination vertex, including a zero-mass adverse atom, matches the closed form; rho0 matches nominal.",
50: "A4 event order and fees are covered, but an explicit ordinary-versus-A4 unchanged-hold multiblock paired path remains incomplete.",
52: "Fixed-action entire ledger scales100-to700 and zero-cost replay uses the same externally fixed action sequence.",
53: "A daily-close trough strictly between decisions recovers, yet remains in drawdown; entry fee contributes to drawdown.",
54: "All fixed benchmark exposures/entry fees and buy-hold dividend/drift handling have exact constructed checks; final liquidation uses the separately verified shared ledger.",
56: "Fixed50% shrink covariance and objective ties have an independent grid reference. An explicit raw-versus-shrink ranking reversal fixture remains missing.",
57: "Literal independent Cartesian192-cell registry passed last boundary batch; current corrected test-file rerun pending.",
58: "Missing/duplicate/truncated/resigned partial campaign/tape structures reject before complete reporting; saved192-cell roundtrip retains exact identities.",
59: "Existing full-kernel A1 uses directly derived repeatedH1 atoms with actualH4 maturity and no earlier feedback; inspectable generated timing case retained.",
60: "A2 whole-law per-fold freeze, shared laws, A4 ordering and A5 routing exist; explicit A5 collision metadata remains a review gap.",
61: "260/520 exact retention passed. k16/k64 fixture used a float count and was corrected to integer configuration; rerun pending resources.",
62: "Fixed panel axes and shared bootstrap streams are checked; fully labeled date/asset/method/cost/fold resampling reference remains partial.",
64: "Existing unequal-fold independent scalar bootstrap checks and fixed linear quantiles establish count-weighted pooling.",
65: "Observed and resampled control denominator below/equal/above1e-12 now discriminate undefined gains; strict5% boundary retained.",
66: "All5000 pilot/campaign index draws reproduce independently of invocation order with local streams.",
69: "Noncash liquidatable-NAV increments telescope independently without booking hypothetical fee reserves; saved H3 endpoints use existing fixed-path bootstrap tests.",
70: "Scalar math.fsum sample-SD with ddof1 independently checks all precision comparisons. Production pre2010/source/cold-start guards and full pilot worker need final integrated receipt.",
71: "Valid negative H1/H2 is explicitly NEGATIVE, supplementing support/mixed/inconclusive/invalid and point-gate/control-family tests.",
72: "Exact checkpoint/certificate/readback semantics are retained. Array-endianness and numerical-versus-path/time partition matrix remain missing.",
73: "Source/config/environment binding, malformed checkpoints and resigned archive/tape payloads have expanded checks; final exact-candidate verification and complete corruption matrix remain pending.",
74: "Failure retains archive prefix and poisoned writer; supervisor durable fallback exists. Before-temp and all journal fault locations remain only partial coverage.",
75: "Two concurrent consumption attempts yielded exactly one winner and one marker. Final corrected test-file rerun and full crash-durability integration remain pending.",
76: "Operation-specific producer, real source review and execution worker now exist; generated final-source preflight/post-consumption integrated receipt remains pending.",
77: "Owned-process supervisor cleanup and archive callback failure are tested; explicit SIGINT at each forecast/ledger/analysis phase remains missing.",
78: "CLI rejects resume, marker reuse is refused before output/value imports, and archive failures cannot automatically continue. Historical artifacts remain immutable.",
79: "Supervisor has independent durable fallback and archive failure preservation; simultaneous failure of every failure-report destination is not fully covered.",
80: "Shared supervisor covers process ownership, observer availability, zero observations, clock reversal and all declared limits; final precision/campaign production dispatch verification pending.",
81: "Saved192-cell evidence roundtrips without optimization; decoded event dollars match independent Decimal accounting. Full saved-report regeneration/provenance corruption matrix remains partial.",
82: "Typed exceptions and retained failures exist; warning/unexpected chained-cause traceback coverage is not complete.",
83: "Generated full192-cell campaign and injected late-cell failure retain completed evidence; this is bounded engineering, not production resource qualification.",
84: "Parent owns new fresh-process controller import isolation tests. Full package import graph/cycle and restricted I/O inventory final receipt pending.",
85: "Parent owns exact source/test/type/lint scope validation. Intentional misspelled marker and every fixture-schema mutation remain separately required.",
87: "Parent implements complete lock/environment binding and non-NumPy drift rejection; final exact environment receipt pending.",
88: "Same fixture/process replay exists; differing PYTHONHASHSEED plus dictionary-order cross-process comparison remains missing.",
89: "No Linux execution evidence. This cannot be closed by macOS tests, emulated metadata or internal review.",
90: "Archive exposes verifier-derived max_library/transitions/pending/feedback counts. Actual fixed maximum-shape time/RSS/worker/controller profile is resource-blocked; membership construction is not this test.",
91: "Scalar/vectorized fee and policy, shared-score cache equivalence pass. Complete allocations/time comparison at declared maximum shapes is resource-blocked.",
92: "Parent owns preserved predecessor hash expansion; retain prior48-identity receipt. Full S6-S9/revision1 inventory result must be independently read before upgrade.",
93: "Controller import isolation is bounded evidence; full suite under denied network/no data mounts needs a real restricted execution receipt.",
94: "New registry validator rejects missing/orphan/duplicateIDs, uncollected nodes, absent/skip/xfail outcomes, orphan requirements and expired/current waivers; complete counts cannot authorize science.",
95: "Scope labels and pure computational report boundary remain explicit; parent owns authoritative verification/admission audit. Full claim-ledger evidence-reference validator remains to be assessed.",
96: "Four actual isolated semantic mutants were killed by assertion failures with zero collection errors, copied-module import proof and unchanged checkout hashes. First import-routing-invalid attempt is retained, not counted.",
}

all_test_receipt = "research/frem_s10_production_20261016/catalog_all_tests_attempt4.log"
rows = []
for old in baseline["obligations"]:
    row = dict(old)
    identity = row["test_id"]
    number = int(identity[-3:])
    row["baseline_status"] = old["status"]
    row["status"] = "VERIFIED" if number in upgrades else old["status"]
    row["nodeids"] = sorted(set(old["nodeids"] + nodes.get(identity, [])))
    row["review_note"] = notes.get(number, "Previously verified bounded case retained unchanged; final candidate verification remains separately required.")
    row["queued_test_families"] = queued.get(identity, [])
    row["current_candidate_runtime_verified"] = False
    row["new_receipts"] = []
    if any("test_catalog_" in node or "test_model_archive_" in node for node in nodes.get(identity, [])):
        row["new_receipts"].append(all_test_receipt)
    if number == 96:
        row["new_receipts"].append("research/frem_s10_production_20261016/semantic_mutants_attempt2/receipt.json")
    if row["queued_test_families"]:
        row["new_receipts"].append("research/frem_s10_production_20261016/catalog_boundaries_tests_attempt1.log")
    row["remaining_evidence"] = (
        "Bounded semantic obligation reviewed. Final changed-source verification is pending; no empirical, Linux or maximum-scale claim follows."
        if number in upgrades else row["review_note"] if number in notes else old["remaining_evidence"]
    )
    rows.append(row)

result = {
    "schema_version": 1,
    "scope": "GENERATED_ENGINEERING_CATALOG_REVIEW",
    "status": "REVIEWED_NOT_FINAL_CANDIDATE_VERIFIED",
    "baseline": str(BASE.relative_to(ROOT)),
    "baseline_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
    "baseline_counts": baseline["counts"],
    "counts": dict(sorted(Counter(row["status"] for row in rows).items())),
    "new_implementations": ["runtime/catalog.py", "experiment/model_archive.py", "scripts/run_semantic_mutants.py"],
    "evidence_limits": [
        "Every status is a reviewed bounded catalog obligation, not a universal correctness proof.",
        "Final candidate source/environment/test binding has not yet run under a passing resource gate.",
        "The last15-case boundary batch was10passed5failed; oracle/config mistakes corrected without rerun. Failed evidence retained.",
        "Semantic mutant attempt1 is INVALID due to wrong imported path; attempt2 is baseline4passed plus4/4 assertion-killed mutants.",
        "No market fit, protected evaluation, independent empirical replication or Linux execution was performed by this review.",
    ],
    "irreducible_current_gates": [
        "Fresh fixed512MiB/free-plus-speculative and zero-pageout admission before further numerical, mutation, full-suite or qualification work.",
        "Maximum-shape time/RSS/storage qualification on the actual final candidate; declared counts do not substitute.",
        "Real Linux same-fixture execution for the cross-platform claim.",
        "Independent empirical reproduction and scientific endpoints require admitted frozen execution; engineering evidence cannot establish them.",
    ],
    "obligations": rows,
    "run_ready": False,
    "scientific_status": "NOT_EVALUATED",
    "authorizes_scientific_execution": False,
}
assert len(rows) == 96 and len({row["test_id"] for row in rows}) == 96
(DEST / "VERIFICATION_CATALOG_REVIEW.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
lines = [
    "# Generated verification catalog review", "",
    "This reconciles all96 obligations against retained, bounded generated evidence. It is not a final-source verification receipt or scientific admission. The predecessor catalog remains unchanged.", "",
    f"Previous: {baseline['counts']}. Reviewed bounded status: {result['counts']}.", "",
    "The changed candidate still needs a fresh resource gate and final integrated verification. The last boundary batch retained10passing and5failing cases; corrected test references/configuration are queued, not reported passing. No new numerical batch was launched after the stop instruction.", "",
    "The archive records every weekly issued law, matured feedback, mixture solver certificates and immutable model state; its counts are independently reconstructed from decoded states. Saved ledger dollars also have a separate Decimal reference. Neither establishes an empirical benefit.", "",
    "| Obligation | Previous | Reviewed | Evidence or remaining gap |", "|---|---|---|---|",
]
for row in rows:
    lines.append(f"| {row['test_id']} | {row['baseline_status']} | {row['status']} | {row['review_note']} |")
lines.extend(["", "Irreducible current gates:", ""] + ["- " + item for item in result["irreducible_current_gates"]])
(DEST / "VERIFICATION_CATALOG_REVIEW.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"counts": result["counts"], "obligations": len(rows), "run_ready": False}))
