# Independent static review

Review scope: root-authored experiment orchestration, governance, reporting,
command-line interface, robustness assays, and manuscript against the portfolio,
risk, mandate, data and evaluation interfaces. Reviewed on 10 October 2026.

**No experiments, optimization calls, model imports, unit tests, simulations or
provider acquisitions were executed for this review.** Source inspection and
syntax parsing cannot establish runtime correctness or empirical validity.

This file records observations at review time. The main author is correcting
items concurrently; a retained finding is evidence of review, not a statement
that a later source revision still contains the defect. The release verification
record should distinguish a reviewed correction from an unexecuted runtime gate.

## Material findings and proposed corrections

| Area | Static observation | Required correction or explicit limitation |
|---|---|---|
| Allocation return contract | Forecasts use cumulative base-currency log returns. Raw allocation arrays are interpreted as simple returns without machine-checked currency, horizon or return-space metadata. | Require an allocation request contract declaring simple-return units, common base currency, asset ordering, horizon and information timestamps; reject a direct undocumented substitution of Student-t log locations. Bounded scenario conversion must state its changed distributional assumptions. |
| Missing-forecast fallback | CLI conversion of JSON null with `np.asarray(..., float)` prevents the supported `expected_returns=None` risk-only path. | Preserve `None` and require `decision_context.forecast_available=False`. Do not invent an expected return for the resulting risk-only allocation. |
| Portfolio research integration | The prediction matrix runner produces predictive evidence. Allocation, portfolio accounting and liability scenario functions are separate entry points. | Supply an explicit prospective adapter connecting dated forecasts, causal risk inputs, dated mandate/master snapshots, allocation decisions and a nonoverlapping execution ledger; alternatively state precisely that the predictive runner does not establish net portfolio utility. No future labels may influence a decision. |
| Cash-flow protection | `reserve_cash` is an externally supplied immediate cash constraint. It does not itself derive the complete ten-payment liability reserve or certify a reserve asset portfolio. | Preserve this distinction. A liability workflow must determine a justified reserve and initial composition, fund the immediate 2033 payment first, and retain funding shortfalls and flexibility requirements. Do not equate an arbitrary positive cash floor with case compliance. |
| Entry-fee accounting | At review, the round-trip ledger used `entry_cost = c*q` despite describing risky weights `q` after entry costs. | For initially all-cash capital use `alpha=1/(1+c*q)`, entry fees `1-alpha`, entry traded fraction `alpha*q`, exit notional `alpha*risky_terminal`, and net wealth `alpha*gross_wealth-exit_cost`. This matches the optimizer's self-financing equation. The data/evaluation author received this correction. |
| Confirmatory inference | Threshold, confidence and bootstrap seed are not all loaded from the frozen protocol; user resampling overrides can alter the reported decision. | Bind the primary confirmatory analysis to the frozen primary comparison and complete inference settings. Mark all deviations exploratory; do not give an altered analysis the primary supported disposition. |
| Saved prediction validation | Reporting recomputes likelihood directly; it does not use the shared prediction validator and full score contract. | Use the shared validator/scoring path so future maturity, actual outcomes, distribution degrees of freedom and execution timestamps are validated consistently. |
| Cross-method report comparisons | A descriptive ranking can be made before verifying that every method and seed has exactly the same observation keys and outcomes. | Require complete matched seed-observation grids and common label evidence before ranking. Retain exclusions as explicit invalid evidence, not silently different sample sizes. |
| Robustness suite configuration | Suite variants mutate an already validated configuration, then call the matrix runner without full revalidation. | Refactor configuration validation to accept a dictionary, validate each merged variant, resolve new relative data paths from the project root, and enforce unique models/seeds and per-matrix budgets. |
| Runtime budget | The matrix timer starts after data preparation and does not check the budget after its final cell. | Start before preparation and check the complete matrix boundary. Native fit calls may still be cooperative rather than hard-interruptible; document that practical limitation. |
| Perturbation validity | Nonfinite noise magnitude is not rejected; a feature entirely absent from training produces a nonfinite median/IQR. | Reject nonfinite severities. Use an explicit recorded fallback for entirely missing training channels, consistent with the train-only scaler; otherwise the assay can corrupt a different mechanism than the one declared. |

## Corrections already observed during this review

- Allocation configuration uses `PortfolioConfig.from_dict`, including nested
  regime policy construction, and allocation results are serialized as
  dataclasses.
- Governance hashes the data files and their required provenance manifests.
- A confirmatory freeze has an atomic consumption receipt which survives failed
  attempts; the matrix consumes it before preparing protected observations.
- Date-block inference computes an overlap bound from observed label maturity,
  in addition to the declared horizon.
- Predictive report aggregation uses UTC dates, and equity plotting sorts ledger
  entries chronologically after validating them.
- Mandate provenance checks both verification/availability and rule-effectivity
  dates. Instrument metadata, classifications and approvals must be dated and
  eligible. A price band cannot become a portfolio cap.
- Mandatory recorded rules without an implemented constraint block allocation.
  Unsupported minimum-holdings rules therefore cannot be silently accepted.
- Solver success is followed by the shared feasibility checker. Risk-only,
  all-cash, unchanged-holdings and rounded-lot candidates face the same hard
  restrictions. Infeasible candidates do not emit certified weights.
- Regime inputs now require actual Boolean quality flags and validate every
  supplied numerical observation before threshold decisions. Nonfinite risk
  multipliers are rejected.

## Manuscript alignment findings

1. Failure to reach a prespecified practical-improvement threshold is not always
   falsification. The proposed decision rule distinguishes a negative interval
   from an inconclusive one; the introduction should use the same distinction.
2. The implemented constraints include individual-security upper bounds and
   group upper/lower bounds. Individual-security lower bounds are not present.
3. The unresolved feasible-set result is `infeasible`, with no weights. Include
   that status when discussing blocked/no-action outcomes.
4. `tradable=False` prevents position changes. An explicit settlement calendar,
   unsettled cash balance and settlement lock ledger are not implemented by
   that flag. Do not claim a settlement engine.
5. Cash-flow matching is implemented in `mandate.py`, with nonnegative units,
   cumulative dated payment coverage and zero-interest carry of excess cash.
   The function does not certify credit quality, executable prices or eligibility.
6. Raw Student-t log-return locations cannot be exponentiated into expected
   simple returns. Student-t exponential moments are not finite; clipping or a
   Gaussian conversion is a separately declared assumption.
7. A decision rule conditioned on supplied recession indicators is not a proven
   recession detector, and no recession is assumed certain.

## Case provenance and preserved negative evidence

`docs/laura_case_evidence.json` contains structured case facts and source scope.
`docs/laura_case_excerpts.json` preserves exact relevant text, the Google document
identifier, current revision, case tab and paragraph indexes. The verified source
is a reproduced case in the user's preparation document, not an independently
authenticated organizer PDF. Current participant trading requirements and the
approved universe remain external inputs.

The same current document reports that APEN memory worsened MSE in all 20
synthetic cases, despite changing 12,015 predictions and reportedly passing
mechanical checks. It also reports graph-memory/world-model deterioration versus
a historical-mean comparator. Those are preserved source-document claims; their
raw runs were not re-audited. `RESEARCH_STATE.json` correctly identifies them as
prior reported evidence. New code and a research restart do not demonstrate that
the mechanisms responsible for those negative outcomes have been fixed.

## Interpretation of this release

Written tests are prospective checks. Source review can identify a visible
equation or interface defect, but an apparent correction remains unexecuted.
Before empirical claims, the later operator must validate the installed
environment, run the retained tests, acquire compliant point-in-time data, verify
the current participant rules, and execute only the declared bounded protocol.

## Final integration addendum

The following corrections were inspected in source after the historical
findings above. They have **not** been verified by executing the package.

| Historical finding | Final source-review disposition |
|---|---|
| Missing forecast-to-allocation bridge | Implemented in `decision.py` for one synchronous hypothetical decision. A required `ReturnContract` enforces declared return space, currency, stable asset order, decision/input-availability times, future entry/exit interval and horizon attestation. |
| Undocumented log/simple conversion | `decide` uses explicitly winsorized log Student-t marginals and a supplied Gaussian copula or explicitly chosen independence. All finite scenarios, probabilities, assumptions, clipping counts and moments are retained. `allocate` requires an explicit simple-return contract. |
| Null forecast incorrectly cast | The CLI now preserves null as `None` for the expressly declared risk-only allocation path. |
| No execution gate for allocations | Both `decide` and `allocate` require `--execute`; they still emit research artifacts only. Existing allocation/decision outputs are protected from overwrite. |
| Checkpoint device assumption | `predict --device` defaults to CPU and passes the device to `Trainer.load`. |
| Frozen input identity | Governance hashes both panel/macro files and required provenance sidecars. |
| Protected retry guard | Confirmatory freeze consumption is atomic and occurs before protected data preparation; failed consumption remains recorded. |
| Configuration mutation bypass | `validate_config` now operates on dictionaries and is called for the matrix and merged suite variants; root-relative paths and per-variant controls are checked. |
| Incomplete matrix wall budget | The timer starts before preparation and checks preparation, fitting, final cells and suite completion. Cooperative native-call limits are explicitly documented. |
| Invalid robustness scales | Nonfinite severities are rejected; all-missing training channels use explicitly recorded zero-median/unit-width fallbacks. |
| Reporting validation/pairing | Reporting uses `score_predictions`, complete seed registries and exact observation/outcome/time pairing before comparisons or ranking. |
| Frozen inference drift | Reporting binds the protocol, analysis source, threshold, confidence, resample count and bootstrap seed; alternative confirmatory blocks must belong to the frozen sensitivity family and remain distinguished from the primary block. |
| Missing mechanism-family correction | `compare_family` requires the complete declared mechanism family and provides Holm correction; `frem compare-family` dispatches to it. |
| Ledger fee inconsistency | The round-trip ledger now uses the self-financing entry fraction `alpha=1/(1+c*q)` and matching entry/exit turnover. |

The new bridge's focused request schema and mathematical assumptions are in
`docs/FORECAST_TO_PORTFOLIO.md`; incomplete request templates contain unknown
inputs rather than fabricated scenario results. `tests/test_decision.py` adds
prospective checks for realized-outcome independence, asset alignment, explicit
dependence, forecast timing, clipping, mandate blocking and CLI execution gates.

Remaining **execution gates** are package installation and dependency
compatibility, execution of the authored tests, validation of provider exports,
empirical calibration and protected research results. Remaining **scope limits**
are a fully asynchronous carried-holdings/settlement engine and certified
liability-asset funding. The implemented synchronous bridge and round-trip
ledger must not be described as those broader systems. Unknown official
participant rules, instrument universe and quantitative funding policy remain
external case inputs, not unexplained missing algorithm code.
