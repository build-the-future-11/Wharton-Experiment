# FREM-S3: completed exploratory campaign

**Verdict: the predeclared combined-memory advantage is not established on the new public sample.** Point effects are mixed and both primary intervals include zero. No tested memory variant strictly improved both windows. The raw FREM base also outscored its additionally calibrated version in both windows. These valid outcomes are retained without retuning or rerunning.

## Executed scope

217 full regression tests passed, including 19 new S3 tests. Two implementation defects were caught before generated or market research outcomes: horizon was inferred instead of hardcoded, and descriptor calculations were made float64 consistently across checkpoint reload. The earlier 15-pass/4-failure test receipt remains intact.

15 fresh generated mechanism cells: five families times three independent path seeds, fourteen memory arms per cell. The predeclared null/recurrence gate passed. 24 learned public fits: four base families times three optimization seeds times two windows. 12 deterministic-baseline cells are separate; their repeated seeds are not independent replicates. All 36 base cells completed, producing 114 public variant prediction cells. Zero failures or omitted cells in the frozen campaign.

The public data consists of twelve newly acquired US current-survivor histories: MSFT, XOM, PG, WMT, MCD, CAT, IBM, GE, HON, PEP, ABT and MMM. A metadata check and 52 prior prediction CSV identity checks found no earlier use of those names in the checked campaign evidence. KO was excluded before acquisition because it appeared in an old engineering source. This is evidence of new-to-campaign observations, not proof nobody has previously seen those securities or that the economic history is independent.

The public windows are January-December 2023 and January-October 2024. All twelve stocks passed the fixed completeness rule: 144 and 120 paired asset-month outcomes. Seeds are optimization repeats, not additional market histories. The configuration, code, scenario family and data request were frozen before generated/new public outcomes. Original 2025 onward protected price history was neither requested nor evaluated. The public sample has survivor, revision and research-coordinate timestamp limitations; it is not the canonical admitted global panel.

## Primary result

Gain is control NLL minus combined-memory NLL; positive favors memory. The existing 0.01-nat practical threshold was retained.

| Window | Memory off | Combined memory | Gain | Exploratory 95% interval |
|---|---:|---:|---:|---|
| fresh_2023 | -1.345176 | -1.331060 | -0.014116 | [-0.036455, +0.006302] |
| fresh_2024 | -1.437064 | -1.444071 | +0.007006 | [-0.007940, +0.022561] |

Both primary three-month block intervals include zero and fail the practical-support criterion. Holm-adjusted p-values for the declared 14-comparison family are retained in the summary; neither primary comparison establishes support. The 2023 six-month sensitivity also includes zero. The 2024 six-month sensitivity is unavailable because ten months cannot provide two six-month blocks. No block length was shortened to force a result. These are short, correlated retrospective histories; intervals are exploratory.

## All memory ablations

Each entry is gain versus the same calibrated FREM checkpoint with memory off. Positive is better. The primary was fixed as combined; the best secondary arm was not substituted after observing outcomes.

| Arm | 2023 gain | 2024 gain |
|---|---:|---:|
| off | +0.000000 | +0.000000 |
| frozen | +0.000000 | +0.000000 |
| recent | -0.000705 | +0.000000 |
| random | -0.010605 | +0.006841 |
| mean | -0.010366 | +0.008279 |
| risk | +0.002436 | -0.001244 |
| combined | -0.014116 | +0.007006 |
| no_confidence | -0.021568 | +0.003633 |
| balanced | -0.014116 | +0.007006 |
| larger | -0.005289 | +0.012700 |
| dual | -0.006473 | +0.013683 |
| cross_asset | +0.007388 | -0.001506 |
| learned | -0.006754 | +0.000564 |
| utility | -0.002627 | +0.002309 |

Mean memory worsened 2023 and improved 2024. Risk memory had the opposite sign. Cross-asset retrieval helped 2023 but hurt 2024. Larger capacity and dual-bank retention helped 2024 by more than 0.01 nats as point estimates, but hurt 2023; these secondary observations do not establish a replicated benefit. Learned keys and utility retention also failed to produce gains in both windows. Balanced same-capacity retention matched the primary in this balanced panel, not a distinct success.

Frozen memory selected zero weights and exactly matched off in both windows. In contrast, combined adaptive memory selected nonzero mean weights in all three seeds of both windows. The remaining issue is therefore generalization of the selected correction, not a disconnected retrieval path or universally zero gate. It also failed to establish a consistent advantage over recent averaging or matched random retrieval.

## Base and calibration comparisons

NLL below, lower is better. All named controls use the common chronological calibration protocol; the raw FREM reference is retained separately, not selected retrospectively as a new primary.

| Model | 2023 NLL | 2024 NLL |
|---|---:|---:|
| frem/off | -1.345176 | -1.437064 |
| no_jepa/off | -1.349918 | -1.447446 |
| ridge/off | -1.255116 | -1.247335 |
| xgboost/off | -1.209832 | -1.334934 |
| zero/off | -1.254592 | -1.361529 |
| historical_mean/off | -1.271172 | -1.393529 |
| Raw FREM before extra calibration | -1.354994 | -1.455920 |

The calibrated FREM base beat zero, historical mean, Ridge and XGBoost on both point estimates. That does not show the new calibration was beneficial: raw FREM was better in both windows. Validation-selected calibration did not generalize here. The no-JEPA control was also descriptively better than calibrated FREM in both windows; this does not isolate the cause or justify retuning the completed experiment. There is no contemporaneous full S2 predecessor refit in this campaign, so no claim of S3 superiority over S2.

## Generated mechanism evidence

Gain versus memory off, averaged over the three independent generated paths. All five families and all seeds are retained. These cases establish implementation behavior under prescribed signals, not financial predictability.

| Family | Combined gain | Recent-average gain | Matched-random gain |
|---|---:|---:|---:|
| null | -0.001842 | -0.000511 | +0.000095 |
| mean_recurrence | +0.236073 | +0.004226 | +0.000139 |
| risk_recurrence | +0.102459 | +0.024663 | +0.024930 |
| abrupt | +0.013390 | +0.010542 | +0.028720 |
| missing_delayed | +0.238341 | +0.009268 | -0.000356 |

The mechanism can exploit planted conditional recurrence. It does not follow that comparable predictive residual structure exists in this stock sample. On the null, combined memory slightly worsened NLL by 0.001842, within the predeclared 0.01 tolerance; this is not zero harm. Abrupt-change gain was smaller than matched random retrieval, further limiting a universal relevance claim.

## Verification and resource evidence

The frozen campaign took 12.39 minutes wall time. All 53 worker processes exited zero. Maximum sampled RSS was 407.672 MiB, under the 768 MiB sampled ceiling, with one worker/thread and nice10. The machine did not pass normal 512 MiB free/speculative admission; execution used the current user's explicit slow-mode authorization and retained its initial resource snapshot.

An independent standard-library audit reconstructed 133,844 saved prediction rows with maximum NLL error 2.22e-15, verified 49 frozen source hashes and 582 predecessor hashes. This is independent arithmetic/source verification, not independent retraining or external review.

## Stage disposition and limits

- A integrity: PASS, with retained corrected pre-experiment failures.
- B base/calibration: executed. Base beats listed simple controls on point scores; extra calibration worsens both fresh windows.
- C generated mechanisms: COMPLETE, declared gate PASS, including small null harm.
- D new public sample: COMPLETE; primary memory advantage unsupported, mixed point effects, statistically inconclusive.
- E canonical confirmation: NOT EXECUTED. No mature supported memory hypothesis and no admitted untouched canonical panel/freeze. Protected tail unchanged.
- F portfolio evidence: NOT EXECUTED. Public adjusted-close research coordinates do not certify an executable investment ledger. No profitability, SOTA, novelty or deployment claim.

No post-result code, gate, threshold, seed, fold or calibration-grid changes were made. No reruns were launched to seek a favorable result. Current S3 protocol is closed with this bounded result. Future work, if separately requested, must be a new protocol on fresh evidence rather than retuning these completed windows.

## Reproduction and artifacts

- `outputs/frem_s3_20261011_attempt1/freeze.json`
- `outputs/frem_s3_20261011_attempt1/completion.json`
- `outputs/frem_s3_20261011_attempt1/mechanism_summary.json`
- `outputs/frem_s3_20261011_attempt1/public_summary.json`
- `research/frem_s3_20261011/verification.json`
- `docs/FREM_S3.md`

Saved predictions, calibration choices, versioned replay states, base checkpoints, raw provider responses, source timestamps and process receipts are under the run directory. Reconstruct scores with `rtk proxy python3 -m scripts.audit_frem_s3 --run outputs/frem_s3_20261011_attempt1 --output research/<new-audit>.json`; it refuses to overwrite an audit file. Replaying training is not permission to claim an untouched new evaluation.
