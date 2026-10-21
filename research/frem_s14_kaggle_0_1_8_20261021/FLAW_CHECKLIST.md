# Evidence and repair checklist after S14 Version 6

The declared model is implemented, but its historical benchmark remains 0/171
cells. No S14 predictive advantage over simple baselines has been measured.
This document distinguishes repairs demonstrated by execution from proposals.

| Flaw | Cause and evidence | Severity | Repair or next action | Required acceptance evidence |
|---|---|---|---|---|
| Earlier intake fixtures failed | Version 5 used flat generated paths, inconsistent with canonical ETF input paths | Resolved engineering blocker | 0.1.8 retains assertions and corrects fixture paths | Version 6: 107/107 warning-strict tests passed in 27.75 seconds |
| Prior GPU qualification failures | Norm rounding and GRU call-interface defects | Resolved for 0.1.8 | Preserved versioned repairs | 19/19 generated CUDA arms complete; in-run independent 222-file semantic verifier passed |
| Dataset violates frozen calendar contract | Hash-matched source: QQQ has 2736 dates; all seven other ETFs have 2735, missing 2011-02-17 | Current scientific execution blocker | Obtain compatible, provenance-verified observations or define a separately versioned missing-data study before any fits | Exact full-cohort calendar/data admission; no silent intersection, invented prices or copied returns |
| Calendar defect was found only after GPU qualification | No independent source-calendar preflight | Repaired operational diagnostic | Added operations/audit_calendar.py outside frozen payload | Three regression tests pass; real eight-file audit reproduces blocker and leaves inputs unchanged |
| No demonstrated baseline improvement | Zero historical S14 cells; generated data cannot answer the market question | Critical evidence gap | Execute complete admissible 171-cell matrix, then independently reconstruct all scores and comparisons | Complete seeds/folds, frozen paired intervals and practical threshold; retain unfavorable outcomes |
| Dataset is an adjusted-close historical proxy | No verified adjustment vintages, terminal returns, spreads or inactive-universe coverage | Restricts claims | Keep proxy labels; obtain separately admitted point-in-time data for investability claims | Timestamp, corporate-action, eligibility and execution-cost evidence |
| More components do not establish novelty or utility | Reservoir, memory, diffusion and routers add complexity | Unresolved scientific risk | Use predeclared mechanism-off arms and matched baselines; select a new mechanism only after valid evidence | Mechanism-specific effect under equal information and declared tuning/compute budgets |
| Local numerical readback is not admitted | Available memory 303.4/289.2/332.9 MiB; increasing pageouts | Verification blocker on this host | Use an admitted host; retain current refusal | Fresh >=512 MiB stable-pageout admission followed by independent semantic verification |
| Manuscript described Version 5 as queued | Status became stale after terminal failures | Repaired reporting defect | Update manuscript and canonical state with retained Version 6 evidence | Compiled paper and artifact-linked values; no efficacy claim |
| Final scientific paper/release incomplete | Core market question remains untested | Release blocker | Complete valid study and claim-to-evidence audit before final disposition | Supported, mixed, negative or inconclusive terminal finding with reproducible evidence |

## What is proven now

Version 6 (360684950) executed privately on GPU T4 x2, with Internet disabled.
Saved notebook cell sources equal the immutable 0.1.8 candidate. The 308 saved
run files were downloaded through the official API; the additional Kaggle log
is retained separately. Integrity verification anchors all 18 payload members,
15 installed Python modules and the qualification manifest to the local release.
The locally reconstructed ZIP is a packaging of downloaded files, not a claimed
server-generated archive. In-run semantic verification passed; it was not rerun
locally because host admission failed.

The temporary S14 token was revoked after verification and its local file removed.
Other account tokens were left intact. No benchmark retry, source-data mutation,
new scientific mechanism or protected-outcome access occurred.

## Next scientific decision

The existing strict benchmark cannot finish on these exact source bytes. This
is not evidence that the model loses to a baseline. The seven missing records
must be recovered with compatible adjustment provenance, or a new data protocol
must explicitly define missingness, return intervals, feature windows and target
horizons. Dropping QQQ's extra date would change the frozen study and would hide
the missing observation unless declared and evaluated as a new study. No choice
has been selected by model scores. A positive result cannot be promised.


### Current frontier reconciled across research chats

The original eight-ETF Version 6 failure above is historical. The separate
research execution chat has already frozen and launched seven-ETF package 0.1.9
as private Kaggle Version 8. Its saved-source identity is recorded in
`research/frem_s14_calendar_successor_20261021/SAVED_IDENTITY.json`. QQQ is excluded
prospectively on source-calendar grounds; retained histories are not intersected
or filled. This narrows the population and is a new study, not a successful
completion of the original eight-ETF matrix. The latest execution-owner check at
10:53 UTC reports RUNNING. No historical results are yet independently verified.
The existing `finish-frem-s14-version-8` monitor owns terminal output collection.
Do not launch another run or change its frozen code, cohort, threshold or budget.
Version 6 evidence and token revocation in this chat are complete. The broader
baseline comparison and final research release remain pending Version 8 evidence.
