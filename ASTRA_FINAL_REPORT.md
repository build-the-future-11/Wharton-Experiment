# ASTRA final report - Wharton, 2026-09-27

## Outcome

**The local competition preparation package is complete within the documented scope. Competition submission remains BLOCKED.** The project is the Wharton Global High School Investment Competition, so the supplied venture-style mission was adapted to investment strategy, funding sensitivity, client alignment and evidence. No venture revenue, customers, interviews, partnerships, student experiences, client balances or trade records were invented.

The authoritative current status is `STATUS.md`. The complete requirement ledger is `ASTRA_CHECKLIST_2026-09-27.md`. Passing local checks is not scientific validation, financial suitability, a paper-ready result or permission to submit.

## Starting state

Starting revision: `8af749dab048b3626090703806f0f2233ce4f210`; tracked worktree clean; `.serena/` already untracked and left untouched. No remote was configured in the repository's prior status. The start snapshot records tracked hashes in `audit/astra_2026-09-27/START_STATE.json`.

Existing work included twelve model implementations, a 1181-cell legacy matrix, 1916 orphan receipts, a spent lockbox, an invalid factorial, a repaired exploratory baseline study, research reports and blocked competition drafts. The prior integrity audit had withdrawn a synthetic forecasting win for leakage and documented degenerate splits and inferred ETF column identities. It had not found the deliverable instruction PDFs in the workspace.

## Work completed

1. Verified the current public Wharton schedule and requirements. Corrected the old note that wrongly placed trading end on November 6; the official FAQ gives December 4 as competition end. Retained the distinction between public dates and private instructions.
2. Found the IPS and Trading Notes instruction PDFs in Downloads. Copied them to ignored local source storage, recorded full SHA-256 and page counts, extracted requirements and visually inspected all eight pages. Proved the duplicate Trading Notes file byte-identical and left all originals untouched.
3. Recovered verified IPS structure, actual Times New Roman 12-point/double-spacing/one-inch margins, 50/500-word limits, title-plus-two-page limit and file-size cap. Recovered the requirement for three exact executed notes with at most 100 words per reflection, submitted through Apply. The full case remains missing; the first name Laura and broad operating/facility objectives are sourced, numerical facts are not.
4. Added an independent, transparent monthly cash-flow sensitivity calculator. It accounts for self-financing rebalancing, purchase/sale costs, cash, market returns, withdrawals and carried arrears. It preserves insolvency. This is **not** an amended M11 implementation, repaired factorial, market backtest or client forecast.
5. Ran every declared combination: five assumed paths, three simple policies and five cost levels, producing 75 complete trajectories. Generated JSON, CSV, a readable table and an offline interactive viewer from the same evidence.
6. Produced written working responses, one-page executive brief, source-formatted IPS draft, eight-slide pitch content, a timed rehearsal script, demonstration script, 60 adversarial questions/answers and a source/evidence appendix. Produced three PDFs and checked their rendered layouts.
7. Added source/artifact checks and explicit submission gates. Updated current README, status, reproduction, claims, limitations and release documentation. Preserved historical research reports and marked the old competition drafts superseded; the old default PDF command no longer emits the superseded IPS.

## New adverse finding

Frozen M11 cash accounting fails a simple diagnostic. Initial cash 1, zero returns/weights/costs and two 0.1 payments should leave cash `[0.9, 0.8]`; `_scenario_cash_paths` reports `[0.8, 0.8]`. It double-deducts the current obligation in reported cash while not carrying the payment deduction into portfolio wealth. Its downstream shortfall routine subtracts obligations again, and its turnover formula measures cross-asset weight differences. D-059 records the finding. Frozen implementations remain byte-identical under D-002. M12's legacy in-sample market scoring and controller dependence likewise do not establish an end-to-end benefit.

## Actual checks and results

| Check | Observed result | Receipt / interpretation |
|---|---|---|
| Full existing plus new accounting tests | 60 passed, 15.55 seconds | `audit/astra_2026-09-27/TESTS.txt`; engineering only |
| Same-machine repaired baseline replay | All 68 saved contrasts matched; no differences within recorded tolerance | `VERIFICATION.json`; not independent reproduction |
| Protected history/model/lockbox/manifest bytes | All 3170 matched starting hashes | `VERIFICATION.json`; no lockbox run performed |
| Declared stress matrix | 75 complete cells, 120 monthly ledger rows each | `runs/competition_stress/RESULTS.json`; assumptions only |
| Offline browser selections | All 75 displayed correctly; no JavaScript errors; no mobile overflow | `BROWSER_QA.json` |
| PDF checks | Brief 1 page; pack 17 pages; IPS 3 pages, 39-word pitch and 288-word statement | `ARTIFACT_QA.json`, `VISUAL_QA.json` |
| Source/artifact consistency | PASS; 60 unique ordered Q&A entries | `PACKAGE_GATE.json`; submission still BLOCKED |
| Python syntax | 144 files parsed; no errors | `STATIC_CHECKS.json`; no configured type/lint system claimed |
| Bounded marker/secret scan of new code/package | No unfinished-code markers or secret-pattern findings | `STATIC_CHECKS.json`; not a universal security certification |
| Existing completion audit | AUDIT OK | Run after new artifacts; pre-existing runpy warning noted below |
| Final diff whitespace check | Passed | `git diff --check`; staged check before checkpoint |

At the 10-bps assumed cost, the deliberately severe funding-failure scenario exhausts all three policies. Funded shares range from 61.47% to 63.10%, with first shortfalls in months 84 or 86. These are conditional arithmetic results, not estimated failure probabilities or actual client outcomes. All less favourable cells remain in the output. The existing repaired forecasting study still does not establish a statistically supported mean-benchmark advantage for its tested baselines.

## Files and provenance

- `output/pdf/WHARTON_EXECUTIVE_BRIEF.pdf`
- `output/pdf/WHARTON_PREPARATION_PACK.pdf`
- `output/pdf/WHARTON_IPS_DRAFT_BLOCKED.pdf`
- `reports/competition_package/` - editable source documents and offline `DECISION_ROOM.html`
- `configs/competition_stress.json`, `src/wharton_lab/competition/stress.py`, `tests/unit/test_competition_stress.py`
- `runs/competition_stress/RESULTS.json`, `SUMMARY.csv`
- `protocol/RECOVERED_DOCUMENTS.json`, `VERIFIED_DELIVERABLE_REQUIREMENTS.md`, `SUBMISSION_GATES.json`
- `audit/astra_2026-09-27/` - start snapshot, exact supplied mission, verification and artifact receipts
- `requirements-verified.txt` - installed environment snapshot, not a portable hashed dependency lock

No unique evidence was removed. `CLEANUP_MANIFEST.md` records supersession and copy/deduplication decisions. Raw ETF data and recovered instruction PDFs remain ignored and local. There was no remote publication or redistribution of the private packet.

## Unresolved gates and deliberately unclaimed work

1. Complete client case: amounts, payment dates, horizon, loss capacity and the complete investment context.
2. Current trading/eligibility rules, allowed-security list, team registration/roster and school documentation.
3. Three actual executed WInS notes and corresponding records. The current date precedes the public first trading day; no competition executions are fabricated.
4. Final-report instructions and numerical rubric; source-grounded final reserve calculations, contribution range and co-sponsor communication.
5. Student review, independent decisions, accurate authorship and required AI attribution; final rendering review after client-specific edits.
6. Explicit external submission authority. No trades, submissions, emails or client contact occurred.

The repaired M01-M12 matrix, new controller/factorial protocol and fresh confirmatory holdout remain **NOT RUN**. They are separate research extensions, not substitutes for missing competition facts. No preprint or arXiv bundle is appropriate on the evidence in this pass. Historical source/feature defects and licensing uncertainties remain visible.

## Reproduce

See `REPRODUCE.md` for full environment and local-source requirements. From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_competition_stress.py
.venv/bin/python scripts/build_competition_artifacts.py
PYTHONPATH=src .venv/bin/python scripts/verify_astra.py
.venv/bin/python scripts/verify_competition_package.py
.venv/bin/python -m pytest -q
bash scripts/audit_completion.sh
node scripts/verify_competition_viewer.cjs
```

The browser command requires Playwright in Node's module path and installed Chrome. This host needed an approved isolated browser launch after sandbox SIGABRT. Poppler needed a local writable fontconfig cache. The old completion module emits a runpy import-order warning but returns AUDIT OK. These environment issues and workarounds are retained in `audit/astra_2026-09-27/ENVIRONMENT_NOTES.md`; none changes the scientific or submission status.
