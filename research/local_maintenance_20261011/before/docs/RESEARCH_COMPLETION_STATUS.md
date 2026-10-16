# Research completion status — 11 October 2026

The market hypothesis is **NOT_EVALUATED** and the paper is **not submission-ready**. See `research/completion_20261011/COMPLETION_RECEIPT.json` for every workflow checkpoint and exact receipts.

## Completed in this revision

- Acquired seven official public regional factor files with 3,666 pre-2025 monthly records. All cover 2008. The data are research portfolios with current-vintage revisions, not historical individual-stock membership or an investable portfolio. Raw and derived data remain local pending redistribution review.
- Independently recomputed the retained four-cell synthetic FREM NLL, MAE, and direction scores using only the Python standard library. Verified exact paired outcomes, timing, split metadata, original source/data hashes, and receipt hashes. FREM's control-minus-candidate NLL is -0.5649268573499813; no model was refitted and the original bootstrap result was preserved.
- Passed 12 new small offline tests and lint for the changed scripts. These are separate from the earlier 163/67/45 model-package tests. The six public-equity adapter tests remain unrun under the resource guard.
- Updated and compiled the 23-page methods/evidence manuscript with 24 references; generated its numerical table from verified artifacts, added relevant financial retrieval neighbors, rendered and reviewed every page, and packaged source plus bibliography. Preserved the original authoring paper and first layout-overflow build.

## Backtest execution

The existing public-equity acquisition contains 24 present-day listings across 12 markets. Its separate six-window/four-model campaign is capped at 24 cells. The latest memory samples were approximately 328, 254, and 163 MiB with growing pageouts; the 512 MiB guard remains unchanged. **No new model cells ran in this revision.** No background task has been scheduled.

The seven regional files are acquisition inputs only. They are not silently passed into the stricter canonical market panel, and no benchmark results are inferred from the existence of downloaded rows. Current official documentation defines the international factors as simple USD returns; the adapter converts percent to fractional units and adds RF to market excess to reconstruct the research market return.

## Reproduce the completed work

Run from the repository root. Use new output directories to retain earlier receipts.

```sh
rtk proxy .venv/bin/python -m unittest tests.test_evidence_audit tests.test_public_regional_factors -v
rtk proxy .venv/bin/python scripts/audit_engineering_evidence.py --output research/NEW_EVIDENCE_AUDIT
rtk proxy .venv/bin/python scripts/acquire_public_regional_factors.py --config configs/public_regional_factors.json --output data/NEW_REGIONAL_INTAKE
rtk proxy .venv/bin/python scripts/build_paper.py --output-dir paper/NEW_REVIEW_BUILD
```

Acquisition normalization never converts numeric outcomes dated 2025 onward. Raw archives can contain later rows; preserve them opaquely and do not use them to select a hypothesis. The standalone manuscript builder supports cached Tectonic or pdfLaTeX/BibTeX and starts from clean intermediates.

## Remaining gates

Admit resources before the six public-equity adapter tests and then execute the declared crisis campaign in a fresh directory. The canonical study still requires truthful point-in-time data, bounded development and the full control/ablation/robustness/OOD/scaling matrix, a scientific freeze, protected confirmation, independent reproduction and a final evidence audit. Full global equities additionally need default-aware outcomes and global holdings/FX accounting. Authorship, affiliations, case/data rights and complete novelty review remain release checks. The existing negative fixture is preserved without retuning.

The local source ZIP is a review artifact. Current arXiv documentation accepts bibliography source or a corresponding compiled bibliography; this package includes both. Local compilation is not arXiv processing or acceptance. No submission was made.

Sources checked: [regional definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_developed.html), [arXiv source requirements](https://info.arxiv.org/help/submit_tex.html). Primary-literature check scope is recorded in `research/completion_20261011/source_audit.json`.
