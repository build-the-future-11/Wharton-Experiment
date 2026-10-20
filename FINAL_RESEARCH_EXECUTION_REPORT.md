# FREM research execution report — 20 October 2026

## Decision

**Submission readiness: NOT READY.** No empirical FREM-S14 result is verified. Private v0.1.3 saved version `359955736` is terminal: Kaggle failed while mounting the declared dataset before notebook execution, produced 0 B, collected zero tests, and completed 0/171 historical cells. The new v0.1.5 candidate is an engineering and observability release, not evidence of predictive improvement, robustness, profitability, or model validity.

## Delivered implementation

Work is on branch `codex/frem-research-execution-20261020`. The terminal v0.1.3 GPU version and all earlier artifacts were preserved.

- Added a full-arm engram retrieval trace: query slot, eligibility, similarity, age, base/graph/final weights, originating decision and maturity times, asset, value, coverage, and weighted memory moments.
- Added trace readback checks for same-asset and mature provenance, known bank origin, capacity, normalized retained weights, and independently reconstructed context moments.
- Kept traces diagnostic-only: they do not enter the model loss or change the issued law.
- Added versioned, non-overwriting candidate release paths; fixed package-version export and suite-wrapper parsing in the read-only plan CLI.

The exact candidate hashes and static-validation receipt are in [ENGINEERING_RECEIPT.json](research/frem_s14_trace_20261020/ENGINEERING_RECEIPT.json). The release assets are [wheel](successors/frem_s14/dist/frem_s14-0.1.5-py3-none-any.whl), [ZIP](successors/frem_s14/dist/frem_s14_kaggle_bundle-0.1.5.zip), [notebook](successors/frem_s14/kaggle/FREM_S14_GPU_0_1_5.ipynb), and [Kaggle metadata](successors/frem_s14/kaggle/kernel-metadata-0.1.5.json).

## Verification actually completed

- Ruff formatting and lint passed for the S14 source tree.
- 57 warning-strict dependency-free standard-library tests passed from source, and the same 57 passed from a clean wheel installation.
- The changed Python modules compiled.
- A fresh `--no-index --no-deps` installation of the 0.1.5 wheel imported as `0.1.5` and accepted the declared historical-suite configuration, reporting 57 arm-seed cells.

The prior 0.1.4 candidate is deliberately retained as failed packaging evidence: its wheel exposed `0.1.3`, and its plan command rejected the suite wrapper. It was not overwritten or relabeled as a success.

## Results and scientific boundaries

There are **no new forecast, portfolio, baseline, ablation, robustness, sensitivity, statistical, or figure results**. The local numerical admission gate remains closed: the latest recorded observation was 99–125 MiB available against the 512 MiB minimum and showed increasing pageouts. No numerical import or test was started locally.

The canonical 30-page manuscript was updated as a prospective protocol and evidence report. It uses the project-local `arxiv.sty`, states the mount failure and 0.1.5 boundaries, and adds no unsupported empirical result. A fresh offline build completed with no overfull boxes or unresolved references and passed rendered-page inspection.

## Ready-to-run GPU qualification

The prior version is finalized as an external dataset-mount failure before notebook execution. Upload the exact 0.1.5 notebook and metadata only after the Kaggle editor loads, with the attached dataset, private visibility, GPU enabled, and Internet disabled. Verify the new saved code and settings before launch. The notebook is self-contained and begins with hash verification and a no-index wheel install.

```bash
cd /Volumes/PRO-BLADE/GitHubRepos/Wharton-Experiment/successors/frem_s14
mkdir -p /private/tmp/frem_s14_upload_0_1_5
cp kaggle/FREM_S14_GPU_0_1_5.ipynb /private/tmp/frem_s14_upload_0_1_5/
cp kaggle/kernel-metadata-0.1.5.json /private/tmp/frem_s14_upload_0_1_5/kernel-metadata.json
kaggle kernels push -p /private/tmp/frem_s14_upload_0_1_5
```

The command is ready for an authenticated host with the Kaggle CLI; this workstation does not currently have that CLI installed, so no upload was attempted or implied.

On the admitted GPU target, preserve the entire generated run directory and require, in order: warning-strict target tests; generated CUDA qualification; immutable qualification verification; strict intake; all 171 declared cells; and independent saved-bundle readback. Any failed or unstarted cell remains in the denominator.

## Required work before a submission package can be called complete

1. Preserve the final v0.1.3 external-mount-failure receipt and do not reinterpret it as model evidence.
2. Create and verify a new exact 0.1.5 saved version, then run and retain its numerical/CUDA trace qualification on an admitted target.
3. Freeze a new execution decision before any 0.1.5 market suite; the trace candidate has a different source identity from v0.1.3.
4. Complete the fixed 171-cell matrix, retain failures, and independently reconstruct inputs, folds, predictions, scores, and ledgers.
5. Generate tables, uncertainty analyses, and figures only from that retained output; update the manuscript solely with those verified findings and limitations.
6. Build and independently compile the final LaTeX submission package after the empirical evidence and reviewer audit pass.
