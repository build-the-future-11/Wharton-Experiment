# FREM-S14: GPU implementation candidate

S14 preserves S13 and all earlier research. It adds a two-dimensional latent stochastic diffusion field, masked multi-offset JEPA, mature-neighbor graph engrams, and training-residual-guided bounded neurogenesis to the reservoir and three-objective routed Student mixture. It retains XGBoost, hybrid XGBoost, ridge, GRU, zero, and mechanism ablations.

**Status: 0.2.4 semantic-reproducibility release candidate; target execution pending.** Version 8 ran the exact 0.1.9 package, but its retained historical metrics are invalid evidence because independent semantic readback failed. Version 0.2.0 repaired deterministic preprocessing and score-vector retention, but it was superseded before upload when a pre-launch audit found stale 0.1.9 runtime identity and stale 2,800-second documentation/configuration inside the release. Versions 0.2.1 through 0.2.3 were preserved unrun during release-audit hardening: 0.2.1 exposed packaged test-style defects, 0.2.2 was built before their corrected bytes were retained, and 0.2.3 exposed invocation-directory-dependent import classification plus a non-fail-fast build wrapper. Version 0.2.4 makes first-party classification explicit and is the clean packaging-only successor. It does not change the architecture, seven-ETF cohort, data hashes, thresholds, folds, arms, seeds, labels, scores, 171-cell denominator, or 13,500-second notebook ceiling. Do not interpret qualification as financial validation. All historical negative, failed and invalid results remain closed.

## Kaggle execution

Use the versioned notebook and metadata emitted by `kaggle/build_release.py` to save a new private Kaggle version. Preserve every prior version, including the terminal 0.1.3 mount failure. The exact release archive is embedded and hash-verified; no separate private payload dataset is needed. Attach `borismarjanovic/price-volume-data-for-all-us-stocks-etfs` and enable a GPU accelerator. Do not duplicate an already running or completed candidate.

The default notebook executes dependency/runtime checks, the complete test directory, a 19-arm generated qualification, independent readback, strict fixed-cohort intake, and the 171-cell historical suite. Any failed engineering qualification stops market fitting. No CPU substitution. The notebook has a 13,500-second total process budget, retains stdout, errors, checkpoints and manifests, and refuses existing run directories. Set `MODE = "qualify"` only to deliberately request engineering qualification alone.

The historical matrix uses seven fixed, hash-allowlisted ETFs, three expanding prefixes, three seeds, 19 arms, 20-observation compounded targets, ten epochs and three-epoch early stopping. QQQ was excluded prospectively after the pre-market source audit; the retained files are not intersected or filled. Each 0.2.x fold has a 4,000-second bound for fitting plus mandatory in-run semantic verification, inside the unchanged 13,500-second notebook ceiling. An exhausted or incomplete matrix is a failure, not a reduced winning matrix. Full package commands are real implementations in `src/frem_s14`; no generated fixture is described as market evidence.

Dataset caveat: adjusted-close ratios are a historical proxy. Point-in-time adjustment vintages, membership, terminal distributions, executable quote prices, spreads and market impact are not established. Proxy ledgers cannot establish investable profitability. The 2025+ boundary remains closed.

## Local commands

```bash
python -m pip install -e '.[dev]'
frem-s14 preflight --output /tmp/s14-fresh-admission.json
python -m pytest tests -W error -q
frem-s14 qualify --device cuda --output /tmp/s14-fresh-qualification
python -m frem_s14.intake --source /kaggle/input --output /tmp/s14-fresh-intake
python -m frem_s14.suite --data /tmp/s14-fresh-intake/panel.csv --qualification /tmp/s14-fresh-qualification --output /tmp/s14-fresh-suite --device cuda
```

Run from this directory after installation, using new output names every time. Numerical imports remain behind resource admission. CPU runs are explicit engineering checks only; CUDA qualification is required for a CUDA campaign. See `SPECIFICATION.md`, `DATA_CONTRACT.md` and `CHECKLIST.md` for exact scope and limitations.

## 0.1.2 evidence and numerical hardening

This candidate additionally closes delayed-decision target leakage, protected-boundary drift,
malformed CSV rows, JSON exponent overflow, nested-manifest omissions, special-file entries,
unverified resume accounting and incomplete benchmark-cell accounting. It preserves the model
components, the 19 arms, three seeds, three folds and every budget/threshold.

Graph engram retrieval uses float32 inside AMP even with a populated bank; ineligible values
are masked before moments. Mixture variance uses within-component variance plus centered
between-component variance. Compounded labels use read-only strided windows; training avoids
unused per-batch metrics transfers. These efficiency changes do not establish measured speedups.

`frem-s14 verify-suite --bundle <downloaded-suite-directory>` independently checks all 171 cells,
all fold configurations, calendars, scores, ledgers and the entire Holm comparison family on an
admitted numerical host. The dependency-free archive command checks a downloaded result ZIP
without extracting it, using a separately trusted local release digest:

```bash
python -m frem_s14.archive --archive downloaded-results.zip --release dist/frem_s14_kaggle_bundle.zip --expected-release-sha256 TRUSTED_RELEASE_SHA256 --output fresh-download-receipt.json
```

Use the actual archive path and `archive_sha256` in the preserved `dist/RELEASE.json`.
Integrity verification can describe a failed or partial run; it never turns that run into a pass.
The 0.1.0 GPU outcome and 0.1.1 local candidate are retained under research records. Test results
for those sources do not transfer to 0.1.2.
