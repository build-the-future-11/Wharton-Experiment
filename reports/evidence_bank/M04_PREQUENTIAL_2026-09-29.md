# M04 maturity-safe prequential evaluation — 29 September 2026

**Status: executed exploratory synthetic evaluation; NOT a market-performance or confirmatory result.**

## Work completed

- Added `src/wharton_lab/evaluation/m04_prequential.py` and its causal regression tests. No original model, loader, legacy protocol, result file, or holdout was modified.
- Activated the canonical Financial APEN residual-memory mechanism by replaying predictions and disclosing outcomes only after they mature. The old executor only called `fit` then `predict`, leaving M04 memory empty.
- Used train-only standardization, maturity-purged training rows, disjoint test blocks, and fresh memory for each fold. The one-step target uses one step of outcome horizon plus two steps of delay. This is an explicit new exploratory protocol, not an amendment to a spent lockbox.
- Included seven arms: zero, historical mean, online mean, the exact frozen ridge backbone, maturity-matched online bias correction, M04 online memory, and a zero-residual-memory ablation.
- Completed **38 regression tests** in an isolated source subset. The original bodies of eight upstream modules were reconstructed from connector reads and verified byte-for-byte against their Git blob hashes. Namespace-package imports avoided loading unrelated models. This is NOT a full-repository integration test.
- Executed **20 dataset/seed cases**, **160 fold evaluations**, **7 arms per fold**, and **16,000 unique evaluated target rows**. These are 1,120 arm/fold cells, NOT 1,120 independent experiments. Only **M04 (one of the twelve core models)** was evaluated here.
- Repeated the full suite in the same environment: **42 protocol/result/prediction/metric files were byte-identical**. Wall-clock run manifests differ as expected. This is deterministic replay, not independent external reproduction.

## Results

The primary contrast is M04 online memory versus its identical frozen ridge backbone. Extra MSE is the mean across five seeds of `100 * (MSE_M04 / MSE_backbone - 1)`; positive means worse. No raw MSE values are pooled across worlds.

| Synthetic world | Seeds | Extra MSE from memory | Seeds beating backbone | Active correction rate |
|---|---:|---:|---:|---:|
| signal | 5 | +7.16% | 0/5 | 82.53% |
| null | 5 | +6.02% | 0/5 | 70.83% |
| regimes | 5 | +5.85% | 0/5 | 75.92% |
| fat_tails | 5 | +4.06% | 0/5 | 71.10% |

Memory changed 12,015 of 16,000 predictions. The zero-residual-memory ablation matched the backbone exactly. There were 15,520 matured test-label updates across the 160 folds. All fit-time labels satisfied the declared maturity cutoff.

**Interpretation:** activating memory does not improve these null-like one-step mean forecasts under the declared default settings; it increases error in all 20 dataset/seed cases. The original SIGNAL generator uses a contemporaneous independently drawn latent variable, not a latent value observable from causal lags. The other worlds likewise do not supply a designed predictable conditional mean. These are useful no-signal robustness checks, not evidence that APEN cannot work on other processes or real markets. A constant-bias unit fixture confirms the mechanism can activate after maturity; it is not financial research evidence.

No post-result hyperparameter tuning was performed. Folds share expanding training histories. Worlds reuse the same seed identifiers. No claim of 160 independent replicates or multiplicity-corrected significance is made.

## Reproduce

From the repository root, with the repository dependencies installed:
```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/causal/test_m04_prequential.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m wharton_lab.evaluation.m04_prequential --out runs/m04_prequential/new_run_20260929
```

The output directory must not already exist. Protected legacy output names are refused. The resolved protocol and source fingerprints are saved before any outcome-bearing evaluation; per-case full predictions and metrics are retained, and a COMPLETED manifest is written only after the suite finishes. The runner refuses changed upstream module hashes and only permits the five previously used development seeds.

The companion execution package retains all original-source subset files, full predictions, metrics, run manifests, test logs, protocol, and deterministic-replay checks. The compact per-case results committed beside this report are `M04_PREQUENTIAL_2026-09-29.csv`. Full prediction CSVs are not committed into the legacy results tree.

## Provenance

- Upstream repository: `build-the-future-11/Wharton-Experiment`.
- Source revision: `8e3d790dd513a93b7d80d5726a8b2708e8615b71`.
- Root tree for the review branch: `31e4c1cac7f08f4ebc42d9ecfea544dbfe5ae0e1`.
- Protocol: `protocol/M04_PREQUENTIAL_EXPLORATORY_2026-09-29.json`.
- Evaluator SHA-256: `f467364dc07cfcaddbd63f575e28957ec6eea1569ed5030341cfffcf78ab2c6c`.
- RESULTS.json SHA-256: `ff63bb8a448f3ef338917316a9dc917751542044cc001bf5e6260e8ecfdafd2b`.
- Runtime: Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0.

## Explicitly not completed

- M01–M03 and M05–M12 causal task-specific evaluations.
- Real ETF evaluation: the exact original cache is not versioned in the repository and was not available in this isolated execution.
- A predictive-market or persistent-regime-error research benchmark; the positive-control fixture is a unit test only.
- Trading, portfolio construction, transaction-cost validation, new confirmatory holdout access, client-case approval, external submission, and full-repository integration tests.

**Promotion gate: do not promote M04 memory to a claimed source of alpha on this evidence. Preserve the negative result and require a separately declared predictive task, matched online controls, and real-data validation before any such claim.**
