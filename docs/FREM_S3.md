# FREM-S3 execution contract

S3 is an additive research lineage. The original FREM, S2, completed data/results, negative findings and protected boundary remain intact.

The canonical pre-execution declaration is `research/frem_s3_20261011/specification.json`; the exact implementation/data request is bound by each run's `freeze.json`. The code snapshot is in `research/frem_s3_20261011/source/`. This freeze is for new-sample exploration, not the canonical protected confirmation.

## Implemented components

- Four causal training partitions: base fit, checkpoint selection, distribution calibration and residual/key fitting. Each boundary purges labels that have not matured.
- Shared mean-shrinkage/Student-t scale calibration across FREM, no-JEPA, Ridge, XGBoost, zero and historical mean. Validation candidates are retained; base fitting never uses the public evaluation outcomes.
- Immutable base predictor and versioned normalized residual episodes, with strict delayed feedback and checkpoint roundtrips.
- Fourteen memory arms: off, frozen, recent average, age/asset-matched random, mean-only, risk-only, combined, no-confidence, balanced-capacity, larger-capacity, dual-bank, cross-asset, learned-key and deployed-NLL utility retention.
- An absolute-age, similarity, effective-neighbor-count and disagreement gate. The same zero-weight fallback remains available.
- Student-t variance-aware risk correction, with shrinkage and bounded scale factors.
- Fresh independent generated mechanism paths: null, recurring mean, recurring variance, abrupt change and missing/delayed feedback, three seeds each. Oracle distributions are diagnostics only.
- Public request names twelve stocks before download, excludes the earlier KO calibration source, and retains original failures. Strict source adapter rejects malformed/nonpositive prices; no outcome-based substitutions or quarantine repair.
- Two new public windows, 2023 and January-October 2024. The source request ends before December 2024 and never requests 2025 prices. These are current-survivor local USD return proxies, not a complete historical market universe.
- All random seeds/folds retained. Paired date scores average seeds before resampling. Three-month primary and six-month sensitivity blocks are fixed. Unavailable sensitivity is reported, not repaired by shortening its block. Holm correction spans the declared comparison family.

## Resource policy

The user's current request authorizes slow execution. Existing normal admission requires 512 MiB free/speculative memory, which is not satisfied at the session snapshot. Record this separately. Each worker runs at nice10 with one BLAS/Torch thread, a sampled 768 MiB RSS ceiling, and a bounded wall time. One worker runs at a time. Process logs, failures and resource receipts are immutable. The campaign ceiling is two hours.

## Commands

Execute a new, uniquely named campaign:

```sh
rtk proxy .venv/bin/python -m scripts.run_frem_s3 --output outputs/<new-s3-run> --execute
```

Independently reconstruct scores and verify original/frozen hashes:

```sh
rtk proxy python3 -m scripts.audit_frem_s3 --run outputs/<s3-run> --output research/<new-audit>.json
```

The launcher does not continue to public acquisition if the predeclared synthetic recurrence/null gate fails. A negative result is terminal for this fixed protocol. Scientific model settings are not changed based on its outcomes.

## Boundaries

Implementation and generated-mechanism success are not market evidence. New public-sample gains are exploratory because of survivor/revision/timestamp limits and short windows. No canonical protected data, global investment simulation, trading, or public submission is included. Admitted point-in-time data and a distinct frozen confirmation remain necessary for those claims.

The independent audit recomputes saved likelihoods without SciPy; it does not independently train the model. All lineage dispositions must cite their direct receipts.
