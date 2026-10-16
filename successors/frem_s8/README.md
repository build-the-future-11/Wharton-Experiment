# FREM S8

Additive development successor: delayed soft gating of equal/contextual density mixtures and a continuous allocation with a committed four-interval holding horizon. The exact hypothesis, assumptions, controls and stopping rules are in [the specification](../../research/frem_s8_20261014/SPECIFICATION.md).

The code and scientific protocol were frozen before 2020–2022 data acquisition. A corrected source freeze handles a 2004-12-31 dividend outside every supported holding, before any market fitting or evaluation score. Both freezes, the failed intake and source archive are retained. No 2019 features, targets or feedback enter S8. The repository-wide freshness audit identifies prior SPY/GLD cache exposure; this remains a matched development evaluation, not globally untouched confirmation.

Tests (single-thread settings are required for the declared compute contract):

```sh
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest tests/s6 tests/s7 tests/s8 -q -p no:cacheprovider
```

The retained market execution is `outputs/frem_s8_2020_2022_20261014_attempt1`. It runs exactly three seeds and all 15 forecast variants, four policies and four costs. The permanent study consumption receipt rejects an accidental repeat. Do not delete it or rerun to seek favorable results.

Recompute the arithmetic audit from saved outputs into a **new** destination (no fits):

```sh
rtk proxy .venv/bin/python research/frem_s8_20261014/audit_saved.py --root outputs/frem_s8_2020_2022_20261014_attempt1 --data data/s8_etf_2020_2022_20261014_attempt2 --freeze research/frem_s8_20261014/freeze2.json --output /private/tmp/s8-audit-new.json
```

Reports, the full comparison table, source/data identities and all failures are under [research/frem_s8_20261014](../../research/frem_s8_20261014). Research ledgers use dividend accrual and endpoint reinvestment, zero cash yield and assumed quote availability. They are not actual fills, an executable mandate or evidence of calibrated funding probability. Original protected 2025+ outcomes remain untouched.
