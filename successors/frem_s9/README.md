# FREM-S9 retrospective horizon/risk diagnostic

Read `research/frem_s9_20261015/SPECIFICATION.md`, `REVIEW.md` and the current receipt before running. This deterministic empirical-scenario model is an exploratory diagnostic using existing pre-2017 training-era ETF data. It is not a novel algorithm or a validated replacement for FREM-S8.

Components: `data.py` implements source/hash/calendar admission and paired horizons; `model.py` implements train-only neighbor/global forecasts and checkpoints; `policy.py` implements finite-action joint/diagonal risk policies and exact cash/dividend accounting; `experiment.py` generates the complete crossed comparison and report; `run.py` enforces resource admission, immutable sources/environment and one-use execution.

Run the bounded engineering tests:

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/s9 -p 'test_*.py' -v
```

After inspecting the current frozen specification and resource state, the launcher uses:

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 -m successors.frem_s9.run --root . --freeze research/frem_s9_20261015/freeze1.json --output outputs/frem_s9_20261015_attempt2
```

Use a fresh output directory. Exit 3 means resource admission blocked execution; exit 1 means an error; exit 0 means the declared diagnostic finished, not that the scientific or investment hypothesis succeeded. A consumed freeze cannot be reused after fitting begins. Do not remove consumption receipts or tune using this version's completed outcomes. Inspect `SUMMARY.json` and every retained control before making claims.

`SPECIFICATION.md` defines assumptions, failure behavior, complexity and the claim boundary. No raw data redistribution, third-party messages, public submission, live trading or paid compute is part of this package.
