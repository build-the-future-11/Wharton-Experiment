# Public equity crisis study

This is a separate exploratory study on downloaded historical prices for 24
present-day listings in 12 markets. It does not satisfy the worldwide historical
equity request, the canonical point-in-time data contract, or the protected
scientific protocol. The code and dataset make no claim to contain delisted firms.

The immutable acquisition request is `data/public_equity_crises_v1/request.json`.
Each response has a hash, URL, retrieval timestamp and receipt. All 24 downloads
succeeded, yielding 158,760 daily source rows. Current metadata quote values in
raw responses are opaque and unused. Requested price history ends before 2025.

## Defined experiment

The configuration is `configs/public_equity_crises.json`. There are six windows:
December 2007–June 2009, July 2009–December 2010, calendar 2011, 2015–2016,
calendar 2020 and calendar 2022. FREM, no-memory, ridge and XGBoost each use seed
11, producing at most 24 cells. Neural models get five epochs with identical
early-stopping rules. These bounded settings are not a tuned or converged
scientific comparison. The total ceiling is 12 hours, with a 300-second fit cap.

Prices are reduced to the last returned daily observation in each calendar month
in the listing's exchange timezone. A missing final observation remains missing;
it is never forward-filled. An incomplete monthly history is disclosed in the
fold coverage receipt. The same included stocks and rows are used for each model.

For target return month M, the decision coordinate is the start of M−1. Inputs
end in M−3, leaving M−2 blind. The forecast concerns the adjusted-close ratio
between the last source observations in M−1 and M. For example, the December
2007 target uses inputs through September, a November 1 decision coordinate,
and the November-to-December adjusted-close ratio. A target is treated as mature
on the third calendar day of the following month. These are explicit research
timing assumptions, not historical publisher timestamps or certified fills.

Training labels must mature strictly before validation begins, and validation
labels strictly before the first test decision. Scalers, checkpoints, model
weights and memory remain fitted on pre-evaluation data. Each later window fits
an expanding history ending before its own evaluation. All dates were selected
before model outcomes. The historical crisis labels are report annotations only.

## Interpretation

Forecast scores use equal stock weight within each month and equal month weight.
The FREM-minus-control analysis reports control NLL minus candidate NLL, so a
positive value favors FREM. A three-month circular block bootstrap uses 2,000
resamples and seed 1901. Intervals are exploratory: there is one optimization seed,
few crisis months, and no correction across the six reported windows.

Each stock also has a separate hypothetical local-currency long/cash ledger.
The rule is long when predicted log return is positive and cash otherwise.
Cash earns zero. Each long month is an independent round-trip at 10 basis points
per side; self-financing wealth grows by `exp(actual) * (1-fee)/(1+fee)`.
The same-cost always-long strategy and gross continuous buy-and-hold are reported
alongside it. Initial wealth is included in drawdown calculations. The median
stock result is a descriptive statistic, not a global portfolio return. No FX
translation, investability, liquidity, taxes, borrow or actual fill is asserted.

Yahoo describes adjusted close as split/dividend adjusted in its
[documentation](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html).
These current-vintage public data may be revised; the study does not certify
historical adjustment availability or reconcile all actions. A blind month does
not cure that revision bias or the survivor-selection bias. Source access is
public; no redistribution license or institutional entitlement is inferred.

## Reproduction

Acquisition and execution refuse existing output paths. Choose new paths for a
justified reproduction and preserve any prior failed attempt.

The initial attempt was blocked before model imports at
`outputs/public_equity_crises_v1_admission_20261011/failure.json`. A separate
600-second resource wait also expired before the new tests could start. Static
checks passed; the new adapter's runtime behavior is not yet verified.

Once memory admission succeeds, verify the six new tests before running models:

```sh
.venv/bin/python -c 'from scripts.public_equity_backtests import resource_gate; print(resource_gate(512))'
TMPDIR=/Volumes/PRO-BLADE/.codex-tmp/wharton-pytest PYTHONWARNINGS=error \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_public_equity_backtests.py
```

The retained source directory can be used directly as `--data
data/public_equity_crises_v1`; it does not need to be downloaded again. The
following also documents acquisition for a separate, explicitly versioned copy:

```sh
.venv/bin/python scripts/public_equity_backtests.py acquire \
  --config configs/public_equity_crises.json --output data/NEW_ACQUISITION
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/public_equity_backtests.py run \
  --config configs/public_equity_crises.json --data data/NEW_ACQUISITION \
  --output outputs/NEW_CAMPAIGN
```

The runner checks three memory samples before scientific imports and before each
cell. It stops below 512 MiB free-plus-speculative memory or on increasing
pageouts. Receipt files distinguish resource stops, incomplete matrices and
completed cells. A source download or a passing test is not a completed backtest.
