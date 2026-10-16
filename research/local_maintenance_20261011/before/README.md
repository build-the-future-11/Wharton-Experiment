# FREM: Causal Fourier–Reservoir Engram Memory

Latest evidence revision: see [research completion status](docs/RESEARCH_COMPLETION_STATUS.md). The 23-page paper now reports the retained negative synthetic finding. Seven regional public datasets were acquired; no new market model cells ran, and submission readiness remains blocked.

**A source-complete, engineering-verified prospective research package. Version 0.1.1, updated 11 October 2026. Scientific status: NOT EVALUATED.**

FREM turns the proposed JEPA/reservoir/Fourier/memory idea into a specific, falsifiable financial forecasting system. The candidate contribution is **retaining bounded episodes according to predictive utility observed only after their outcomes become available**. It must beat the same predictor without memory and simpler memory-retention policies before that contribution can be claimed.

This delivery includes the forecasting model, training and inference, seven baseline families, ablations, data adapters, statistical analysis, portfolio constraints and fallbacks, Laura Gao’s funding requirements, a paper, and the Codex execution handoff. The earlier core verification passed 163 FREM tests, 67 RE-JEPA v0.3 tests, 45 RENG-JEPA v0.1 tests, full-scope Ruff, static validation, and package builds. Those are engineering checks only: no admitted market dataset was supplied, no market model fit or backtest ran, and no protected outcome was accessed. There are no trained market weights or invented performance tables.

## Start here

1. Read `RESEARCH_STATE.json` before changing the project. It preserves the question, evidence status, prior reported failures and remaining gates.
2. Read `docs/CODEX_HANDOFF.md` for the exact next execution sequence and `docs/DATA_GUIDE.md` for acquisition and normalization.
3. Read `paper/main.pdf` for the mathematical manuscript, or rebuild it from `paper/main.tex` and `paper/references.bib`.
4. Read `docs/STATIC_VERIFICATION.md` for what was checked and what remains unverified.
5. Complete the external data and mandate inputs before attempting the dependent commands. The supplied templates intentionally reject missing evidence.

## What the system implements

| Requested idea | Implemented interpretation |
|---|---|
| JEPA for financial prediction | Temporal context encoder, future target encoder updated by EMA, latent predictor, anti-collapse penalties, supervised probabilistic return head. |
| Reservoir computing | Fixed leaky recurrent state with an operator-norm contraction bound; its trainable readout is combined with the temporal encoder. |
| Modified Fourier anomaly model | Causal trailing Hann-windowed DFT, band descriptors, train-only spectral surprise reference, learned band weights, and explicit episode writes. The DFT itself remains standard. |
| N-grams / engrams | Quantized short return motifs and stable memory keys; bounded retrieval, matured residuals, delayed utility credit, aging and alternative eviction policies. |
| XGBoost | A standalone gradient-tree baseline and optional residual corrector with a separate purged calibration interval. |
| Choosing how to learn | A bounded controller for learning rate, auxiliary-loss weight and the matured memory expert. This is a declared feedback policy, not a newly learned backpropagation algorithm. |
| Spatiotemporal selection | Trailing feature/time gates, asset and sector/market context, causal peer summaries, and timestamp-constrained episodic memory. |
| Stochastic finance and allocation | Student-t log-return uncertainty, explicitly bounded simple-return scenarios, covariance/CVaR risk, dated mandate constraints, self-financing transaction costs, fallbacks, and liability reserve calculations. |

The package does not call stock dynamics a known physical law. It supplies stochastic assumptions and testable stability results; it does not contain a fitted continuous-time SDE or an unverified physics-informed PDE loss. Reservoir contraction proves forgetting of initial state under its assumptions, not profitable prediction from almost no observations.

## The information contract

The primary target is cumulative **base-currency log total return**, from the first executable close strictly after the decision to the close `H` executable intervals later. Publication, corporate-action, FX and outcome-availability times are explicit. Unlabeled inference uses only the latest available trailing context, with future fields marked unknown.

CSV and parquet panels are accepted through explicit column mappings. Corporate actions, currencies, historical eligibility and feature availability remain economic requirements, regardless of file format. All preprocessing fits on the training partition. Development removes the protected tail before building windows and excludes earlier samples whose labels mature inside that tail.

See `docs/DATA_GUIDE.md` for mandatory columns, example manifests, raw-price versus total-return-index conventions, bond cash flows, exchange calendars, vintage macro data and data rights. `data/templates/` contains empty headers and deliberately incomplete manifest inputs. Empty headers are not a dataset.

## Forecasts and portfolio decisions

The forecasting and investment questions are separate. A better likelihood score is not a profitable strategy. The `decide` bridge takes saved distribution parameters, an explicit common horizon, dated instrument metadata, current holdings and a reviewed mandate. It records its bounded scenario assumptions before deriving simple-return moments and allocation weights. The separate `allocate` command accepts already supplied simple-return moments/scenarios with their return-space and currency contract.

For finite-degree Student-t **log** returns, the positive exponential moment diverges. Therefore, `exp(mu)-1` is not an expected simple return. Bounded scenario conversion is a declared change to the distribution, and its caps and dependence assumptions must be assessed. See `docs/FORECAST_TO_PORTFOLIO.md`.

Portfolio code enforces eligible instruments, market/sector/asset-class limits, security caps, group floors, cash, FX exposure, turnover, available liquidity, tradability and provenance. Candidate and fallback portfolios are checked after solving, and lot rounding is checked again. A recession probability is an external dated input or scenario, not a fact inferred merely from the current narrative. No broker or live-order integration is provided.

The included realizable return ledger is deliberately defined for synchronized, nonoverlapping long-only round trips with cash. It rejects overlapping or asynchronous horizons. Forecasting multiple markets does not make their exchange closes simultaneous. Actual individual-bond defaults with zero recovery, full settlement accounting and arbitrary asynchronous holdings require a broader execution model before making claims about them.

## Laura Gao’s mandate

The recovered user preparation document supplies these cash-flow facts:

- $300,000 at the beginning of 2027, then $150,000 at the beginning of 2028.
- No other pre-2033 projection contributions or withdrawals.
- At the beginning of 2033, establish the reserve before paying the first operating amount or funding the facility.
- Ten nominal $50,000 beginning-of-year payments from 2033 through 2042.
- In 2031, communicate a credible range for the 2033 facility contribution, conditional on explicit assumptions and financial flexibility.

The reserve is an annuity due under a constant-rate illustration, with pathwise reserve sizing and security cash-flow matching also implemented. WInS trading-period gains/losses do not alter the specified long-term starting contributions.

`configs/mandates/laura_gao.pending.json` preserves the known facts and refuses allocation until the actual participant approved-security list, trading requirements, current holdings and numerical team policy are supplied. The user document is a copied case, not an independently authenticated organizer PDF. Travel and recipe ideas are not evidence of an approved investment universe. The Taiwan exchange’s commonly cited 10% daily price band is not a blanket 10% foreign-investor portfolio cap. Sources and distinctions are in `docs/LAURA_MANDATE.md` and `docs/MARKET_RULES.md`.

## Installation and verification commands

These commands reproduce the local engineering gate. They were run successfully with Python 3.12 and the committed `uv.lock`; Python 3.11 remains in the declared compatibility range but was not separately exercised in this session. Default experiments use CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
python scripts/validate_source.py
python -m pytest -q
frem plan --config configs/first_execution.json
```

After normalizing a licensed panel to `data/raw/panel.parquet` with its manifest and reviewing the protected boundary:

```bash
frem inspect-data --path data/raw/panel.parquet --before 2025-01-01T00:00:00Z
frem run --config configs/first_execution.json --phase development --execute
```

The printed run directory identifies all subsequent evidence. Use `frem compare --run <run-directory>` and `frem report --run <run-directory>` only for a complete matrix. The full `compare-family` requires every declared secondary mechanism control; the small compatibility preset intentionally does not contain that full family. Commands with angle brackets require the actual printed path and are documentation conventions.

Do not jump directly from installation to the protected confirmation. The handoff specifies environment capture, data checks, bounded first execution, development, freeze, one protected use, failure handling, analysis and paper completion.

## Presets and budget scope

| Configuration | Intended purpose | Declared size |
|---|---|---|
| `configs/first_execution.json` | First bounded compatibility exercise | Four methods, one seed, two neural epochs; 900-second matrix cap. |
| `configs/development.json` | Initial research development matrix | Eight methods × three seeds = 24 cells; 12-hour matrix cap. |
| `configs/robustness_suite.json` | Optional, separately selected sensitivity/perturbation work | Twelve variants × three seeds = 36 cells; separate 12-hour suite cap. |
| `configs/confirmatory.json` | Candidate final evaluation plan | Eighteen methods × five seeds = 90 cells; 24-hour matrix cap; requires a matching freeze. |

These are proposed ceilings, not measured costs or an authorization to run every preset. Wall limits are cooperative; some native estimator operations need an external scheduler for a hard deadline. Passing a nominal shared cap is not proof that tuning opportunities or achieved compute are equal. The clearest mechanistic comparison is FREM versus its otherwise matched memory-off control.

## Project map

| Location | Contents |
|---|---|
| `src/frem/` | Model, data, training, inference, memory, baseline, risk, portfolio, research governance and CLI implementations. |
| `tests/` | Mathematical, temporal, serialization, data, metrics, funding and constraint checks; 163 currently pass. |
| `configs/` | Candidate bounded experiments and pending mandate/instrument inputs. |
| `research/` | Prospective protocol and exact original user research pipeline; future freezes and append-only decisions belong here. |
| `paper/` | Methods/protocol manuscript, bibliography, compiled PDF and reproducible source. |
| `docs/` | Mathematical specifications, acquisition, financial rules, independent review, static verification and Codex handoff. |
| `scripts/` | Static validation, paper-build, manifest, and bounded synthetic efficiency-verification helpers. |
| `RELEASE_MANIFEST.json` | Packaged file hashes, evidence status and release identity. |

## Evidence discipline

There are no admitted-market accuracy, profitability, or data-efficiency results in this release. Local synthetic timing measurements are explicitly bounded engineering evidence and do not establish research performance. The earlier APEN/graph-memory failures reported in the user’s preparation document remain recorded as prior reported evidence whose raw runs were not re-audited. A new implementation is not proof that those failures are solved.

The next researcher must keep failed runs, complete seed families and protected-tail consumption receipts. A valid negative result closes this version’s tested claim; a new mechanism requires a successor hypothesis. Consult `docs/RESEARCH_PIPELINE.md`, `AGENTS.md` and the canonical state before modifying the project.
