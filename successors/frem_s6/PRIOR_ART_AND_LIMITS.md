# FREM-S6 source audit and evidence limits

Reviewed 13 October 2026. This is an additive design/source audit. No model was
trained by this audit, no protected outcome was read, and no old campaign was
reopened. The S5-R1 pilot remains closed with unsupported memory-specific benefit.

## What the requested ideas can mean precisely

| Requested idea | Defensible implementation target | Claim that is not established |
|---|---|---|
| Reservoir encoded engram kernel | Bounded recurrent sequence states used as keys for strictly matured dated episodes; finite-capacity retrieval; explicit context/no-context modes | A new reservoir or memory primitive, biological engrams, or a profitable predictor |
| JEPA | Learn predictions of target representations under an explicit target-encoder and anti-collapse contract | Transferring vision results directly to financial returns |
| Rebound ripples | Delay/accumulate updates, quantize update vectors, retain quantization residuals and apply a predeclared temporal schedule | That delayed errors are harmless, or that an error-feedback theorem automatically covers a changed optimizer |
| Network divergence | Explicit bounded branches or experts with declared routing and regularization | Unbounded autonomous network growth, guaranteed useful specialization |
| Mathematical proofing and exampling | Algebraic invariants, finite-difference checks, exact small references, adversarial examples, and falsifiable contract tests | Proof of future returns, causation, calibration, or the absence of every defect |
| Non-Gaussian probability zones | Proper normalized heavy-tail/mixture predictive densities and coverage diagnostics; scenarios with stated assumptions | Calibrated recession/funding probabilities merely because a density was produced |
| Stochastic calculus / physics for speed | A stated discretization or structured operator with an error/stability argument and measured runtime against a matched implementation | A generic speed or economic advantage from physics terminology |
| Dual / multinational operation | Separate forecasting and policy paths; explicit currency, clocks, eligibility, instrument metadata, FX and unified accounting | Investable global aggregation from independent local-currency survivor backtests |
| Longer-horizon decisions | Dated liabilities and cost-aware multi-period policy tests under matched exposure | Market manipulation or guaranteed wins; price-impact cost modeling does not establish permission or capacity to move markets |

The implementation target column is a design interpretation of the request, not
a declaration that every item has been implemented or tested. The S6 execution
receipt and code define the actual delivered subset.

## Primary prior art checked

1. **Reservoirs.** Herbert Jaeger's 2001 echo-state report, with the 2010 erratum,
   is prior art for recurrent reservoirs and readout learning. The author's
   [publication index](https://www.ai.rug.nl/minds/pubs/) and indexed
   [corrected report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)
   were located; a direct full-PDF fetch timed out in this audit. No theorem from
   that PDF is claimed as independently rechecked here. A contractivity proof
   for the particular implemented recurrence should be written and tested
   separately; spectral radius alone is not a proof for every nonlinear update.
2. **Engram.** Cheng et al., *Conditional Memory via Scalable Lookup* (submitted
   12 January 2026; current page reports v2 dated 12 July 2026), describes
   N-gram lookup memory for language models. Reservoir-keyed financial episodes
   are a different adaptation. Calling this project's memory “engram” does not
   make it an implementation of that system. [Primary paper](https://arxiv.org/abs/2601.07372v2).
3. **Associative memory.** Ramsauer et al., *Hopfield Networks is All You Need*
   (2020), connects continuous associative-memory updates and attention. A
   retrieval kernel should be compared against established nearest-neighbor,
   recent-loss and fixed-ensemble controls. [Primary paper](https://arxiv.org/abs/2008.02217).
4. **JEPA.** Assran et al., *Self-Supervised Learning from Images with a
   Joint-Embedding Predictive Architecture* (2023), predicts target
   representations from context blocks. It establishes image-learning prior
   art, not financial effectiveness. [Primary paper](https://arxiv.org/abs/2301.08243).
5. **Quantized error feedback.** Karimireddy et al., *Error Feedback Fixes
   SignSGD and other Gradient Compression Schemes* (ICML 2019), retains the
   compression error and feeds it into the next update. S6 temporal smoothing,
   clipping, delays or altered compression require their own assumptions and
   checks; the paper's convergence conclusions cannot simply be copied.
   [Primary proceedings paper](https://proceedings.mlr.press/v97/karimireddy19a.html).
6. **Conditional branches.** Shazeer et al., *Outrageously Large Neural Networks:
   The Sparsely-Gated Mixture-of-Experts Layer* (2017), uses a router over
   subnetworks. Conditional branching is established prior art. A local CPU
   implementation needs measured costs before claiming computational savings.
   [Primary paper](https://arxiv.org/abs/1701.06538).
7. **XGBoost.** Chen and Guestrin, *XGBoost: A Scalable Tree Boosting System*
   (2016), supplies the tree-boosting baseline and systems prior art. A real
   XGBoost arm must run its declared library, retain its fit settings and share
   the same information/tuning budget; a fallback estimator is not XGBoost.
   [Primary paper](https://arxiv.org/abs/1603.02754).
8. **Causal limits.** Schölkopf et al., *Towards Causal Representation Learning*
   (2021), distinguishes causal structure from statistical representation
   learning. Time ordering, feature masking, correlation tests and good
   predictions do not identify intervention effects without causal assumptions
   and identification evidence. [Primary paper](https://arxiv.org/abs/2102.11107).
9. **Execution impact.** Almgren and Chriss, *Optimal Liquidation* (1997
   manuscript), considers volatility risk and costs from temporary/permanent
   impact. Impact is a cost/risk modeling problem here; no live execution or
   market-manipulation mechanism is in scope.
   [Authors' paper record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=53501).

These sources defeat claims that memory, reservoirs, JEPA, mixture routing,
boosting or error feedback are new individually. Their combination remains a
candidate engineering design. A new name or combination is not novelty evidence.
This audit is a targeted prior-art pass, not an exhaustive novelty search.

## Local data inventory, without protected-outcome inspection

The following counts were read from manifests; the factor CSV's row count was
also checked with the standard library. Raw equity responses and factor ZIP
contents were not opened. Current quote metadata in retained Yahoo responses
must remain opaque and unused.

| Local source | Available evidence | Admission and reuse boundary |
|---|---|---|
| `data/raw/panel.parquet` and sidecar | Both absent on 13 October | Canonical point-in-time execution remains blocked |
| `data/public_equity_crises_v1/manifest.json` | 24 stocks, 12 listing markets, 158,760 daily observations; acquisition timestamp 11 October 2026 | Retrospective current-survivor selection; revised adjusted-return proxies; incomplete action/availability/default provenance; no FX. Protected 2025+ observations were excluded by acquisition. Completed crisis outcomes are spent development evidence |
| `outputs/frem_s3_20261011_attempt1/public_data/manifest.json` | 12 US stocks, 75,216 daily observations; acquisition timestamp 11 October 2026; requested history ends before December 2024 | S3/S4/S5 history is already inspected development data. The manifest's inherited single-seed/five-epoch limitation is stale for S3 training; use S3's specification and receipts for actual three-seed/twelve-epoch ceilings. Do not call this fresh confirmation |
| `data/public_regional_factors_20261011/monthly.csv` | 3,666 normalized monthly rows: US factors July 1926–December 2024 (1,182); developed, developed ex-US, Europe, Japan, Asia Pacific ex-Japan and North America July 1990–December 2024 (414 each) | Current-vintage aggregate research portfolios; overlapping regions; no stock eligibility/default/actions/execution contract. Raw ZIPs can contain 2025+ rows and must remain opaque |
| `data/engineering/yahoo_calibrated_v1/` | Existing explicitly synthetic engineering fixture | Engineering-only; not additional market histories or an independent financial sample |

The regional factor manifest records Dartmouth source last-modified dates of
23–25 September 2026 and retrieval on 11 October. They are revision timestamps,
not historical publication-availability proofs. No column is automatically a
tradable security. In particular, a market-excess factor must be reconciled with
the associated risk-free series before interpreting a total return; SMB/HML are
long-short research factors, not long-only assets.

“All datasets” therefore cannot mean all world markets or all available
post-2024 outcomes. A source inventory, provenance contract, coverage/exposure
register and separate bounded protocol are required for any new analysis. No
old negative result is eligible for iterative tuning until positive.

## Resource and scientific gates

`scripts/public_equity_backtests.py::resource_gate` samples macOS free plus
speculative memory three times and rejects if the minimum is below its configured
floor or pageouts increase. The normal fresh-training floor remains 512 MiB.
S5's recorded 128 MiB / 300-second cap applied to a standard-library replay and
does not relax that fresh-training gate. Earlier S2/S3/S4 slow-mode deviations
were campaign-specific historical authorizations.

The implementation session should take and retain a new resource snapshot
before importing the scientific stack. This document does not assert current
headroom from an old observation. Bounded standard-library checks can establish
invariants without establishing market performance. Tiny generated tests should
be labeled constructed engineering tests, not scientific backtests.

For fresh research, first record a material successor hypothesis and a finite
budget. Freeze preprocessing, forecasts, context modes, specialist inclusion,
model-selection objective, training trials, seeds, baseline family, chronology,
costs, exposure limits and stopping rules before outcomes. A zero-context mode
must remove the specified context consistently; it must state whether earlier
labels, persistent reservoir state, asset identity or memory remain available.
Comparing a context-free input with an unchanged contextual state is not a clean
zero-context ablation.

Use matched date-level observations and retain all seeds/failures. Compare fixed
averaging and recent-loss routing before accepting memory complexity. Separate
forecast quality from policy/exposure/turnover/fee effects. Preserve insufficient
precision instead of shortening bootstrap blocks until an interval passes.

The canonical protocol is `PROPOSED_NOT_FROZEN`; the existing protected boundary
is 1 January 2025. A successor must not read that tail while defining its
hypothesis or selecting hyperparameters. Complete scientific freeze and
protected-use bookkeeping precede confirmation. No runtime invariant or
synthetic pass proves financial calibration, causal discovery or profitability.

## Laura Gao: fresh public verification and exact remaining inputs

The official [competition homepage](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/),
[rules](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/)
and [FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/)
were checked on 13 October 2026. Current pages send registered teams to
SurveyMonkey Apply for the 2026–2027 case, trading requirements and deliverable
instructions. The public pages inspected did not supply an authenticated Laura
case amendment or an approved-instrument list. No credentials or private portal
were used. The homepage describes judging by strategy, client alignment,
research and explanation; maximizing simulated return is not the competition's
sole objective. The rules also require attribution of included AI-generated
material and prohibit submitting it as one's own work.

The repository's `configs/mandates/laura_gao.pending.json` remains the case
arithmetic source, backed by a user-document reproduction verified 10 October:
$300,000 at the beginning of 2027, $150,000 at the beginning of 2028, and ten
nominal USD $50,000 operating payments at the beginnings of 2033–2042. Those
facts were re-read locally, not independently authenticated with the organizer
today. At zero investment return, $450,000 of contributions is $50,000 short of
the $500,000 payments before any facility spending. This arithmetic is not a
prediction.

An executable or competition-certified allocation still requires:

1. Dated official case, amendments, trading rules and approved instruments.
2. Decision time, existing holdings, cash, NAV, settlement and available capital.
3. Instrument IDs, currency, execution prices/spreads/fees, liquidity, tradability,
   timestamps, corporate actions and aligned FX; approval provenance per asset.
4. Reviewed cash, concentration, market/sector, turnover, liquidity and FX limits;
   explicit permitted asset classes and leverage/shorting policy.
5. A numerical funding-confidence policy, flexibility buffer and facility
   spending rule; these are preferences/mandate inputs, not values an optimizer
   may invent.
6. Synchronized forecasts/scenarios with matching issuance and outcome horizons,
   measured calibration, and known source/model version. Current S5 historical
   forecasts do not establish October 2026 expected returns or buy quantities.

The supported decision process is to price the dated operating reserve and
assess diversified growth only for the affordable surplus under a reviewed
policy. This audit supplies no ticker weights, trade sizes, price targets or
funding-confidence number. It does not certify that any prior candidate ticker
is currently eligible. No trade, broker action, public submission or third-party
message was performed.
