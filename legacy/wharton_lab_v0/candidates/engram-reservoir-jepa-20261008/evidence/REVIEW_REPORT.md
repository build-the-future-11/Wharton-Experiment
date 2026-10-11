# Wharton A-20261008-R2 adversarial implementation review

This was a separate line-by-line review pass by the implementing agent. It is not an independent-person review or scientific certification. Baseline source is preserved in Library ZIP version 0, SHA-256 `3cc74ee86718acc4f51040e4a8f5fb67995199bcfb40f4834fdadbbc21ce8e3c`.

## Reproduced findings and repairs

| Finding | Reproduction | Repair |
|---|---|---|
| Future liabilities classified as missed payments | $100 cash, $80 obligation due after a one-step horizon produced $80 unpaid/shortfall | Separate amounts due by horizon from future balances; reserve/report treatment is an explicit required planner policy |
| Silent zero cash-yield assumption | Omitting cash yield succeeded | Required per-step cash yield and return convention; required planner cash-yield sequence |
| Invalid inference clock accepted | Prediction with NaN as-of succeeded | Reject nonfinite clocks and require explicit maximum context age at CLI inference |
| Partially replaced model remained fitted after failed refit | Poisoning eligible training labels caused failure but left fitted=True | Mark unfitted at refit start; validate eligible data and fail closed |

All four failed before repair (`review-reproduced-failures.log`) and then passed (`review-fixed-regressions.log`). The future-reserve policy was made explicit during the repair so nominal reserve or report-only treatment cannot be selected implicitly.

## Line-by-line coverage

- `data.py`: checked CSV schema/order, hash admission, lawful-use/source attestations, protected-history boundaries, aware timestamps, strictly ordered observation availability, contexts strictly before decisions, compounded H-step labels and maturity delay. Timestamp and unknown-vintage limitations remain explicit. A row-level manifest cannot prove vendor vintages or licensing; no market data is admitted here.
- `model.py`: checked train/calibration chronology, boundary purging, input/latent normalization fit only on training rows, stop-gradient EMA targets, training-only future sequences, deterministic encoder/reservoir, XGBoost head budgets, out-of-fit calibration residuals, maturity-filtered memory, batch invariance and checkpointing. Poisoned calibration future targets leave encoder weights identical; poisoned unavailable contexts/labels leave fitted state and memory identical. Memory-off predicts the exact uncorrected head. The finite reservoir context/reset convention is explicit and is not original-spec fidelity.
- `pipeline.py`: checked origin/label alignment, disjoint test windows, per-origin refitting, retained raw predictions/failures, source/data/checkpoint identity, asset-order checks and context age. Existing all-matured ridge/mean baselines are stronger-history references, not described as matched fits. Added exact matched-training-row ridge and raw XGBoost comparators, plus a same-head memory-off contrast. Tree counts match; total encoder compute does not. No tuning or winner selection occurs.
- `accounting.py`: checked every cash/holdings mutation, post-fee rebalance equation, executed-notional fees, splits/dividends, before-trade/after-market payment timing, scheduled-to-arrears movement, zero/partial/full payment, write-off reporting, terminal mark/liquidation and closed-state behavior. Added uniquely identified beginning-of-step contributions; deposits increase both gross wealth and equity, and remain separate from market gains. All supported payment/unpaid/timing/terminal policies stay explicit.
- `planning.py`: checked forecast conditioning reaches scenarios; bootstrap uses only supplied observed history; allocations are common across scenarios; cash, fee and obligations flow through the same ledger; failures have no asset-zero fallback; fractional CVaR, future-liability policy and unpaid/write-off metrics do not double-count obligations. This remains a restricted common-action controller with no approved client objective, tail calibration or interim lateness penalty.
- `__main__.py`: exercised evaluate → checkpoint → forecast → plan and ledger commands, missing assumptions, checksum/order failures and stale inputs. Local pickle is restricted by expected hash and source revision but remains a trusted-artifact format; do not load arbitrary external pickle.
- `client_contract.py`: pinned recovered working-source identity; tested BOY contribution timing, ten 2033–2042 operating payments, immediate first payment (annuity due), explicit reserve returns/flexibility buffer and mechanical-ceiling language. No tax/return/asset policy was invented.

## Recovery findings

The Oct7 model handoff matches all 12 current `model.py` hashes. It does not verify the corrected M11 successor `mpc.py` in an inaccessible archive. The old Wharton GYIC mount path is absent. Local Downloads contain the IPS and Trading Notes instruction PDFs and a literature-review DOCX; they are not the original detailed Engram specification or admitted market panel. Library searches recovered the current cash-flow source snapshot and reserve memo. The overlay says Engram is an unadmitted proposal. The original mechanism was not recovered and fidelity is not claimed.

Library ZIP materialization was attempted through the supported helper for the integrated M04/M11 candidate, development backtest and CORE state; all three returned HTTP403. No archive bytes were read or source code imported. The parsed textual Library readbacks and twelve-file hash comparison are retained. M04’s 14.26% degradation remains a source-reported negative development result with the raw-French-vintage caveats intact.

## Remaining scope limits

No market-alpha result, full benchmark, fresh confirmation, competition approval or live action. No independent-person audit. No prescribed interim-payment lateness penalty, taxes, settlement mechanics, FX, lot rounding, or cash-account segregation enforcement for an actual portfolio. The 2031 credible-range method and long-run asset/scenario mapping require an adopted protocol and admitted data. Missing market/vendor availability and corporate-action contracts cannot be repaired by synthetic fixtures.

Forecasting uses log returns and therefore rejects returns at or below -100%. A dataset with such default/delisting observations cannot be admitted by silently dropping them; handling them needs a specified alternative transform/target contract. The accounting engine separately supports a total-loss mark. Synthetic tests do not resolve this market-data limitation.
