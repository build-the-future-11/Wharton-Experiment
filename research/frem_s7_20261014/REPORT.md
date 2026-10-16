# S6 cause audit and S7 fresh-development result

**S7 was built and tested, but it did not improve the primary endpoint. Preserve this negative result.**

## Measured S6 causes

1. The six-expert router scored each expert independently and softmaxed historical losses. That is different from optimizing the likelihood of their mixture. Adaptive-volatility weight fell from 1/6 to 0.11016 on average. Its scale averaged 0.32227 of the zero expert scale: weak standalone likelihood does not imply no complementary density value.

| Fixed-forecast intervention on spent S6 predictions | NLL |
|---|---:|
| memory | -2.748738155 |
| equal | -2.753844941 |
| restore_vol_weight | -2.753905213 |
| vol_scale_sqrt5 | -2.747900710 |

Restoring only adaptive-volatility weight to 1/6 (rescaling other weights proportionally) removes essentially all of the memory deficit. This is a deterministic attribution of the saved forecast difference, not a proof of future usefulness. Simply multiplying its scale by sqrt(5) worsened the score and was not adopted. The original implementation estimates that expert scale from daily returns while forecasting roughly weekly outcomes; the mismatch is real, but correcting it mechanically is not established as a mixture improvement.

2. The old study omitted weeks unless both Monday opens existed. It was explicitly an included-episode study, not continuous calendar performance. On those episodes SPY returned -4.8976%, TLT +0.1716%, GLD -8.5393%. Between the same January 23, 2017 and December 31, 2018 endpoint opens, their continuous price returns were +10.0644%, +0.2576%, +4.7355%. Thus missing holiday intervals changed the measured market exposure materially. Dividends were also excluded. No old result is overwritten or relabeled.

3. The signal policy kept an 80% risky anchor even when predictive means were poor. At zero costs the S6 signal ledger ended at $95,248.49, versus $96,228.29 for equal-weight assets. Thus gross exposure/allocation, not fees alone, caused the loss. At 10 bps it ended at $94,643.60: $604.89 additional terminal-wealth drag relative to the zero-cost counterfactual. Nominal fees were $625.89; fees and terminal drag differ because fees alter subsequent holdings. Fixed expert averaging was only $2.67 lower, so the memory mechanism did not explain the common portfolio loss.

## Frozen successor

S7 changes the router objective to the likelihood of the full expert mixture, regularized toward equal weights with 12 date-equivalent pseudocounts. It compares contextual and global mixture estimates and retains an ungated control. The deployed contextual strength is selected from 0, .25, .5, 1 using 2018 only; nonzero must improve both chronological halves by more than 0.01 nats. The candidate is allowed to select zero.

A separate finite-candidate cost-aware policy chooses hold, cash, an 80% equal anchor, or the old signal allocation using predicted log means, diagonal variance penalty and estimated two-way fees. Every forecast is crossed with both policies; cash/equal-asset/buy-and-hold controls are retained. This is a research heuristic, not a calibrated portfolio optimizer.

Training: 2005–2016 with purged labels/future targets. Scale calibration: fully mature 2017 observations. Router selection: 2018. Evaluation: 51 continuous weekly intervals in 2019, acquired only after freeze2. Three base fits/seeds (11/29/47); the forecast variants share those experts. First common trading session per week to the next, actual 3–10 day horizons; dividend cash accrued at ex-date and reinvested at interval end. No gaps in evaluation holdings. Payment timing, taxes and actual fills are not modeled.

This is newly acquired S6/S7 development data, not universally unseen market history or independent confirmation. Current-vintage survivor ETFs and assumed quote availability remain limitations. Original 2025+ protected outcomes remain untouched. No global-equity or Laura-funding claim.

## Fresh results

| Forecast | NLL (lower better) | Old signal wealth, 10 bps | Cost-aware wealth, 10 bps |
|---|---:|---:|---:|
| s7 | -2.628128323 | $119,187.29 | $100,000.00 |
| equal | -2.628128323 | $119,187.29 | $100,000.00 |
| memory | -2.641469469 | $119,241.51 | $100,000.00 |
| recent | -2.640514774 | $119,281.78 | $100,000.00 |
| mixture_ungated | -2.640997196 | $119,250.40 | $100,000.00 |
| mixture_global | -2.640988221 | $119,250.24 | $100,000.00 |
| zero | -2.635825319 | $115,989.32 | $100,000.00 |
| ridge | -2.659157440 | $119,851.85 | $100,000.00 |
| trend | -2.653770317 | $118,342.41 | $110,285.99 |
| adaptive_volatility | -1.908108787 | $115,989.32 | $100,000.00 |
| reservoir_jepa | -2.658838820 | $120,025.80 | $100,000.00 |
| xgboost_hist | -2.656900443 | $118,899.40 | $100,000.00 |
| cash benchmark | N/A | $100,000.00 | N/A |
| buy_hold benchmark | N/A | $119,874.88 | N/A |
| equal_weight benchmark | N/A | $120,165.90 | N/A |

All three seeds chose alpha=0. S7 therefore exactly matched fixed averaging: gain 0 and intervals [0,0]. The S6 memory control beat S7 by 0.013341 nats on this later period; S7-minus-S6 gain intervals were [-0.020372,-0.005931] (3-date) and [-0.020378,-0.006084] (6-date). These are descriptive development intervals, not corrected confirmatory claims.

Ungated mixture routing improved on equal averaging descriptively but did not beat S6 memory. Global and contextual mixture scores were nearly identical. Neither fact authorizes post-result substitution for the frozen gated primary.

At 10 bps the S7 cost-aware policy remained entirely in cash, paying zero fees and finishing at $100,000. The equal-asset benchmark finished at $120,165.90. At zero costs the same cost-aware forecast policy finished at $119,926.29. The predicted benefit/cost threshold was too conservative for the realized rally; hindsight cannot establish that the forecast could have anticipated it. S7 signal policy finished at $119,187.29. Reduced trading did not establish improved investment value.

## Verification and preservation

125 S6/S7 tests passed after retaining one test-fixture failure. S7 worker completed in 11.52 seconds, peak RSS 169.03 MiB on macOS. Explicit user RAM override from the immediately preceding run remained recorded; one-thread execution and a two-hour parent timeout were retained.

Independent standard-library audit: 7,344 forecast NLL values reconstructed (maximum error 4.44e-15), 108 ledgers verified (maximum residual 2.91e-11), 2,334 labels reconciled to retained opens/dividends. All 22 frozen S6 identities and all 16 S7 freeze identities unchanged. This is not independent retraining.

Artifacts: diagnosis.json, freeze1.json, freeze2.json, AUDIT.json, acquisition/coverage files, all per-seed checkpoints and selection gains, predictions, ensembles, costs, benchmarks, process receipts and original test failure. ALL_BASELINES.csv contains every forecast/policy/cost combination.

## Terminal disposition

Diagnosis complete; successor implemented and tested; improvement not established. No completed protocol reopened, no favorable seed omitted, and no post-2019 tuning. A further successor would need a new hypothesis and a later evaluation period frozen before acquisition. Potential next hypotheses are expert-quality/forecast calibration and economically validated policy selection; these remain proposals, not demonstrated fixes.
