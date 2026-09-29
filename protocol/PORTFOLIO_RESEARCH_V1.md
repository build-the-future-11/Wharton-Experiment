# Post-lockbox portfolio research protocol, v1

Status: exploratory synthetic benchmark, not a fresh confirmatory lockbox.
Source basis: upstream `8e3d790dd513a93b7d80d5726a8b2708e8615b71`.
This protocol does not authorize trading, submissions, or reuse of withdrawn evidence.
The original protocol, lockbox records, and historical result files are preserved.

## Fixed design before the production run

Evaluate M01–M12, each full model and one specifically named ablation, using five
seeds 2026092901–2026092905 in four worlds. Expected production cells: 480.
A disjoint-seed preflight may check software interfaces. It is not pooled into
production evidence, and no favorable production configuration is selected.
The runner snapshots exact source hashes and configuration before executing cells.

Each synthetic panel has 320 timesteps and five anonymous assets. At decision
origin t, only returns strictly before t are input features. Training target
origins run from 20 through 235 inclusive. The outer test origins are 240–299.
Every five-step training target ends no later than 239. A single common RMS scale
is fitted on returns 1–239, never on test returns. Model-specific inputs are
explicitly recorded; linear baselines use the corresponding information set.

Innovations are 0.5 times a common noise plus sqrt(0.75) times asset-specific noise.
The null world is IID with innovation scale 0.01. The linear-signal world has
AR coefficient 0.65 with scale 0.01. The regime-shift world changes coefficient
0.5 to -0.3 and innovation scale 0.008 to 0.022 at timestep 240. Heavy tails use
variance-normalized Student-t innovations with four degrees of freedom and AR
coefficient 0.2. These are constructed stress settings, not market observations.

## Tasks and counterfactuals

| ID | Primary measurement and baseline | Predeclared ablation |
|---|---|---|
| M01 | Pinball across 0.05/0.25/0.50/0.75/0.95; ridge with training-only calibration quantiles; empirical quantiles also reported | No regime feature |
| M02 | Negative date-level Spearman on future residualized returns; residual ridge ranker | Fit raw-return labels instead of residuals |
| M03 | Cross-asset next-return MSE; linear PCA-AR(1) loadings | Historical defective single-snapshot loading target |
| M04 | Next-return MSE; information-matched full-training ridge; same-backbone output retained | No matured episodic memory |
| M05 | Next-return MSE; information-matched ridge | Linear experts only instead of linear/Gaussian experts |
| M06 | Next-return MSE; information-matched flattened-window ridge | Remove predictive latent loss, retain auxiliary and variance/covariance terms |
| M07 | Frobenius error to realized five-step future sample covariance; trailing 20-step sample covariance; global covariance also reported | Zero context, retaining the same SPD-head architecture |
| M08 | Mean asset-wise next-return MSE; information-matched multioutput ridge | Zero graph edges during training and prediction |
| M09 | Next-return MSE; information-matched ridge | Fixed rather than updated group weights |
| M10 | Joint five-step, five-asset energy score; conditional ridge residual bootstrap; unconditional bootstrap also reported | No conditioning information |
| M11 | CVaR95 of total unpaid currency obligations; cash-only and equal-weight policies | Require full investment rather than allow cash |
| M12 | Point MSE; information-matched ridge; separately recorded composed-planner stress output | No episodic memory |

Covariance targets are finite-window realized covariances, not an oracle
conditional population covariance. Energy scores use the distinct-pair
U-statistic estimator. All losses are explicitly task-labelled; losses across
tasks and units are never ranked against one another.

For M03, refit at each test origin using only history ending at t-1. This is a
prequential experiment, not a static untouched holdout. Its linear factor
baseline uses exactly the same history and factor estimates. Other predictive
models remain frozen through their contiguous test blocks. Different model
families therefore must not be ranked on a single aggregated leaderboard.
M04 reserves the final 30% of training origins to accumulate matured calibration
residuals, leaving its backbone fixed; the main ridge baseline can use all rows.
M06 internally purges target-horizon overlap before its validation boundary.

M11 fits one shared open-loop five-period policy to 64 training bootstrap paths.
M12 fits its scenario head to genuine full paths and exercises the composed
scenario-to-planner route, but is not jointly trained. Policy evaluation resets
initial cash to 1 for each nonoverlapping heldout five-step window. Obligations
are [0.2,0.2,0.2,0.2,0.23], fees are 5 basis points and asset caps 0.4. Cash earns
zero. This is a hypothetical liability stress test, not a client case or trading
recommendation. An open-loop plan is not a full receding-horizon policy.
The controller objective is CVaR95(unpaid / initial wealth) plus 1e-4 times mean
squared risky weight. The risk-concentration coefficient is fixed, not tuned.

## Statistical and integrity rules

Use the simulation seed as the independent replication unit. Adjacent test
origins and overlapping forward windows are not independent replicates. Report
all five paired seed outcomes per world, their mean and a clearly exploratory
95% t interval for paired loss differences. The Gaussian approximation may be
poor with five seeds. No multiplicity-controlled positive discovery is declared.
No claim of equality is inferred merely from an interval including zero.
Report ablation effects with their sign convention, including harmful components.

Record failed cells with exception text and absent measurements, never zeros.
Every numerical result must reproduce from saved predictions, targets, and
loss arrays, with source/config/data/output SHA256 values. Preserve per-cell
wall times as local measurements, not cross-hardware speed comparisons.
No hyperparameter sweep or post-hoc best-seed selection is part of this run.

## Explicit non-claims

This run does not establish market alpha, client suitability, novelty of all
twelve named methods, stability at large scale, or submission readiness of
independent papers. Graph M08 is currently a supervised predictor with a frozen
graph. Repaired M07 is conditional SPD regression, not a new joint-embedding
objective. M09's fusion and M12's latent recurrence are fixed random features.
M05's one-batch run does not establish dynamic expert lifecycle benefits.
A genuine final publication claim requires evidence appropriate to that claim;
legacy names and complete execution receipts are not substitutes.

## Numerical preflight amendment before production

The first disjoint-seed preflight (seed 911) exposed SLSQP iteration-limit failure
in the nonsmooth CVaR controller. It is preserved as incomplete/failed preflight
evidence. Production M11 and M12 use an explicit finite candidate bank instead,
not a hidden fallback. Fractions 0, .25, .5, .75, 1 of the risky budget, equal-weight
and each single-asset capped tilt, plus linearly increasing/decreasing exposure
schedules, are evaluated under the exact same training ledger objective.
Only feasible policies enter the bank; duplicate policies are removed. The
fully-invested ablation uses only fully invested feasible candidates. The method
finds the minimum *within that bank*, not a continuous/global optimum.
Optional SLSQP remains available in the library, with convergence reported
separately from the existence of a verified feasible policy.
This is an engineering repair; no test-set performance selected the bank.
