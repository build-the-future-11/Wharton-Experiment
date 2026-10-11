# RE-JEPA architecture and novelty boundary

## Precise research statement

Existing forecasting systems can be sample-inefficient when histories are short
or regimes shift. RE-JEPA proposes a causal reservoir-JEPA whose prediction is
augmented by (a) stochastic rollouts in its own latent dynamics and (b) bounded
retrieval of Fourier-keyed episodes credited only after outcomes mature. The
testable claim is not that simulation creates information. It is that these two
inductive biases may use limited information more effectively than a matched
backbone without them.

## Forward chronology

For asset features x[1:t], the model:

1. Computes a trailing Fourier-band descriptor from x[1:t].
2. Updates a fixed contractive reservoir r_t.
3. Encodes raw, reservoir, and spectral state into context latent z_t.
4. Uses z_t as a query into engrams with `outcome_available_at <= t`.
5. Samples K latent future paths from a learned stochastic transition model.
6. Maps each set of worlds through M learned interpretation experts.
7. Uses a gate based only on z_t to combine those interpretations.
8. Fuses z_t, imagined state, and matured engram retrieval.
9. Applies a growth expert for Student-t return location/scale and a protection
   expert for loss probability, downside magnitude, volatility and drawdown
   exposure; a declared location-scale transform supplies a downside quantile.

The EMA target encoder maps a later, causally defined target window into a
training target latent. It is never used to expose future data at inference.

## What “chooses how it interprets it” means

The model has several interpretation experts. Their parameters and gate are
learned only on development data. At an inference time t, the gate sees z_t,
not the realized return after t. The exact checkpoint, gate, world count, random
seed policy, and scenario aggregation are frozen before protected evaluation.
This is conditional inference, not test-time tuning.

## What “engram backpropagation on a Fourier transfer” means

The current Fourier descriptor contributes to the query key. Similarity weights
over already-matured engrams remain differentiable with respect to the current
query and fusion layers. Stored past keys and values are detached immutable
records. A new record can receive utility credit only after its forecast outcome
becomes available. No gradient travels backward into an unknown future outcome
or silently rewrites a retained episode.

## Separate return and loss systems

The growth expert estimates distributional return. The protection expert
estimates downside probability and magnitude. They share the same causal latent
information but have separate parameters and losses. The portfolio layer then
solves an explicit trade-off; neither expert can hide the other objective in a
single opaque score. A single-head control tests whether separation helps.

## International portfolio layer

Forecasts are base-currency total returns with explicit FX and availability.
Eligibility, exchange calendars, corporate actions, delistings, trading costs,
liquidity, and market/sector/currency caps are point-in-time inputs. The optimizer
cannot include an instrument merely because it exists today.

The long-term goal layer is separate from WInS performance. It evaluates the
2033 operating-reserve funding probability and the 2031 credible facility range
under declared return assumptions. It never inserts WInS gains into the case
projection.

## Implemented research-system boundary

The v0.2 package also implements the surrounding system required to test the
architecture: point-in-time validation and hashed snapshots, Laura funding and
three-sleeve calculations, stress-conditioned simulator diagnostics, temporal
knowledge-graph relations, representation-collapse probes, purged walk-forward
splits, simple baselines, scenario-CVaR allocation, FX/exposure analysis,
immutable experiment manifests, complete receipts, an evidence-bound report
generator and a nine-command CLI. None of those modules turns absent data into
evidence; empirical outputs remain blocked until a lawful panel and actual
competition universe are admitted.

## Novelty boundary

JEPA latent prediction, world-model imagination, reservoir computing, Fourier
time-series features, episodic memory, multi-objective learning, and
liability-aware allocation all have prior art. The candidate contribution is
their exact causal coupling: stochastic latent worlds plus delayed-utility,
spectral-keyed immutable engrams inside a matched reservoir-JEPA, evaluated
specifically under predeclared low-data conditions.

That novelty is not established by writing code. Recent time-series JEPA work
reports that gains can vary materially by backbone, so a negative or mixed
result is plausible and scientifically acceptable.

## Primary failure modes

- Latent worlds may be uncalibrated hallucinations rather than useful scenarios.
- More parameters or compute may explain any gain.
- The interpreter gate may collapse to one expert.
- Engram retrieval may memorize noise or duplicate current information.
- Fourier descriptors may add no useful signal beyond the reservoir.
- Shared features may cause growth and protection gradients to interfere.
- Thin international histories may introduce survivorship, FX, or calendar bias.
- A forecasting gain may not survive costs or improve Laura's funding outcomes.

Each failure has a predeclared mechanism-off or simpler baseline in the protocol.

