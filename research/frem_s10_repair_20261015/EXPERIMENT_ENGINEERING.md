# Generated experiment implementation — bounded evidence

The pure experiment layer now enumerates the exact 192 cells, builds six separate causal weekly model states, retains immutable decision laws, replays every opening/dividend/fill/close/final-liquidation event, and publishes immutable, hashed evidence. It has no market reader or executable market CLI. Production source, freeze, resource, precision and execution-consumption admission remain external gates.

`experiment/matrix.py` preserves the planner's exact 128 strategy / 48 benchmark / 16 ablation identities. `forecasts.py` starts eligible prequential issues in 2008, supplies only matured labels, keeps models continuous across A/B, and retains the selected decision laws. Policy/cost arms and unchanged-forecast ablations reuse those immutable laws. A1 retains direct-H4 maturity while using repeated H1 components; A2 freezes the entire first-fold law separately in each fold; A4 uses the prospectively recorded CL-15 phases; A5 removes the exact-hold alternative.

`experiment/replay.py` starts every portfolio fold with 100000 cash, uses actual entering shares for dividends, retains actual daily-close NAV separately from hypothetical liquidatable boundaries, and charges the final actual exit once. Every dated result includes standardized ES/variogram/CRPS/MSE, raw per-asset CRPS/MSE, fixed 50/80/95-percent interval diagnostics with inclusive endpoints/ties, and separate asset/basket loss events and probabilities. Scores are cached only for the same retained law object and outcome index within one fixed-transform campaign; bitwise cached/uncached equivalence is tested.

The `KERNEL_REPLAY` input boundary reconstructs all origin metadata and x/g/d/H1 arrays and the burn-in transform from the provided panel, comparing complete payloads rather than trusting IDs. Source dividends require exact admitted asset ownership, including zero-amount event IDs. Close timestamps are supplied explicitly; no historical close time is invented. `KERNEL_REPLAY` describes the numerical path and is not proof of data authority.

`campaign_steps` yields one result at a time, allowing durable publication before the next cell. A failed cell retains its successful event/decision/score prefix and the last immutable state, including a completed A4 sale before a subsequent failed selection. A short stream creates an explicit incomplete-matrix failure. `experiment/artifacts.py` writes source/input/tape/attempt identities, typed arrays with shape/dtype/unit context, per-frame forecast files and per-cell evidence. Completion requires the exact inventory; missing, unexpected, changed and misordered artifacts fail. The verifier reads one payload at a time and reconstructs each forecast frame to check its binary numerical identity. Tape identities use bounded per-law binary64 component hashes and small ordered metadata rather than materializing millions of tagged floats.

`experiment/evidence.py` selects the fixed 10-bps endpoint columns, checks exact origin/time/anchor pairing and robust/nominal forecast identity, and requires the complete 192-cell matrix. It rejects generated-engineering evidence and retains the full-fold minimum-block requirement. The endpoint statistics remain separate from fitting and replay.

## Verified scope

- 58 owned constructed tests passed with warnings treated as errors in 15.64 seconds. Measured child peak RSS was 111394816 bytes, about 106.2 MiB.
- The full 192-cell fixture uses two blocks per fold and three-atom constructed forecast laws. It exercises the real policy, six benchmarks, exact fee roots, event ledgers, full cell publication and inventory verification. It is not a full-scale fitted campaign.
- Separate tests run the real six-variant forecast kernel, verify causal maturity and future-target perturbation invariance, and connect the real kernel to policy selection and ledger execution on a tiny generated schedule.
- Explicit-config strict mypy, Ruff and formatting passed. Exact source hashes and log references are retained in `experiment_engineering_verification.json`.
- Failed diagnostics remain retained. `/usr/bin/time -l` failed its sandbox-denied `sysctl kern.clockrate` after pytest passed; the final measurement uses Python's child resource usage. Earlier Ruff findings were import ordering, an unescaped test regex and function complexity; they were repaired without changing the scientific mechanism.

## Remaining boundaries

No market values, protected outcomes or market fits were accessed. No profitability, empirical support, source license, or full-length/520-atom feasibility conclusion follows. The source/timestamp provenance adapter, authorized supervised production invocation, full-scale resource profiling, complete weekly mixture-solve timing/certificate archival, saved JSON-to-endpoint regeneration, final figures/reporting, and independent reproduction remain unfinished. The old results and protocols remain intact.
