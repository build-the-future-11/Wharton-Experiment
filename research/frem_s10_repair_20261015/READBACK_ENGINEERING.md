# Saved evidence reconstruction

This supplements the earlier experiment engineering receipt without replacing its
history. The new `experiment.readback.load_campaign(directory, manifest_sha256)`
reconstructs saved complete or failed-prefix campaigns as immutable typed records.
It first verifies the exact artifact inventory and forecast-frame semantic hashes,
then checks scientific fields, numeric array units and dimensions, event ordering,
state identities, decision/score pairing, accounting conservation, fill certificates,
and actual versus hypothetical NAV marks. It does not read source markets, execute
models, select policies, or confer source authority.

Nine final focused tests passed with warnings treated as errors in 14.20 seconds.
Strict mypy, Ruff lint and format checks passed. Exact paths, hashes, logs and retained
earlier failures are recorded in `readback_engineering_verification.json`.

The roundtrip covers all 192 cells with real policy selection, benchmark rules,
transaction fees and ordered ledger accounting over constructed three-atom forecast
tapes and two decision blocks per fold. It preserves terminal state hashes, exact
boundary increments, score arrays and event attribution. Tests reject corrupted
array metadata, recomputed file hashes hiding a changed semantic state, and a failed
partial falsely labeled complete. An explicit execution guard proves readback does
not call fitting or policy selection. An integer-valued initial capital case checks
that serialization retains the numeric type required by typed scientific hashes.

The existing immutable input, tape and campaign test fixtures now have session scope
so the readback module reuses their generated replay. This changes test setup lifetime
only; it has no effect on scientific model behavior or the frozen protocol.

The actual replay roundtrip is intentionally separate from the analysis adapter's
24/25-row constructed retained-evidence tests. It does not establish a single full
market-length raw-data-to-paper execution. Full source authority, admitted production
invocation, market-length compute profiling, and scientific precision/freeze gates
remain separate work. No empirical effect or positive scientific result is claimed.
