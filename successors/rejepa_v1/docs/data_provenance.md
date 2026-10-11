# Data provenance and admission

No market dataset is admitted. `rejepa.data` implements the executable
point-in-time validator and deterministic CSV snapshot writer.

An admitted row must carry stable asset identity, event/availability/outcome
times, market, currency, eligibility, point-in-time membership, positive price,
total return, volume and source identity. Non-USD rows require an FX conversion.
The validator rejects timestamps that travel backward, duplicate asset-date
keys, non-finite observations, invalid prices and missing FX. It warns when
delisting or corporate-action lineage is absent.

The snapshot writer sorts deterministically, normalizes UTC timestamps, writes
canonical bytes and produces a SHA-256 manifest containing provider, usage
right, retrieval time, coverage and schema. Passing this validator does not
prove a provider license or full survivorship coverage; those remain human and
source audits.

Fundamental, rate-curve, inflation, macro and classification observations use a
separate vintage-aware validator. It requires observation time, public
availability time, source, vintage, frequency and finite value. As-of selection
admits only vintages published by the decision time, so a later revision cannot
replace the value that was actually known historically.

Fundamentals and macro series must use historical publication/vintage times.
Current membership cannot be backfilled. Forward-filled values retain their
original availability time and cannot be presented as newly observed data.
