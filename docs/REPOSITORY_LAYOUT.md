# Repository layout

The checkout contains three separate prospective research lineages and one
read-only predecessor archive:

| Path | Lineage | Current evidence boundary |
|---|---|---|
| `src/frem/` | FREM v0.1 | Runtime-tested source; no admitted-market model cell executed |
| `successors/rejepa_v1/` | RE-JEPA v0.3 | Synthetic engineering compatibility only |
| `successors/reng_jepa_v01/` | RENG-JEPA v0.1 | Synthetic engineering compatibility only |
| `legacy/wharton_lab_v0/` | Original Wharton lab and candidates | Historical invalid, negative, null, and engineering evidence |

Generated caches, local environments, dependency payloads, paper
intermediates, render previews, and distribution archives are ignored. Runtime
receipts, experiment manifests, source datasets explicitly labeled for
engineering use, reviewed PDFs, and negative/failure evidence remain tracked.

`RESEARCH_STATE.json` is the canonical cross-lineage state. Each successor also
retains its own state and receipts. Similar names do not make the successor
protocols interchangeable: RE-JEPA v0.3 uses the Student-t/NLL protocol, while
RENG-JEPA v0.1 implements the later quantile-score manuscript.
