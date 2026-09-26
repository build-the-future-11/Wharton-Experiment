# Trading Notes — PROPOSED only (not FINAL analysis)

**Status:** BLOCKED_DATA — no verified executed WInS trades in this workspace.

A FINAL Trading Notes Analysis requires three **genuinely executed** WInS trades with:
- Exact original note text
- Executed timestamp
- Source record / blotter export
- Reflection ≤100 words per note

## Evidence-collection workflow
1. Export blotter from WInS (CSV/PDF) into `data/wins/executions/`.
2. Record SHA-256 of each source file in `data/wins/executions/MANIFEST.yaml`.
3. For each of three trades, create `data/wins/executions/trade_0N.json` with fields: timestamp, symbol, side, quantity, price if available, note_text_exact, source_file, source_hash.
4. Only then draft reflections and assemble the FINAL analysis PDF.
5. Until then, keep proposals under `data/wins/proposed/` and label every note **PROPOSED**.

## PROPOSED placeholders (not executed)
None populated — populating fake executions is prohibited.
