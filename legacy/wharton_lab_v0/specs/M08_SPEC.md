# M08 — Dynamic Graph Financial JEPA

## Graph
- Built from **trailing shrinkage-correlation** of historical node returns (fit history only).
- **Frozen** adjacency after `fit`; `predict` reuses snapshot (optional `shuffled_edges` control).

## Model
- Temporal **GRU node encoder**
- **Message passing** on frozen graph
- **Future latent** scalar head per node
- **Node mask** support for missing entities

## Distinction from M06
Graph construction and message passing are required; not a sequence-only JEPA.
