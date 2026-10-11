# M12 — LAF-GMJEPA

## Objective
Composed stack for joint market modeling and liability-aware planning.

## Components
1. **Dynamic graph state** — trailing correlation kNN features.
2. **Episodic memory** — slot-based write/read.
3. **Latent world model** — recurrent latent transition.
4. **Scenario head** — M10 flow-matching module.
5. **Liability planner** — M11 MPC (optional via ablation).

## Training modes
- **Composed** (default): market encoder + ridge head; scenario head trained on encoded features; planner fitted only when liabilities supplied.
- **Joint** (`joint_training=True`): scenario head targets aligned with labels.

## Ablations
| Flag | Effect |
|------|--------|
| `no_graph` | Raw returns instead of graph features |
| `no_memory` | Skip memory read/write |
| `no_latent` | Pass-through obs into latent dim |
| `no_liability` | Equal-weight planner stub |

## Invariant
Market forecasts from `predict_market` must **not** change when only the liability schedule changes (same `X`).

## Files
- `src/wharton_lab/models/m12/model.py`
- `tests/unit/test_m12.py`
