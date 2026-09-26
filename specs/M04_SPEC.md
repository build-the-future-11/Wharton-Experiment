# M04 — Financial APEN

## Lineage

Reimplementation of the **Synthica / financial APEN** pattern: episodic memory of **prediction residuals** keyed by **historical context**, with delayed maturity — not a port of physics APEN PDE machinery.

**Gap vs original physics APEN:** no continuous adjoint field; discrete bounded store, cosine similarity retrieval, and scalar residual correction.

## Mechanism

1. **Backbone** \(g(x)\): Ridge (default) or small MLP.
2. At decision time \(t\), log context key \(\kappa_t = \mathrm{normalize}(\phi(x_t))\).
3. **Value stored:** residual \(e_t = g(x_t) - y^{\mathrm{obs}}_t\) once outcome known at enqueue; episode enters **PendingOutcomeQueue** until \(t + H + D\).
4. After maturity, episode moves to **BoundedMemoryStore** (FIFO eviction at capacity).
5. **Predict:** retrieve top-\(k\) similar keys with `decision_time <= as_of`; gate on similarity, novelty, confidence; correct:
   \[
   \hat{y}_t = g(x_t) - \conf \cdot \mean_{j \in \mathcal{N}_k} e_j.
   \]

## Audit logs

`insertion_log`, `retrieval_log`, `eviction_log` carry as-of timestamps.
