"""M05 Q-APEN model."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m05.config import M05Config
from wharton_lab.models.m05.experts import (
    ExpertSpec,
    expert_variance,
    fit_expert_local,
    init_weights,
    predict_expert,
)
from wharton_lab.models.m05.lifecycle import (
    LifecycleState,
    MaturedLossRecord,
    PendingMaturity,
    cosine_similarity,
    maybe_create_expert,
    merge_experts,
    prune_worst,
)
from wharton_lab.models.m05.quantized_memory import QuantizedExpertMemory


class QAPENModel(BaseModel):
    MODEL_ID: ClassVar[str] = "M05"
    metadata: ClassVar[ModelMetadata] = ModelMetadata(
        model_id="M05",
        title="Q-APEN — Budgeted Quantized Adaptive Expert Ensemble",
        version="0.1.0",
        description="Top-k sparse routing over linear/Gaussian experts with quantized memory and causal lifecycle.",
        inputs=("features",),
        outputs=("point_forecast",),
        training_constraints=("matured_losses_for_lifecycle",),
        metrics=("routing_entropy", "disagreement"),
        tags=("ensemble", "quantized", "causal"),
    )

    def __init__(self, config: M05Config | None = None) -> None:
        self.config = config or M05Config()
        self._rng = np.random.default_rng(self.config.random_state)
        self._memory = QuantizedExpertMemory(
            self.config.memory_cap_bytes, self.config.codebook_size
        )
        self._specs: list[ExpertSpec] = []
        self._router_w: np.ndarray | None = None
        self._lifecycle = LifecycleState()
        self._n_features: int | None = None
        self._fitted = False

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_memory=True,
            requires_maturity=True,
            max_entities=None,
        )

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def _kind_cycle(self, idx: int) -> str:
        kinds = self.config.expert_types
        return kinds[idx % len(kinds)]

    def _ensure_experts(self, n_features: int) -> None:
        if self._specs:
            return
        self._n_features = n_features
        for i in range(min(2, self.config.max_experts)):
            kind = self._kind_cycle(i)
            spec = ExpertSpec(expert_id=f"e{i}", kind=kind, n_features=n_features)
            self._specs.append(spec)
            w = init_weights(kind, n_features, self._rng)
            self._memory.pack(spec.expert_id, w)
        self._router_w = self._rng.normal(scale=0.1, size=(n_features, len(self._specs)))

    def _route(self, X: np.ndarray) -> np.ndarray:
        assert self._router_w is not None
        logits = X @ self._router_w
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        k = min(self.config.top_k, probs.shape[1])
        top_idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
        mask = np.zeros_like(probs)
        rows = np.arange(probs.shape[0])[:, None]
        mask[rows, top_idx] = probs[rows, top_idx]
        s = mask.sum(axis=1, keepdims=True)
        s[s < 1e-12] = 1.0
        return mask / s

    def _expert_preds(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = []
        vars_ = []
        for spec in self._specs:
            w = self._memory.unpack(spec.expert_id)
            if w is None:
                w = init_weights(spec.kind, spec.n_features, self._rng)
            preds.append(predict_expert(spec.kind, w, X))
            vars_.append(expert_variance(spec.kind, w, X))
        return np.stack(preds, axis=1), np.stack(vars_, axis=1)

    def _aggregate(self, weights: np.ndarray, preds: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        # Reliability must vary by expert: a common row-wise factor cancels.
        center = (weights * preds).sum(axis=1, keepdims=True)
        reliability = 1.0 / (self.config.disagreement_eps + np.maximum(vars_, 0) + (preds-center)**2)
        reliable_weights = weights * reliability
        denom = reliable_weights.sum(axis=1)
        return (reliable_weights * preds).sum(axis=1) / np.maximum(denom, 1e-12)

    def _lifecycle_update(self, batch_loss: float) -> None:
        ready = self._lifecycle.advance_step()
        for spec in self._specs:
            self._lifecycle.queue_loss(spec.expert_id, batch_loss)
        if len(self._lifecycle.matured) < self.config.min_matured_before_lifecycle:
            return
        if len(self._specs) >= self.config.max_experts:
            to_prune = prune_worst(
                [s.expert_id for s in self._specs],
                self._lifecycle.matured,
                self.config.prune_worst_fraction,
            )
            for eid in to_prune:
                if len(self._specs) <= 2:
                    break
                self._specs = [s for s in self._specs if s.expert_id != eid]
                self._memory.remove(eid)
        else:
            new_spec = maybe_create_expert(
                self._specs,
                self.config.max_experts,
                batch_loss,
                self.config.create_loss_threshold,
                self._n_features or 1,
                self._rng,
                self._kind_cycle,
            )
            if new_spec is not None:
                w = init_weights(new_spec.kind, new_spec.n_features, self._rng)
                if self._memory.pack(new_spec.expert_id, w):
                    self._specs.append(new_spec)
        # merge similar — only same kind / same weight dim
        if len(self._specs) >= 2:
            for i in range(len(self._specs)):
                for j in range(i + 1, len(self._specs)):
                    if self._specs[i].kind != self._specs[j].kind:
                        continue
                    wi = self._memory.unpack(self._specs[i].expert_id)
                    wj = self._memory.unpack(self._specs[j].expert_id)
                    if wi is None or wj is None:
                        continue
                    if wi.shape != wj.shape:
                        continue
                    if cosine_similarity(wi, wj) >= self.config.merge_similarity_threshold:
                        new_id, merged = merge_experts(
                            self._specs[i].expert_id,
                            self._specs[j].expert_id,
                            wi,
                            wj,
                            f"m{len(self._specs)}",
                        )
                        if self._memory.pack(new_id, merged):
                            self._memory.remove(self._specs[j].expert_id)
                            self._specs[j] = ExpertSpec(
                                expert_id=new_id,
                                kind=self._specs[i].kind,
                                n_features=self._specs[i].n_features,
                            )
                        break
        # keep router width aligned after create/merge/prune
        if self._router_w is None or self._router_w.shape[1] != len(self._specs):
            nf = self._n_features or (self._router_w.shape[0] if self._router_w is not None else 1)
            self._router_w = self._rng.normal(scale=0.1, size=(nf, max(1, len(self._specs))))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> QAPENModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._ensure_experts(X.shape[1])
        assert self._router_w is not None
        if self._router_w.shape[1] != len(self._specs):
            self._router_w = self._rng.normal(
                scale=0.1, size=(X.shape[1], len(self._specs))
            )
        for spec in self._specs:
            w = self._memory.unpack(spec.expert_id)
            if w is None:
                continue
            w_new = fit_expert_local(spec.kind, w, X, y)
            if not self._memory.pack(spec.expert_id, w_new):
                self._memory.pack(spec.expert_id, w)
        weights = self._route(X)
        preds, vars_ = self._expert_preds(X)
        agg = self._aggregate(weights, preds, vars_)
        loss = float(np.mean((agg - y) ** 2))
        self._lifecycle_update(loss)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self._specs:
            self._ensure_experts(X.shape[1])
        weights = self._route(X)
        preds, vars_ = self._expert_preds(X)
        return self._aggregate(weights, preds, vars_)

    def memory_nbytes(self) -> int:
        return self._memory.used_bytes()

    def pending_maturity_count(self) -> int:
        return len(self._lifecycle.pending)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "memory": self._memory.serialize(),
            "specs": [
                {"expert_id": s.expert_id, "kind": s.kind, "n_features": s.n_features}
                for s in self._specs
            ],
            "router_w": None if self._router_w is None else self._router_w.tolist(),
            "lifecycle": {
                "pending": [p.__dict__ for p in self._lifecycle.pending],
                "matured": [m.__dict__ for m in self._lifecycle.matured],
                "step": self._lifecycle.step,
            },
            "fitted": self._fitted,
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M05Config.from_dict(state["config"])
        self._rng = np.random.default_rng(self.config.random_state)
        self._memory = QuantizedExpertMemory(
            self.config.memory_cap_bytes, self.config.codebook_size
        )
        self._memory.deserialize(state["memory"])
        self._specs = [
            ExpertSpec(e["expert_id"], e["kind"], e["n_features"]) for e in state["specs"]
        ]
        rw = state.get("router_w")
        self._router_w = None if rw is None else np.asarray(rw, dtype=float)
        lc = state["lifecycle"]
        self._lifecycle = LifecycleState(
            pending=[PendingMaturity(**p) for p in lc.get("pending", [])],
            matured=[MaturedLossRecord(**m) for m in lc.get("matured", [])],
            step=int(lc.get("step", 0)),
        )
        self._fitted = bool(state.get("fitted", False))
