"""M04: Financial APEN — memory-corrected predictions."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Union

import numpy as np
from sklearn.linear_model import Ridge

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m04.config import M04Config
from wharton_lab.models.m04.memory import (
    BoundedMemoryStore,
    PendingOutcomeQueue,
    confidence_from_similarities,
    novelty_gate,
)


class _TinyMLP:
    def __init__(self, n_in: int, hidden: int, rng: np.random.Generator) -> None:
        self.W1 = rng.normal(scale=0.1, size=(n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(scale=0.1, size=hidden)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        h = np.tanh(X @ self.W1 + self.b1)
        return h @ self.w2


class M04FinancialAPEN(BaseModel):
    MODEL_ID = "M04"

    metadata = ModelMetadata(
        model_id="M04",
        title="Financial APEN",
        version="0.1.0",
        description="Context-keyed memory of prediction residuals; ridge/MLP backbone correction.",
        inputs=("features", "context_key"),
        outputs=("corrected_prediction",),
        training_constraints=("pending_maturity", "as_of_retrieval"),
        metrics=("mse", "memory_hit_rate"),
        tags=("memory", "apen", "residual"),
    )

    def __init__(self, config: Optional[M04Config] = None) -> None:
        self.config = config or M04Config()
        self._backbone: Union[Ridge, _TinyMLP, None] = None
        self._pending = PendingOutcomeQueue(
            self.config.horizon_steps, self.config.maturity_delay
        )
        self._memory = BoundedMemoryStore(self.config.memory_capacity)
        self._fitted = False
        self._n_features = 0
        self._context_dim = 0

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_quantiles=False,
            supports_ranking=False,
            supports_factors=False,
            supports_memory=True,
            requires_maturity=True,
        )

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def _context_key(self, X: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if extra is not None:
            e = np.asarray(extra, dtype=float).reshape(1, -1)
            key = np.concatenate([X.mean(axis=0), e.ravel()])
        else:
            key = X.mean(axis=0)
        return key / (np.linalg.norm(key) + 1e-8)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> M04FinancialAPEN:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._n_features = X.shape[1]
        rng = np.random.default_rng(self.config.random_state)
        if self.config.use_mlp_backbone:
            self._backbone = _TinyMLP(self._n_features, self.config.mlp_hidden, rng)
            for _ in range(80):
                pred = self._backbone.predict(X)
                grad = (pred - y) / len(y)
                h = np.tanh(X @ self._backbone.W1 + self._backbone.b1)
                self._backbone.w2 -= 0.05 * (h.T @ grad)
                dh = np.outer(grad, self._backbone.w2) * (1 - h**2)
                self._backbone.W1 -= 0.05 * (X.T @ dh)
                self._backbone.b1 -= 0.05 * dh.sum(axis=0)
        else:
            self._backbone = Ridge(alpha=self.config.ridge_alpha)
            self._backbone.fit(X, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def _backbone_predict(self, X: np.ndarray) -> np.ndarray:
        if self._backbone is None:
            raise RuntimeError("Not fitted")
        if isinstance(self._backbone, Ridge):
            return self._backbone.predict(X)
        return self._backbone.predict(X)

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        base = self._backbone_predict(X)
        as_of: datetime = kwargs.get("as_of", datetime(2020, 1, 1))
        context_extra = kwargs.get("context_extra")
        corrected = base.copy()
        for i in range(X.shape[0]):
            key = self._context_key(X[i : i + 1], context_extra)
            hits = self._memory.retrieve_similar(
                key, as_of, k=self.config.retrieval_k
            )
            if not hits:
                continue
            sims = [s for _, s in hits]
            max_sim = max(sims)
            if max_sim < self.config.similarity_threshold:
                continue
            if novelty_gate(max_sim, self.config.novelty_threshold):
                continue
            conf = confidence_from_similarities(sims, self.config.min_confidence)
            if conf <= 0:
                continue
            resid_corr = float(np.mean([ep.residual for ep, _ in hits]))
            corrected[i] = base[i] - conf * resid_corr
        return corrected

    def log_prediction(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray],
        decision_time: datetime,
        *,
        row_index: int = 0,
    ) -> tuple[float, int]:
        """Log residual at decision time; enqueue until matured."""
        pred = self._backbone_predict(X[row_index : row_index + 1])[0]
        residual = float(pred - (y_true[row_index] if y_true is not None else pred))
        if y_true is not None:
            residual = float(pred - y_true[row_index])
        key = self._context_key(X[row_index : row_index + 1])
        eid = self._pending.enqueue(decision_time, key, residual)
        return pred, eid

    def advance_time(self, as_of: datetime) -> int:
        """Move matured episodes into long-term memory."""
        matured = self._pending.pop_matured(as_of)
        for ep in matured:
            self._memory.insert(ep)
        return len(matured)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "backbone": self._backbone,
            "backbone_type": "ridge" if isinstance(self._backbone, Ridge) else "mlp",
            "memory_episodes": self._memory.episodes,
            "pending_heap": self._pending._heap,
            "pending_counter": self._pending._counter,
            "fitted": self._fitted,
            "n_features": self._n_features,
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M04Config.from_dict(state["config"])
        self._backbone = state["backbone"]
        self._memory = BoundedMemoryStore(self.config.memory_capacity)
        self._memory.episodes = list(state["memory_episodes"])
        self._pending = PendingOutcomeQueue(
            self.config.horizon_steps, self.config.maturity_delay
        )
        self._pending._heap = list(state["pending_heap"])
        self._pending._counter = state["pending_counter"]
        self._fitted = state["fitted"]
        self._n_features = state["n_features"]
