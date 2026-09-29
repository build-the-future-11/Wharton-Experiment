"""M09 — Regime-Robust Multi-Timescale Representation."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m01.regimes import causal_regime_ids, trailing_volatility
from wharton_lab.models.m09.config import M09Config
from wharton_lab.models.m09.fusion import FusionModule
from wharton_lab.models.m09 import groupdro as gd


class M09Model(BaseModel):
    MODEL_ID = "M09"
    metadata = ModelMetadata(
        model_id="M09",
        title="Regime-Robust Multi-Timescale Representation",
        version="0.1.0",
        description="Fast returns/vol + slow trailing market state with fusion and GroupDRO.",
        inputs=("returns_features", "market_slow_state"),
        outputs=("point_forecast",),
        training_constraints=("causal_regimes_only", "no_fabricated_macro"),
        metrics=("mse", "worst_group_mse"),
        tags=("multi_timescale", "groupdro", "regime_robust"),
    )

    def __init__(self, config: Optional[M09Config] = None):
        self.config = config or M09Config()
        self.fusion: Optional[FusionModule] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.group_weights_: Optional[np.ndarray] = None
        self._fitted = False

    def _split_streams(self, X: np.ndarray, returns_col: int = 0) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        fd = min(self.config.fast_dim, X.shape[1])
        fast = X[:, :fd]
        rets = X[:, returns_col] if X.shape[1] > returns_col else fast[:, 0]
        rets = np.asarray(rets, dtype=float).ravel()
        n = len(rets)
        slow_vol = trailing_volatility(rets, self.config.slow_window)
        slow_vol = np.nan_to_num(slow_vol, nan=0.0)
        slow_mean = np.zeros(n, dtype=float)
        w = self.config.slow_window
        for i in range(n):
            hist = rets[max(0, i - w) : i]
            slow_mean[i] = float(np.mean(hist)) if len(hist) else 0.0
        slow = np.column_stack([slow_vol, slow_mean])
        return fast, slow

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        returns_col: int = 0,
        **kwargs: Any,
    ) -> M09Model:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        fast, slow = self._split_streams(X, returns_col=returns_col)
        self.fusion = FusionModule(
            fast.shape[1], slow.shape[1], hidden=self.config.fusion_hidden, seed=self.config.random_state
        )
        Z = self.fusion.transform(fast, slow)
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        rets = X[:, returns_col]
        groups = causal_regime_ids(
            rets,
            vol_window=self.config.vol_window,
            vol_threshold=self.config.vol_threshold,
        )
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Finite, aligned X and y are required")
        if self.config.groupdro_steps < 1 or self.config.ridge_alpha < 0:
            raise ValueError("Positive iterations and nonnegative regularization required")
        n_groups = 4
        counts = np.bincount(groups, minlength=n_groups)
        present = counts > 0
        q = present.astype(float) / present.sum()
        base_weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if base_weight.shape != y.shape or np.any(base_weight < 0) or not np.isfinite(base_weight).all():
            raise ValueError("Invalid sample weights")
        masses = np.bincount(groups, weights=base_weight, minlength=n_groups)
        if np.any(masses[present] <= 0):
            raise ValueError("Every present group needs positive sample mass")
        penalty = self.config.ridge_alpha * np.eye(Z.shape[1])
        penalty[-1, -1] = 0.0  # unpenalized intercept
        for _ in range(self.config.groupdro_steps):
            used_q = q.copy()
            weights = base_weight * q[groups] / masses[groups]
            lhs = Z.T @ (weights[:, None] * Z) + penalty
            rhs = Z.T @ (weights * y)
            coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            residual_sq = (y - Z @ coef) ** 2
            losses = np.bincount(groups, weights=base_weight * residual_sq, minlength=n_groups)
            losses[present] /= masses[present]
            log_q = np.log(q[present]) + self.config.groupdro_eta * losses[present]
            log_q -= log_q.max()
            q[present] = np.exp(log_q) / np.exp(log_q).sum()
        # Store the weights used for the final regression, not an unused next update.
        q = used_q
        self._history_X = X[-max(self.config.slow_window, self.config.vol_window):].copy()
        self._returns_col = returns_col

        self.coef_ = coef
        self.intercept_ = float(coef[-1]) if len(coef) else 0.0
        self.group_weights_ = q
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._fitted or self.fusion is None or self.coef_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        history = getattr(self, "_history_X", np.empty((0, X.shape[1])))
        if history is None:
            history = np.empty((0, X.shape[1]))
        joined = np.concatenate([history, X], axis=0)
        fast, slow = self._split_streams(joined, returns_col=kwargs.get("returns_col", getattr(self, "_returns_col", 0)))
        fast, slow = fast[len(history):], slow[len(history):]
        Z = self.fusion.transform(fast, slow)
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        return Z @ self.coef_

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_ranking=True, supports_memory=False)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "coef": None if self.coef_ is None else self.coef_.tolist(),
            "fusion": None if self.fusion is None else self.fusion.state_dict(),
            "group_weights": None if self.group_weights_ is None else self.group_weights_.tolist(),
            "fitted": self._fitted,
            "history_X": getattr(self, "_history_X", None),
            "returns_col": getattr(self, "_returns_col", 0),
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M09Config(**state["config"])
        self.coef_ = None if state["coef"] is None else np.asarray(state["coef"])
        self.group_weights_ = None if state["group_weights"] is None else np.asarray(state["group_weights"])
        self._fitted = bool(state["fitted"])
        self._history_X = state.get("history_X", None)
        self._returns_col = state.get("returns_col", 0)
        if state["fusion"] is not None:
            self.fusion = FusionModule(1, 2, hidden=self.config.fusion_hidden)
            self.fusion.load_state_dict(state["fusion"])

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def smoke_forward(self, n_features: int = 4, n_samples: int = 64) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        X = rng.normal(scale=0.01, size=(n_samples, max(n_features, 2)))
        y = X[:, 0] * 0.5 + rng.normal(scale=0.01, size=n_samples)
        self.fit(X, y)
        pred = self.predict(X[:8])
        return {
            "pred_shape": np.asarray(pred).shape,
            "finite": bool(np.all(np.isfinite(pred))),
            "group_weights_sum": float(self.group_weights_.sum()) if self.group_weights_ is not None else None,
        }
