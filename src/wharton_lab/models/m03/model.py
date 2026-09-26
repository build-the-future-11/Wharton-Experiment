"""M03: Conditional nonlinear factor model."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m03.config import M03Config
from wharton_lab.models.m03.factors import fit_ar1, forecast_ar1, pca_factors
from wharton_lab.models.m03.mlp import LoadingMLP


class M03ConditionalNonlinearFactor(BaseModel):
    MODEL_ID = "M03"

    metadata = ModelMetadata(
        model_id="M03",
        title="Conditional Nonlinear Factor Model",
        version="0.1.0",
        description="MLP loadings beta(c); PCA factors on train; AR(1) factor forecasts at predict.",
        inputs=("characteristics", "returns_panel_train"),
        outputs=("expected_return",),
        training_constraints=("forecast_factors_only_at_predict",),
        metrics=("mse",),
        tags=("factor", "mlp", "nonlinear"),
    )

    def __init__(self, config: Optional[M03Config] = None) -> None:
        self.config = config or M03Config()
        self._mlp: Optional[LoadingMLP] = None
        self._factor_history: Optional[np.ndarray] = None
        self._ar_params: list[tuple[float, float]] = []
        self._pca = None
        self._fitted = False
        self._n_char = 0

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_quantiles=False,
            supports_ranking=False,
            supports_factors=True,
            supports_memory=False,
            requires_maturity=False,
        )

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        returns_panel: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> M03ConditionalNonlinearFactor:
        """
        X: (N, d) characteristics at train snapshot
        y: (N,) realized returns to explain (same period as factor train end)
        returns_panel: (T, N) for PCA factor learning
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._n_char = X.shape[1]
        self._ar_params = []
        if returns_panel is None:
            # synthetic single-period factors from cross-section
            returns_panel = y[None, :]

        factors, pca = pca_factors(returns_panel, self.config.n_factors)
        K = factors.shape[1]
        self._pca = pca
        self._factor_history = factors

        for k in range(K):
            if len(factors) >= self.config.ar1_fit_min:
                self._ar_params.append(fit_ar1(factors[:, k]))
            else:
                self._ar_params.append((0.0, 0.0))

        # Target loadings: approximate via cross-section regression targets
        # r_i ≈ beta_i · f_last
        f_last = factors[-1]
        denom = np.sum(f_last**2) + 1e-8
        target_loadings = np.outer(y, f_last) / denom  # (N, K) rank-1 proxy

        self._mlp = LoadingMLP(
            n_char=self._n_char,
            n_factors=int(K),
            hidden=self.config.mlp_hidden,
            lr=self.config.mlp_lr,
            random_state=self.config.random_state,
        )
        self._mlp.fit(X, target_loadings, epochs=self.config.mlp_epochs)
        self._fitted = True
        return self

    def forecast_factors(self, steps: int = 1) -> np.ndarray:
        if self._factor_history is None:
            raise RuntimeError("Not fitted")
        hist = self._factor_history
        forecasts = []
        for k, (c, phi) in enumerate(self._ar_params):
            forecasts.append(forecast_ar1(hist[:, k], c, phi, steps=steps))
        return np.stack(forecasts, axis=1)  # (steps, K)

    def predict(
        self,
        X: np.ndarray,
        *,
        forecast_steps: int = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        if not self._fitted or self._mlp is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        betas = self._mlp.forward(X)  # (N, K)
        f_fore = self.forecast_factors(steps=forecast_steps)[-1]
        return (betas * f_fore[None, :]).sum(axis=1)

    def smoke_forward(self, n_features: int = 4, n_samples: int = 32) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        n = max(n_samples, 16)
        X = rng.normal(size=(n, n_features))
        y = rng.normal(size=n)
        panel = rng.normal(size=(40, n))
        self.fit(X, y, returns_panel=panel)
        pred = self.predict(X[:8])
        pred = np.asarray(pred)
        return {
            "pred_shape": pred.shape,
            "finite": bool(np.all(np.isfinite(pred))),
        }

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "mlp": self._mlp.state_dict() if self._mlp else None,
            "factor_history": self._factor_history,
            "ar_params": self._ar_params,
            "pca": self._pca,
            "fitted": self._fitted,
            "n_char": self._n_char,
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M03Config.from_dict(state["config"])
        self._factor_history = state["factor_history"]
        self._ar_params = state["ar_params"]
        self._pca = state["pca"]
        self._fitted = state["fitted"]
        self._n_char = state["n_char"]
        if state["mlp"] is not None:
            self._mlp = LoadingMLP(self._n_char, self.config.n_factors)
            self._mlp.load_state_dict(state["mlp"])
