"""M10 — Conditional Flow-Matching Market Scenario Model."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m10.config import M10Config
from wharton_lab.models.m10 import flow


class M10Model(BaseModel):
    MODEL_ID = "M10"
    metadata = ModelMetadata(
        model_id="M10",
        title="Conditional Flow-Matching Market Scenario Model",
        version="0.1.0",
        description="Conditional vector field with Euler ODE; multi-asset paths.",
        inputs=("market_context",),
        outputs=("scenario_paths",),
        metrics=("energy_score",),
        tags=("flow_matching", "scenarios"),
    )

    def __init__(self, config: Optional[M10Config] = None):
        self.config = config or M10Config()
        self._W: Optional[np.ndarray] = None
        self._context_mean: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._fitted = False
        self._rng = np.random.default_rng(self.config.random_state)

    def _context(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        d = min(self.config.context_dim, X.shape[1])
        return X[:, :d]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> M10Model:
        """y: (n, n_assets * path_steps) flattened paths or (n, n_assets)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Finite aligned training arrays are required")
        if self.config.n_ode_steps < 1 or self.config.training_tau_samples < 1 or self.config.ridge_alpha < 0:
            raise ValueError("Invalid flow solver or training configuration")
        ctx = self._context(X)
        self._context_mean = ctx.mean(axis=0)
        self._y_mean = y.mean(axis=0)
        ctx_c = ctx - self._context_mean
        n, out_dim = y.shape
        d = ctx_c.shape[1]
        in_dim = out_dim + d + 2
        self._feature_version = 2
        F_list = []
        V_list = []
        for i in range(n):
            y_i = y[i]
            for _ in range(self.config.training_tau_samples):
                z = flow.sample_base_noise(
                    y_i.shape,
                    self._rng,
                    student_t_df=self.config.student_t_df,
                    scale=self.config.noise_scale,
                )
                tau = float(self._rng.uniform(0, 1))
                x_tau = flow.bridge_sample(z, y_i, tau)
                v_tgt = flow.bridge_velocity(z, y_i)
                F_list.append(np.concatenate([x_tau, ctx_c[i], [tau, 1.0]]))
                V_list.append(v_tgt)
        F = np.vstack(F_list)
        V = np.vstack(V_list)
        penalty = self.config.ridge_alpha * np.eye(in_dim)
        penalty[-1, -1] = 0
        self._W = np.linalg.lstsq(
            F.T @ F + penalty,
            F.T @ V, rcond=None,
        )[0]
        self._fitted = True
        return self

    def _velocity_fn(self, context_row: np.ndarray) -> Callable[[np.ndarray, float], np.ndarray]:
        W = self._W
        d = len(context_row)
        out_dim = W.shape[1]

        def v_fn(z: np.ndarray, tau: float) -> np.ndarray:
            z = np.asarray(z, dtype=float).ravel()
            if z.size != out_dim:
                raise ValueError("Velocity state dimension mismatch")
            feat = np.concatenate([z, context_row, [tau, 1.0]]) if getattr(self, "_feature_version", 1) == 2 else np.concatenate([z, context_row])
            return feat @ W

        return v_fn

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._fitted or self._W is None:
            raise RuntimeError("Model not fitted")
        n_paths = int(kwargs.get("n_paths", 16))
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        ctx = self._context(X)
        if ctx.shape[1] != len(self._context_mean) or not np.isfinite(ctx).all():
            raise ValueError("Context dimension mismatch or nonfinite input")
        if ctx.shape[0] == 1 and n_paths > 1:
            ctx = np.repeat(ctx, n_paths, axis=0)
        out_dim = self._W.shape[1]
        paths = []
        for i in range(len(ctx)):
            c = ctx[i] - self._context_mean
            v_fn = self._velocity_fn(c)
            z0 = flow.sample_base_noise(
                (out_dim,),
                self._rng,
                student_t_df=self.config.student_t_df,
                scale=self.config.noise_scale,
            )
            y_hat = flow.euler_integrate(v_fn, z0, n_steps=self.config.n_ode_steps)
            paths.append(y_hat)
        return np.vstack(paths)

    def sample_paths(self, X: np.ndarray, n_paths: int = 32) -> np.ndarray:
        """Joint multi-step paths reshaped (n_paths, path_steps, n_assets)."""
        flat = self.predict(X[:1], n_paths=n_paths)
        k = self.config.n_assets
        h = self.config.path_steps
        if flat.shape[1] != k * h:
            raise ValueError("Trained target dimension does not equal path_steps * n_assets; no future padding is allowed")
        return flat.reshape(n_paths, h, k)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_quantiles=False)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "W": None if self._W is None else self._W.tolist(),
            "context_mean": None if self._context_mean is None else self._context_mean.tolist(),
            "y_mean": None if self._y_mean is None else self._y_mean.tolist(),
            "fitted": self._fitted,
            "rng_state": self._rng.bit_generator.state,
            "feature_version": getattr(self, "_feature_version", 1),
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M10Config(**state["config"])
        self._W = None if state["W"] is None else np.asarray(state["W"])
        self._context_mean = None if state["context_mean"] is None else np.asarray(state["context_mean"])
        self._y_mean = None if state["y_mean"] is None else np.asarray(state["y_mean"])
        self._fitted = bool(state["fitted"])
        self._rng = np.random.default_rng(self.config.random_state)
        if state.get("rng_state") is not None:
            self._rng.bit_generator.state = state["rng_state"]
        self._feature_version = state.get("feature_version", 1)

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def smoke_forward(self, n_features: int = 8, n_samples: int = 24) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        k, h = self.config.n_assets, self.config.path_steps
        X = rng.normal(size=(n_samples, n_features))
        y = rng.normal(size=(n_samples, k * h))
        self.fit(X, y)
        paths = self.sample_paths(X, n_paths=4)
        return {
            "path_shape": paths.shape,
            "finite": bool(np.all(np.isfinite(paths))),
        }
