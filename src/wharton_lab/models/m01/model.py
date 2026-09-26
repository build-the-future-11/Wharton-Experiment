"""M01: Regime-conditioned distributional gradient boosting."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HAS_HIST = True
except ImportError:  # pragma: no cover
    _HAS_HIST = False

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m01.config import DEFAULT_QUANTILES, M01Config
from wharton_lab.models.m01.noncrossing import enforce_non_crossing
from wharton_lab.models.m01.regimes import causal_regime_ids, soft_regime_weights


def average_pinball_loss(
    y: np.ndarray,
    q_preds: np.ndarray,
    quantiles: Sequence[float],
) -> float:
    """Mean pinball loss averaged over quantiles."""
    y = np.asarray(y, dtype=float).ravel()
    q_preds = np.asarray(q_preds, dtype=float)
    taus = np.asarray(quantiles, dtype=float)
    losses = []
    for j, tau in enumerate(taus):
        e = y - q_preds[:, j]
        losses.append(np.mean(np.maximum(tau * e, (tau - 1) * e)))
    return float(np.mean(losses))


def _make_regressor(config: M01Config, alpha: float):
    if config.use_hist_gb and _HAS_HIST:
        # HistGradientBoosting uses quantile via loss='quantile' and quantile=alpha
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=alpha,
            max_iter=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
    )


class M01RegimeDistributionalGBM(BaseModel):
    MODEL_ID = "M01"

    metadata = ModelMetadata(
        model_id="M01",
        title="Regime-Conditioned Distributional Gradient Boosting",
        version="0.1.0",
        description="Quantile GBM with causal volatility/sign regimes and non-crossing repair.",
        inputs=("features", "past_returns_for_regime"),
        outputs=("quantiles", "point"),
        training_constraints=("causal_regimes", "no_future_smoothing"),
        metrics=("pinball_loss",),
        tags=("quantile", "regime", "gbm"),
    )

    def __init__(self, config: Optional[M01Config] = None) -> None:
        self.config = config or M01Config()
        self.quantiles_ = tuple(float(q) for q in self.config.quantiles)
        self._regressors: dict[tuple[int, float], Any] = {}
        self._global_regressors: dict[float, Any] = {}
        self._calibration: dict[float, float] = {}
        self._fitted = False
        self._n_features: int = 0
        self._regime_as_feature: bool = False

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_quantiles=True,
            supports_ranking=False,
            supports_factors=False,
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
        past_returns: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> M01RegimeDistributionalGBM:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._n_features = X.shape[1]
        if past_returns is None:
            if X.shape[1] > self.config.return_sign_feature_col:
                past_returns = X[:, self.config.return_sign_feature_col]
            else:
                past_returns = y
        past_returns = np.asarray(past_returns, dtype=float).ravel()

        self._regressors.clear()
        self._global_regressors.clear()

        X_fit = X
        if self.config.use_regimes and self.config.regime_as_feature:
            # Single conditioned booster: regime id is a causal feature.
            regimes = causal_regime_ids(
                past_returns,
                vol_window=self.config.vol_window,
                vol_threshold=self.config.vol_threshold,
            ).astype(float)
            X_fit = np.column_stack([X, regimes])
            for alpha in self.quantiles_:
                if not self.config.distributional_vs_point and abs(alpha - 0.5) > 1e-9:
                    continue
                reg = _make_regressor(self.config, alpha)
                reg.fit(X_fit, y, sample_weight=sample_weight)
                self._global_regressors[alpha] = reg
            # Point-only: mirror median into other slots at predict time if needed
            if not self.config.distributional_vs_point:
                med = self._global_regressors.get(0.5)
                for alpha in self.quantiles_:
                    self._global_regressors.setdefault(alpha, med)
            self._fitted = True
            self._n_features = X_fit.shape[1]
            self._regime_as_feature = True
            return self

        self._regime_as_feature = False
        for alpha in self.quantiles_:
            reg = _make_regressor(self.config, alpha)
            reg.fit(X, y, sample_weight=sample_weight)
            self._global_regressors[alpha] = reg

        if self.config.use_regimes:
            if self.config.soft_vs_hard:
                # Soft: one weighted fit per quantile (mixture weights as sample_weight)
                weights = soft_regime_weights(
                    past_returns,
                    vol_window=self.config.vol_window,
                    vol_threshold=self.config.vol_threshold,
                )
                # Use max soft weight as sample weight proxy (fast, keeps soft conditioning)
                sw = weights.max(axis=1)
                if sample_weight is not None:
                    sw = sw * sample_weight
                for alpha in self.quantiles_:
                    reg = _make_regressor(self.config, alpha)
                    reg.fit(X, y, sample_weight=sw)
                    self._global_regressors[alpha] = reg
            else:
                regimes = causal_regime_ids(
                    past_returns,
                    vol_window=self.config.vol_window,
                    vol_threshold=self.config.vol_threshold,
                )
                for alpha in self.quantiles_:
                    for rid in range(4):
                        mask = regimes == rid
                        if mask.sum() < self.config.min_samples_leaf:
                            continue
                        sw = sample_weight[mask] if sample_weight is not None else None
                        reg = _make_regressor(self.config, alpha)
                        reg.fit(X[mask], y[mask], sample_weight=sw)
                        self._regressors[(rid, alpha)] = reg

        if self.config.calibrate:
            q_hat = self._predict_quantiles_raw(X, past_returns)
            for j, alpha in enumerate(self.quantiles_):
                resid = y - q_hat[:, j]
                self._calibration[alpha] = float(np.median(resid))

        self._fitted = True
        return self

    def _predict_quantiles_raw(
        self, X: np.ndarray, past_returns: Optional[np.ndarray]
    ) -> np.ndarray:
        n = X.shape[0]
        k = len(self.quantiles_)
        out = np.zeros((n, k), dtype=float)
        if past_returns is None:
            past_returns = X[:, self.config.return_sign_feature_col]
        past_returns = np.asarray(past_returns, dtype=float).ravel()

        X_pred = X
        if getattr(self, "_regime_as_feature", False) and self.config.use_regimes:
            regimes = causal_regime_ids(
                past_returns,
                vol_window=self.config.vol_window,
                vol_threshold=self.config.vol_threshold,
            ).astype(float)
            X_pred = np.column_stack([X, regimes])

        for j, alpha in enumerate(self.quantiles_):
            base = self._global_regressors[alpha].predict(X_pred)
            if (
                self.config.use_regimes
                and self._regressors
                and not getattr(self, "_regime_as_feature", False)
            ):
                if self.config.soft_vs_hard:
                    w = soft_regime_weights(
                        past_returns,
                        vol_window=self.config.vol_window,
                        vol_threshold=self.config.vol_threshold,
                    )
                    blend = np.zeros(n, dtype=float)
                    for rid in range(4):
                        key = (rid, alpha)
                        if key not in self._regressors:
                            continue
                        blend += w[:, rid] * self._regressors[key].predict(X)
                    has_any = (w.sum(axis=1) > 0).astype(float)
                    out[:, j] = has_any * blend + (1 - has_any) * base
                else:
                    regimes = causal_regime_ids(
                        past_returns,
                        vol_window=self.config.vol_window,
                        vol_threshold=self.config.vol_threshold,
                    )
                    pred = base.copy()
                    for rid in range(4):
                        key = (rid, alpha)
                        if key not in self._regressors:
                            continue
                        mask = regimes == rid
                        if mask.any():
                            pred[mask] = self._regressors[key].predict(X[mask])
                    out[:, j] = pred
            else:
                out[:, j] = base

        if self.config.calibrate:
            for j, alpha in enumerate(self.quantiles_):
                out[:, j] += self._calibration.get(alpha, 0.0)
        return out

    def predict_quantiles(
        self,
        X: np.ndarray,
        *,
        past_returns: Optional[np.ndarray] = None,
        non_crossing_method: str = "sort",
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        q = self._predict_quantiles_raw(X, past_returns)
        if self.config.distributional_vs_point:
            q = enforce_non_crossing(q, method=non_crossing_method)
        return q

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        q = self.predict_quantiles(
            X,
            past_returns=kwargs.get("past_returns"),
            non_crossing_method=kwargs.get("non_crossing_method", "sort"),
        )
        med_idx = int(np.argmin(np.abs(np.array(self.quantiles_) - 0.5)))
        return q[:, med_idx]

    def pinball_loss(
        self,
        y: np.ndarray,
        X: np.ndarray,
        past_returns: Optional[np.ndarray] = None,
    ) -> float:
        q = self.predict_quantiles(X, past_returns=past_returns)
        return average_pinball_loss(y, q, self.quantiles_)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "quantiles": self.quantiles_,
            "regressors": self._regressors,
            "global_regressors": self._global_regressors,
            "calibration": self._calibration,
            "fitted": self._fitted,
            "n_features": self._n_features,
            "regime_as_feature": getattr(self, "_regime_as_feature", False),
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M01Config.from_dict(state["config"])
        self.quantiles_ = tuple(state["quantiles"])
        self._regressors = state["regressors"]
        self._global_regressors = state["global_regressors"]
        self._calibration = state["calibration"]
        self._fitted = state["fitted"]
        self._n_features = state["n_features"]
        self._regime_as_feature = bool(state.get("regime_as_feature", self.config.regime_as_feature))
