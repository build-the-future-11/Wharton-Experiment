"""M02: Factor-residualized cross-sectional learning-to-rank."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m02.config import M02Config
from wharton_lab.models.m02.factors import residualize_labels, rolling_ols_exposure
from wharton_lab.models.m02.metrics import date_level_spearman_ic
from wharton_lab.models.m02.ranker import PairwiseLogisticRanker


class M02FactorResidualRanker(BaseModel):
    MODEL_ID = "M02"

    metadata = ModelMetadata(
        model_id="M02",
        title="Factor-Residualized Cross-Sectional Learning-to-Rank",
        version="0.1.0",
        description="Rolling OLS exposures; pairwise ranker on residualized forward labels.",
        inputs=("features", "returns", "factor_returns", "date_ids"),
        outputs=("rank_scores",),
        training_constraints=("historical_factors_only", "label_at_maturity"),
        metrics=("spearman_ic",),
        tags=("ranking", "factor", "cross_section"),
    )

    def __init__(self, config: Optional[M02Config] = None) -> None:
        self.config = config or M02Config()
        self._ranker: Optional[PairwiseLogisticRanker] = None
        self._fitted = False
        self._n_features = 0

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_quantiles=False,
            supports_ranking=True,
            supports_factors=True,
            supports_memory=False,
            requires_maturity=True,
        )

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        asset_returns: Optional[np.ndarray] = None,
        factor_returns: Optional[np.ndarray] = None,
        date_ids: Optional[np.ndarray] = None,
        realized_future_factors: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> M02FactorResidualRanker:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._n_features = X.shape[1]
        if date_ids is None:
            date_ids = np.zeros(len(y), dtype=int)
        date_ids = np.asarray(date_ids)

        labels = y
        betas = kwargs.get("betas")
        if betas is not None and realized_future_factors is not None:
            betas = np.asarray(betas, dtype=float)
            rff = np.asarray(realized_future_factors, dtype=float)
            if betas.shape == rff.shape and len(betas) == len(y):
                labels = residualize_labels(y, betas, rff)

        self._ranker = PairwiseLogisticRanker(
            n_features=self._n_features,
            hidden=self.config.rank_hidden,
            lr=self.config.rank_lr,
            l2=self.config.l2,
            random_state=self.config.random_state,
        )
        rng = np.random.default_rng(self.config.random_state)
        self._ranker.fit_pairs(
            X,
            labels,
            date_ids,
            epochs=self.config.rank_epochs,
            max_pairs=self.config.max_pairs_per_date,
            rng=rng,
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._fitted or self._ranker is None:
            raise RuntimeError("Model not fitted")
        return self._ranker.predict(X)

    def spearman_ic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        date_ids: np.ndarray,
    ) -> float:
        scores = self.predict(X)
        return date_level_spearman_ic(scores, y, date_ids)

    @staticmethod
    def compute_betas_from_panel(
        asset_returns: np.ndarray,
        factor_returns: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Returns (T, N, F) betas for panel residualization pipeline."""
        return rolling_ols_exposure(asset_returns, factor_returns, window)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "ranker": self._ranker.state_dict() if self._ranker else None,
            "fitted": self._fitted,
            "n_features": self._n_features,
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M02Config.from_dict(state["config"])
        self._fitted = state["fitted"]
        self._n_features = state["n_features"]
        if state["ranker"] is not None:
            self._ranker = PairwiseLogisticRanker(self._n_features)
            self._ranker.load_state_dict(state["ranker"])
