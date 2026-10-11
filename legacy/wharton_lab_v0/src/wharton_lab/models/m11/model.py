"""M11 — Liability-Conditioned Distributional MPC."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.exceptions import BlockedClientError
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m11.config import M11Config
from wharton_lab.models.m11.mpc import MPCResult, PortfolioState, solve_mpc


class M11Model(BaseModel):
    MODEL_ID = "M11"
    metadata = ModelMetadata(
        model_id="M11",
        title="Liability-Conditioned Distributional MPC",
        version="0.1.0",
        description="CVaR funding shortfall MPC with non-anticipative first action.",
        inputs=("holdings", "cash", "scenario_returns", "liability_schedule"),
        outputs=("target_weights",),
        training_constraints=("requires_liability_schedule",),
        metrics=("funding_shortfall",),
        tags=("mpc", "liabilities"),
    )

    def __init__(self, config: Optional[M11Config] = None):
        self.config = config or M11Config()
        self._last_result: Optional[MPCResult] = None
        self._scenario_returns: Optional[np.ndarray] = None
        self._obligations: Optional[np.ndarray] = None
        self._state: Optional[PortfolioState] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        liability_schedule: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> M11Model:
        """X: (n_scenarios, horizon, n_assets) scenario returns; y unused."""
        if liability_schedule is None:
            liability_schedule = kwargs.get("obligations")
        if liability_schedule is None:
            raise BlockedClientError(
                "M11 requires an explicit client liability_schedule; will not invent client wealth."
            )
        self._scenario_returns = np.asarray(X, dtype=float)
        self._obligations = np.asarray(liability_schedule, dtype=float).ravel()
        if self._scenario_returns.ndim != 3:
            raise ValueError("X must be (n_scenarios, horizon, n_assets)")
        H = self._scenario_returns.shape[1]
        if self._obligations.shape[0] != H:
            raise ValueError("liability_schedule length must match scenario horizon")
        self.config.horizon = H
        cash = float(kwargs.get("cash", 1.0))
        k = self._scenario_returns.shape[-1]
        raw_holdings = kwargs.get("holdings")
        if raw_holdings is None:
            holdings = np.zeros(k, dtype=float)
        else:
            holdings = np.asarray(raw_holdings, dtype=float).ravel()
        self._state = PortfolioState(holdings=holdings, cash=cash)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._fitted or self._scenario_returns is None or self._obligations is None:
            raise RuntimeError("Model not fitted")
        if self._state is None:
            raise RuntimeError("Portfolio state missing")
        scenarios = np.asarray(X, dtype=float) if X is not None and X.size else self._scenario_returns
        try:
            result = solve_mpc(
                state=self._state,
                scenario_returns=scenarios,
                obligations=self._obligations,
                horizon=self.config.horizon,
                cvar_alpha=self.config.cvar_alpha,
                max_weight=self.config.max_weight,
                tcost_bps=self.config.transaction_cost_bps,
            )
        except Exception as exc:
            return self._handle_infeasible(exc)

        if not result.success:
            return self._handle_infeasible(RuntimeError(result.message))
        self._last_result = result
        return result.first_action_weights

    def _handle_infeasible(self, exc: Exception) -> np.ndarray:
        k = self._scenario_returns.shape[-1] if self._scenario_returns is not None else 1
        if self._state is not None:
            h = self._state.holdings
            h_sum = 0.0 if h is None else float(np.asarray(h).sum())
            port = self._state.cash + h_sum
            if port <= 0:
                raise BlockedClientError(f"Insolvent client state: {exc}") from exc
        w = np.zeros(k)
        w[0] = 1.0
        return w

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(requires_maturity=False)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "scenario_returns": None
            if self._scenario_returns is None
            else self._scenario_returns.tolist(),
            "obligations": None if self._obligations is None else self._obligations.tolist(),
            "state": None
            if self._state is None
            else {"holdings": self._state.holdings.tolist(), "cash": self._state.cash},
            "fitted": self._fitted,
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M11Config(**state["config"])
        self._scenario_returns = None if state["scenario_returns"] is None else np.asarray(state["scenario_returns"])
        self._obligations = None if state["obligations"] is None else np.asarray(state["obligations"])
        if state["state"] is not None:
            self._state = PortfolioState(
                holdings=np.asarray(state["state"]["holdings"]),
                cash=float(state["state"]["cash"]),
            )
        self._fitted = bool(state["fitted"])

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def smoke_forward(self, n_features: int = 3, n_samples: int = 8) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        H, k = self.config.horizon, 3
        X = rng.normal(scale=0.01, size=(n_samples, H, k))
        liabilities = np.full(H, 0.05)
        self.fit(X, np.zeros(n_samples), liability_schedule=liabilities, cash=1.0)
        w = self.predict(X)
        return {"weight_shape": w.shape, "finite": bool(np.all(np.isfinite(w))), "sum_weights": float(w.sum())}
