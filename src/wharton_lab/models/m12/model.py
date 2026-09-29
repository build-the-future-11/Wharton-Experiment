"""M12 — LAF-GMJEPA composed stack."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.exceptions import BlockedClientError
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m10.model import M10Model
from wharton_lab.models.m10.config import M10Config
from wharton_lab.models.m11.model import M11Model
from wharton_lab.models.m11.config import M11Config
from wharton_lab.models.m12.components import DynamicGraphState, EpisodicMemory, LatentWorldModel
from wharton_lab.models.m12.config import M12Config


class M12Model(BaseModel):
    MODEL_ID = "M12"
    metadata = ModelMetadata(
        model_id="M12",
        title="LAF-GMJEPA",
        version="0.1.0",
        description="Graph + episodic memory + latent world model + scenarios + liability planner.",
        inputs=("market_features", "optional_liability_schedule"),
        outputs=("market_forecast", "allocation"),
        training_constraints=("composed_or_joint", "market_invariant_to_liability"),
        tags=("composed", "ablations"),
    )

    def __init__(self, config: Optional[M12Config] = None):
        self.config = config or M12Config()
        self.graph = DynamicGraphState(k=self.config.graph_k)
        self.memory = EpisodicMemory(self.config.memory_slots, self.config.latent_dim, self.config.random_state)
        self.latent = LatentWorldModel(self.config.latent_dim, self.config.random_state)
        self.scenario_head = M10Model(
            M10Config(n_assets=self.config.n_assets, random_state=self.config.random_state)
        )
        self.planner = M11Model(M11Config(horizon=5))
        self._market_coef: Optional[np.ndarray] = None
        self._fitted = False

    def _encode_market(self, X: np.ndarray, *, update_memory: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape
        rets = X.reshape(n, -1, max(1, d // max(1, self.config.n_assets)))[:, :, 0]
        if self.config.no_graph:
            graph_feat = rets
        else:
            graph_feat = self.graph.encode(rets)
        z = np.zeros((n, self.config.latent_dim))
        for i in range(n):
            obs = graph_feat[i]
            if not self.config.no_memory:
                mem = self.memory.read(obs)
                mem_slice = mem[: obs.size] if mem.size >= obs.size else np.pad(mem, (0, obs.size - mem.size))
                obs = obs + 0.1 * mem_slice
            if not self.config.no_latent:
                z[i] = self.latent.step(z[i - 1] if i else np.zeros(self.config.latent_dim), obs)
            else:
                z[i] = obs[: self.config.latent_dim] if obs.size >= self.config.latent_dim else np.pad(
                    obs, (0, self.config.latent_dim - obs.size)
                )
            if not self.config.no_memory and update_memory:
                self.memory.write(obs)
        return np.hstack([graph_feat, z])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> M12Model:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        enc = self._encode_market(X, update_memory=True)
        enc = np.hstack([enc, np.ones((len(enc), 1))])
        self._market_coef = np.linalg.lstsq(enc, y, rcond=None)[0]

        scenario_targets = kwargs.get("scenario_targets")
        if self.config.joint_training:
            raise NotImplementedError("Joint training is not implemented; use the explicitly composed model")
        scenario_options = {"n_assets": self.config.n_assets, "random_state": self.config.random_state}
        scenario_options.update(kwargs.get("scenario_config", {}))
        if scenario_options["n_assets"] != self.config.n_assets:
            raise ValueError("Scenario and market asset counts must match")
        self.scenario_head = M10Model(M10Config(**scenario_options))
        if scenario_targets is not None:
            targets = np.asarray(scenario_targets, dtype=float)
            expected = self.config.n_assets * self.scenario_head.config.path_steps
            if targets.shape != (len(X), expected):
                raise ValueError("scenario_targets must contain genuine full multi-asset future paths")
            self.scenario_head.fit(enc[:, :-1], targets)

        if not self.config.no_liability and kwargs.get("liability_schedule") is not None:
            scen = kwargs.get("scenario_returns")
            if scen is not None:
                self.planner.fit(
                    scen,
                    y,
                    liability_schedule=kwargs["liability_schedule"],
                    cash=float(kwargs.get("cash", 1.0)),
                )
        self._fitted = True
        return self

    def predict_market(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self._market_coef is None:
            raise RuntimeError("Model not fitted")
        enc = self._encode_market(X)
        enc = np.hstack([enc, np.ones((len(enc), 1))])
        return enc @ self._market_coef

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.predict_market(X, **kwargs)

    def plan(
        self,
        scenario_returns: np.ndarray,
        liability_schedule: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.config.no_liability:
            k = scenario_returns.shape[-1]
            return np.full(k, 1.0 / k)
        k = scenario_returns.shape[-1]
        self.planner.fit(
            scenario_returns,
            np.zeros(scenario_returns.shape[0]),
            liability_schedule=liability_schedule,
            cash=float(kwargs.get("cash", 1.0)),
            holdings=kwargs.get("holdings", np.zeros(k)),
        )
        return self.planner.predict(scenario_returns)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_memory=not self.config.no_memory, supports_factors=True)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "market_coef": None if self._market_coef is None else self._market_coef.tolist(),
            "memory": self.memory.memory.tolist(),
            "fitted": self._fitted,
            "latent_A": self.latent.A,
            "scenario_head": self.scenario_head._serialize_state(),
            "planner": self.planner._serialize_state(),
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.__init__(M12Config(**state["config"]))
        self._market_coef = None if state["market_coef"] is None else np.asarray(state["market_coef"])
        self.memory.memory = np.asarray(state["memory"])
        self._fitted = bool(state["fitted"])
        if "latent_A" in state:
            self.latent.A = np.asarray(state["latent_A"])
        if "scenario_head" in state:
            self.scenario_head._deserialize_state(state["scenario_head"])
        if "planner" in state:
            self.planner._deserialize_state(state["planner"])

    def config_dict(self) -> Mapping[str, Any]:
        return {**self.config.to_dict(), "scenario_model": self.scenario_head.config_dict(), "planner_model": self.planner.config_dict()}

    def smoke_forward(self, n_features: int = 10, n_samples: int = 32) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        X = rng.normal(scale=0.01, size=(n_samples, n_features))
        y = rng.normal(scale=0.01, size=n_samples)
        self.fit(X, y)
        pred = self.predict(X[:4])
        return {"pred_shape": pred.shape, "finite": bool(np.all(np.isfinite(pred)))}
