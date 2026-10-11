"""Abstract model interface."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Union

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata


class BaseModel(ABC):
    """All lab models implement fit, predict, serialization, and smoke checks."""

    MODEL_ID: ClassVar[str] = "base"
    metadata: ClassVar[ModelMetadata]

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> BaseModel:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        ...

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        ...

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_id": self.MODEL_ID, "state": self._serialize_state()}
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> BaseModel:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("model_id") != cls.MODEL_ID:
            raise ValueError(
                f"Expected model_id {cls.MODEL_ID}, got {payload.get('model_id')}"
            )
        inst = cls()
        inst._deserialize_state(payload["state"])
        return inst

    @abstractmethod
    def _serialize_state(self) -> Mapping[str, Any]:
        ...

    @abstractmethod
    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        ...

    def smoke_forward(self, n_features: int = 4, n_samples: int = 32) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_samples, n_features))
        y = rng.normal(size=n_samples)
        self.fit(X, y)
        pred = self.predict(X[:8])
        pred = np.asarray(pred)
        return {
            "pred_shape": pred.shape,
            "finite": bool(np.all(np.isfinite(pred))),
        }

    def config_dict(self) -> Mapping[str, Any]:
        return {}

    def to_json_metadata(self) -> str:
        cap = self.capabilities()
        return json.dumps(
            {
                "model_id": self.MODEL_ID,
                "capabilities": cap.__dict__,
                "config": dict(self.config_dict()),
            },
            indent=2,
        )
