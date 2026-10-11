"""M07 Eigen-JEPA — forward covariance / eigenspace prediction."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m07.config import M07Config
from wharton_lab.models.m07.metrics import (
    chordal_distance,
    flag_small_eigengaps,
    projector_from_eigenvectors,
)
from wharton_lab.models.m07.psd import PSDCovHead
from wharton_lab.models.torch_utils import pick_device, require_torch, set_deterministic_seed

torch = require_torch()
nn = torch.nn


class _ReturnEncoder(nn.Module):
    def __init__(self, window: int, n_assets: int, hidden: int, latent: int) -> None:
        super().__init__()
        flat = window * n_assets
        self.net = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class EigenJEPAModel(BaseModel):
    MODEL_ID: ClassVar[str] = "M07"
    metadata: ClassVar[ModelMetadata] = ModelMetadata(
        model_id="M07",
        title="Eigen-JEPA",
        version="0.1.0",
        description="Predict forward covariance and eigenspaces for windows 20/60 with PSD heads.",
        inputs=("return_windows",),
        outputs=("leading_eigenvalues", "projectors"),
        training_constraints=("psd_covariance",),
        metrics=("chordal", "projector_fro"),
        tags=("jepa", "covariance", "eigen"),
    )

    def __init__(self, config: M07Config | None = None) -> None:
        self.config = config or M07Config()
        set_deterministic_seed(self.config.random_state)
        self.device = pick_device(self.config.device)
        n = self.config.n_assets
        self.encoders = nn.ModuleDict(
            {
                str(w): _ReturnEncoder(w, n, self.config.hidden_dim, self.config.latent_dim)
                for w in self.config.windows
            }
        ).to(self.device)
        self.cov_heads = nn.ModuleDict(
            {str(w): PSDCovHead(n) for w in self.config.windows}
        ).to(self.device)
        self._opt = torch.optim.Adam(
            list(self.encoders.parameters()) + list(self.cov_heads.parameters()),
            lr=self.config.lr,
        )
        self._last_eigvals: dict[str, np.ndarray] = {}
        self._last_projectors: dict[str, np.ndarray] = {}
        self._last_eigengap_flags: dict[str, np.ndarray] = {}
        self._fitted = False

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_factors=True)

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def _panel_from_X(self, X: np.ndarray) -> dict[int, np.ndarray]:
        """X: (n_samples, window * n_assets) or (n_samples, window, n_assets)."""
        X = np.asarray(X, dtype=float)
        n_assets = self.config.n_assets
        out: dict[int, np.ndarray] = {}
        for w in self.config.windows:
            if X.ndim == 3 and X.shape[1] == w:
                out[w] = X
            else:
                need = w * n_assets
                flat = X if X.ndim == 2 else X.reshape(X.shape[0], -1)
                if flat.shape[1] < need:
                    pad = np.zeros((flat.shape[0], need - flat.shape[1]))
                    flat = np.concatenate([flat, pad], axis=1)
                out[w] = flat[:, :need].reshape(-1, w, n_assets)
        return out

    def _empirical_cov(self, window: np.ndarray) -> np.ndarray:
        # window (w, n_assets) -> sample cov across time
        x = window - window.mean(axis=0, keepdims=True)
        return (x.T @ x) / max(window.shape[0] - 1, 1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> EigenJEPAModel:
        panels = self._panel_from_X(X)
        for _ in range(self.config.epochs):
            self.encoders.train()
            total_loss = torch.zeros((), device=self.device)
            count = 0
            for w, arr in panels.items():
                t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
                z = self.encoders[str(w)](t)
                pred_cov = self.cov_heads[str(w)]()
                # target covs per sample averaged
                targets = []
                for i in range(t.shape[0]):
                    targets.append(self._empirical_cov(t[i].detach().cpu().numpy()))
                target = torch.as_tensor(
                    np.mean(targets, axis=0), dtype=torch.float32, device=self.device
                )
                total_loss = total_loss + nn.functional.mse_loss(pred_cov, target)
                count += 1
            self._opt.zero_grad()
            (total_loss / max(count, 1)).backward()
            self._opt.step()
        self._update_eigen_cache(panels)
        self._fitted = True
        return self

    def _update_eigen_cache(self, panels: dict[int, np.ndarray]) -> None:
        self.cov_heads.eval()
        with torch.no_grad():
            for w in self.config.windows:
                cov = self.cov_heads[str(w)]().cpu().numpy()
                evals, evecs = np.linalg.eigh(cov)
                self._last_eigvals[str(w)] = evals
                k = min(3, evecs.shape[1])
                P = projector_from_eigenvectors(evecs[:, -k:])
                self._last_projectors[str(w)] = P
                self._last_eigengap_flags[str(w)] = flag_small_eigengaps(
                    evals, self.config.eigengap_threshold
                )

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        panels = self._panel_from_X(X)
        self._update_eigen_cache(panels)
        # return leading eigenvalues concatenated across windows
        parts = []
        for w in self.config.windows:
            parts.append(self._last_eigvals[str(w)][-3:])
        return np.concatenate(parts)

    def predict_covariance(self, window: int) -> np.ndarray:
        with torch.no_grad():
            return self.cov_heads[str(window)]().cpu().numpy()

    def last_projector(self, window: int) -> np.ndarray:
        return self._last_projectors[str(window)]

    def chordal_to_empirical(self, X: np.ndarray, window: int) -> float:
        panels = self._panel_from_X(X)
        arr = panels[window]
        emp = np.mean([self._empirical_cov(a) for a in arr], axis=0)
        evals, evecs = np.linalg.eigh(emp)
        k = min(3, evecs.shape[1])
        P_emp = projector_from_eigenvectors(evecs[:, -k:])
        P_pred = self.last_projector(window)
        return chordal_distance(P_pred, P_emp)

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "encoders": self.encoders.state_dict(),
            "cov_heads": self.cov_heads.state_dict(),
            "fitted": self._fitted,
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M07Config.from_dict(state["config"])
        set_deterministic_seed(self.config.random_state)
        self.device = pick_device(self.config.device)
        n = self.config.n_assets
        self.encoders = nn.ModuleDict(
            {
                str(w): _ReturnEncoder(w, n, self.config.hidden_dim, self.config.latent_dim)
                for w in self.config.windows
            }
        ).to(self.device)
        self.cov_heads = nn.ModuleDict(
            {str(w): PSDCovHead(n) for w in self.config.windows}
        ).to(self.device)
        self.encoders.load_state_dict(state["encoders"])
        self.cov_heads.load_state_dict(state["cov_heads"])
        self._opt = torch.optim.Adam(
            list(self.encoders.parameters()) + list(self.cov_heads.parameters()),
            lr=self.config.lr,
        )
        self._fitted = bool(state.get("fitted", False))
