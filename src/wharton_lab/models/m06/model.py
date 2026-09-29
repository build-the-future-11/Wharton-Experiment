"""M06 FI-JEPA model."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m06.config import M06Config
from wharton_lab.models.m06.modules import (
    AuxHead,
    ContextEncoder,
    HorizonPredictor,
    TargetEncoder,
    ema_update,
    vicreg_loss,
)
from wharton_lab.models.torch_utils import pick_device, require_torch, set_deterministic_seed

torch = require_torch()
nn = torch.nn


class FIJEPAModel(BaseModel):
    MODEL_ID: ClassVar[str] = "M06"
    metadata: ClassVar[ModelMetadata] = ModelMetadata(
        model_id="M06",
        title="FI-JEPA — Multi-Horizon Financial-Informed JEPA",
        version="0.1.0",
        description="Context/target JEPA with horizons 1/5/20, financial aux heads, VICReg anti-collapse.",
        inputs=("sequence_features",),
        outputs=("point_forecast",),
        training_constraints=("no_ema_on_held_out_future",),
        metrics=("latent_mse", "vicreg", "aux_return_vol"),
        tags=("jepa", "self-supervised", "multi-horizon"),
    )

    def __init__(self, config: M06Config | None = None) -> None:
        self.config = config or M06Config()
        set_deterministic_seed(self.config.random_state)
        self.device = pick_device(self.config.device)
        d_in = self.config.input_dim
        h, z = self.config.hidden_dim, self.config.latent_dim
        self.context = ContextEncoder(d_in, h, z).to(self.device)
        self.target = TargetEncoder(d_in, h, z).to(self.device)
        self.target.load_state_dict(self.context.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictors = nn.ModuleDict(
            {str(ho): HorizonPredictor(z, ho) for ho in self.config.horizons}
        ).to(self.device)
        self.aux = AuxHead(z).to(self.device)
        self._opt = torch.optim.Adam(
            list(self.context.parameters())
            + list(self.predictors.parameters())
            + list(self.aux.parameters()),
            lr=self.config.lr,
        )
        self._fitted = False

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_factors=True)

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def _to_sequences(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            return X
        n, f = X.shape
        sl = self.config.seq_len
        d = self.config.input_dim
        if f != sl * d:
            # pad or trim features into seq_len chunks
            need = sl * d
            if f < need:
                pad = np.zeros((n, need - f))
                X = np.concatenate([X, pad], axis=1)
            else:
                X = X[:, :need]
        return X.reshape(n, sl, d)

    def _future_slices(self, seq: torch.Tensor) -> dict[int, torch.Tensor]:
        """Build target windows for each horizon from trailing timesteps."""
        out: dict[int, torch.Tensor] = {}
        t_len = seq.shape[1]
        for ho in self.config.horizons:
            end = min(t_len, ho + 1)
            out[ho] = seq[:, -end:, :]
        return out

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> FIJEPAModel:
        seq_np = self._to_sequences(X)
        y = np.asarray(y, dtype=float).ravel()
        n = seq_np.shape[0]
        split = max(1, int(n * (1.0 - self.config.holdout_future_frac)))
        # Explicit forward targets overlap adjacent rows. Purge training rows
        # whose target horizon would reach the internal validation origin.
        train_end = split
        if kwargs.get("future_windows") is not None:
            train_end = split - max(self.config.horizons) + 1
        if train_end < 2:
            raise ValueError("Not enough matured training rows after horizon purge")
        train_idx = np.arange(train_end)
        hold_idx = np.arange(split, n)

        seq_t = torch.as_tensor(seq_np, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        self.predictive_coeff_ = float(kwargs.get("predictive_coeff", 1.0))
        if self.predictive_coeff_ < 0: raise ValueError("predictive_coeff must be nonnegative")
        future_windows = kwargs.get("future_windows")
        self.target_mode_ = "explicit_future_windows" if future_windows is not None else "in_context_self_distillation_not_future"
        future_t = {}
        if future_windows is not None:
            for ho in self.config.horizons:
                arr = np.asarray(future_windows[ho], dtype=float)
                if arr.shape != (n, ho, self.config.input_dim) or not np.isfinite(arr).all():
                    raise ValueError("future_windows must have shape (n, horizon, input_dim)")
                future_t[ho] = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        ret_aux = y_t
        vol_aux = torch.sqrt(torch.relu(y_t ** 2) + 1e-6)

        for _ in range(self.config.epochs):
            self.context.train()
            for start in range(0, len(train_idx), self.config.batch_size):
                idx = train_idx[start : start + self.config.batch_size]
                batch = seq_t[idx]
                z_c = self.context(batch)
                futures = {ho: arr[idx] for ho, arr in future_t.items()} if future_t else self._future_slices(batch)
                loss = torch.zeros((), device=self.device)
                for ho, fut in futures.items():
                    with torch.no_grad():
                        z_t = self.target(fut)
                    z_p = self.predictors[str(ho)](z_c)
                    loss = loss + self.predictive_coeff_ * nn.functional.mse_loss(z_p, z_t)
                pred_ret, pred_vol = self.aux(z_c)
                loss = loss + self.config.aux_coeff * (
                    nn.functional.mse_loss(pred_ret, ret_aux[idx])
                    + nn.functional.mse_loss(pred_vol, vol_aux[idx])
                )
                loss = loss + vicreg_loss(z_c, self.config.vicreg_coeff)
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
                ema_update(self.target, self.context, self.config.ema_momentum)

            # Diagnostics only: neither gradients nor optimizer/EMA updates use holdout.
            self.validation_loss_ = None
            if len(hold_idx) > 0:
                self.context.eval()
                with torch.no_grad():
                    z_c = self.context(seq_t[hold_idx])
                    predicted, _ = self.aux(z_c)
                    self.validation_loss_ = float(nn.functional.mse_loss(predicted, y_t[hold_idx]).cpu())

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        self.context.eval()
        seq_np = self._to_sequences(X)
        with torch.no_grad():
            seq_t = torch.as_tensor(seq_np, dtype=torch.float32, device=self.device)
            z_c = self.context(seq_t)
            ret, _ = self.aux(z_c)
            out = ret.cpu().numpy()
        return np.asarray(out, dtype=float).ravel()

    def state_dict_numpy(self) -> dict[str, Any]:
        return {k: v.detach().cpu().numpy() for k, v in self.context.state_dict().items()}

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "context": self.context.state_dict(),
            "target": self.target.state_dict(),
            "predictors": self.predictors.state_dict(),
            "aux": self.aux.state_dict(),
            "fitted": self._fitted,
            "target_mode": getattr(self, "target_mode_", "legacy_in_context"),
            "predictive_coeff": getattr(self, "predictive_coeff_", 1.0),
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M06Config.from_dict(state["config"])
        set_deterministic_seed(self.config.random_state)
        self.device = pick_device(self.config.device)
        d_in, h, z = self.config.input_dim, self.config.hidden_dim, self.config.latent_dim
        self.context = ContextEncoder(d_in, h, z).to(self.device)
        self.target = TargetEncoder(d_in, h, z).to(self.device)
        self.predictors = nn.ModuleDict(
            {str(ho): HorizonPredictor(z, ho) for ho in self.config.horizons}
        ).to(self.device)
        self.aux = AuxHead(z).to(self.device)
        self.context.load_state_dict(state["context"])
        self.target.load_state_dict(state["target"])
        self.predictors.load_state_dict(state["predictors"])
        self.aux.load_state_dict(state["aux"])
        for p in self.target.parameters():
            p.requires_grad = False
        self._opt = torch.optim.Adam(
            list(self.context.parameters())
            + list(self.predictors.parameters())
            + list(self.aux.parameters()),
            lr=self.config.lr,
        )
        self._fitted = bool(state.get("fitted", False))
        self.target_mode_ = state.get("target_mode", "legacy_in_context")
        self.predictive_coeff_ = state.get("predictive_coeff", 1.0)
