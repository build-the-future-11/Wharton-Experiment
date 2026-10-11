"""Chronological benchmark baselines with explicit uncertainty calibration.

These are architecture controls, not reproductions of published SOTA numbers.
They share context/target semantics with FREM and contain no network downloads.
"""

from __future__ import annotations

import json
import math
import time as walltime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import TrainOnlyRobustScaler, WindowData
from .model import FixedReservoir
from .training import _xgboost_class, prediction_frame

BASELINE_NAMES = ("zero", "ridge", "reservoir_ridge", "xgboost", "gru", "dlinear", "patch_transformer")


class GRURegressor(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.sequence = nn.GRU(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, state = self.sequence(x)
        return self.head(state[-1]).squeeze(-1)


class DLinearRegressor(nn.Module):
    """Shared linear trend/seasonal projection over an observed context window."""

    def __init__(self, features: int, context: int) -> None:
        super().__init__()
        self.trend = nn.Linear(context, 1)
        self.seasonal = nn.Linear(context, 1)
        self.combine = nn.Linear(features, 1)
        self.kernel = min(7, context if context % 2 else context - 1)

    def forward(self, x: Tensor) -> Tensor:
        channels = x.transpose(1, 2)
        pad = self.kernel // 2
        trend = F.avg_pool1d(F.pad(channels, (pad, pad), mode="replicate"), self.kernel, stride=1)
        prediction = self.trend(trend) + self.seasonal(channels - trend)
        return self.combine(prediction.squeeze(-1)).squeeze(-1)


class PatchTransformerRegressor(nn.Module):
    """Small joint-channel patch Transformer; not the original PatchTST recipe."""

    def __init__(self, features: int, context: int, hidden: int, patch_length: int) -> None:
        super().__init__()
        self.patch_length = min(context, patch_length)
        self.patch_count = math.ceil(context / self.patch_length)
        self.left_padding = self.patch_count * self.patch_length - context
        self.project = nn.Linear(features * self.patch_length, hidden)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, hidden))
        heads = 4 if hidden % 4 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.left_padding:
            x = torch.cat((x[:, :1].expand(-1, self.left_padding, -1), x), dim=1)
        patches = x.reshape(x.shape[0], self.patch_count, -1)
        tokens = self.project(patches) + self.position
        mask = torch.triu(
            torch.ones(self.patch_count, self.patch_count, device=x.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.transformer(tokens, mask=mask)
        return self.head(encoded[:, -1]).squeeze(-1)


class Baseline:
    def __init__(
        self,
        name: str,
        *,
        seed: int = 0,
        alpha: float = 1.0,
        hidden_dim: int = 64,
        reservoir_dim: int = 64,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scale_floor: float = 1e-4,
        calibration_fraction: float = 0.2,
        max_wall_seconds: float | None = None,
        device: str = "cpu",
        n_estimators: int = 200,
        max_depth: int = 3,
        patch_length: int = 8,
    ) -> None:
        if name not in BASELINE_NAMES:
            raise ValueError(f"Unknown baseline {name!r}; choose from {BASELINE_NAMES}")
        if min(hidden_dim, reservoir_dim, epochs, batch_size, n_estimators, max_depth, patch_length) < 1:
            raise ValueError("Baseline sizes, epochs, and batches must be positive")
        if alpha < 0 or learning_rate <= 0 or weight_decay < 0 or scale_floor <= 0:
            raise ValueError("Baseline optimizer/regularization parameters are invalid")
        if not 0 < calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in (0, 0.5)")
        if max_wall_seconds is not None and max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive or None")
        self.name = name
        self.parameters = {
            "seed": seed,
            "alpha": alpha,
            "hidden_dim": hidden_dim,
            "reservoir_dim": reservoir_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scale_floor": scale_floor,
            "calibration_fraction": calibration_fraction,
            "max_wall_seconds": max_wall_seconds,
            "device": device,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "patch_length": patch_length,
        }
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA baseline requested but unavailable")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS baseline requested but unavailable")
        self.estimator: Any = None
        self.reservoir: FixedReservoir | None = None
        self.scaler = TrainOnlyRobustScaler()
        self.fitted = False
        self._fit_started = False
        self.feature_names: list[str] = []
        self.asset_names: list[str] = []
        self.context = 0
        self.features = 0
        self.fitted_through = -1
        self.scale = scale_floor
        self.student_df = 5.0
        self.fit_report: dict[str, Any] = {}
        self._started_at: float | None = None
        self._passes: list[dict[str, Any]] = []

    @property
    def neural(self) -> bool:
        return self.name in {"gru", "dlinear", "patch_transformer"}

    def _exceeded(self) -> bool:
        budget = self.parameters["max_wall_seconds"]
        return (
            budget is not None
            and self._started_at is not None
            and walltime.monotonic() - self._started_at >= budget
        )

    def _new_neural(self) -> nn.Module:
        if self.name == "gru":
            return GRURegressor(self.features, self.parameters["hidden_dim"])
        if self.name == "dlinear":
            return DLinearRegressor(self.features, self.context)
        if self.name == "patch_transformer":
            return PatchTransformerRegressor(
                self.features, self.context, self.parameters["hidden_dim"], self.parameters["patch_length"]
            )
        raise ValueError("Requested neural constructor for a non-neural baseline")

    @torch.no_grad()
    def _features(self, x: np.ndarray) -> np.ndarray:
        if self.name == "reservoir_ridge":
            if self.reservoir is None:
                raise RuntimeError("Fixed reservoir has not been initialized")
            rows = []
            for start in range(0, len(x), self.parameters["batch_size"]):
                context = torch.as_tensor(
                    x[start : start + self.parameters["batch_size"]], dtype=torch.float32, device=self.device
                )
                states = self.reservoir(context)
                rows.append(
                    torch.cat((states[:, -1], states.mean(dim=1), context[:, -1]), dim=-1).cpu().numpy()
                )
            return np.concatenate(rows, axis=0)
        if self.name == "xgboost":
            return np.concatenate(
                (x[:, -1], x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1), x[:, -1] - x[:, 0]),
                axis=1,
            )
        return x.reshape(len(x), -1)

    def _fit_point(self, x: np.ndarray, y: np.ndarray, label: str) -> Any:
        if self._exceeded():
            raise TimeoutError("Baseline wall budget exhausted before an essential fit pass")
        if self.name == "zero":
            self._passes.append({"pass": label, "rows": len(y), "epochs": 0, "gradient_steps": 0})
            return None
        if self.neural:
            torch.manual_seed(self.parameters["seed"])
            estimator = self._new_neural().to(self.device)
            optimizer = torch.optim.AdamW(
                estimator.parameters(),
                lr=self.parameters["learning_rate"],
                weight_decay=self.parameters["weight_decay"],
            )
            steps = 0
            completed = 0
            last_loss = None
            for epoch in range(self.parameters["epochs"]):
                if self._exceeded():
                    break
                estimator.train()
                seen = 0
                for start in range(0, len(y), self.parameters["batch_size"]):
                    if self._exceeded():
                        break
                    inputs = torch.as_tensor(
                        x[start : start + self.parameters["batch_size"]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    targets = torch.as_tensor(
                        y[start : start + self.parameters["batch_size"]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    prediction = estimator(inputs)
                    loss = F.mse_loss(prediction, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Baseline produced a non-finite objective")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    last_loss = float(loss.detach())
                    steps += 1
                    seen += len(targets)
                if seen == len(y):
                    completed += 1
            if steps == 0:
                raise TimeoutError("Baseline budget produced no optimizer update")
            estimator.eval()
            self._passes.append(
                {
                    "pass": label,
                    "rows": len(y),
                    "epochs": completed,
                    "gradient_steps": steps,
                    "last_batch_mse": last_loss,
                    "wall_limited": self._exceeded(),
                }
            )
            return estimator
        features = self._features(x)
        if self.name == "xgboost":
            estimator = _xgboost_class()(
                n_estimators=self.parameters["n_estimators"],
                max_depth=self.parameters["max_depth"],
                learning_rate=0.03,
                reg_lambda=10.0,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=1,
                random_state=self.parameters["seed"],
            )
        else:
            from sklearn.linear_model import Ridge

            estimator = Ridge(alpha=self.parameters["alpha"])
        estimator.fit(features, y)
        self._passes.append(
            {
                "pass": label,
                "rows": len(y),
                "epochs": None,
                "gradient_steps": 0,
                "wall_limited": self._exceeded(),
            }
        )
        return estimator

    @torch.no_grad()
    def _predict_point(self, estimator: Any, x: np.ndarray) -> np.ndarray:
        if self.name == "zero":
            return np.zeros(len(x), dtype=np.float64)
        if self.neural:
            estimator.eval()
            pieces = []
            for start in range(0, len(x), self.parameters["batch_size"]):
                inputs = torch.as_tensor(
                    x[start : start + self.parameters["batch_size"]], dtype=torch.float32, device=self.device
                )
                pieces.append(estimator(inputs).cpu().numpy())
            return np.concatenate(pieces).astype(np.float64)
        return np.asarray(estimator.predict(self._features(x)), dtype=np.float64)

    def fit(self, train: WindowData) -> dict[str, Any]:
        if self._fit_started:
            raise RuntimeError("Each baseline instance fits one split; use a new instance to refit")
        self._fit_started = True
        self._started_at = walltime.monotonic()
        if len(train) < 2 or not np.isfinite(train.y).all():
            raise ValueError("Baseline fitting requires at least two finite labeled windows")
        self.context, self.features = train.x.shape[1:]
        self.feature_names, self.asset_names = list(train.feature_names), list(train.asset_names)
        self.fitted_through = int(train.label_time.max())
        if self.fitted_through >= np.iinfo(np.int64).max - 1:
            raise ValueError("Training labels contain unavailable maturity")
        if self.name == "reservoir_ridge":
            self.reservoir = FixedReservoir(
                self.features,
                self.parameters["reservoir_dim"],
                leak=0.3,
                contraction=0.9,
                seed=self.parameters["seed"],
            ).to(self.device)
        times = np.unique(train.time)
        boundary_index = int(len(times) * (1 - self.parameters["calibration_fraction"]))
        boundary_index = min(max(boundary_index, 1), len(times) - 1)
        earlier = (
            np.flatnonzero(train.label_time < times[boundary_index])
            if len(times) > 1
            else np.array([], dtype=int)
        )
        later = (
            np.flatnonzero(train.time >= times[boundary_index]) if len(times) > 1 else np.array([], dtype=int)
        )
        calibration_note = "conservative_training_target_scale_no_inner_holdout"
        if self.name == "zero":
            residuals = train.y.copy()
            calibration_note = "zero_forecast_training_outcomes"
        elif len(times) >= 8 and len(earlier) >= 16 and len(later) >= 5:
            inner = train.subset(earlier)
            inner_scaler = TrainOnlyRobustScaler().fit(inner)
            inner_estimator = self._fit_point(inner_scaler.transform_x(inner.x), inner.y, "inner_scale_fit")
            predictions = self._predict_point(inner_estimator, inner_scaler.transform_x(train.x[later]))
            residuals = train.y[later] - predictions
            calibration_note = "purged_chronological_training_tail_with_earlier_only_scaler"
        else:
            residuals = train.y - float(np.mean(train.y))
            # Very short samples cannot support a credible extra calibration split.
            # A zero-forecast error scale avoids an artificially perfect fit scale.
            residuals = np.r_[residuals, train.y]
        self.scaler.fit(train)
        contexts = self.scaler.transform_x(train.x)
        self.estimator = self._fit_point(contexts, train.y, "full_training_refit")
        residual_rms = float(np.sqrt(np.mean(np.square(residuals))))
        # Student-t scale is not its standard deviation: Var = scale²*nu/(nu-2).
        self.scale = max(
            self.parameters["scale_floor"], residual_rms * math.sqrt((self.student_df - 2) / self.student_df)
        )
        self.fitted = True
        neural_parameters = (
            sum(parameter.numel() for parameter in self.estimator.parameters()) if self.neural else None
        )
        self.fit_report = {
            "baseline": self.name,
            "status": "fitted",
            "train_rows": len(train),
            "fitted_through": self.fitted_through,
            "scale": self.scale,
            "scale_calibration": calibration_note,
            "passes": self._passes,
            "trainable_parameters": neural_parameters,
            "elapsed_seconds": walltime.monotonic() - self._started_at,
            "max_wall_seconds": self.parameters["max_wall_seconds"],
            "budget_note": "Wall clock checked between neural minibatches and estimator calls; a single tree/linear fit cannot be forcibly interrupted here",
        }
        return dict(self.fit_report)

    def predict(self, data: WindowData) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Fit/load baseline before prediction")
        if len(data) == 0 or int(data.time.min()) <= self.fitted_through:
            raise ValueError("Forecast windows must follow all baseline fitting outcomes")
        if list(data.feature_names) != self.feature_names or list(data.asset_names) != self.asset_names:
            raise ValueError("Prediction schema/asset mapping differs from baseline training")
        if data.x.shape[1:] != (self.context, self.features):
            raise ValueError("Prediction context shape differs from baseline training")
        prediction = self._predict_point(self.estimator, self.scaler.transform_x(data.x))
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Baseline produced non-finite forecasts")
        return prediction_frame(data, prediction, np.full(len(data), self.scale), self.student_df)

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted baseline")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "baseline.json").exists():
            raise FileExistsError("Choose a fresh baseline checkpoint directory")
        if self.neural:
            torch.save(self.estimator.state_dict(), directory / "model.pt")
        elif self.name == "xgboost":
            self.estimator.save_model(str(directory / "model.ubj"))
        elif self.name != "zero":
            import joblib

            joblib.dump(self.estimator, directory / "estimator.joblib")
        if self.reservoir is not None:
            torch.save(self.reservoir.state_dict(), directory / "reservoir.pt")
        state = {
            "format_version": 1,
            "name": self.name,
            "parameters": self.parameters,
            "context": self.context,
            "features": self.features,
            "feature_names": self.feature_names,
            "asset_names": self.asset_names,
            "fitted_through": self.fitted_through,
            "scale": self.scale,
            "student_df": self.student_df,
            "scaler": self.scaler.to_dict(),
            "fit_report": self.fit_report,
        }
        (directory / "baseline.json").write_text(
            json.dumps(state, indent=2, allow_nan=False), encoding="utf-8"
        )


def make_baseline(name: str, seed: int = 0, **kwargs: Any) -> Baseline:
    return Baseline(name, seed=seed, **kwargs)


def load_baseline(path: str | Path, *, device: str | None = None, trust_joblib: bool = False) -> Baseline:
    """Load a baseline checkpoint; sklearn/joblib needs explicit local trust.

    Pickle-based sklearn artifacts can execute code on load. Tensor weights and
    native XGBoost files do not require pickle loading. This project never loads
    arbitrary downloaded joblib artifacts implicitly.
    """
    directory = Path(path)
    state = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
    if state.get("format_version") != 1:
        raise ValueError("Unsupported baseline checkpoint format")
    parameters = dict(state["parameters"])
    if device is not None:
        parameters["device"] = device
    baseline = Baseline(state["name"], **parameters)
    for field in (
        "context",
        "features",
        "feature_names",
        "asset_names",
        "fitted_through",
        "scale",
        "student_df",
        "fit_report",
    ):
        setattr(baseline, field, state[field])
    baseline.scaler = TrainOnlyRobustScaler.from_dict(state["scaler"])
    if baseline.neural:
        baseline.estimator = baseline._new_neural().to(baseline.device)
        baseline.estimator.load_state_dict(
            torch.load(directory / "model.pt", map_location=baseline.device, weights_only=True)
        )
        baseline.estimator.eval()
    elif baseline.name == "xgboost":
        baseline.estimator = _xgboost_class()()
        baseline.estimator.load_model(str(directory / "model.ubj"))
    elif baseline.name != "zero":
        if not trust_joblib:
            raise ValueError(
                "Loading this local sklearn artifact requires trust_joblib=True; never trust unverified pickle files"
            )
        import joblib

        baseline.estimator = joblib.load(directory / "estimator.joblib")
    if baseline.name == "reservoir_ridge":
        baseline.reservoir = FixedReservoir(
            baseline.features, baseline.parameters["reservoir_dim"], seed=baseline.parameters["seed"]
        ).to(baseline.device)
        baseline.reservoir.load_state_dict(
            torch.load(directory / "reservoir.pt", map_location=baseline.device, weights_only=True)
        )
    baseline.fitted = baseline._fit_started = True
    return baseline
