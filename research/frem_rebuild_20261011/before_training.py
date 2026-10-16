"""Explicit fit/predict lifecycle, chronological feedback, and safe checkpoints.

Nothing trains at import time. Calling fit is an experiment and must be a later,
deliberate execution step. Default prediction freezes weights, scaler, spectral
reference, memory, and controller. Online mode changes only delayed memory and
expert weighting; it does not silently take gradient steps.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import time as walltime
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .controller import AdaptiveController, ControllerConfig
from .data import TrainOnlyRobustScaler, WindowData
from .memory import DAY_NS, EngramMemory, Retrieval
from .model import MEMORY_FEATURES, FREMModel, ModelConfig, predictive_loss, representation_losses


@dataclass
class TrainingConfig:
    seed: int = 0
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    jepa_weight: float = 0.1
    variance_weight: float = 0.01
    covariance_weight: float = 0.001
    ema_momentum: float = 0.99
    early_stopping_patience: int = 5
    min_delta: float = 1e-5
    device: str = "cpu"
    scaler_clip: float | None = 12.0
    memory_utility_rate: float = 0.1
    memory_half_life_days: float = 180.0
    memory_temperature: float = 0.2
    pending_capacity: int = 100_000
    controller_patience: int = 2
    controller_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    max_wall_seconds: float | None = None
    xgboost_estimators: int = 200
    xgboost_max_depth: int = 3

    def __post_init__(self) -> None:
        for field in (
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "jepa_weight",
            "variance_weight",
            "covariance_weight",
            "ema_momentum",
            "min_delta",
            "memory_utility_rate",
            "memory_half_life_days",
            "memory_temperature",
            "controller_lr_factor",
            "min_learning_rate",
        ):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in (
            "epochs",
            "batch_size",
            "early_stopping_patience",
            "pending_capacity",
            "controller_patience",
            "xgboost_estimators",
            "xgboost_max_depth",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Optimizer settings are invalid")
        if min(self.jepa_weight, self.variance_weight, self.covariance_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if not 0 <= self.ema_momentum < 1 or self.min_delta < 0:
            raise ValueError("EMA/early-stopping settings are invalid")
        if self.max_wall_seconds is not None and (
            not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be positive or None")
        if self.scaler_clip is not None and (not math.isfinite(self.scaler_clip) or self.scaler_clip <= 0):
            raise ValueError("scaler_clip must be positive or None")


def time_groups(timestamps: np.ndarray) -> Iterator[np.ndarray]:
    """Yield all rows at each decision time together, preserving row order."""
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.size == 0:
        return
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Decision rows must be chronological")
    boundaries = np.r_[0, np.flatnonzero(np.diff(timestamps) != 0) + 1, len(timestamps)]
    for start, end in itertools.pairwise(boundaries):
        yield np.arange(start, end, dtype=np.int64)


def date_balanced_mean(values: np.ndarray, timestamps: np.ndarray, asset_ids: np.ndarray) -> float:
    """Mean within each UTC-date/asset, then across assets, then across UTC dates.

    Match the primary evaluation even when markets decide at different UTC times
    or an asset has several observations in one date. Training minibatch losses
    remain row-weighted; this reduction applies to tuning/evaluation.
    """
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps)
    asset_ids = np.asarray(asset_ids)
    if (
        values.ndim != 1
        or timestamps.shape != values.shape
        or asset_ids.shape != values.shape
        or not len(values)
        or not np.isfinite(values).all()
        or not np.issubdtype(timestamps.dtype, np.integer)
        or not np.issubdtype(asset_ids.dtype, np.integer)
    ):
        raise ValueError(
            "Date-balanced scores require aligned finite losses and integer timestamp/asset keys"
        )
    dates = pd.to_datetime(timestamps, unit="ns", utc=True).normalize()
    if dates.isna().any() or np.any(asset_ids < 0):
        raise ValueError("Date-balanced scores require valid dates and nonnegative asset keys")
    rows = pd.DataFrame({"utc_date": dates, "asset_id": asset_ids, "loss": values})
    per_asset_date = rows.groupby(["utc_date", "asset_id"], sort=True)["loss"].mean()
    return float(per_asset_date.groupby(level="utc_date").mean().mean())


def prediction_frame(
    data: WindowData,
    prediction: np.ndarray,
    scale: np.ndarray,
    student_df: np.ndarray | float = 5.0,
    **extra: Any,
) -> pd.DataFrame:
    """Reporting actuals is separate from constructing a model forecast."""
    frame = pd.DataFrame(
        {
            "time": data.time.copy(),
            "label_time": data.label_time.copy(),
            "asset_id": data.asset_id.copy(),
            "asset": np.asarray(data.asset_names, dtype=object)[data.asset_id],
            "prediction": prediction,
            "scale": scale,
            "student_df": student_df,
            "actual": data.y.copy(),
            **extra,
        }
    )
    if data.entry_time is not None:
        frame["entry_time"] = data.entry_time
    if data.exit_time is not None:
        frame["exit_time"] = data.exit_time
    return frame


def _xgboost_class() -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError(
            "XGBoost was explicitly requested. Install the 'xgboost' project extra; no substitute model is silently used."
        ) from error
    return XGBRegressor


class Trainer:
    def __init__(
        self, model_config: ModelConfig | dict[str, Any], training_config: TrainingConfig | dict[str, Any]
    ) -> None:
        self.model_config = ModelConfig(**model_config) if isinstance(model_config, dict) else model_config
        self.training_config = (
            TrainingConfig(**training_config) if isinstance(training_config, dict) else training_config
        )
        device = torch.device(self.training_config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose device='cpu' explicitly")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if device.type == "mps" and self.model_config.use_spectral:
            raise ValueError("Use CPU/CUDA for this spectral implementation; MPS FFT support is not assumed")
        torch.manual_seed(self.training_config.seed)
        self.device = device
        self.model = FREMModel(self.model_config).to(device)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.scaler = TrainOnlyRobustScaler(clip=self.training_config.scaler_clip)
        self.controller = self._new_controller()
        self.memory = self._new_memory()
        self.fitted = False
        self._fit_started = False
        self.training_cutoff: int | None = None
        self.fitted_through: int | None = None
        self.online_watermark: int | None = None
        self.feature_names: list[str] = []
        self.asset_names: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.fit_report: dict[str, Any] = {}
        self.residual_model: Any = None
        self.gradient_steps = 0
        self._started_at: float | None = None

    def _new_controller(self) -> AdaptiveController:
        training = self.training_config
        return AdaptiveController(
            ControllerConfig(
                enabled=self.model_config.use_adaptive_controller,
                base_learning_rate=training.learning_rate,
                min_learning_rate=training.min_learning_rate,
                lr_factor=training.controller_lr_factor,
                patience=training.controller_patience,
                improvement_tolerance=training.min_delta,
                base_jepa_weight=training.jepa_weight,
            )
        )

    def _new_memory(self) -> EngramMemory:
        model, training = self.model_config, self.training_config
        return EngramMemory(
            key_dim=self.model.key_dim,
            capacity=model.memory_capacity,
            neighbors=model.memory_neighbors,
            same_asset=model.memory_same_asset,
            exact_ngram=model.memory_exact_ngram,
            retention=model.memory_retention,
            utility_rate=training.memory_utility_rate,
            half_life_days=training.memory_half_life_days,
            temperature=training.memory_temperature,
            pending_capacity=training.pending_capacity,
            surprise_threshold=model.memory_surprise_threshold,
            seed=training.seed,
        )

    def _budget_exceeded(self) -> bool:
        limit = self.training_config.max_wall_seconds
        return (
            limit is not None
            and self._started_at is not None
            and walltime.monotonic() - self._started_at >= limit
        )

    def _tensor(self, array: Any, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    def _validate_schema(self, data: WindowData, *, labeled: bool) -> None:
        if len(data) == 0:
            raise ValueError("Empty window sets cannot train or produce forecasts")
        if data.x.shape[1:] != (self.model_config.context_length, self.model_config.input_dim):
            raise ValueError("Window context shape does not match ModelConfig")
        if labeled and data.future_x.shape[1:] != (self.model_config.horizon, self.model_config.input_dim):
            raise ValueError("Future target shape does not match ModelConfig")
        if labeled and not np.isfinite(data.y).all():
            raise ValueError("Fit/validation/calibration require finite executable return targets")
        if self.feature_names and list(data.feature_names) != self.feature_names:
            raise ValueError("Feature name/order differs from the fitted model")
        if self.asset_names and list(data.asset_names) != self.asset_names:
            raise ValueError("Asset ID mapping differs from training; align asset_names explicitly")
        if np.any(data.label_time <= data.time):
            raise ValueError("Targets must become available strictly after their decision")
        if np.any(np.diff(data.time) < 0):
            raise ValueError("Window rows must be sorted chronologically")

    def _sync_expert(self) -> None:
        with torch.no_grad():
            self.model.memory_expert_weight.fill_(self.controller.memory_weight)

    def _advance(self, timestamp: int) -> None:
        if not self.model_config.use_memory:
            return
        events = self.memory.advance(timestamp)
        for event in events:
            assert event.outcome is not None
            self.controller.observe_matured(
                actual=event.outcome,
                base_prediction=event.base_prediction,
                memory_prediction=event.memory_prediction,
                scale=event.scale,
                label_time=event.label_time,
                as_of_time=timestamp,
                had_memory=bool(event.retrieved_ids),
            )
        self._sync_expert()

    def _retrieve(self, x: Tensor, data: WindowData, indices: np.ndarray) -> tuple[Tensor, list[Retrieval]]:
        if not self.model_config.use_memory:
            empty = [Retrieval(np.zeros(MEMORY_FEATURES, dtype=np.float32), [], []) for _ in indices]
            return x.new_zeros(len(indices), MEMORY_FEATURES), empty
        keys, codes = self.model.make_keys(x)
        key_array, code_array = keys.cpu().numpy(), codes.cpu().numpy()
        retrievals = [
            self.memory.retrieve(
                key_array[row],
                asset_id=int(data.asset_id[index]),
                decision_time=int(data.time[index]),
                ngram_code=int(code_array[row]),
            )
            for row, index in enumerate(indices)
        ]
        return self._tensor(np.stack([value.features for value in retrievals])), retrievals

    def _schedule(
        self, output: dict[str, Tensor], data: WindowData, indices: np.ndarray, retrievals: list[Retrieval]
    ) -> None:
        if not self.model_config.use_memory:
            return
        fields = {
            name: output[name].detach().cpu().numpy()
            for name in (
                "key",
                "ngram_code",
                "prediction",
                "base_prediction",
                "memory_expert_prediction",
                "scale",
                "spectral_surprise",
            )
        }
        for row, index in enumerate(indices):
            actual = float(data.y[index])
            self.memory.schedule(
                key=fields["key"][row],
                asset_id=int(data.asset_id[index]),
                decision_time=int(data.time[index]),
                label_time=int(data.label_time[index]),
                base_prediction=float(fields["base_prediction"][row]),
                prediction=float(fields["prediction"][row]),
                memory_prediction=float(fields["memory_expert_prediction"][row]),
                scale=float(fields["scale"][row]),
                ngram_code=int(fields["ngram_code"][row]),
                novelty=float(fields["spectral_surprise"][row]),
                outcome=actual if math.isfinite(actual) else None,
                retrieval=retrievals[row],
            )

    def _gradient_step(self, batches: list[tuple[dict[str, Tensor], Tensor, Tensor]]) -> dict[str, float]:
        output = {
            name: torch.cat([batch[0][name] for batch in batches], dim=0)
            for name in ("prediction", "scale", "df", "latent", "predicted_latent")
        }
        target = torch.cat([batch[1] for batch in batches], dim=0)
        future_target = torch.cat([batch[2] for batch in batches], dim=0)
        supervised = predictive_loss(output, target, self.model_config.use_stochastic)
        regularizers = representation_losses(output["predicted_latent"], future_target, output["latent"])
        loss = supervised
        if self.model_config.use_jepa:
            loss = (
                loss
                + self.controller.jepa_weight * regularizers["jepa"]
                + self.training_config.variance_weight * regularizers["variance"]
                + self.training_config.covariance_weight * regularizers["covariance"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training objective; the run must be recorded as failed")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.training_config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise FloatingPointError("Optimizer produced a non-finite parameter; checkpoint is invalid")
        if self.model_config.use_jepa:
            self.model.update_ema(self.training_config.ema_momentum)
        self.gradient_steps += 1
        gates = torch.cat([batch[0]["feature_gate"].detach().flatten() for batch in batches])
        if self.model_config.use_feature_gate:
            p = gates.clamp(1e-6, 1 - 1e-6)
            gate_entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        else:
            gate_entropy = gates.new_tensor(0)
        return {
            "loss": float(loss.detach()),
            "predictive_loss": float(supervised.detach()),
            "jepa_loss": float(regularizers["jepa"].detach()) if self.model_config.use_jepa else 0.0,
            "variance_penalty": float(regularizers["variance"].detach())
            if self.model_config.use_jepa
            else 0.0,
            "covariance_penalty": float(regularizers["covariance"].detach())
            if self.model_config.use_jepa
            else 0.0,
            "latent_variance": float(output["latent"].detach().var(0, unbiased=False).mean()),
            "gate_entropy": float(gate_entropy),
            "gradient_norm": float(norm),
            "rows": len(target),
            "regularizer_samples": int(regularizers["regularizer_samples"].detach()),
        }

    def _snapshot(self) -> dict[str, Any]:
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "controller": copy.deepcopy(self.controller.to_dict()),
            "memory": copy.deepcopy(self.memory.to_dict()),
        }

    def _restore_snapshot(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.controller = AdaptiveController.from_dict(state["controller"])
        self.memory = EngramMemory.from_dict(state["memory"])

    def fit(
        self, train: WindowData, validation: WindowData | None = None, calibration: WindowData | None = None
    ) -> dict[str, Any]:
        if self._fit_started:
            raise RuntimeError(
                "A Trainer may fit exactly one chronological split; create a fresh instance for another run"
            )
        self._fit_started = True
        self._started_at = walltime.monotonic()
        self._validate_schema(train, labeled=True)
        self.feature_names, self.asset_names = list(train.feature_names), list(train.asset_names)
        self.training_cutoff = int(np.max(train.label_time))
        if self.training_cutoff >= np.iinfo(np.int64).max - 1:
            raise ValueError("Training labels contain an unavailable-maturity sentinel")
        self.fitted_through = self.training_cutoff
        if validation is not None:
            self._validate_schema(validation, labeled=True)
            if int(validation.time.min()) <= self.training_cutoff:
                raise ValueError("Purge training labels that mature at/after the validation start")
            self.fitted_through = max(self.fitted_through, int(validation.label_time.max()))
        if self.model_config.residual_xgboost and calibration is None:
            raise ValueError(
                "The hybrid requires a separate chronological calibration WindowData after tuning validation"
            )
        if calibration is not None:
            self._validate_schema(calibration, labeled=True)
            if int(calibration.time.min()) <= self.fitted_through:
                raise ValueError(
                    "Purge earlier labels so residual calibration starts after all tuning outcomes"
                )
            if not self.model_config.residual_xgboost:
                raise ValueError("Calibration was supplied but residual_xgboost is disabled")
        scaled_train = self.scaler.fit_transform(train)
        if self.model.spectral is not None:

            def reference_batches() -> Iterator[Tensor]:
                for start in range(0, len(train), self.training_config.batch_size):
                    if self._budget_exceeded():
                        raise TimeoutError(
                            "Wall budget exhausted fitting the training-only spectral reference"
                        )
                    yield self._tensor(scaled_train.x[start : start + self.training_config.batch_size])

            self.model.spectral.fit_reference(reference_batches())
        best_score = math.inf
        best_snapshot: dict[str, Any] | None = None
        best_epoch = 0
        bad_epochs = 0
        completed_epochs = 0
        stop_reason = "epoch_budget"
        for epoch in range(1, self.training_config.epochs + 1):
            if self._budget_exceeded():
                stop_reason = "wall_budget"
                break
            self.model.train()
            self.memory = self._new_memory()
            # Replaying the same training history across epochs must not be
            # presented to the expert updater as additional independent evidence.
            self.controller.memory_logit = 0.0
            self.controller.observation_count = 0
            self.controller.last_outcome_availability = -(1 << 63)
            self._sync_expert()
            accumulated: list[tuple[dict[str, Tensor], Tensor, Tensor]] = []
            accumulated_rows = 0
            statistics: list[dict[str, float]] = []
            training_errors: list[float] = []
            seen = 0
            memory_hits = 0
            for indices in time_groups(train.time):
                if self._budget_exceeded():
                    stop_reason = "wall_budget"
                    break
                timestamp = int(train.time[indices[0]])
                self._advance(timestamp)
                x = self._tensor(scaled_train.x[indices])
                context, retrievals = self._retrieve(x, train, indices)
                output = self.model(x, self._tensor(train.asset_id[indices], torch.long), context)
                target = self._tensor(train.y[indices])
                if self.model_config.use_jepa:
                    future_target = self.model.encode_target(self._tensor(scaled_train.future_x[indices]))
                else:
                    future_target = torch.zeros_like(output["latent"])
                accumulated.append((output, target, future_target))
                accumulated_rows += len(indices)
                memory_hits += sum(bool(value.identifiers) for value in retrievals)
                training_errors.extend(
                    (target.detach() - output["prediction"].detach()).cpu().numpy().tolist()
                )
                # Every row sharing this timestamp was predicted from the same
                # memory/controller state before any same-time record is queued.
                self._schedule(output, train, indices, retrievals)
                seen += len(indices)
                if accumulated_rows >= self.training_config.batch_size:
                    statistics.append(self._gradient_step(accumulated))
                    accumulated, accumulated_rows = [], 0
            if accumulated:
                statistics.append(self._gradient_step(accumulated))
            if not statistics:
                stop_reason = "wall_budget"
                break
            # All observed training labels are legally known at fit completion.
            # A partial epoch contains only its seen pending events, never unseen
            # rows inserted retroactively without predictions.
            self._advance(self.training_cutoff + 1)
            if not self.model_config.use_stochastic:
                scale = max(
                    self.model_config.scale_floor, float(np.sqrt(np.mean(np.square(training_errors))))
                )
                self.model.deterministic_scale.fill_(scale)
            total_rows = sum(row["rows"] for row in statistics)
            summary = {
                key: sum(row[key] * row["rows"] for row in statistics) / total_rows
                for key in statistics[0]
                if key not in {"rows", "regularizer_samples"}
            }
            summary.update(
                {
                    "epoch": epoch,
                    "rows": seen,
                    "gradient_steps": self.gradient_steps,
                    "memory_hit_fraction": memory_hits / max(seen, 1),
                    "memory": self.memory.diagnostics(),
                    "complete_epoch": seen == len(train),
                    "regularizer_samples": sum(row["regularizer_samples"] for row in statistics),
                }
            )
            self.model.eval()
            if seen == len(train):
                completed_epochs += 1
            if validation is not None and not self._budget_exceeded():
                forecasts = self._forecast(validation, online_update=False, enforce_cutoff=False)
                from scipy.stats import t as student_t

                losses = -student_t.logpdf(
                    validation.y,
                    df=forecasts.student_df.to_numpy(dtype=np.float64),
                    loc=forecasts.prediction.to_numpy(dtype=np.float64),
                    scale=forecasts.scale.to_numpy(dtype=np.float64),
                )
                score = date_balanced_mean(losses, validation.time, validation.asset_id)
                if not math.isfinite(score):
                    raise FloatingPointError("Validation log score is non-finite")
                summary["validation_nll"] = score
                summary["validation_rows"] = len(validation)
                summary["validation_dates"] = len(np.unique(validation.time // DAY_NS))
                summary["validation_timestamps"] = len(np.unique(validation.time))
                summary["validation_weighting"] = (
                    "equal_UTC_date_then_equal_asset_then_equal_within_asset_rows"
                )
                if score < best_score - self.training_config.min_delta:
                    best_score, best_epoch, bad_epochs = score, epoch, 0
                    best_snapshot = self._snapshot()
                else:
                    bad_epochs += 1
                decision = self.controller.observe_validation(score)
                for group in self.optimizer.param_groups:
                    group["lr"] = decision["learning_rate"]
                summary["controller"] = decision
            elif validation is None:
                best_epoch = epoch
                best_snapshot = self._snapshot()
            elif best_snapshot is None:
                # Preserve diagnostics locally, then fail below if no tuning
                # validation completed; these are not selected checkpoints.
                best_epoch = epoch
                best_snapshot = self._snapshot()
                summary["validation_skipped"] = "wall_budget"
            self.history.append(summary)
            if stop_reason == "wall_budget" or self._budget_exceeded():
                stop_reason = "wall_budget"
                break
            if validation is not None and bad_epochs >= self.training_config.early_stopping_patience:
                stop_reason = "early_stopping"
                break
        if best_snapshot is None or self.gradient_steps == 0:
            raise TimeoutError("No usable gradient-updated state was produced within the training budget")
        if validation is not None and math.isinf(best_score):
            self.fit_report = {
                "status": "incomplete",
                "reason": "no_completed_tuning_validation",
                "gradient_steps_executed": self.gradient_steps,
                "history": self.history,
            }
            raise TimeoutError(
                "Weights were updated, but no tuning validation completed; this run cannot produce a selected research checkpoint"
            )
        self._restore_snapshot(best_snapshot)
        self.model.eval()
        self.fitted = True
        self.online_watermark = self.fitted_through
        self.fit_report = {
            "status": "fitted",
            "scientific_claim": "training completed; held-out evidence required",
            "stop_reason": stop_reason,
            "requested_epochs": self.training_config.epochs,
            "completed_epochs": completed_epochs,
            "attempted_epochs": len(self.history),
            "selected_epoch": best_epoch,
            "gradient_steps_executed": self.gradient_steps,
            "best_validation_nll": None if math.isinf(best_score) else best_score,
            "validation_selection_completed": not math.isinf(best_score),
            "validation_weighting": "equal_UTC_date_then_equal_asset_then_equal_within_asset_rows",
            "elapsed_seconds": walltime.monotonic() - self._started_at,
            "training_cutoff": self.training_cutoff,
            "fitted_through": self.fitted_through,
            "train_rows": len(train),
            "validation_rows": 0 if validation is None else len(validation),
            "validation_dates": 0 if validation is None else len(np.unique(validation.time // DAY_NS)),
            "validation_timestamps": 0 if validation is None else len(np.unique(validation.time)),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "fixed_buffer_elements": sum(b.numel() for b in self.model.buffers()),
            "memory": self.memory.diagnostics(),
            "history": self.history,
            "memory_residual_reference": "training-trajectory base forecasts; fixed keys, evolving readout residuals",
        }
        if calibration is not None:
            if self._budget_exceeded():
                self.fitted = False
                raise TimeoutError(
                    "Budget exhausted before required hybrid calibration; hybrid checkpoint is incomplete"
                )
            self.fit_residual(calibration)
            self.fit_report["elapsed_seconds"] = walltime.monotonic() - self._started_at
        return copy.deepcopy(self.fit_report)

    @torch.no_grad()
    def _forecast(
        self, data: WindowData, *, online_update: bool, enforce_cutoff: bool = True
    ) -> pd.DataFrame:
        self._validate_schema(data, labeled=False)
        if enforce_cutoff:
            known_cutoffs = [
                value for value in (self.fitted_through, self.online_watermark) if value is not None
            ]
            cutoff = max(known_cutoffs) if known_cutoffs else None
            if cutoff is None or int(data.time.min()) <= cutoff:
                raise ValueError(
                    "Forecast decisions must follow all training/tuning/calibration information and the previous decision batch after any online adaptation"
                )
        # Intentionally access only x here. future_x is not read, validated,
        # standardized, or forwarded to any prediction path.
        contexts = self.scaler.transform_x(data.x)
        predictions = np.empty(len(data), dtype=np.float64)
        scales = np.empty(len(data), dtype=np.float64)
        degrees = np.empty(len(data), dtype=np.float64)
        base = np.empty(len(data), dtype=np.float64)
        gates = np.empty(len(data), dtype=np.float64)
        surprises = np.empty(len(data), dtype=np.float64)
        hits = np.zeros(len(data), dtype=np.int64)
        self.model.eval()
        for indices in time_groups(data.time):
            timestamp = int(data.time[indices[0]])
            if online_update:
                self._advance(timestamp)
            x = self._tensor(contexts[indices])
            memory_context, retrievals = self._retrieve(x, data, indices)
            output = self.model(x, self._tensor(data.asset_id[indices], torch.long), memory_context)
            predictions[indices] = output["prediction"].cpu().numpy()
            scales[indices] = output["scale"].cpu().numpy()
            degrees[indices] = output["df"].cpu().numpy()
            base[indices] = output["base_prediction"].cpu().numpy()
            gates[indices] = output["memory_gate"].cpu().numpy()
            surprises[indices] = output["spectral_surprise"].cpu().numpy()
            hits[indices] = [len(value.identifiers) for value in retrievals]
            if online_update:
                self._schedule(output, data, indices, retrievals)
                self.online_watermark = timestamp
        if self.residual_model is not None:
            features = self._residual_features(contexts, predictions, scales)
            correction = np.asarray(self.residual_model.predict(features), dtype=np.float64)
            limit = self.model_config.xgboost_residual_clip
            predictions += self.model_config.xgboost_residual_weight * np.clip(correction, -limit, limit)
        frame = prediction_frame(
            data,
            predictions,
            scales,
            degrees,
            base_prediction=base,
            memory_gate=gates,
            memory_neighbors=hits,
            spectral_surprise=surprises,
        )
        frame.attrs["diagnostics"] = {
            "online_update": online_update,
            "memory_hit_fraction": float(np.mean(hits > 0)),
            "mean_memory_gate": float(gates.mean()),
            "memory": self.memory.diagnostics(),
        }
        return frame

    def predict(self, data: WindowData, online_update: bool = False) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Fit or load a completed Trainer before prediction")
        if self.model_config.residual_xgboost and self.residual_model is None:
            raise RuntimeError("The requested hybrid has no fitted residual calibration model")
        if online_update and data.metadata.get("mode") == "unlabeled_inference":
            raise ValueError(
                "Latest unlabeled windows have no declared executable maturity; use frozen prediction until an online event schedule is supplied"
            )
        return self._forecast(data, online_update=online_update)

    def observe_outcomes(self, outcomes: pd.DataFrame, *, as_of_time: int) -> dict[str, Any]:
        """Supply real live outcomes to explicit online pending predictions.

        Required columns: time, asset_id, actual, available_at. A label published
        exactly at as_of_time remains ineligible until a later decision.
        """
        required = {"time", "asset_id", "actual", "available_at"}
        if not required <= set(outcomes):
            raise ValueError(f"Feedback is missing columns: {sorted(required - set(outcomes))}")
        if not self.fitted or not self.model_config.use_memory:
            raise RuntimeError("Live feedback requires a fitted memory-enabled model")
        for row in outcomes.itertuples(index=False):
            self.memory.provide_feedback(
                decision_time=int(row.time),
                asset_id=int(row.asset_id),
                outcome=float(row.actual),
                available_at=int(row.available_at),
                as_of_time=int(as_of_time),
            )
        self._advance(int(as_of_time))
        self.online_watermark = (
            int(as_of_time) if self.online_watermark is None else max(self.online_watermark, int(as_of_time))
        )
        return self.memory.diagnostics()

    @staticmethod
    def _residual_features(x: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                x[:, -1],
                x.mean(axis=1),
                x.std(axis=1),
                (x[:, -1] - x[:, 0]),
                prediction[:, None],
                scale[:, None],
            ),
            axis=1,
        )

    def fit_residual(self, calibration: WindowData) -> dict[str, Any]:
        if not self.fitted or not self.model_config.residual_xgboost:
            raise RuntimeError("Residual fitting requires a fitted residual_xgboost-enabled model")
        if self.residual_model is not None:
            raise RuntimeError("Residual calibration is single-use; create a new run to refit it")
        self._validate_schema(calibration, labeled=True)
        if self.fitted_through is None or int(calibration.time.min()) <= self.fitted_through:
            raise ValueError("Residual calibration must begin after all neural tuning outcomes")
        # This internal call permits the requested-but-not-yet-fitted hybrid.
        forecasts = self._forecast(calibration, online_update=False)
        x = self.scaler.transform_x(calibration.x)
        features = self._residual_features(x, forecasts.prediction.to_numpy(), forecasts.scale.to_numpy())
        cls = _xgboost_class()
        estimator = cls(
            n_estimators=self.training_config.xgboost_estimators,
            max_depth=self.training_config.xgboost_max_depth,
            learning_rate=0.03,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=10.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=self.training_config.seed,
        )
        estimator.fit(features, calibration.y - forecasts.prediction.to_numpy())
        self.residual_model = estimator
        self.fitted_through = int(calibration.label_time.max())
        self.online_watermark = self.fitted_through
        report = {
            "rows": len(calibration),
            "start": int(calibration.time.min()),
            "fitted_through": self.fitted_through,
            "correction_weight": self.model_config.xgboost_residual_weight,
            "correction_clip": self.model_config.xgboost_residual_clip,
            "status": "fitted_on_dedicated_later_calibration",
            "scale_note": "Neural scale is retained; hybrid calibration is not guaranteed",
        }
        self.fit_report["residual_xgboost"] = report
        self.fit_report["fitted_through"] = self.fitted_through
        return report

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        if not self.fitted:
            raise RuntimeError("An unfitted/incomplete Trainer cannot be saved as a usable checkpoint")
        if self.model_config.residual_xgboost and self.residual_model is None:
            raise RuntimeError(
                "A hybrid checkpoint is incomplete until its separate residual calibration is fitted"
            )
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "state.json").exists() and not overwrite:
            raise FileExistsError(
                "Checkpoint exists; choose a new artifact directory or explicit overwrite=True"
            )
        torch.save(self.model.state_dict(), directory / "model.pt")
        optimizer_state = {"optimizer": self.optimizer.state_dict(), "cpu_rng": torch.get_rng_state()}
        if torch.cuda.is_available():
            optimizer_state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(optimizer_state, directory / "optimizer.pt")
        if self.residual_model is not None:
            self.residual_model.save_model(str(directory / "residual_xgboost.ubj"))
        state = {
            "format_version": 1,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "scaler": self.scaler.to_dict(),
            "memory": self.memory.to_dict(),
            "controller": self.controller.to_dict(),
            "training_cutoff": self.training_cutoff,
            "fitted_through": self.fitted_through,
            "online_watermark": self.online_watermark,
            "feature_names": self.feature_names,
            "asset_names": self.asset_names,
            "fit_report": self.fit_report,
            "gradient_steps": self.gradient_steps,
            "history": self.history,
            "residual_fitted": self.residual_model is not None,
        }
        (directory / "state.json").write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> Trainer:
        directory = Path(path)
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        config = dict(state["training_config"])
        if device is not None:
            config["device"] = device
        trainer = cls(ModelConfig(**state["model_config"]), TrainingConfig(**config))
        trainer.model.load_state_dict(
            torch.load(directory / "model.pt", map_location=trainer.device, weights_only=True)
        )
        optimization = torch.load(directory / "optimizer.pt", map_location=trainer.device, weights_only=True)
        trainer.optimizer.load_state_dict(optimization["optimizer"])
        trainer.scaler = TrainOnlyRobustScaler.from_dict(state["scaler"])
        trainer.memory = EngramMemory.from_dict(state["memory"])
        trainer.controller = AdaptiveController.from_dict(state["controller"])
        for name in (
            "training_cutoff",
            "fitted_through",
            "online_watermark",
            "feature_names",
            "asset_names",
            "fit_report",
            "gradient_steps",
            "history",
        ):
            setattr(trainer, name, state[name])
        if state["residual_fitted"]:
            trainer.residual_model = _xgboost_class()()
            trainer.residual_model.load_model(str(directory / "residual_xgboost.ubj"))
        trainer.fitted = trainer._fit_started = True
        trainer.model.eval()
        # Restoring a checkpoint restores its RNG stream explicitly; deterministic
        # prediction itself uses no dropout or stochastic sampling.
        torch.set_rng_state(optimization["cpu_rng"].cpu())
        if trainer.device.type == "cuda" and "cuda_rng" in optimization:
            torch.cuda.set_rng_state_all([value.cpu() for value in optimization["cuda_rng"]])
        return trainer
