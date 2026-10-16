"""FREM-S2: frozen-reference residuals and validation log-score calibration.

This additive development successor makes no novelty or performance claim.
See research/frem_rebuild_20261011/specification.json for its declared contract.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from .calibration_diagnostics import calibration_diagnostics
from .data import WindowData
from .memory import EngramMemory
from .model import ModelConfig
from .training import Trainer, TrainingConfig, date_balanced_mean, time_groups

MEMORY_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
SCALE_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)


def residual_partitions(data: WindowData) -> tuple[WindowData, WindowData, WindowData]:
    """60/20/20 decision-date split, purging labels against each next start."""
    dates = np.unique(data.time)
    if len(dates) < 15:
        raise ValueError("Frozen residual fitting requires at least 15 decision timestamps")
    first, second = dates[int(0.6 * len(dates))], dates[int(0.8 * len(dates))]
    masks = (
        (data.time < first) & (data.label_time < first),
        (data.time >= first) & (data.time < second) & (data.label_time < second),
        data.time >= second,
    )
    parts = tuple(data.subset(mask) for mask in masks)
    if any(len(part) == 0 for part in parts):
        raise ValueError("Purging left an empty base-fit, checkpoint-selection or residual partition")
    return parts


def choose_calibration(frame: pd.DataFrame, residual: np.ndarray, *, use_memory: bool) -> dict:
    """Finite declared search on validation, with deterministic conservative ties."""
    if residual.shape != (len(frame),) or not np.isfinite(residual).all():
        raise ValueError("Calibration residuals must be aligned and finite")
    candidates = []
    for weight in MEMORY_WEIGHTS if use_memory else (0.0,):
        for multiplier in SCALE_MULTIPLIERS:
            losses = -student_t.logpdf(
                frame.actual.to_numpy(),
                df=frame.student_df.to_numpy(),
                loc=frame.prediction.to_numpy() + weight * residual,
                scale=frame.scale.to_numpy() * multiplier,
            )
            score = date_balanced_mean(losses, frame.time.to_numpy(), frame.asset_id.to_numpy())
            candidates.append({"weight": weight, "scale_multiplier": multiplier, "validation_nll": score})
    selected = min(
        candidates,
        key=lambda row: (row["validation_nll"], row["weight"], abs(math.log(row["scale_multiplier"]))),
    )
    return {"selected": dict(selected), "candidates": candidates}


class FrozenResidualForecaster:
    """One selected base checkpoint, immutable heldout memory, two calibrated views."""

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig) -> None:
        if model_config.residual_xgboost or not model_config.use_stochastic:
            raise ValueError("Frozen residual successor requires Student-t output and no tree hybrid")
        self.base = Trainer(
            replace(model_config, use_memory=False, return_parameterization="train_standardized"),
            training_config,
        )
        self.memory = EngramMemory(
            key_dim=self.base.model.key_dim,
            capacity=model_config.memory_capacity,
            neighbors=model_config.memory_neighbors,
            same_asset=model_config.memory_same_asset,
            exact_ngram=model_config.memory_exact_ngram,
            retention="fifo",
            half_life_days=training_config.memory_half_life_days,
            temperature=training_config.memory_temperature,
            pending_capacity=training_config.pending_capacity,
            seed=training_config.seed,
        )
        self.fitted = False
        self._fit_started = False
        self.fitted_through: int | None = None
        self.fit_report: dict = {}
        self.calibration: dict = {}
        self._started_at: float | None = None

    def _check_budget(self) -> None:
        limit = self.base.training_config.max_wall_seconds
        if (
            limit is not None
            and self._started_at is not None
            and time.monotonic() - self._started_at >= limit
        ):
            raise TimeoutError("Frozen residual fit exhausted its shared wall budget")

    @torch.no_grad()
    def _keys(self, data: WindowData, indices: np.ndarray):
        contexts = self.base.scaler.transform_x(data.x[indices])
        keys, codes = self.base.model.make_keys(self.base._tensor(contexts))
        return keys.cpu().numpy(), codes.cpu().numpy()

    @torch.no_grad()
    def _residuals(self, data: WindowData, *, fitting: bool = False) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(len(data), dtype=np.float64)
        counts = np.zeros(len(data), dtype=np.int64)
        for indices in time_groups(data.time):
            if fitting:
                self._check_budget()
            keys, codes = self._keys(data, indices)
            for row, index in enumerate(indices):
                retrieval = self.memory.retrieve(
                    keys[row],
                    asset_id=int(data.asset_id[index]),
                    decision_time=int(data.time[index]),
                    ngram_code=int(codes[row]),
                )
                values[index] = float(retrieval.features[0])
                counts[index] = len(retrieval.identifiers)
        return values, counts

    def fit(self, train: WindowData, validation: WindowData) -> dict:
        if self._fit_started:
            raise RuntimeError("Create a fresh frozen residual forecaster for another fit")
        self._fit_started = True
        self._started_at = time.monotonic()
        self.base._validate_schema(train, labeled=True)
        self.base._validate_schema(validation, labeled=True)
        if int(train.label_time.max()) >= int(validation.time.min()):
            raise ValueError("Residual training labels must mature before calibration begins")
        core, selection, residual = residual_partitions(train)
        self.base.fit(core, selection)
        self._check_budget()
        forecasts = self.base.predict(residual)
        # No gradient or dropout is used after this point. No outcome is used by
        # the base fit/selection before it is predicted in the later partition.
        for indices in time_groups(residual.time):
            self._check_budget()
            self.memory.advance(int(residual.time[indices[0]]))
            keys, codes = self._keys(residual, indices)
            for row, index in enumerate(indices):
                mu = float(forecasts.prediction.iloc[index])
                self.memory.schedule(
                    key=keys[row],
                    asset_id=int(residual.asset_id[index]),
                    decision_time=int(residual.time[index]),
                    label_time=int(residual.label_time[index]),
                    base_prediction=mu,
                    prediction=mu,
                    memory_prediction=mu,
                    scale=float(forecasts.scale.iloc[index]),
                    ngram_code=int(codes[row]),
                    novelty=0.0,
                    outcome=float(residual.y[index]),
                )
        self.memory.advance(int(validation.time.min()))
        if self.memory.pending:
            raise ValueError("Residual holdout contains outcomes not matured before calibration")
        self._check_budget()
        validation_forecasts = self.base.predict(validation)
        corrections, counts = self._residuals(validation, fitting=True)
        self.calibration = {
            "memory": choose_calibration(validation_forecasts, corrections, use_memory=True),
            "no_memory": choose_calibration(validation_forecasts, corrections, use_memory=False),
        }
        self._check_budget()
        self.fitted_through = int(validation.label_time.max())
        self.fit_report = {
            "status": "fitted",
            "lineage": "FREM-S2",
            "base": self.base.fit_report,
            "partition_rows": [len(core), len(selection), len(residual)],
            "partition_max_label_time": [int(part.label_time.max()) for part in (core, selection, residual)],
            "partition_min_decision_time": [int(part.time.min()) for part in (core, selection, residual)],
            "residual_reference": "selected_frozen_base_on_later_purged_holdout_eval_mode",
            "refitted_after_residuals": False,
            "controller": "validation_student_t_nll_grid",
            "retention": "fifo_no_utility_credit",
            "memory": self.memory.diagnostics(),
            "calibration": self.calibration,
            "calibration_diagnostics": calibration_diagnostics(
                self.calibration["memory"]["candidates"],
                counts.tolist(),
                self.memory.to_dict()["records"],
                int(validation.time.min()),
            ),
            "fitted_through": self.fitted_through,
            "elapsed_seconds": time.monotonic() - self._started_at,
            "interpretation": "development fit; held-out evidence required",
        }
        self.fitted = True
        return self.fit_report

    def predict(self, data: WindowData, *, use_memory: bool = True) -> pd.DataFrame:
        if not self.fitted or self.fitted_through is None:
            raise RuntimeError("Fit or load the complete successor before prediction")
        if not len(data) or int(data.time.min()) <= self.fitted_through:
            raise ValueError("Forecasts must follow every calibration outcome")
        frame = self.base.predict(data)
        corrections, counts = self._residuals(data)
        chosen = self.calibration["memory" if use_memory else "no_memory"]["selected"]
        frame["base_prediction"] = frame.prediction.copy()
        frame["memory_residual"] = corrections
        frame["memory_neighbors"] = counts
        frame["memory_gate"] = chosen["weight"]
        frame["prediction"] = frame.prediction + chosen["weight"] * corrections
        frame["scale"] = frame.scale * chosen["scale_multiplier"]
        return frame

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an incomplete successor as a fitted checkpoint")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.base.save(path / "base")
        state = {
            "format_version": 1,
            "memory": self.memory.to_dict(),
            "calibration": self.calibration,
            "fitted_through": self.fitted_through,
            "fit_report": self.fit_report,
        }
        (path / "successor.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> FrozenResidualForecaster:
        path = Path(path)
        state = json.loads((path / "successor.json").read_text())
        if state["format_version"] != 1:
            raise ValueError("Unsupported frozen residual checkpoint format")
        base = Trainer.load(path / "base", device="cpu")
        result = cls(base.model_config, base.training_config)
        result.base = base
        result.memory = EngramMemory.from_dict(state["memory"])
        result.calibration = state["calibration"]
        result.fitted_through = state["fitted_through"]
        result.fit_report = state["fit_report"]
        result.fitted = result._fit_started = True
        return result
