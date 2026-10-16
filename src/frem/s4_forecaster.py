"""S4 recent fitted base, calibration, immutable residual reference and key metric."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .s3 import KeyMetric, apply_calibration, calibrate


def partitions(data):
    dates = np.unique(data.time)
    if len(dates) < 96:
        raise ValueError("S4 requires 96 pre-validation dates")
    dates = dates[-96:]
    bounds = [int(dates[0]), int(dates[48]), int(dates[60]), int(dates[72]), int(data.label_time.max()) + 1]
    result = []
    for lower, upper in itertools.pairwise(bounds):
        part = data.subset((data.time >= lower) & (data.time < upper) & (data.label_time < upper))
        if not len(part):
            raise ValueError("Empty purged S3 partition")
        result.append(part)
    return result


class S4Base:
    def __init__(self, name, seed):
        self.name, self.seed = name, seed
        self.fitted = False

    def fit(self, train):
        if self.fitted:
            raise RuntimeError("New fit requires a new instance")
        from .baselines import make_baseline
        from .model import ModelConfig
        from .training import Trainer, TrainingConfig

        core, selection, calibration, residual = partitions(train)
        self.train_mean = float(np.mean(core.y))
        self.constant_scale = max(1e-4, float(np.sqrt(np.mean(core.y**2))) * np.sqrt(3 / 5))
        self.predictor = None
        if self.name in {"frem", "no_jepa"}:
            self.predictor = Trainer(
                ModelConfig(
                    input_dim=train.x.shape[-1],
                    context_length=train.x.shape[1],
                    horizon=train.future_x.shape[1],
                    hidden_dim=32,
                    reservoir_dim=32,
                    spectral_bands=4,
                    memory_capacity=128,
                    memory_neighbors=4,
                    use_memory=False,
                    use_jepa=self.name != "no_jepa",
                    return_parameterization="train_standardized",
                    reservoir_seed=self.seed,
                ),
                TrainingConfig(
                    seed=self.seed,
                    epochs=5,
                    batch_size=32,
                    early_stopping_patience=3,
                    max_wall_seconds=240,
                    device="cpu",
                ),
            )
            fit = self.predictor.fit(core, selection)
        elif self.name in {"ridge", "xgboost"}:
            self.predictor = make_baseline(
                self.name, seed=self.seed, n_estimators=200, max_depth=3, max_wall_seconds=240, device="cpu"
            )
            fit = self.predictor.fit(core)
        elif self.name in {"zero", "historical_mean"}:
            fit = {"name": self.name, "train_rows": len(core), "df": 5, "scale": self.constant_scale}
        else:
            raise ValueError("Unknown S3 base")
        self.calibration = calibrate(self.raw(calibration))
        self.residual_data = residual
        self.residual_frame = apply_calibration(self.raw(residual), self.calibration)
        self.descriptor_center = core.x.astype(np.float64).mean(axis=(0, 1))
        self.descriptor_unit = np.maximum(core.x.astype(np.float64).std(axis=(0, 1)), 1e-6)
        descriptor = self.descriptors(residual)
        z = (
            (self.residual_frame.actual - self.residual_frame.prediction) / self.residual_frame.scale
        ).to_numpy()
        self.metric = KeyMetric().fit(descriptor, z, residual.label_time, int(residual.label_time.max()) + 1)
        self.fitted_through = int(residual.label_time.max())
        self.fit_report = {
            "base": fit,
            "calibration": self.calibration,
            "partition_rows": [len(p) for p in (core, selection, calibration, residual)],
            "partition_max_label": [
                int(p.label_time.max()) for p in (core, selection, calibration, residual)
            ],
            "base_frozen_before_residuals": True,
            "metric": self.metric.to_dict(),
        }
        self.fitted = True
        return self.fit_report

    def raw(self, data):
        if self.predictor is not None:
            return self.predictor.predict(data)
        from .training import prediction_frame

        mean = 0.0 if self.name == "zero" else self.train_mean
        return prediction_frame(data, np.full(len(data), mean), np.full(len(data), self.constant_scale), 5.0)

    def predict(self, data):
        if not self.fitted or int(data.time.min()) <= self.fitted_through:
            raise ValueError("Prediction must follow all fitted outcomes")
        return apply_calibration(self.raw(data), self.calibration)

    def descriptors(self, data):
        x = (data.x.astype(np.float64) - self.descriptor_center) / self.descriptor_unit
        return np.concatenate((x[:, -1], x.mean(axis=1), x.std(axis=1)), axis=1)

    def keys(self, data, learned=False):
        if learned:
            return self.metric.transform(self.descriptors(data))
        if self.name in {"frem", "no_jepa"}:
            import torch

            with torch.no_grad():
                x = self.predictor.scaler.transform_x(data.x)
                keys, _ = self.predictor.model.make_keys(self.predictor._tensor(x))
                return keys.cpu().numpy()
        return self.descriptors(data)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        if self.predictor is not None:
            self.predictor.save(directory / "base")
        state = {
            "name": self.name,
            "seed": self.seed,
            "fit_report": self.fit_report,
            "descriptor_center": self.descriptor_center.tolist(),
            "descriptor_unit": self.descriptor_unit.tolist(),
            "train_mean": self.train_mean,
            "constant_scale": self.constant_scale,
            "fitted_through": self.fitted_through,
        }
        (directory / "s4.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        return hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()

    @classmethod
    def load(cls, directory, *, trust_joblib=False):
        from .baselines import load_baseline
        from .training import Trainer

        directory = Path(directory)
        state = json.loads((directory / "s4.json").read_text())
        result = cls(state["name"], state["seed"])
        result.predictor = (
            Trainer.load(directory / "base", device="cpu")
            if result.name in {"frem", "no_jepa"}
            else load_baseline(directory / "base", trust_joblib=trust_joblib)
            if result.name in {"ridge", "xgboost"}
            else None
        )
        result.fit_report = state["fit_report"]
        result.calibration = state["fit_report"]["calibration"]
        for field in ("train_mean", "constant_scale", "fitted_through"):
            setattr(result, field, state[field])
        for field in ("descriptor_center", "descriptor_unit"):
            setattr(result, field, np.asarray(state[field]))
        result.metric = KeyMetric()
        for field in ("center", "unit", "coef"):
            setattr(result.metric, field, np.asarray(state["fit_report"]["metric"][field]))
        result.metric.fitted_through = state["fit_report"]["metric"]["fitted_through"]
        result.fitted = True
        return result
