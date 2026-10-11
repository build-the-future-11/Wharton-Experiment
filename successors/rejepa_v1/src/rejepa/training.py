"""Deterministic phase-one training and frozen inference utilities."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch

from .engram import SpectralEngramMemory
from .model import REJEPA, REJEPAOutput, rejepa_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    patience: int = 15
    minimum_delta: float = 1e-5
    seed: int = 11
    world_seed: int = 1009
    downside_threshold: float = 0.0
    latent_weight: float = 1.0
    growth_weight: float = 1.0
    protection_weight: float = 1.0
    diversity_floor: float = 0.05

    def __post_init__(self) -> None:
        if min(self.epochs, self.patience) < 1:
            raise ValueError("epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if self.gradient_clip_norm <= 0 or self.minimum_delta < 0:
            raise ValueError("Invalid stopping or clipping configuration")
        weights = (self.latent_weight, self.growth_weight, self.protection_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if sum(weights) <= 0:
            raise ValueError("At least one loss weight must be positive")


@dataclass(frozen=True)
class TrainingBatch:
    context: torch.Tensor
    future_target: torch.Tensor
    realized_return: torch.Tensor
    as_of_ns: int | None = None
    horizon: int | torch.Tensor | None = None

    def validate(self, input_dim: int) -> None:
        if self.context.ndim != 3 or self.future_target.ndim != 3:
            raise ValueError("Context and target must be rank-three tensors")
        if self.context.shape[0] != self.future_target.shape[0]:
            raise ValueError("Context and target batch sizes differ")
        if self.context.shape[-1] != input_dim or self.future_target.shape[-1] != input_dim:
            raise ValueError("Batch feature dimension does not match model")
        if self.realized_return.shape != (self.context.shape[0],):
            raise ValueError("realized_return must have one value per row")
        tensors = (self.context, self.future_target, self.realized_return)
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Training batch contains non-finite values")
        if self.horizon is not None:
            raw = torch.as_tensor(self.horizon)
            if raw.ndim > 1 or (raw.ndim == 1 and raw.shape != (self.context.shape[0],)):
                raise ValueError("horizon must be scalar or have one value per row")
            if not torch.isfinite(raw).all() or (raw <= 0).any():
                raise ValueError("horizon values must be finite and positive")
            if raw.is_floating_point() and not torch.equal(raw, raw.round()):
                raise ValueError("horizon values must be integers")


@dataclass
class TrainingResult:
    best_epoch: int
    completed_epochs: int
    best_validation_loss: float
    stopped_early: bool
    history: list[dict[str, float]] = field(default_factory=list)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _evaluate(
    model: REJEPA,
    batches: list[TrainingBatch],
    config: TrainingConfig,
    memory: SpectralEngramMemory | None,
) -> float:
    model.eval()
    weighted_total = 0.0
    sample_count = 0
    with torch.inference_mode():
        for index, batch in enumerate(batches):
            output = model(
                batch.context,
                future_target=batch.future_target,
                memory=memory,
                as_of_ns=batch.as_of_ns,
                generator=torch.Generator(device=batch.context.device).manual_seed(config.world_seed + index),
                horizon=batch.horizon,
            )
            losses = rejepa_loss(
                output,
                batch.realized_return,
                student_df=model.config.student_df,
                downside_threshold=config.downside_threshold,
                latent_weight=config.latent_weight,
                growth_weight=config.growth_weight,
                protection_weight=config.protection_weight,
                diversity_floor=config.diversity_floor,
            )
            value = float(losses["total"].item())
            if not math.isfinite(value):
                raise FloatingPointError("Validation loss became non-finite")
            batch_size = batch.context.shape[0]
            weighted_total += value * batch_size
            sample_count += batch_size
    return weighted_total / sample_count


def fit_model(
    model: REJEPA,
    train_batches: list[TrainingBatch],
    validation_batches: list[TrainingBatch],
    config: TrainingConfig,
    memory: SpectralEngramMemory | None = None,
) -> TrainingResult:
    """Fit a model without mutating engrams or using validation outcomes for updates."""
    if not train_batches or not validation_batches:
        raise ValueError("Training and validation each require at least one batch")
    for batch in [*train_batches, *validation_batches]:
        batch.validate(model.config.input_dim)
        if memory is not None and batch.as_of_ns is None:
            raise ValueError("Every memory-backed batch requires as_of_ns")
    seed_everything(config.seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        weighted_training_total = 0.0
        training_sample_count = 0
        for index, batch in enumerate(train_batches):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch.context,
                future_target=batch.future_target,
                memory=memory,
                as_of_ns=batch.as_of_ns,
                generator=torch.Generator(device=batch.context.device).manual_seed(
                    config.world_seed + epoch * len(train_batches) + index
                ),
                horizon=batch.horizon,
            )
            losses = rejepa_loss(
                output,
                batch.realized_return,
                student_df=model.config.student_df,
                downside_threshold=config.downside_threshold,
                latent_weight=config.latent_weight,
                growth_weight=config.growth_weight,
                protection_weight=config.protection_weight,
                diversity_floor=config.diversity_floor,
            )
            total = losses["total"]
            if not torch.isfinite(total):
                raise FloatingPointError("Training loss became non-finite")
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Gradient norm became non-finite")
            optimizer.step()
            model.update_target_encoder()
            batch_size = batch.context.shape[0]
            weighted_training_total += float(total.detach().item()) * batch_size
            training_sample_count += batch_size
        validation_loss = _evaluate(model, validation_batches, config, memory)
        training_loss = weighted_training_total / training_sample_count
        history.append(
            {"epoch": float(epoch), "training_loss": training_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - config.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = deepcopy(model.state_dict())
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        best_epoch=best_epoch,
        completed_epochs=len(history),
        best_validation_loss=best_loss,
        stopped_early=len(history) < config.epochs,
        history=history,
    )


def predict_frozen(
    model: REJEPA,
    context: torch.Tensor,
    *,
    seed: int,
    memory: SpectralEngramMemory | None = None,
    as_of_ns: int | None = None,
    horizon: int | torch.Tensor | None = None,
) -> REJEPAOutput:
    """Run inference with a declared seed and prove the checkpoint did not adapt."""
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    with torch.inference_mode():
        output = model(
            context,
            memory=memory,
            as_of_ns=as_of_ns,
            generator=torch.Generator(device=context.device).manual_seed(seed),
            horizon=horizon,
        )
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items()):
        raise RuntimeError("Frozen inference mutated model state")
    return output
