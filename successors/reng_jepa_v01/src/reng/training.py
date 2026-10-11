"""Chronological Stage A/Stage B training and immutable memory construction."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .memory import EngramBank, EngramRecord
from .model import RENGJEPA, stage_a_loss, stage_b_loss


@dataclass
class TrainingHistory:
    stage: str
    records: list[dict[str, float | int | str]] = field(default_factory=list)
    best_epoch: int | None = None
    stop_reason: str | None = None
    elapsed_seconds: float = 0.0


class WarmupCosine:
    def __init__(
        self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float = 0.05
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup = max(1, round(self.total_steps * warmup_fraction))
        self.base = [group["lr"] for group in optimizer.param_groups]
        self.step_number = 0

    def step(self) -> None:
        self.step_number += 1
        if self.step_number <= self.warmup:
            multiplier = self.step_number / self.warmup
        else:
            progress = (self.step_number - self.warmup) / max(self.total_steps - self.warmup, 1)
            multiplier = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for group, base in zip(self.optimizer.param_groups, self.base, strict=True):
            group["lr"] = base * multiplier


def _finite_or_fail(loss: torch.Tensor, model: torch.nn.Module) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Non-finite gradient")


def fit_stage_a(
    model: RENGJEPA,
    batches: list[dict[str, torch.Tensor | int]],
    validation_batches: list[dict[str, torch.Tensor | int]],
    *,
    epochs: int = 60,
    patience: int = 10,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    wall_seconds: float | None = None,
) -> TrainingHistory:
    if not batches or not validation_batches:
        raise ValueError("Stage A requires fitting and stopping blocks")
    parameters = list(model.backbone.parameters()) + list(model.stage_a_heads.parameters())
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = WarmupCosine(optimizer, epochs * len(batches))
    history = TrainingHistory("stage_a")
    start = time.monotonic()
    best_loss = float("inf")
    best_state = None
    stale = 0
    updates = max(epochs * len(batches), 1)
    update = 0
    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            output = model.stage_a(
                batch["context"], batch["future"], batch["asset_mask"], int(batch["horizon"])
            )
            losses = stage_a_loss(
                output,
                batch["simple_return"] / batch["scale"],
                use_jepa=model.config.use_jepa_objective,
            )
            losses["total"].backward()
            _finite_or_fail(losses["total"], model)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            momentum = 0.99 + (0.999 - 0.99) * min(update / max(updates - 1, 1), 1.0)
            model.backbone.update_target(momentum)
            update += 1
            training_loss += float(losses["total"].detach())
            if wall_seconds is not None and time.monotonic() - start > wall_seconds:
                history.stop_reason = "wall_time"
                history.elapsed_seconds = time.monotonic() - start
                return history
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for batch in validation_batches:
                output = model.stage_a(
                    batch["context"], batch["future"], batch["asset_mask"], int(batch["horizon"])
                )
                validation_loss += float(
                    stage_a_loss(output, batch["simple_return"] / batch["scale"])["total"]
                )
        validation_loss /= len(validation_batches)
        history.records.append(
            {
                "epoch": epoch,
                "train_loss": training_loss / len(batches),
                "validation_loss": validation_loss,
            }
        )
        relative = abs(validation_loss - best_loss) / max(abs(best_loss), 1.0)
        if validation_loss < best_loss and (not math.isfinite(best_loss) or relative > 1e-4):
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            history.stop_reason = "early_stopping"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history.stop_reason = history.stop_reason or "max_epochs"
    history.elapsed_seconds = time.monotonic() - start
    return history


def fit_target_standardizer(
    targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    values = targets[mask]
    if len(values) < 2:
        raise ValueError("Target standardization requires at least two valid rows")
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-3)
    return mean.detach(), scale.detach()


def build_engram_bank(
    model: RENGJEPA,
    batches: list[dict[str, object]],
    *,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    freeze: bool = True,
) -> EngramBank:
    bank = EngramBank(max_records=model.config.memory_max_records)
    model.eval()
    with torch.no_grad():
        for batch in batches:
            horizon = int(batch["horizon"])
            output = model.stage_a(batch["context"], batch["future"], batch["asset_mask"], horizon)
            features = output.features
            standardized_target = (features.target_latent - target_mean) / target_scale
            outcome = batch["log_return"] / batch["scale"]
            volatility = batch["realized_volatility"] / batch["scale"]
            drawdown = batch["worst_interim_return"] / batch["scale"]
            value = torch.cat(
                [
                    standardized_target,
                    outcome.unsqueeze(-1),
                    volatility.unsqueeze(-1),
                    drawdown.unsqueeze(-1),
                ],
                dim=-1,
            )
            asset_ids = tuple(batch["asset_ids"])
            for b in range(len(features.base_context)):
                if b % model.config.memory_stride != 0:
                    continue
                for asset_index, asset_id in enumerate(asset_ids):
                    if not bool(features.asset_mask[b, asset_index]):
                        continue
                    origin_ns = int(batch["decision_ns"][b])
                    available_ns = int(batch["outcome_available_ns"][b, asset_index])
                    for branch in range(3):
                        record_id = hashlib.sha256(
                            f"{asset_id}|{horizon}|{branch}|{origin_ns}|{available_ns}".encode()
                        ).hexdigest()[:20]
                        bank.insert(
                            EngramRecord(
                                asset_id=asset_id,
                                horizon=horizon,
                                branch=branch,
                                origin_ns=origin_ns,
                                available_ns=available_ns,
                                context=features.base_context[b, asset_index, branch]
                                .detach()
                                .cpu(),
                                spectrum=features.spectrum[b, asset_index].detach().cpu(),
                                value=value[b, asset_index].detach().cpu(),
                                record_id=record_id,
                            )
                        )
    if freeze:
        bank.freeze()
    return bank


def fit_stage_b_member(
    model: RENGJEPA,
    member_index: int,
    batches: list[dict[str, object]],
    validation_batches: list[dict[str, object]],
    bank: EngramBank,
    *,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    epochs: int = 80,
    patience: int = 10,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    wall_seconds: float | None = None,
) -> TrainingHistory:
    if member_index >= len(model.members):
        raise IndexError("Unknown ensemble member")
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    member = model.members[member_index]
    optimizer = torch.optim.AdamW(
        member.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = WarmupCosine(optimizer, epochs * len(batches))
    history = TrainingHistory(f"stage_b_member_{member_index}")
    start = time.monotonic()
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        member.train()
        train_loss = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = model.backbone(
                    batch["context"], batch["future"], batch["asset_mask"], int(batch["horizon"])
                )
            output = member(
                features,
                batch["scale"],
                tuple(batch["asset_ids"]),
                batch["decision_ns"],
                int(batch["horizon"]),
                bank,
            )
            target = (features.target_latent - target_mean) / target_scale
            loss = stage_b_loss(
                output, target, batch["log_return"], batch["simple_return"], batch["asset_mask"]
            )
            loss["total"].backward()
            _finite_or_fail(loss["total"], member)
            torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
            optimizer.step()
            member.project_decoder()
            scheduler.step()
            train_loss += float(loss["total"].detach())
            if wall_seconds is not None and time.monotonic() - start > wall_seconds:
                history.stop_reason = "wall_time"
                history.elapsed_seconds = time.monotonic() - start
                return history
        member.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_batches:
                features = model.backbone(
                    batch["context"], batch["future"], batch["asset_mask"], int(batch["horizon"])
                )
                output = member(
                    features,
                    batch["scale"],
                    tuple(batch["asset_ids"]),
                    batch["decision_ns"],
                    int(batch["horizon"]),
                    bank,
                )
                target = (features.target_latent - target_mean) / target_scale
                validation_loss += float(
                    stage_b_loss(
                        output,
                        target,
                        batch["log_return"],
                        batch["simple_return"],
                        batch["asset_mask"],
                    )["total"]
                )
        validation_loss /= len(validation_batches)
        history.records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(batches),
                "validation_loss": validation_loss,
            }
        )
        relative = abs(validation_loss - best_loss) / max(abs(best_loss), 1.0)
        if validation_loss < best_loss and (not math.isfinite(best_loss) or relative > 1e-4):
            best_loss, best_state, history.best_epoch, stale = (
                validation_loss,
                copy.deepcopy(member.state_dict()),
                epoch,
                0,
            )
        else:
            stale += 1
        if stale >= patience:
            history.stop_reason = "early_stopping"
            break
    if best_state is not None:
        member.load_state_dict(best_state)
    history.stop_reason = history.stop_reason or "max_epochs"
    history.elapsed_seconds = time.monotonic() - start
    return history


def fitting_date_block_bootstrap(dates: np.ndarray, expected_block: int, seed: int) -> np.ndarray:
    unique = np.unique(np.asarray(dates))
    if len(unique) < 2:
        raise ValueError("Block bootstrap requires multiple fitting dates")
    rng = np.random.default_rng(seed)
    output = []
    while len(output) < len(unique):
        start = int(rng.integers(len(unique)))
        length = int(rng.geometric(1.0 / expected_block))
        output.extend(unique[(start + offset) % len(unique)] for offset in range(length))
    return np.asarray(output[: len(unique)])
