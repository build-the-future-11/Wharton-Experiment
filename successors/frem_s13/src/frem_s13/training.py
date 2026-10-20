"""Chronological minibatches, finite gradients, early stopping and epoch resume."""

import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import S13, nll, objective
from .runtime import digest, write_json


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def budget(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("declared campaign wall-clock budget exhausted")


def tensor(value, device, dtype=None):
    return torch.as_tensor(value, dtype=dtype, device=device)


class GRUForecast(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(config.input_dim, config.latent, batch_first=True)
        self.head = nn.Linear(config.latent, 2)

    def forward(self, x, times, assets):
        _, hidden = self.gru(x)
        raw = self.head(hidden[-1]).float()
        mu, scale = raw[:, :1], F.softplus(raw[:, 1:]) + 1e-3
        return {
            "mu": mu,
            "scale": scale,
            "weights": torch.ones(len(x), 3, 1, device=x.device),
            "mean": mu[:, 0],
            "std": scale[:, 0] * (self.config.student_df / (self.config.student_df - 2)) ** 0.5,
        }


@torch.no_grad()
def populate(model, samples, selected, device):
    if not isinstance(model, S13) or model.config.arm == "no_memory":
        return
    for ix in samples.batches(selected):
        x, _ = samples.batch(ix)
        # Keys are fixed spectral/statistical coordinates, independent of learned parameters.
        fixed, _ = model.encoder.components(tensor(x, device))
        model.memory.append(
            fixed,
            tensor(samples.y[ix] / samples.target_scale, device, torch.float32),
            tensor(samples.end[ix], device),
            tensor(samples.asset[ix], device),
        )


@torch.no_grad()
def predict(model, samples, indices, device, deadline=float("inf")):
    model.eval()
    results = {}
    for ix in samples.batches(indices):
        budget(deadline)
        x, _ = samples.batch(ix)
        output = model(
            tensor(x, device), tensor(samples.time[ix], device), tensor(samples.asset[ix], device)
        )
        loss = nll(
            output,
            tensor(samples.y[ix] / samples.target_scale, device, torch.float32),
            model.config.student_df,
        )
        for name in ("mu", "scale", "weights", "mean", "std"):
            results.setdefault(name, []).append(output[name].cpu().numpy())
        results.setdefault("nll", []).append(loss.cpu().numpy() + np.log(samples.target_scale))
    out = {key: np.concatenate(value) for key, value in results.items()}
    for name in ("mu", "scale", "mean", "std"):
        out[name] *= samples.target_scale
    if any(not np.isfinite(value).all() for value in out.values()):
        raise FloatingPointError("nonfinite prediction")
    return out


def save_checkpoint(path, payload):
    path = Path(path)
    if path.exists() or path.with_suffix(".sha256.json").exists():
        raise FileExistsError("immutable checkpoint already exists")
    # Exclusive temp file + atomic publication; an interrupted temp is retained as evidence.
    temp = path.with_suffix(".pending")
    with temp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)
    temp.unlink()
    write_json(path.with_suffix(".sha256.json"), {"sha256": digest(path)})


def load_checkpoint(path, binding):
    path = Path(path)
    if json.loads(path.with_suffix(".sha256.json").read_text())["sha256"] != digest(path):
        raise ValueError("checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["binding"] != binding:
        raise ValueError("resume source/data/config/runtime/seed/arm differs")
    return state


def train_neural(samples, config, arm, seed, directory, deadline, binding, resume=None):
    seed_all(seed, config.device)
    cfg = replace(config.model, arm=arm)
    model = (GRUForecast(cfg) if arm == "gru" else S13(cfg, seed)).to(config.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    best, stale, first, prior_mse = float("inf"), 0, 0, 0.0
    best_state, history = None, []
    if resume:
        state = load_checkpoint(resume, binding)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        best, stale, first, prior_mse = (
            state["best"],
            state["stale"],
            state["epoch"] + 1,
            state["prior_mse"],
        )
        best_state, history = state["best_state"], state["history"]
    for epoch in range(first, config.epochs):
        if stale >= config.patience:
            break
        budget(deadline)
        growth = None
        if (
            isinstance(model, S13)
            and arm not in ("no_neurogenesis", "no_reservoir")
            and epoch > 0
            and epoch % cfg.growth_every == 0
            and prior_mse > cfg.growth_threshold
        ):
            growth = model.encoder.grow()
            model.target.active.copy_(model.encoder.active)
        model.train()
        if isinstance(model, S13):
            model.memory.reset()
        written = np.zeros(len(samples.train), dtype=bool)
        sum_loss, squared, count = 0.0, 0.0, 0
        for ix in samples.batches(samples.train):
            budget(deadline)
            eligible = (samples.end[samples.train] < samples.time[ix].min()) & ~written
            populate(model, samples, samples.train[eligible], config.device)
            written |= eligible
            x, future = samples.batch(ix, future=arm not in ("gru", "no_jepa"))
            optimizer.zero_grad(set_to_none=True)
            y = tensor(samples.y[ix] / samples.target_scale, config.device, torch.float32)
            with torch.autocast(device_type=config.device, dtype=torch.float16, enabled=config.amp):
                output = model(
                    tensor(x, config.device),
                    tensor(samples.time[ix], config.device),
                    tensor(samples.asset[ix], config.device),
                )
                if arm == "gru":
                    loss = nll(output, y, cfg.student_df).mean()
                else:
                    loss, _ = objective(
                        model,
                        output,
                        y,
                        tensor(future, config.device) if future is not None else None,
                        config,
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            if isinstance(model, S13):
                model.ema_update()
            sum_loss += float(loss.detach()) * len(ix)
            squared += float((output["mean"].detach() - y).square().sum())
            count += len(ix)
        # Complete the training-only bank before validation; never write validation/test labels.
        populate(model, samples, samples.train[~written], config.device)
        validation = predict(model, samples, samples.validation, config.device, deadline)
        score = float(validation["nll"].mean())
        prior_mse = squared / count
        if score < best:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "loss": sum_loss / count,
                "validation_nll": score,
                "training_mse": prior_mse,
                "growth": growth,
            }
        )
        payload = {
            "binding": binding,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
            "best": best,
            "stale": stale,
            "epoch": epoch,
            "prior_mse": prior_mse,
            "best_state": best_state,
            "history": history,
            "preprocessing": {
                "center": tensor(samples.center, "cpu"),
                "scale": tensor(samples.scale, "cpu"),
                "target_scale": samples.target_scale,
            },
        }
        save_checkpoint(Path(directory) / f"epoch_{epoch:03d}.pt", payload)
    if best_state is None:
        raise RuntimeError("no valid epoch completed")
    model.load_state_dict(best_state)
    write_json(
        Path(directory) / "training.json",
        {
            "history": history,
            "best_validation_nll": best,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
    )
    return model
