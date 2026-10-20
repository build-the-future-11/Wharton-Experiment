"""Reservoir + spectral engram + EMA JEPA + multiobjective expert routing."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """Masked fixed reservoirs have a Euclidean contraction bound below one."""

    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        g = torch.Generator().manual_seed(seed)
        matrices = []
        for _ in config.leaks:
            w = torch.randn(config.reservoir, config.reservoir, generator=g)
            w *= config.contraction / torch.linalg.matrix_norm(w, ord=2)
            matrices.append(w)
        self.register_buffer("recurrent", torch.stack(matrices))
        self.register_buffer(
            "incoming",
            torch.randn(len(config.leaks), config.reservoir, config.input_dim, generator=g)
            / math.sqrt(config.input_dim),
        )
        self.register_buffer(
            "active",
            torch.arange(config.reservoir)
            < (config.reservoir if config.arm == "fixed_full_capacity" else config.initial_neurons),
        )
        self.register_buffer("leaks", torch.tensor(config.leaks).view(1, -1, 1))
        width = (
            config.input_dim * (3 + config.bands)
            + config.ngram
            + len(config.leaks) * config.reservoir
        )
        self.projection = nn.Sequential(
            nn.Linear(width, 2 * config.latent),
            nn.LayerNorm(2 * config.latent),
            nn.SiLU(),
            nn.Linear(2 * config.latent, config.latent),
        )

    def components(self, x):
        # FFT and recurrent states deliberately remain float32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            spectrum = torch.fft.rfft(x - x.mean(1, keepdim=True), dim=1, norm="ortho")
            bands = torch.stack(
                [
                    part.abs().square().mean(1).log1p()
                    for part in torch.tensor_split(spectrum[:, 1:], self.config.bands, dim=1)
                ],
                1,
            )
            if self.config.arm == "no_spectral":
                bands = bands * 0
            ngram = x[:, -self.config.ngram :, 0].sign() * math.sqrt(self.config.ngram_bonus)
            if self.config.arm == "no_ngram":
                ngram = ngram * 0
            fixed = torch.cat(
                [x[:, -1], x.mean(1), x.std(1, correction=0), bands.flatten(1), ngram], 1
            )
            state = x.new_zeros(x.shape[0], len(self.config.leaks), self.config.reservoir)
            if self.config.arm != "no_reservoir":
                mask = self.active.to(x.dtype)
                for step in x.unbind(1):
                    activation = torch.einsum("bki,kji->bkj", state, self.recurrent)
                    activation = activation + torch.einsum("bi,kji->bkj", step, self.incoming)
                    state = ((1 - self.leaks) * state + self.leaks * torch.tanh(activation)) * mask
            return fixed, state.flatten(1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.config.context, self.config.input_dim):
            raise ValueError("context tensor shape differs from frozen model configuration")
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError("context must contain finite floating-point features")
        fixed, state = self.components(x)
        return self.projection(torch.cat([fixed, state], 1)), fixed

    @torch.no_grad()
    def grow(self):
        before = int(self.active.sum())
        after = min(before + self.config.growth_step, self.config.reservoir)
        self.active[:after] = True
        return before, after


class Engram(nn.Module):
    """Bounded FIFO bank with fixed coordinates and strict outcome maturity."""

    def __init__(self, config, key_width):
        super().__init__()
        self.config = config
        c = config.memory_capacity
        self.register_buffer("keys", torch.zeros(c, key_width))
        self.register_buffer("values", torch.zeros(c))
        self.register_buffer("ends", torch.zeros(c, dtype=torch.int64))
        self.register_buffer("assets", torch.zeros(c, dtype=torch.int64))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def reset(self):
        self.count.zero_()
        self.cursor.zero_()

    @torch.no_grad()
    def append(self, keys, values, ends, assets):
        if not (len(keys) == len(values) == len(ends) == len(assets)):
            raise ValueError("memory record dimensions differ")
        if (
            keys.ndim != 2
            or keys.shape[1] != self.keys.shape[1]
            or values.ndim != 1
            or ends.ndim != 1
            or assets.ndim != 1
            or ends.dtype != torch.int64
            or assets.dtype != torch.int64
        ):
            raise ValueError("memory record types/shapes differ from the contract")
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError("nonfinite memory record")
        for key, value, end, asset in zip(keys, values, ends, assets, strict=True):
            slot = int(self.cursor)
            self.keys[slot] = F.normalize(key.float(), dim=0)
            self.values[slot], self.ends[slot], self.assets[slot] = value, end, asset
            self.cursor.fill_((slot + 1) % self.config.memory_capacity)
            self.count.fill_(min(int(self.count) + 1, self.config.memory_capacity))

    def forward(self, keys, times, assets):
        count = int(self.count)
        if count == 0 or self.config.arm == "no_memory":
            return keys.new_zeros(len(keys), 3)
        similarity = F.normalize(keys.float(), dim=1) @ self.keys[:count].T
        valid = (self.ends[:count][None] < times[:, None]) & (
            self.assets[:count][None] == assets[:, None]
        )
        # Timestamp ages are measured in days, not rows/asset count.
        age = (times[:, None] - self.ends[:count][None]).clamp_min(0).float() / 86400000000
        logits = (
            similarity / self.config.memory_temperature
            - math.log(2) * age / self.config.memory_half_life
        )
        logits = logits.masked_fill(~valid, -1e30)
        top, indices = logits.topk(min(self.config.neighbors, count), dim=1)
        selected_valid = valid.gather(1, indices)
        weight = torch.softmax(top, 1) * selected_valid
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-12)
        values = self.values[:count][indices]
        mean = (weight * values).sum(1)
        variance = (weight * (values - mean[:, None]).square()).sum(1)
        return torch.stack(
            [mean, (variance + 1e-8).sqrt(), selected_valid.sum(1) / self.config.neighbors], 1
        )


class S13(nn.Module):
    def __init__(self, config, seed=11):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config, seed)
        self.target = deepcopy(self.encoder)
        self.target.requires_grad_(False)
        key_width = config.input_dim * (3 + config.bands) + config.ngram
        self.memory = Engram(config, key_width)
        self.predictor = nn.Sequential(
            nn.Linear(config.latent, config.latent),
            nn.SiLU(),
            nn.Linear(config.latent, config.latent),
        )
        self.expert = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.latent + 3, config.latent),
                    nn.SiLU(),
                    nn.Linear(config.latent, 4),
                )
                for _ in range(config.experts)
            ]
        )
        # Separate gates for return likelihood, downside probability and tail quantile.
        self.router = nn.Linear(config.latent + 3, 3 * config.experts)

    @torch.no_grad()
    def ema_update(self):
        for target, online in zip(self.target.parameters(), self.encoder.parameters(), strict=True):
            target.lerp_(online, 1 - self.config.ema)
        for target, online in zip(self.target.buffers(), self.encoder.buffers(), strict=True):
            target.copy_(online)

    def forward(self, x, times, assets):
        if (
            times.shape != (len(x),)
            or assets.shape != (len(x),)
            or times.dtype != torch.int64
            or assets.dtype != torch.int64
        ):
            raise ValueError("decision timestamps and asset IDs must be int64 vectors")
        z, key = self.encoder(x)
        memory = self.memory(key.detach(), times, assets).to(z.dtype)
        fused = torch.cat([z, memory], 1)
        parameters = torch.stack([expert(fused) for expert in self.expert], 1).float()
        mu = parameters[..., 0] + memory[:, :1].float()
        scale = F.softplus(parameters[..., 1]) + 1e-3
        p_loss = torch.sigmoid(parameters[..., 2])
        tail = mu - F.softplus(parameters[..., 3])
        weights = self.router(fused).float().reshape(-1, 3, self.config.experts).softmax(-1)
        alpha = (self.config.router_cap - 1 / self.config.experts) / (1 - 1 / self.config.experts)
        weights = alpha * weights + (1 - alpha) / self.config.experts
        if self.config.arm == "uniform_router":
            weights = weights * 0 + 1 / self.config.experts
        mean = (weights[:, 0] * mu).sum(-1)
        variance = (
            weights[:, 0]
            * (scale.square() * self.config.student_df / (self.config.student_df - 2) + mu.square())
        ).sum(-1) - mean.square()
        return {
            "z": z.float(),
            "predicted": self.predictor(z).float(),
            "mu": mu,
            "scale": scale,
            "weights": weights,
            "mean": mean,
            "std": variance.clamp_min(1e-8).sqrt(),
            "p_loss": (weights[:, 1] * p_loss).sum(-1),
            "tail": (weights[:, 2] * tail).sum(-1),
            "memory": memory.float(),
        }

    @torch.no_grad()
    def scenarios(self, output, count):
        if type(count) is not int or count < 1:
            raise ValueError("positive scenario count required")
        indices = torch.multinomial(output["weights"][:, 0], count, replacement=True)
        mu, scale = output["mu"].gather(1, indices), output["scale"].gather(1, indices)
        noise = torch.distributions.StudentT(self.config.student_df).sample(mu.shape).to(mu.device)
        return mu + scale * noise


def nll(output, y, df):
    law = torch.distributions.StudentT(df, output["mu"], output["scale"])
    return -torch.logsumexp(output["weights"][:, 0].log() + law.log_prob(y[:, None]), 1)


def objective(model, output, y, future, run_config):
    predictive = nll(output, y, model.config.student_df).mean()
    risk = (output["p_loss"] - (y < 0).float()).square().mean()
    residual = y - output["tail"]
    risk = risk + torch.maximum(0.05 * residual, -0.95 * residual).mean()
    jepa = predictive.new_zeros(())
    if model.config.arm != "no_jepa":
        with torch.no_grad():
            target, _ = model.target(future)
        z = output["z"]
        jepa = F.mse_loss(output["predicted"], target.float())
        if len(z) > 1:
            centered = z - z.mean(0)
            covariance = centered.T @ centered / (len(z) - 1)
            variance = F.relu(1 - torch.sqrt(z.var(0, correction=1) + 1e-4)).mean()
            off = covariance - torch.diag_embed(covariance.diagonal())
            jepa = jepa + variance + off.square().sum() / z.shape[1]
    balance = (output["weights"].mean(0) - 1 / model.config.experts).square().mean()
    total = predictive + run_config.jepa_weight * jepa + run_config.balance_weight * balance
    if model.config.arm != "no_risk":
        total = total + run_config.risk_weight * risk
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite objective")
    return total, {
        "nll": float(predictive.detach()),
        "jepa": float(jepa.detach()),
        "risk": float(risk.detach()),
        "balance": float(balance.detach()),
    }
