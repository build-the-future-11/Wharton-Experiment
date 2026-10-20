"""Reservoir + spectral engram + EMA JEPA + multiobjective expert routing."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics import StochasticField


class Encoder(nn.Module):
    """Masked fixed reservoirs have a Euclidean contraction bound below one."""

    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        g = torch.Generator().manual_seed(seed)
        matrices = []
        for _ in config.leaks:
            w = torch.randn(config.reservoir, config.reservoir, generator=g, dtype=torch.float64)
            # Normalize in float64, then reserve a dimension-scaled float32 rounding margin.
            # The declared contraction bound and its acceptance threshold remain unchanged.
            margin = 1 + 32 * config.reservoir * torch.finfo(torch.float32).eps
            w *= config.contraction / (torch.linalg.matrix_norm(w, ord=2) * margin)
            matrices.append(w.float())
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
        self.register_buffer("growth_cross", torch.zeros(config.reservoir))
        self.register_buffer("growth_sum", torch.zeros(config.reservoir))
        self.register_buffer("growth_square", torch.zeros(config.reservoir))
        self.register_buffer("growth_residual", torch.zeros(2))
        self.register_buffer("growth_count", torch.zeros(()))
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

    def components(self, x, fixed_only=False):
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
            if fixed_only:
                return fixed, None
            state = x.new_zeros(x.shape[0], len(self.config.leaks), self.config.reservoir)
            self.proposals = x.new_zeros(x.shape[0], self.config.reservoir)
            if self.config.arm != "no_reservoir":
                mask = self.active.to(x.dtype)
                # Input projection is one batched kernel rather than one per time step.
                projected = torch.einsum("bti,kji->btkj", x, self.incoming)
                for step in projected.unbind(1):
                    activation = torch.einsum("bki,kji->bkj", state, self.recurrent)
                    activation = activation + step
                    proposed = torch.tanh(activation)
                    state = ((1 - self.leaks) * state + self.leaks * proposed) * mask
                self.proposals = proposed.mean(1).detach()
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
        n = self.growth_count.clamp_min(1)
        covariance = self.growth_cross - self.growth_sum * self.growth_residual[0] / n
        variance = (self.growth_square - self.growth_sum.square() / n).clamp_min(0)
        residual_variance = (
            self.growth_residual[1] - self.growth_residual[0].square() / n
        ).clamp_min(0)
        utility = covariance.abs() / (variance * residual_variance).sqrt().clamp_min(1e-8)
        utility = utility.masked_fill(self.active, -1)
        selected = torch.argsort(utility, descending=True, stable=True)[: after - before]
        self.active[selected] = True
        for value in (
            self.growth_cross,
            self.growth_sum,
            self.growth_square,
            self.growth_residual,
            self.growth_count,
        ):
            value.zero_()
        return before, after

    @torch.no_grad()
    def observe_growth(self, proposals, residual):
        if not self.training:
            raise ValueError("growth utility accepts training observations only")
        proposals, residual = proposals.detach().float(), residual.detach().float()
        self.growth_cross.add_((proposals * residual[:, None]).sum(0))
        self.growth_sum.add_(proposals.sum(0))
        self.growth_square.add_(proposals.square().sum(0))
        self.growth_residual.add_(torch.stack([residual.sum(), residual.square().sum()]))
        self.growth_count.add_(len(residual))


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
        # Decision times are retained solely for post-run provenance. They are
        # not an input to retrieval; maturity remains controlled by ``ends``.
        self.register_buffer("decisions", torch.full((c,), -1, dtype=torch.int64))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def reset(self):
        self.count.zero_()
        self.cursor.zero_()

    @torch.no_grad()
    def append(self, keys, values, ends, assets, decisions=None):
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
        if decisions is None:
            # Direct unit-level bank use has no associated decision clock. A
            # trace marks such synthetic records as unavailable rather than
            # inventing an origin time.
            decisions = torch.full_like(ends, -1)
        if decisions.shape != ends.shape or decisions.dtype != torch.int64:
            raise ValueError("memory decision times differ from the record contract")
        n, capacity = len(keys), self.config.memory_capacity
        if not n:
            return
        # Retain exactly the last capacity records, matching scalar FIFO wrap semantics.
        keep = min(n, capacity)
        slots = (torch.arange(n - keep, n, device=self.cursor.device) + self.cursor) % capacity
        self.keys.index_copy_(0, slots, F.normalize(keys[-keep:].float(), dim=1))
        self.values.index_copy_(0, slots, values[-keep:])
        self.ends.index_copy_(0, slots, ends[-keep:])
        self.assets.index_copy_(0, slots, assets[-keep:])
        self.decisions.index_copy_(0, slots, decisions[-keep:])
        self.cursor.copy_((self.cursor + n) % capacity)
        self.count.copy_((self.count + n).clamp_max(capacity))

    def forward(self, keys, times, assets, capture_trace=False):
        if (
            keys.ndim != 2
            or keys.shape[1] != self.keys.shape[1]
            or times.shape != (len(keys),)
            or assets.shape != (len(keys),)
            or times.dtype != torch.int64
            or assets.dtype != torch.int64
            or not torch.isfinite(keys).all()
        ):
            raise ValueError("invalid retrieval query")
        # Graph matmul is autocast-eligible even when its operands are float32.
        # Keep masked logits and moment arithmetic in float32 with a populated bank.
        with torch.autocast(device_type=keys.device.type, enabled=False):
            return self._retrieve(keys.float(), times, assets, capture_trace=capture_trace)

    def _retrieve(self, keys, times, assets, *, capture_trace):
        count = int(self.count)
        if count == 0 or self.config.arm == "no_memory":
            summary = keys.new_zeros(len(keys), 3)
            if not capture_trace:
                return summary
            empty_float = keys.new_empty((len(keys), 0))
            empty_int = times.new_empty((len(keys), 0))
            return summary, {
                "slot": empty_int,
                "eligible": torch.empty((len(keys), 0), dtype=torch.bool, device=keys.device),
                "similarity": empty_float,
                "age_days": empty_float,
                "base_weight": empty_float,
                "graph_weight": empty_float,
                "weight": empty_float,
                "episode_decision": empty_int,
                "episode_maturity": empty_int,
                "episode_asset": empty_int,
                "episode_value": empty_float,
                "memory_mean": summary[:, 0],
                "memory_std": summary[:, 1],
                "memory_coverage": summary[:, 2],
                "retained_count": torch.full(
                    (len(keys),), count, dtype=torch.int64, device=keys.device
                ),
                "graph_applied": torch.zeros(len(keys), dtype=torch.bool, device=keys.device),
            }
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
        base_weight = torch.softmax(top, 1) * selected_valid
        base_weight = base_weight / base_weight.sum(1, keepdim=True).clamp_min(1e-12)
        graph_weight = base_weight
        if self.config.arm != "no_graph":
            # One personalized diffusion step on the query's already mature neighbors.
            nodes = self.keys[:count][indices]
            edges = nodes @ nodes.transpose(1, 2) / self.config.memory_temperature
            pair_valid = selected_valid[:, :, None] & selected_valid[:, None, :]
            graph = edges.masked_fill(~pair_valid, -1e30).softmax(-1) * pair_valid
            graph = graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
            diffused = torch.bmm(base_weight[:, None, :], graph).squeeze(1)
            graph_weight = diffused
            weight = (1 - self.config.graph_mix) * base_weight + self.config.graph_mix * diffused
        else:
            weight = base_weight
        values = torch.where(selected_valid, self.values[:count][indices], 0.0)
        mean = (weight * values).sum(1)
        variance = (weight * (values - mean[:, None]).square()).sum(1)
        summary = torch.stack(
            [
                mean,
                (variance + 1e-8).sqrt() * selected_valid.any(1),
                selected_valid.sum(1) / self.config.neighbors,
            ],
            1,
        )
        if not capture_trace:
            return summary
        selected_similarity = similarity.gather(1, indices)
        selected_age = age.gather(1, indices)
        return summary, {
            "slot": indices,
            "eligible": selected_valid,
            "similarity": selected_similarity,
            "age_days": selected_age,
            "base_weight": base_weight,
            "graph_weight": graph_weight,
            "weight": weight,
            "episode_decision": self.decisions[:count][indices],
            "episode_maturity": self.ends[:count][indices],
            "episode_asset": self.assets[:count][indices],
            "episode_value": values,
            "memory_mean": summary[:, 0],
            "memory_std": summary[:, 1],
            "memory_coverage": summary[:, 2],
            "retained_count": torch.full(
                (len(keys),), count, dtype=torch.int64, device=keys.device
            ),
            "graph_applied": torch.full(
                (len(keys),), self.config.arm != "no_graph", dtype=torch.bool, device=keys.device
            ),
        }


class S14(nn.Module):
    def __init__(self, config, seed=11):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config, seed)
        self.target = deepcopy(self.encoder)
        self.target.requires_grad_(False)
        key_width = config.input_dim * (3 + config.bands) + config.ngram
        self.memory = Engram(config, key_width)
        self.field = StochasticField(config)
        self.mask_token = nn.Parameter(torch.zeros(config.input_dim))
        self.predictor = nn.Sequential(
            nn.Linear(config.latent, config.latent),
            nn.SiLU(),
            nn.Linear(config.latent, config.latent),
        )
        self.expert = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.latent + 4, config.latent),
                    nn.SiLU(),
                    nn.Linear(config.latent, 4),
                )
                for _ in range(config.experts)
            ]
        )
        # Separate gates for return likelihood, downside probability and tail quantile.
        self.router = nn.Linear(config.latent + 4, 3 * config.experts)

    @torch.no_grad()
    def ema_update(self):
        for target, online in zip(self.target.parameters(), self.encoder.parameters(), strict=True):
            target.lerp_(online, 1 - self.config.ema)
        for target, online in zip(self.target.buffers(), self.encoder.buffers(), strict=True):
            target.copy_(online)

    def forward(self, x, times, assets, capture_trace=False):
        if (
            times.shape != (len(x),)
            or assets.shape != (len(x),)
            or times.dtype != torch.int64
            or assets.dtype != torch.int64
        ):
            raise ValueError("decision timestamps and asset IDs must be int64 vectors")
        z, key = self.encoder(x)
        proposals = self.encoder.proposals
        retrieval = self.memory(key.detach(), times, assets, capture_trace=capture_trace)
        if capture_trace:
            memory, retrieval_trace = retrieval
        else:
            memory = retrieval
        if self.config.arm == "no_spde":
            advanced, field_variance = z, z.new_zeros(self.config.latent)
        else:
            advanced, field_variance = self.field(z)
        uncertainty = field_variance.mean().sqrt().expand(len(z), 1)
        fused = torch.cat([advanced.to(z.dtype), memory, uncertainty.to(z.dtype)], 1)
        context_z = z
        if self.training and self.config.arm not in ("no_jepa", "no_mask"):
            width = max(1, int(self.config.context * self.config.mask_fraction))
            starts = torch.randint(self.config.context - width + 1, (len(x), 1), device=x.device)
            positions = torch.arange(self.config.context, device=x.device)[None]
            mask = ((positions >= starts) & (positions < starts + width))[:, :, None]
            masked = torch.where(mask, self.mask_token[None, None].to(x.dtype), x)
            context_z, _ = self.encoder(masked)
        predictions, variances = [], []
        offsets = (
            self.config.target_offsets if self.training and self.config.arm != "no_jepa" else ()
        )
        for offset in offsets:
            if self.config.arm == "no_spde":
                predicted = self.predictor(context_z).float()
                var = torch.ones(self.config.latent, device=z.device)
            else:
                predicted, var = self.field(context_z, offset / self.config.horizon)
            predictions.append(predicted)
            variances.append(var)
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
            * (
                scale.square() * self.config.student_df / (self.config.student_df - 2)
                + (mu - mean[:, None]).square()
            )
        ).sum(-1)
        output = {
            "z": z.float(),
            "growth_proposals": proposals,
            "predicted": torch.stack(predictions, 1) if predictions else None,
            "latent_variance": torch.stack(variances) if variances else None,
            "masked_z": context_z.float(),
            "mu": mu,
            "scale": scale,
            "weights": weights,
            "mean": mean,
            "std": variance.clamp_min(1e-8).sqrt(),
            "p_loss": (weights[:, 1] * p_loss).sum(-1),
            "tail": (weights[:, 2] * tail).sum(-1),
            "memory": memory.float(),
        }
        if capture_trace:
            output["retrieval_trace"] = {
                name: value.detach() for name, value in retrieval_trace.items()
            }
        return output

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


def objective(model, output, y, future, run_config, collect_metrics=True):
    predictive = nll(output, y, model.config.student_df).mean()
    risk = (output["p_loss"] - (y < 0).float()).square().mean()
    residual = y - output["tail"]
    risk = risk + torch.maximum(0.05 * residual, -0.95 * residual).mean()
    jepa = predictive.new_zeros(())
    if model.config.arm != "no_jepa":
        if future.ndim != 4 or future.shape[1] != len(model.config.target_offsets):
            raise ValueError("JEPA requires one future context per registered offset")
        with torch.no_grad():
            target, _ = model.target(future.flatten(0, 1))
            target = target.reshape(len(future), future.shape[1], -1).float()
        z = output["z"]
        offsets = [-1] if model.config.arm == "no_multihorizon" else range(future.shape[1])
        terms = []
        for index in offsets:
            mean = output["predicted"][:, index]
            if model.config.arm == "no_spde":
                term = F.mse_loss(mean, target[:, index])
            else:
                term = model.field.latent_nll(
                    target[:, index], mean, output["latent_variance"][index]
                )
            terms.append(term + (1 - F.cosine_similarity(mean, target[:, index], dim=1)).mean())
        jepa = torch.stack(terms).mean()
        if len(z) > 1:
            for embedding in (z, output["masked_z"]):
                centered = embedding - embedding.mean(0)
                covariance = centered.T @ centered / (len(z) - 1)
                variance = F.relu(1 - torch.sqrt(embedding.var(0, correction=1) + 1e-4)).mean()
                off = covariance - torch.diag_embed(covariance.diagonal())
                jepa = jepa + 0.5 * (variance + off.square().sum() / z.shape[1])
    balance = (output["weights"].mean(0) - 1 / model.config.experts).square().mean()
    total = predictive + run_config.jepa_weight * jepa + run_config.balance_weight * balance
    if model.config.arm != "no_risk":
        total = total + run_config.risk_weight * risk
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite objective")
    if not collect_metrics:
        return total, {}
    return total, {
        "nll": float(predictive.detach()),
        "jepa": float(jepa.detach()),
        "risk": float(risk.detach()),
        "balance": float(balance.detach()),
    }
