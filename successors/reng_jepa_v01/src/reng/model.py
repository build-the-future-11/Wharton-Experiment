"""Paper-faithful RENG-JEPA neural architecture."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .graph import GraphContext, signed_correlation_adjacency
from .memory import EngramBank, RetrievalResult, SpectralEngramRetriever
from .probability import GaussianComponent, mixture_log_prob
from .reservoir import MultiscaleReservoir
from .spectral import CausalSpectralChannels


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        leading = sequence.shape[:-2]
        flattened = sequence.reshape(-1, sequence.shape[-2], sequence.shape[-1])
        _, hidden = self.gru(flattened)
        return self.projection(hidden[-1]).reshape(*leading, -1)


@dataclass
class FeatureState:
    base_context: torch.Tensor
    online_latent: torch.Tensor
    target_latent: torch.Tensor
    spectrum: torch.Tensor
    adjacency: torch.Tensor
    asset_mask: torch.Tensor


@dataclass
class MemberOutput:
    gate_logits: torch.Tensor
    latent_mean: torch.Tensor
    latent_diagonal: torch.Tensor
    latent_factors: torch.Tensor
    return_mean: torch.Tensor
    return_diagonal: torch.Tensor
    return_factors: torch.Tensor
    growth_mean: torch.Tensor
    target_decoded_return: torch.Tensor
    memory_entropy: torch.Tensor
    memory_missing: torch.Tensor

    def return_components(self, batch_index: int) -> list[GaussianComponent]:
        return [
            GaussianComponent(
                self.return_mean[batch_index, branch],
                self.return_diagonal[batch_index, branch],
                self.return_factors[batch_index, branch],
            )
            for branch in range(self.return_mean.shape[1])
        ]


@dataclass
class StageAOutput:
    prediction: torch.Tensor
    auxiliary_return: torch.Tensor
    features: FeatureState


class RENGFeatureBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = SequenceEncoder(
            config.input_channels, config.encoder_hidden, config.latent_dim
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.spectral = CausalSpectralChannels(
            config.context_length, config.spectral_bins, config.latent_dim
        )
        self.reservoir = MultiscaleReservoir(config.input_channels, config.reservoir_dim)
        self.graph = GraphContext(config.latent_dim)
        self.horizon_embedding = nn.Embedding(len(config.horizons), config.horizon_dim)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must lie in [0, 1]")
        for target, online in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)

    def forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        asset_mask: torch.Tensor,
        horizon: int,
    ) -> FeatureState:
        if context.ndim != 4 or context.shape[-2:] != (self.config.context_length, 18):
            raise ValueError("context must be (batch, assets, context_length, 18)")
        if future.ndim != 4 or future.shape[-2:] != (horizon, 18):
            raise ValueError("future block shape does not match horizon")
        if context.shape[:2] != asset_mask.shape or future.shape[:2] != asset_mask.shape:
            raise ValueError("Asset masks do not align")
        try:
            horizon_index = self.config.horizons.index(horizon)
        except ValueError as exc:
            raise ValueError(f"Unknown horizon {horizon}") from exc
        online = self.context_encoder(context)
        with torch.no_grad():
            target = self.target_encoder(future)
        spectral, complex_spectrum = self.spectral(context[..., 0])
        reservoirs = self.reservoir(context)
        observed = context[..., 6] > 0.5
        lookback = min(self.config.graph_lookback, context.shape[-2])
        adjacency = signed_correlation_adjacency(
            context[..., -lookback:, 0],
            observed[..., -lookback:],
            min_pairs=min(self.config.graph_min_pairs, lookback),
            threshold=self.config.graph_threshold,
            scale=self.config.graph_scale,
        )
        graph = self.graph(online, adjacency)
        denominator = asset_mask.sum(dim=1, keepdim=True).clamp_min(1).to(online.dtype)
        global_context = (online * asset_mask.unsqueeze(-1)).sum(
            dim=1, keepdim=True
        ) / denominator.unsqueeze(-1)
        global_context = global_context.expand_as(online)
        horizon_vector = self.horizon_embedding.weight[horizon_index]
        horizon_vector = horizon_vector.view(1, 1, -1).expand(*online.shape[:2], -1)
        branches = []
        for branch in range(3):
            if self.config.use_reservoir:
                reservoir = reservoirs[..., branch, :]
            else:
                reservoir = torch.zeros_like(reservoirs[..., branch, :])
            if self.config.use_fourier_features:
                frequency = spectral[..., branch, :]
            else:
                frequency = torch.zeros_like(spectral[..., branch, :])
            branches.append(
                torch.cat(
                    [online, reservoir, frequency, graph, global_context, horizon_vector], dim=-1
                )
            )
        base = torch.stack(branches, dim=2)
        if base.shape[-1] != self.config.base_context_dim:
            raise AssertionError("Base interpreter dimension drift")
        return FeatureState(base, online, target.detach(), complex_spectrum, adjacency, asset_mask)


class StageAHeads(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        merged = config.base_context_dim * 3
        self.predictor = nn.Sequential(nn.Linear(merged, 128), nn.GELU(), nn.Linear(128, 32))
        self.auxiliary = nn.Sequential(nn.Linear(merged, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, base_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        merged = base_context.flatten(start_dim=-2)
        return self.predictor(merged), self.auxiliary(merged).squeeze(-1)


class EnsembleMember(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        coefficient = 0.5 if config.use_spectral_kernel else 0.0
        self.retrievers = nn.ModuleList(
            [
                SpectralEngramRetriever(branch, spectral_coefficient=coefficient)
                for branch in range(3)
            ]
        )
        self.interpreter_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.interpreter_dim, 128),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(128, 128),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(128, 197),
                )
                for _ in range(3)
            ]
        )
        self.gate = nn.Sequential(nn.Linear(606, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.growth = nn.Sequential(
            nn.Linear(config.interpreter_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.decoder = nn.Parameter(torch.empty(len(config.horizons), 32))
        nn.init.xavier_uniform_(self.decoder)
        self.decoder_bias = nn.Parameter(torch.zeros(len(config.horizons)))

    @torch.no_grad()
    def project_decoder(self) -> None:
        norm = torch.linalg.vector_norm(self.decoder, dim=-1, keepdim=True)
        self.decoder.mul_(torch.clamp(5.0 / norm.clamp_min(1e-12), max=1.0))

    def _retrieve(
        self,
        features: FeatureState,
        bank: EngramBank | None,
        asset_ids: tuple[str, ...],
        decision_ns: torch.Tensor,
        horizon: int,
        permute_seed: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, assets = features.base_context.shape[:2]
        memories = features.base_context.new_zeros(batch, assets, 3, 32)
        entropy = features.base_context.new_zeros(batch, assets, 3)
        missing = torch.ones(
            batch, assets, 3, dtype=torch.bool, device=features.base_context.device
        )
        if bank is None or not self.config.use_memory:
            return memories, entropy, missing
        for b in range(batch):
            query_ns = int(decision_ns[b])
            for asset_index, asset_id in enumerate(asset_ids):
                for branch, retriever in enumerate(self.retrievers):
                    records = bank.eligible(asset_id, horizon, branch, query_ns)
                    override = None
                    if permute_seed is not None:
                        override = bank.permuted_values(
                            asset_id, horizon, branch, query_ns, permute_seed + b + asset_index
                        )
                    result: RetrievalResult = retriever(
                        features.base_context[b, asset_index, branch],
                        features.spectrum[b, asset_index],
                        records,
                        query_ns=query_ns,
                        min_records=self.config.memory_min_records,
                        override_values=override,
                    )
                    memories[b, asset_index, branch] = result.value
                    entropy[b, asset_index, branch] = result.entropy
                    missing[b, asset_index, branch] = result.missing
        return memories, entropy, missing

    def forward(
        self,
        features: FeatureState,
        scales: torch.Tensor,
        asset_ids: tuple[str, ...],
        decision_ns: torch.Tensor,
        horizon: int,
        bank: EngramBank | None = None,
        *,
        permute_memory_seed: int | None = None,
    ) -> MemberOutput:
        memory, entropy, missing = self._retrieve(
            features, bank, asset_ids, decision_ns, horizon, permute_memory_seed
        )
        interpreter_input = torch.cat([features.base_context, memory], dim=-1)
        raw = torch.stack(
            [
                self.interpreter_heads[branch](interpreter_input[:, :, branch])
                for branch in range(3)
            ],
            dim=1,
        )
        cursor = 0
        latent_mean = 6.0 * torch.tanh(raw[..., cursor : cursor + 32] / 6.0)
        cursor += 32
        latent_std = 0.05 + 2.95 * torch.sigmoid(raw[..., cursor : cursor + 32])
        cursor += 32
        latent_factors = 0.5 * torch.tanh(raw[..., cursor : cursor + 128]).reshape(
            *raw.shape[:3], 32, 4
        )
        cursor += 128
        residual_loading = 0.5 * torch.tanh(raw[..., cursor : cursor + 4])
        cursor += 4
        residual_std = 0.02 + 1.98 * torch.sigmoid(raw[..., cursor])
        decoder_index = self.config.horizons.index(horizon)
        decoder = self.decoder[decoder_index]
        return_mean = (
            torch.einsum("d,bjad->bja", decoder, latent_mean) + self.decoder_bias[decoder_index]
        )
        return_mean = return_mean * scales.unsqueeze(1)
        projected_diag = torch.einsum("d,bjad->bja", decoder.square(), latent_std.square())
        return_diagonal = (projected_diag + residual_std.square()) * scales.unsqueeze(1).square()
        projected_factor = torch.einsum("d,bjadk->bjak", decoder, latent_factors)
        if not self.config.use_cross_asset_factors:
            projected_factor = torch.zeros_like(projected_factor)
            residual_loading = torch.zeros_like(residual_loading)
        return_factors = torch.cat([projected_factor, residual_loading], dim=-1) * scales.unsqueeze(
            1
        ).unsqueeze(-1)
        mask = features.asset_mask.to(interpreter_input.dtype)
        branch_summaries = []
        for branch in range(3):
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_input = (interpreter_input[:, :, branch] * mask.unsqueeze(-1)).sum(
                dim=1
            ) / denominator
            mean_variance = (latent_std[:, branch].square().mean(dim=-1) * mask).sum(
                dim=1, keepdim=True
            ) / denominator
            mean_entropy = (entropy[:, :, branch] * mask).sum(dim=1, keepdim=True) / denominator
            branch_summaries.append(torch.cat([mean_input, mean_variance, mean_entropy], dim=-1))
        gate_logits = self.gate(torch.cat(branch_summaries, dim=-1))
        if self.config.uniform_gate:
            gate_logits = torch.zeros_like(gate_logits)
        growth_input = interpreter_input.flatten(start_dim=2)
        growth_mean = scales * self.growth(growth_input).squeeze(-1)
        target_decoded_return = (
            torch.einsum("d,bad->ba", decoder, features.target_latent)
            + self.decoder_bias[decoder_index]
        ) * scales
        return MemberOutput(
            gate_logits=gate_logits,
            latent_mean=latent_mean,
            latent_diagonal=latent_std.square(),
            latent_factors=latent_factors,
            return_mean=return_mean,
            return_diagonal=return_diagonal,
            return_factors=return_factors,
            growth_mean=growth_mean,
            target_decoded_return=target_decoded_return,
            memory_entropy=entropy,
            memory_missing=missing,
        )


class RENGJEPA(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.backbone = RENGFeatureBackbone(config)
        self.stage_a_heads = StageAHeads(config)
        self.members = nn.ModuleList(
            [EnsembleMember(config) for _ in range(config.ensemble_members)]
        )

    def stage_a(
        self, context: torch.Tensor, future: torch.Tensor, asset_mask: torch.Tensor, horizon: int
    ) -> StageAOutput:
        features = self.backbone(context, future, asset_mask, horizon)
        prediction, auxiliary = self.stage_a_heads(features.base_context)
        return StageAOutput(prediction, auxiliary, features)

    def forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        asset_mask: torch.Tensor,
        scales: torch.Tensor,
        asset_ids: tuple[str, ...],
        decision_ns: torch.Tensor,
        horizon: int,
        banks: tuple[EngramBank | None, ...] | None = None,
    ) -> tuple[MemberOutput, ...]:
        features = self.backbone(context, future, asset_mask, horizon)
        if banks is None:
            banks = (None,) * len(self.members)
        if len(banks) != len(self.members):
            raise ValueError("One memory bank is required per ensemble member")
        return tuple(
            member(features, scales, asset_ids, decision_ns, horizon, bank)
            for member, bank in zip(self.members, banks, strict=True)
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def vicreg_penalty(latent: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected = latent[mask]
    if len(selected) < 2:
        raise ValueError("VICReg moments require at least two valid examples")
    variance = selected.var(dim=0, unbiased=False)
    variance_penalty = (1.0 - torch.sqrt(variance + 1e-4)).clamp_min(0).mean()
    centered = selected - selected.mean(dim=0)
    covariance = centered.T @ centered / max(len(selected) - 1, 1)
    covariance = covariance - torch.diag(torch.diagonal(covariance))
    covariance_penalty = covariance.square().sum() / latent.shape[-1]
    return variance_penalty, covariance_penalty


def stage_a_loss(
    output: StageAOutput,
    standardized_simple_return: torch.Tensor,
    *,
    use_jepa: bool = True,
) -> dict[str, torch.Tensor]:
    mask = output.features.asset_mask
    prediction = (
        F.mse_loss(output.prediction[mask], output.features.target_latent[mask])
        if use_jepa
        else output.prediction.sum() * 0
    )
    variance, covariance = vicreg_penalty(output.features.online_latent, mask)
    auxiliary = F.mse_loss(output.auxiliary_return[mask], standardized_simple_return[mask])
    total = prediction + variance + 0.01 * covariance + 0.1 * auxiliary
    return {
        "total": total,
        "jepa": prediction,
        "variance": variance,
        "covariance": covariance,
        "auxiliary": auxiliary,
    }


def stage_b_loss(
    output: MemberOutput,
    target_latent: torch.Tensor,
    normalized_log_return: torch.Tensor,
    simple_return: torch.Tensor,
    asset_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = target_latent.shape[0]
    latent_terms = []
    return_terms = []
    decode_terms = []
    growth_terms = []
    for b in range(batch):
        valid = asset_mask[b]
        latent_value = target_latent[b, valid].reshape(-1)
        latent_components = [
            GaussianComponent(
                output.latent_mean[b, branch, valid].reshape(-1),
                output.latent_diagonal[b, branch, valid].reshape(-1),
                output.latent_factors[b, branch, valid].reshape(
                    -1, output.latent_factors.shape[-1]
                ),
            )
            for branch in range(3)
        ]
        latent_log_prob, _ = mixture_log_prob(
            latent_value, output.gate_logits[b], latent_components
        )
        latent_terms.append(-latent_log_prob / latent_value.numel())
        return_components = [
            GaussianComponent(
                output.return_mean[b, branch, valid],
                output.return_diagonal[b, branch, valid],
                output.return_factors[b, branch, valid],
            )
            for branch in range(3)
        ]
        return_log_prob, _ = mixture_log_prob(
            normalized_log_return[b, valid], output.gate_logits[b], return_components
        )
        return_terms.append(-return_log_prob / int(valid.sum()))
        decode_terms.append(
            F.mse_loss(output.target_decoded_return[b, valid], normalized_log_return[b, valid])
        )
        growth_terms.append(F.mse_loss(output.growth_mean[b, valid], simple_return[b, valid]))
    latent_nll = torch.stack(latent_terms).mean()
    return_nll = torch.stack(return_terms).mean()
    decode = torch.stack(decode_terms).mean()
    growth = torch.stack(growth_terms).mean()
    total = latent_nll + return_nll + 0.1 * decode + growth
    return {
        "total": total,
        "latent_nll": latent_nll,
        "return_nll": return_nll,
        "decode": decode,
        "growth": growth,
    }
