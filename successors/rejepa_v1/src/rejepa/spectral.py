"""Causal Fourier descriptors for spectral engram keys."""

from __future__ import annotations

import torch
from torch import nn


class CausalFourierBank(nn.Module):
    """Summarize only the supplied trailing context in fixed frequency bands."""

    def __init__(self, features: int, bands: int) -> None:
        super().__init__()
        if features < 1 or bands < 1:
            raise ValueError("features and bands must be positive")
        self.features = features
        self.bands = bands

    @property
    def output_dim(self) -> int:
        return self.features * self.bands

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.features:
            raise ValueError("Expected [batch, time, features]")
        if sequence.shape[1] < 2:
            raise ValueError("At least two trailing observations are required")
        if not torch.isfinite(sequence).all():
            raise ValueError("Fourier input must be finite")
        length = sequence.shape[1]
        centered = sequence - sequence.mean(dim=1, keepdim=True)
        window = torch.hann_window(length, periodic=False, device=sequence.device, dtype=sequence.dtype).view(
            1, length, 1
        )
        spectrum = torch.fft.rfft(centered * window, dim=1)
        power = torch.log1p(spectrum.real.square() + spectrum.imag.square())[:, 1:, :]
        chunks = torch.tensor_split(power, self.bands, dim=1)
        summaries = []
        for chunk in chunks:
            if chunk.shape[1] == 0:
                summaries.append(
                    torch.zeros(
                        sequence.shape[0],
                        self.features,
                        device=sequence.device,
                        dtype=sequence.dtype,
                    )
                )
            else:
                summaries.append(chunk.mean(dim=1))
        return torch.cat(summaries, dim=-1)
