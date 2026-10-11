"""Past-only Fourier amplitude, phase and time-frequency diagnostics."""

from __future__ import annotations

import torch


def amplitude_phase(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence.ndim != 3 or sequence.shape[1] < 4 or not torch.isfinite(sequence).all():
        raise ValueError("Expected finite [batch, time, features] with at least four steps")
    centered = sequence - sequence.mean(dim=1, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=1)[:, 1:, :]
    return spectrum.abs(), torch.angle(spectrum)


def causal_time_frequency_energy(sequence: torch.Tensor, window: int, step: int) -> torch.Tensor:
    if window < 4 or step < 1 or sequence.ndim != 3 or sequence.shape[1] < window:
        raise ValueError("Invalid sequence, window or step")
    frames = sequence.unfold(dimension=1, size=window, step=step)
    frames = frames.permute(0, 1, 3, 2)
    centered = frames - frames.mean(dim=2, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=2)
    return (spectrum.real.square() + spectrum.imag.square()).mean(dim=2)
