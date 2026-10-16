"""Explicit immutable forecast configuration for the declared S10 arms."""

from __future__ import annotations

from dataclasses import dataclass, replace

from frem_s10.errors import ContractError


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    library_minimum: int = 64
    library_window: int = 520
    neighbors: int = 32
    concentration_floor: int = 16
    half_life_weeks: float = 52.0
    feedback_window: int = 104
    global_minimum: int = 26
    l2_coefficient: float = 0.05
    shrink_pseudocount: float = 64.0
    maximum_blend: float = 0.5
    no_shrink: bool = False
    repeat_h1: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        fixed = (self.library_minimum == 64, self.library_window in (260, 520),
                 (self.neighbors, self.concentration_floor) in ((16, 8), (32, 16), (64, 32)),
                 self.half_life_weeks == 52.0, self.feedback_window == 104,
                 self.global_minimum == 26, self.l2_coefficient == 0.05,
                 self.shrink_pseudocount == 64.0, self.maximum_blend == 0.5,
                 isinstance(self.no_shrink, bool), isinstance(self.repeat_h1, bool))
        if not all(fixed):
            raise ContractError("forecast configuration differs from declared S10 arms")

    def variant(self, name: str) -> ForecastConfig:
        """Return only one of the four forecast-changing ablations plus no-shrink."""
        if name in ("primary", "contextual_stack", "frozen_state", "force_liquidate", "no_exact_hold"):
            return self
        if name == "repeat_h1":
            return replace(self, repeat_h1=True)
        if name == "window260":
            return replace(self, library_window=260)
        if name == "neighbors16":
            return replace(self, neighbors=16, concentration_floor=8)
        if name == "neighbors64":
            return replace(self, neighbors=64, concentration_floor=32)
        if name == "no_shrink":
            return replace(self, no_shrink=True)
        raise ContractError(f"unknown forecast ablation: {name}")
